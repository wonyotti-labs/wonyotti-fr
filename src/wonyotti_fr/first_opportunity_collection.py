from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .close_economics import ECONOMIC_FEATURES, current_economic_values
from .close_effect import (
    CLOSE_FEATURES,
    IndexedEvents,
    close_effect_label,
    immediate_close,
    validate_close_record,
)
from .close_flow import FLOW_FEATURES, FLOW_WINDOWS, flow_features
from .common import save_json
from .context_position import CONTEXT_FEATURES, context_features
from .continuation_inputs import PENDING_FEATURES, pending_values
from .engine import TradingEngine
from .event_backtest import iter_events
from .event_features import MARKET_FEATURES, event_features
from .journal import canonical
from .minute_data import validate_minutes
from .streaming_backtest import ParquetRows

EXPANDED_FIRST_FEATURES = CLOSE_FEATURES+ECONOMIC_FEATURES+PENDING_FEATURES+CONTEXT_FEATURES+FLOW_FEATURES


def first_context_lookup(bars):
    validate_minutes(bars, 5)
    original = event_features(bars)
    context, flow = context_features(bars), flow_features(bars)
    table = pd.concat([original[MARKET_FEATURES], context[CONTEXT_FEATURES], flow[list(FLOW_WINDOWS)]], axis=1)
    times = bars.end.astype('datetime64[ns, UTC]').array.asi8
    return {int(time): value for time, value in zip(times, table.to_numpy(float), strict=True)}


def expanded_first_values(captured, state, risk, context):
    values = np.asarray(captured, dtype=float)
    if values.shape != (len(CLOSE_FEATURES),):
        raise ValueError('과거 첫 기회의 실제 관리 특징 차원 오류')
    time = pd.Timestamp(state['last_end']).value
    vector = context.get(time//(300*10**9)*(300*10**9))
    if vector is None or not np.isfinite(values).all() or not np.isfinite(vector).all():
        return None
    if not np.array_equal(vector[:len(MARKET_FEATURES)], values[:len(MARKET_FEATURES)]):
        raise ValueError('과거 첫 기회의 원래 확정 시장 특징 불일치')
    economic = current_economic_values(state, risk)
    pending = pending_values([state['pending']])[0]
    extra = vector[len(MARKET_FEATURES):]
    flow = extra[len(CONTEXT_FEATURES):]
    result = np.r_[values, economic, pending, extra, flow*np.sign(state['quantity'])]
    if len(result) != len(EXPANDED_FIRST_FEATURES) or not np.isfinite(result).all():
        raise ValueError('과거 첫 기회의 추가 특징·현재 숫자 오류')
    return result


def compare_replayed_prefix(actual: Path, expected: Path, complete: bool):
    a, b = pq.ParquetFile(actual), pq.ParquetFile(expected)
    size = a.metadata.num_rows
    if size > b.metadata.num_rows or (complete and size != b.metadata.num_rows):
        raise ValueError('첫 기회 수집 재생의 원래 행 수 불일치')
    if not size:
        return
    if a.schema_arrow.names != b.schema_arrow.names:
        raise ValueError('첫 기회 수집 재생의 원래 열 불일치')
    previous = b.iter_batches(batch_size=8192)
    for batch in a.iter_batches(batch_size=8192):
        prior = next(previous)
        pd.testing.assert_frame_equal(batch.to_pandas(), prior.slice(0, len(batch)).to_pandas(), check_exact=True)


def collect_first_opportunities(bars, policy, config, reference, output, journal, cutoff, context, *, max_new=None):
    validate_minutes(bars, 1)
    if (config.bar_seconds != 60 or config.signal_delay_bars != 0 or policy.manager.features != CLOSE_FEATURES
        or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        or (max_new is not None and (type(max_new) is not int or max_new < 1))):
        raise ValueError('과거 첫 기회 수집의 시세·특징·지연·처리 한도 오류')
    output.mkdir(parents=True, exist_ok=False)
    writers = {name: ParquetRows(output/f'{name}.parquet', 8192) for name in ['equity', 'trades', 'fills', 'management_membership']}
    trades = pd.read_parquet(reference/'trades.parquet').to_dict('records')
    entries = [trade['entry_time'] for trade in trades]
    if len(set(entries)) != len(entries):
        raise ValueError('과거 첫 기회의 기준 거래 중복')
    prior_net = np.r_[0., np.cumsum([trade['net_pnl'] for trade in trades])]
    events, engine = IndexedEvents(bars), TradingEngine(config)
    rows, positions, captured, ended, counts = [], {}, [], {}, Counter()
    original = policy.feature_values

    def record_features(bar, view):
        values = original(bar, view)
        captured.append(np.asarray(values, dtype=float).copy())
        return values

    policy.feature_values = record_features
    policy.prepare(bars)
    computed, processed, saved_count = 0, 0, journal.count()
    try:
        for index, event in enumerate(iter_events(bars)):
            captured.clear()
            result = engine.step(event, policy, final=index == len(bars)-1)
            processed = index+1
            for trade in result['closed_trades']:
                ended[trade['entry_time']] = result['time']
            for name, field in [('trades', 'closed_trades'), ('fills', 'fills')]:
                for item in result.pop(field):
                    writers[name].append(item)
            result.pop('rejected')
            writers['equity'].append(result)
            counts['boundaries'] += 1
            state = engine.state
            if not captured or not state['quantity']:
                counts['excluded_'+('flat' if not state['quantity'] else result.get('policy_event', 'risk_or_no_decision'))] += 1
                continue
            if len(captured) != 1:
                raise ValueError('과거 첫 기회의 관리 입력 호출 수 오류')
            entry = state['active_trade']['entry_time']
            number = state['closed_trades']
            if number >= len(trades) or trades[number]['entry_time'] != entry or not np.isclose(state['closed_net'], prior_net[number], rtol=0, atol=1e-7):
                raise ValueError('과거 첫 기회의 기준 거래·현금 연결 오류')
            values = expanded_first_values(captured[0], state, config, context)
            position = positions.setdefault(entry, {'position_entry_time': entry, 'direction': int(np.sign(state['quantity'])),
                'management_rows': 0, 'available_rows': 0, 'eligible_rows': 0, 'first_available_time': None,
                'reference_equity': None, 'first_eligible_time': None, 'first_opportunity_index': -1})
            available = values is not None
            eligible = available and state['pending'] != 'exit'
            first_available = available and position['first_available_time'] is None
            selected = eligible and position['first_eligible_time'] is None
            if selected and len(rows) >= saved_count and max_new is not None and computed >= max_new:
                counts['excluded_processing_limit'] += 1
                break
            if first_available:
                position['first_available_time'] = event['end']
                position['reference_equity'] = state['cash']+state['quantity']*state['last_close']
            position['management_rows'] += 1
            position['available_rows'] += int(available)
            position['eligible_rows'] += int(eligible)
            membership_index = writers['management_membership'].rows
            writers['management_membership'].append({'decision_time': event['end'], 'position_entry_time': entry,
                'original_intent': state['pending'], 'available': available, 'eligible': eligible,
                'first_available': first_available, 'selected_first': selected, 'opportunity_index': membership_index})
            counts['management_rows'] += 1
            if not selected:
                continue
            position['first_eligible_time'] = event['end']
            position['first_opportunity_index'] = membership_index
            snapshot = engine.snapshot()
            opportunity = {'decision_time': event['end'], 'position_entry_time': entry, 'start': index+1,
                'trade_index': number, 'original_intent': state['pending'], 'state': snapshot,
                'first_available_time': position['first_available_time'], 'reference_equity': position['reference_equity'],
                'opportunity_index': membership_index, **dict(zip(EXPANDED_FIRST_FEATURES, map(float, values), strict=True))}
            sequence = len(rows)
            record = journal.read(sequence, opportunity)
            if record is None:
                if index+1 == len(events) or pd.Timestamp(events[index+1]['end']) >= cutoff:
                    trace, outcome = None, {'label_status': 'outside_training_boundary'}
                else:
                    trace = immediate_close(events, index+1, snapshot, config, cutoff)
                    outcome = close_effect_label(opportunity, trace, trades[number], config, cutoff)
                record = json.loads(canonical({'opportunity': opportunity, 'close': trace, 'outcome': outcome}))
                validate_close_record(opportunity, record, trades[number], config, cutoff, events)
                journal.append(sequence, opportunity, record)
                computed += 1
            validate_close_record(opportunity, record, trades[number], config, cutoff, events)
            row = {key: value for key, value in opportunity.items() if key != 'state'} | record['outcome']
            row['first_target_common_bps'] = (row['close_advantage_pnl']/position['reference_equity']*10000 if row['label_status'] == 'closed' else None)
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f'첫 기회 {len(rows)}개 / {processed}/{len(bars)}분 재생', flush=True)
    finally:
        policy.feature_values = original
        for writer in writers.values():
            writer.close()
    complete = processed == len(bars)
    for name in ['equity', 'trades', 'fills']:
        compare_replayed_prefix(output/f'{name}.parquet', reference/f'{name}.parquet', complete)
    if journal.count() != len(rows):
        raise ValueError('과거 첫 기회의 저널 초과·누락')
    if complete and engine.snapshot() != json.loads((reference/'final_state.json').read_text()):
        raise ValueError('과거 첫 기회의 기준 최종 상태 불일치')
    for row in rows:
        if row['position_entry_time'] in ended and 'continue_end' in row and pd.Timestamp(row['continue_end']) != pd.Timestamp(ended[row['position_entry_time']]):
            raise ValueError('과거 첫 기회의 자연 종료 봉 불일치')
    outcome = {row['position_entry_time']: row for row in rows}
    population = []
    for trade in trades:
        entry = trade['entry_time']
        position = positions.get(entry, {'position_entry_time': entry, 'direction': trade['direction'],
            'management_rows': 0, 'available_rows': 0, 'eligible_rows': 0, 'first_available_time': None,
            'reference_equity': None, 'first_eligible_time': None, 'first_opportunity_index': -1})
        record = outcome.get(entry)
        reason = ('no_management_opportunity' if not position['management_rows'] else
            'unavailable_features_only' if not position['available_rows'] else 'no_eligible_opportunity') if record is None else record['label_status']
        population.append({**position, 'has_eligible_opportunity': record is not None, 'reason': reason,
            'first_target_common_bps': record.get('first_target_common_bps') if record else 0.,
            'natural_exit_time': trade['exit_time'], 'natural_exit_reason': trade['exit_reason']})
    save_json(output/'final_state.json', engine.snapshot())
    save_json(output/'parity.json', {'complete': complete, 'processed_bars': processed, 'all_replayed_outputs_exact': True,
        'final_state_exact': complete, 'counts': dict(counts), 'selected_first': len(rows)})
    return rows, population, dict(counts), complete
