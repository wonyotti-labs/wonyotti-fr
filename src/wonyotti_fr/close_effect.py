from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig, PolicyDecision, TradingEngine
from .event_backtest import iter_events
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .journal import canonical
from .minute_data import prepare_minute_period, validate_minutes
from .minute_inputs import MINUTE_FEATURES
from .order_history_boost import OrderHistoryBoostModels
from .outcome_journal import OutcomeJournal
from .streaming_backtest import ParquetRows

CLOSE_PERIOD = ['2021-01-01', '2022-01-01']
CLOSE_FEATURES = OrderHistoryBoostModels.features


class IndexedEvents:
    def __init__(self, bars):
        self.time, self.end = bars.time.array, bars.end.array
        names = ['open', 'high', 'low', 'close', 'funding_rate']
        if {'count', 'volume'} <= set(bars):
            names += ['count', 'volume']
        self.columns = {k: bars[k].to_numpy() for k in names}
        self.features = bars[MARKET_FEATURES].to_numpy(dtype=float)
        extra = set(MINUTE_FEATURES) & set(bars)
        if extra and extra != set(MINUTE_FEATURES):
            raise ValueError('청산 비교의 확정 분봉 특징 누락')
        self.minute = bars[MINUTE_FEATURES].to_numpy(dtype=float) if extra else None

    def __len__(self):
        return len(self.time)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('청산 비교의 시세 인덱스 오류')
        result = {k: float(v[index]) for k, v in self.columns.items()}
        result.update(time=self.time[index].isoformat(), end=self.end[index].isoformat(),
            features=[float(v) if np.isfinite(v) else None for v in self.features[index]])
        if self.minute is not None:
            result['minute_features'] = [float(v) if np.isfinite(v) else None for v in self.minute[index]]
        return result


def immediate_close(events, start, state, config, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if (config.bar_seconds != 60 or config.signal_delay_bars != 0 or not state['quantity']
        or state['completed'] or state.get('deferred_intents') or type(start) is not int
        or not 0 <= start < len(events) or events[start]['time'] != state['last_end']
        or cutoff.tzinfo is None or cutoff.utcoffset().total_seconds() or cutoff != cutoff.floor('min')):
        raise ValueError('청산 반사실의 계좌·지연·시간 오류')
    engine = TradingEngine(config, state)
    engine.state['pending'] = 'exit'
    engine.state.pop('pending_reduction_fraction', None)
    fills, funding, residual, processed = [], 0., 0., 0
    status, end = 'right_censored', cutoff
    for index in range(start, len(events)):
        event = events[index]
        if pd.Timestamp(event['end']) >= cutoff:
            break
        funding += engine.state['quantity']*event['open']*event['funding_rate']
        result = engine.step(event, lambda bar, view: PolicyDecision('exit' if view['direction'] else 'hold', {}, 'label_close'))
        if result['fills'] and (event.get('count', 1) <= 0 or event.get('volume', 1) <= 0):
            raise ValueError('청산 반사실의 무거래 체결 오류')
        fills.extend(result['fills'])
        processed += 1
        residual = max(residual, abs(result['accounting_residual']))
        if result['closed_trades']:
            if (len(result['closed_trades']) != 1 or engine.state['quantity']
                or result['closed_trades'][0]['entry_time'] != state['active_trade']['entry_time']):
                raise ValueError('청산 반사실의 포지션 종료 연결 오류')
            status, end = 'closed', pd.Timestamp(event['end'])
            break
    expected = state['cash']-sum(f['delta_quantity']*f['price']+f['fee'] for f in fills)-funding
    if (len(fills) > 1 or not np.isclose(expected, engine.state['cash'], rtol=0, atol=1e-7)
        or not np.isclose(funding, engine.state['total_funding']-state['total_funding'], rtol=0, atol=1e-7)
        or any(not np.isclose(f['fee'], abs(f['delta_quantity'])*f['price']*config.fee_bps/10000, rtol=0, atol=1e-9) for f in fills)):
        raise ValueError('청산 반사실의 독립 현금 흐름 불일치')
    if status == 'closed' and (len(fills) != 1 or abs(fills[0]['delta_quantity']+state['quantity']) > 1e-10):
        raise ValueError('청산 반사실의 잔량·체결 수량 오류')
    return {'status': status, 'label_end': end, 'final_cash': engine.state['cash'],
        'funding_cost': funding, 'fills': fills, 'observed_minutes': processed,
        'remaining_quantity': engine.state['quantity'], 'max_accounting_residual': residual}


def close_effect_label(opportunity, trace, trade, config, cutoff):
    state = opportunity['state']
    equity = state['cash']+state['quantity']*state['last_close']
    if (not np.isfinite(equity) or equity <= 0 or trade['entry_time'] != state['active_trade']['entry_time']
        or trade['direction'] != int(np.sign(state['quantity']))):
        raise ValueError('청산 순효과의 현재 포지션·순자산 오류')
    # 봉 안 손절은 봉 종료 시각, 시가 청산은 다음 분 경계까지 정답에 포함한다.
    natural_end = pd.Timestamp(trade['exit_time'])
    if trade['exit_reason'] not in {'intrabar_stop', 'end_of_test'}:
        natural_end += pd.Timedelta(minutes=1)
    decision = pd.Timestamp(opportunity['decision_time'])
    if natural_end <= decision:
        raise ValueError('청산 순효과의 과거 종료 연결 오류')
    closed = trace['status'] == 'closed' and natural_end < cutoff and trade['exit_reason'] != 'end_of_test'
    continued_cash = config.initial_equity+state['closed_net']+trade['net_pnl'] if closed else None
    difference = trace['final_cash']-continued_cash if closed else None
    return {'label_status': 'closed' if closed else 'right_censored',
        'label_end': max(pd.Timestamp(trace['label_end']), natural_end) if closed else cutoff,
        'continue_end': natural_end, 'continue_exit_reason': trade['exit_reason'],
        'decision_equity': float(equity), 'close_cash': trace['final_cash'] if trace['status'] == 'closed' else None,
        'continue_cash': continued_cash, 'close_advantage_pnl': difference,
        'close_advantage_bps': difference/equity*10000 if closed else None}


def validate_close_record(opportunity, record, trade, config, cutoff, events):
    if canonical(record['opportunity']) != canonical(opportunity):
        raise ValueError('청산 기회의 저장 상태·입력 불일치')
    start, state = opportunity['start'], opportunity['state']
    values = [opportunity[k] for k in CLOSE_FEATURES]
    if any(v is None for v in values):
        expected = {'label_status': 'unavailable_features'}
    elif start == len(events) or pd.Timestamp(events[start]['end']) >= cutoff:
        expected = {'label_status': 'outside_training_boundary'}
    else:
        trace = record['close']
        count, fills = trace['observed_minutes'], trace['fills']
        if (type(count) is not int or not 0 < count <= len(events)-start
            or trace['status'] not in {'closed', 'right_censored'}
            or not np.isfinite(trace['max_accounting_residual']) or not 0 <= trace['max_accounting_residual'] < 1e-7
            or len(fills) != int(trace['status'] == 'closed')):
            raise ValueError('청산 반사실의 저장 종료·지원 오류')
        last = events[start+count-1]
        funding = sum(state['quantity']*events[i]['open']*events[i]['funding_rate'] for i in range(start, start+count))
        if pd.Timestamp(last['end']) >= cutoff:
            raise ValueError('청산 반사실의 정답 경계 초과')
        if fills:
            fill = fills[0]
            price = last['open']*(1-np.sign(state['quantity'])*config.slippage_bps/10000)
            fee = abs(state['quantity'])*price*config.fee_bps/10000
            if (fill['time'] != last['time'] or last.get('count', 1) <= 0 or last.get('volume', 1) <= 0
                or pd.Timestamp(trace['label_end']) != pd.Timestamp(last['end'])
                or not np.allclose([fill['delta_quantity'], fill['price'], fill['fee'], trace['remaining_quantity']],
                                   [-state['quantity'], price, fee, 0.], rtol=0, atol=1e-9)):
                raise ValueError('청산 반사실의 저장 체결·비용 오류')
        elif (pd.Timestamp(trace['label_end']) != cutoff or trace['remaining_quantity'] != state['quantity']
              or (start+count < len(events) and pd.Timestamp(events[start+count]['end']) < cutoff)):
            raise ValueError('청산 반사실의 미확정 경계·잔량 오류')
        cash = state['cash']-sum(f['delta_quantity']*f['price']+f['fee'] for f in fills)-funding
        if not np.allclose([cash, funding], [trace['final_cash'], trace['funding_cost']], rtol=0, atol=1e-7):
            raise ValueError('청산 반사실의 저장 현금 흐름 오류')
        expected = close_effect_label(opportunity, trace, trade, config, cutoff)
    if canonical(record['outcome']) != canonical(expected):
        raise ValueError('청산 순효과의 저장 정답 오류')
    if expected['label_status'] in {'outside_training_boundary', 'unavailable_features'} and record['close'] is not None:
        raise ValueError('청산 순효과의 제외 행에 반사실 존재')


def collect_close_effects(bars, policy, config, reference, output, journal, cutoff, max_new=None, *, decision_seconds=300):
    validate_minutes(bars, 1)
    if (config.bar_seconds != 60 or config.signal_delay_bars != 0
        or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        or policy.manager.features != CLOSE_FEATURES or type(decision_seconds) is not int or decision_seconds not in (60, 300)):
        raise ValueError('청산 기회 수집의 시세·특징·지연 오류')
    output.mkdir(parents=True, exist_ok=False)
    writers = {name: ParquetRows(output/f'{name}.parquet', 8192) for name in ['equity', 'trades', 'fills']}
    source_trades = pd.read_parquet(reference/'trades.parquet').to_dict('records')
    prior_net = np.r_[0., np.cumsum([t['net_pnl'] for t in source_trades])]
    events, engine, rows, counts = IndexedEvents(bars), TradingEngine(config), [], Counter()
    captured, ended = [], {}
    original = policy.feature_values

    def record_features(bar, view):
        values = original(bar, view)
        if pd.Timestamp(bar['end']).value % (decision_seconds*10**9) == 0:
            captured.append(np.asarray(values, dtype=float).copy())
        return values

    policy.feature_values = record_features
    policy.prepare(bars)
    computed, processed = 0, 0
    try:
        for index, event in enumerate(iter_events(bars)):
            captured.clear()
            result = engine.step(event, policy, final=index == len(bars)-1)
            processed = index+1
            for trade in result['closed_trades']:
                ended[trade['entry_time']] = result['time']
            for name, key in [('trades', 'closed_trades'), ('fills', 'fills')]:
                for item in result.pop(key):
                    writers[name].append(item)
            result.pop('rejected')
            writers['equity'].append(result)
            if pd.Timestamp(event['end']).value % (decision_seconds*10**9):
                continue
            counts['boundaries'] += 1
            if not captured or not engine.state['quantity']:
                reason = 'flat' if not engine.state['quantity'] else result.get('policy_event', 'risk_or_no_decision')
                counts['excluded_'+reason] += 1
                continue
            if len(captured) != 1 or captured[0].shape != (len(CLOSE_FEATURES),):
                raise ValueError('청산 기회의 실제 관리 입력 호출·차원 오류')
            counts['opportunities'] += 1
            state = engine.snapshot()
            trade_index = state['closed_trades']
            if trade_index >= len(source_trades) or not np.isclose(state['closed_net'], prior_net[trade_index], rtol=0, atol=1e-7):
                raise ValueError('청산 기회의 원래 거래 인덱스 누락')
            opportunity = {'decision_time': event['end'], 'position_entry_time': state['active_trade']['entry_time'],
                'start': index+1, 'trade_index': trade_index, 'original_intent': state['pending'], 'state': state,
                **{k: float(v) if np.isfinite(v) else None for k, v in zip(CLOSE_FEATURES, captured[0], strict=True)}}
            number = len(rows)
            record = journal.read(number, opportunity)
            if record is None:
                if max_new is not None and computed >= max_new:
                    counts['opportunities'] -= 1
                    counts['excluded_processing_limit'] += 1
                    break
                if not np.isfinite(captured[0]).all():
                    trace, outcome = None, {'label_status': 'unavailable_features'}
                elif index+1 == len(events) or pd.Timestamp(events[index+1]['end']) >= cutoff:
                    trace, outcome = None, {'label_status': 'outside_training_boundary'}
                else:
                    trace = immediate_close(events, index+1, state, config, cutoff)
                    outcome = close_effect_label(opportunity, trace, source_trades[trade_index], config, cutoff)
                record = json.loads(canonical({'opportunity': opportunity, 'close': trace, 'outcome': outcome}))
                validate_close_record(opportunity, record, source_trades[trade_index], config, cutoff, events)
                journal.append(number, opportunity, record)
                computed += 1
            validate_close_record(opportunity, record, source_trades[trade_index], config, cutoff, events)
            rows.append({k: v for k, v in opportunity.items() if k != 'state'} | record['outcome'])
            if len(rows) % 1000 == 0:
                print(f'청산 순효과 {len(rows)}개 대조, {processed}/{len(bars)}분 재생', flush=True)
    finally:
        policy.feature_values = original
        for writer in writers.values():
            writer.close()
    complete = processed == len(bars)
    for name in ['equity', 'trades', 'fills']:
        actual = pd.read_parquet(output/f'{name}.parquet')
        expected = pd.read_parquet(reference/f'{name}.parquet')
        if not complete:
            expected = expected.iloc[:len(actual)].reset_index(drop=True)
        if len(actual):
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        elif name == 'equity':
            raise ValueError('청산 비교의 원래 경로 재생 누락')
    if journal.count() != len(rows):
        raise ValueError('청산 기회의 초과·누락 저장 원장')
    if complete and engine.snapshot() != json.loads((reference/'final_state.json').read_text()):
        raise ValueError('청산 비교의 원래 최종 상태 불일치')
    for row in rows:
        if row['position_entry_time'] in ended and 'continue_end' in row:
            if pd.Timestamp(row['continue_end']) != pd.Timestamp(ended[row['position_entry_time']]):
                raise ValueError('청산 비교의 실제 종료 봉 경계 불일치')
    save_json(output/'final_state.json', engine.snapshot())
    save_json(output/'parity.json', {'complete': complete, 'processed_bars': processed,
        'all_replayed_outputs_exact': True, 'final_state_exact': complete, 'counts': dict(counts)})
    return rows, dict(counts), complete


def run_close_effect_labels(reference: Path, market: Path, features: Path, output: Path,
                            *, resume: Path | None = None, max_opportunities: int | None = None,
                            decision_seconds: int = 300, legacy_labels: Path | None = None) -> Path:
    frozen, policy = load_selection(reference)
    if (frozen['protocol'] != 'exit_move_v54' or (max_opportunities is not None and (type(max_opportunities) is not int or max_opportunities < 1))
        or type(decision_seconds) is not int or decision_seconds not in (60, 300)
        or (decision_seconds == 60) != (legacy_labels is not None)):
        raise ValueError('청산 순효과의 고정 부모·처리 한도 오류')
    previous = reference/'candidate-00'
    settings = {'reference': str(reference), 'reference_sha256': sha256(reference/'frozen_selection.json'),
        'reference_outputs_sha256': {n: sha256(previous/n) for n in ['equity.parquet', 'trades.parquet', 'fills.parquet', 'final_state.json']},
        'market': str(market), 'features': str(features), 'market_manifest_sha256': sha256(market/'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features/'manifest-5m.json'), 'period': CLOSE_PERIOD,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V59.md')),
        'implementation_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'new_models_fitted': False, 'forced_boundary_closes': False}
    if decision_seconds == 60:
        settings.update(decision_seconds=60, legacy_labels=str(legacy_labels), legacy_files_sha256=sha256(legacy_labels/'files.json'),
            protocol_sha256=sha256(Path('docs/EXPERIMENT_V74.md')))
    label = 'minute-close-effect-labels' if decision_seconds == 60 else 'close-effect-labels'
    out = resume if resume is not None else new_run(output, label, settings)
    if resume is not None and (json.loads((out/'manifest.json').read_text())['settings'] != settings
        or ((out/'summary.json').exists() and json.loads((out/'summary.json').read_text())['complete'])):
        raise ValueError('청산 순효과 재개의 변경 입력 또는 완료 실행')
    print(f'보유 유지와 청산의 순효과 정답: {out}', flush=True)
    try:
        if decision_seconds == 60:
            from .minute_close_effects import (
                preserve_generation_source,
                validate_legacy_reference,
                verify_legacy_grid,
            )
            preserve_generation_source(out, settings['implementation_sha256'])
            legacy = validate_legacy_reference(legacy_labels, settings)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', *CLOSE_PERIOD, minute_inputs=True)
        if (out/'input_verification.json').exists() and canonical(checks) != canonical(json.loads((out/'input_verification.json').read_text())):
            raise ValueError('청산 순효과의 시세 검증 변경')
        save_json(out/'input_verification.json', checks)
        replay = out/f'replay-{uuid4().hex[:12]}'
        with OutcomeJournal(out/'outcomes.sqlite', settings) as journal:
            rows, counts, complete = collect_close_effects(bars, policy, EngineConfig(**frozen['risk']), previous,
                replay, journal, pd.Timestamp('2021-12-31', tz='UTC'), max_opportunities, decision_seconds=decision_seconds)
            journal.verify()
        frame = pd.DataFrame(rows).reindex(columns=sorted(pd.DataFrame(rows).columns))
        for name in ['decision_time', 'position_entry_time', 'label_end', 'continue_end']:
            if name in frame:
                frame[name] = pd.to_datetime(frame[name], utc=True)
        frame.to_parquet(out/'opportunity_ledger.parquet', index=False)
        if decision_seconds == 60:
            save_json(out/'legacy_parity.json', verify_legacy_grid(frame, legacy, complete))
            if sha256(legacy_labels/'files.json') != settings['legacy_files_sha256']:
                raise ValueError('분별 청산 생성 중 기존 원장 지문 변경')
            if any(sha256(legacy_labels/n) != h for n, h in json.loads((legacy_labels/'files.json').read_text()).items()):
                raise ValueError('분별 청산 생성 중 기존 출력 변경')
            if any(sha256(Path(__file__).parent/n) != h for n, h in settings['implementation_sha256'].items()):
                raise ValueError('분별 청산 생성 중 구현 변경')
        train = frame[frame.label_status.eq('closed')].reset_index(drop=True) if len(frame) else frame.copy()
        train.to_parquet(out/'training_labels.parquet', index=False)
        support = []
        if len(train):
            for kind, groups in [('month', train.groupby(train.decision_time.dt.strftime('%Y-%m'))), ('direction', train.groupby('direction'))]:
                for key, part in groups:
                    support.append({'kind': kind, 'group': str(key), 'rows': len(part), 'positions': part.position_entry_time.nunique(),
                        'mean_close_advantage_bps': part.close_advantage_bps.mean(), 'positive': int(part.close_advantage_bps.gt(0).sum()),
                        'negative': int(part.close_advantage_bps.lt(0).sum()), 'zero': int(part.close_advantage_bps.eq(0).sum())})
        save_json(out/'support.json', support)
        save_json(out/'summary.json', {'complete': complete, 'opportunities': len(frame), 'closed': len(train),
            'statuses': frame.label_status.value_counts().to_dict() if len(frame) else {}, 'counts': counts,
            'replay': str(replay), 'replay_parity_sha256': sha256(replay/'parity.json'), 'profitability_accepted': False,
            'forced_boundary_closes': False, 'new_models_fitted': False})
        files = ['manifest.json', 'input_verification.json', 'outcomes.sqlite',
            'opportunity_ledger.parquet', 'training_labels.parquet', 'support.json', 'summary.json']
        if decision_seconds == 60:
            files.append('legacy_parity.json')
        save_json(out/'files.json', {n: sha256(out/n) for n in files})
        print(f'청산 순효과 {len(train)}/{len(frame)}개 확정, 전체 완료: {complete}', flush=True)
    except Exception as error:
        save_json(out/f'failure-{uuid4().hex[:12]}.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
