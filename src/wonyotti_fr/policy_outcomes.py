from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import iter_events
from .event_research import load_selection
from .journal import canonical
from .label_weighting import lifecycle_weights
from .lifecycle_edge import lifecycle_outcome
from .minute_data import prepare_minute_period
from .net_edge_labels import potential_entries
from .outcome_journal import OutcomeJournal
from .pullback_diagnostics import verify_fill_activity

OUTCOME_PERIOD = ['2021-01-01', '2022-01-01']


def validate_outcome(opportunity, record, cutoff):
    outcome, fills = record['outcome'], record['fills']
    status = outcome.get('label_status')
    if status not in {'closed', 'right_censored', 'outside_training_boundary'}:
        raise ValueError('진입 효용의 정답 상태 오류')
    if status == 'outside_training_boundary':
        if pd.Timestamp(opportunity['decision_time'])+pd.Timedelta(minutes=1) < cutoff or fills:
            raise ValueError('진입 효용의 경계 밖 기회 오류')
        return
    end = pd.Timestamp(outcome['label_end'])
    decision = pd.Timestamp(opportunity['decision_time'])
    if end.tzinfo is None or end.utcoffset().total_seconds() or not decision < end <= cutoff:
        raise ValueError('진입 효용의 정답 종료 시각 오류')
    if not np.isfinite(outcome['max_accounting_residual']) or not 0 <= outcome['max_accounting_residual'] < 1e-7:
        raise ValueError('진입 효용의 회계 잔차 오류')
    previous = decision
    quantity = 0.
    for fill in fills:
        stamp = pd.Timestamp(fill['time'])
        if (stamp.tzinfo is None or stamp.utcoffset().total_seconds() or not previous <= stamp <= end
            or not np.isfinite([fill[k] for k in ['delta_quantity', 'price', 'fee']]).all()
            or fill['price'] <= 0 or fill['fee'] < 0):
            raise ValueError('진입 효용의 체결 시간·숫자 오류')
        quantity += fill['delta_quantity']
        if opportunity['order_direction']*quantity < -1e-10:
            raise ValueError('진입 효용의 포지션 반전 오류')
        previous = stamp
    if status == 'closed':
        if (len(fills) < 2 or fills[0]['reason'] != 'entry' or abs(quantity) > 1e-10
            or end >= cutoff or fills[-1]['reason'] == 'end_of_test'
            or pd.Timestamp(outcome['entry_time']) != pd.Timestamp(fills[0]['time'])
            or pd.Timestamp(outcome['exit_time']) != pd.Timestamp(fills[-1]['time'])
            or outcome['exit_reason'] != fills[-1]['reason'] or outcome['fills'] != len(fills)):
            raise ValueError('진입 효용의 자연 종료·체결 연결 오류')
        notional = abs(fills[0]['delta_quantity'])*fills[0]['price']
        if notional <= 0:
            raise ValueError('진입 효용의 최초 명목액 오류')
        fees = sum(f['fee'] for f in fills)
        net = -sum(f['delta_quantity']*f['price']+f['fee'] for f in fills)-outcome['funding_cost']
        expected = [notional, fees, net, net/notional*10000]
        actual = [outcome[k] for k in ['entry_notional', 'fees', 'net_pnl', 'net_bps']]
        if not np.isfinite(expected+actual).all() or not np.allclose(expected, actual, rtol=0, atol=1e-7):
            raise ValueError('진입 효용의 독립 체결 손익 오류')
    elif (end != cutoff or outcome['net_bps'] is not None or not np.isfinite(outcome['remaining_quantity'])
          or abs(outcome['remaining_quantity']-quantity) > 1e-10):
        raise ValueError('진입 효용의 미확정 정답·잔량 오류')


def outcome_frame(records):
    frame = pd.DataFrame(records)
    frame = frame.reindex(columns=sorted(frame.columns))
    for name in ['signal_time', 'decision_time', 'entry_time', 'exit_time', 'label_end']:
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def run_policy_outcomes(reference: Path, market: Path, features: Path, output: Path,
                        *, resume: Path | None = None, max_opportunities: int | None = None) -> Path:
    frozen, policy = load_selection(reference)
    if frozen['protocol'] != 'exit_move_v54' or (max_opportunities is not None
        and (type(max_opportunities) is not int or max_opportunities < 1)):
        raise ValueError('현재 정책 진입 효용의 기반·처리 수 오류')
    settings = {'reference': str(reference), 'reference_sha256': sha256(reference/'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V55.md')), 'period': OUTCOME_PERIOD,
        'market': str(market), 'features': str(features), 'market_manifest_sha256': sha256(market/'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features/'manifest-5m.json'),
        'implementation_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'all_opportunities': True, 'forced_boundary_closes': False, 'new_models_fitted': False}
    if resume is not None:
        out = resume
        if json.loads((out/'manifest.json').read_text())['settings'] != settings:
            raise ValueError('현재 정책 진입 효용 재개의 입력·정책·코드 변경')
        if (out/'summary.json').exists() and json.loads((out/'summary.json').read_text())['complete']:
            raise ValueError('현재 정책 진입 효용은 이미 완료됐습니다.')
    else:
        out = new_run(output, 'policy-outcome-labels', settings)
    print(f'현재 정책의 독립 진입 효용 정답: {out}', flush=True)
    try:
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', *OUTCOME_PERIOD, minute_inputs=True)
        opportunities, counts = potential_entries(bars, policy)
        if resume is None:
            save_json(out/'input_verification.json', checks)
            opportunities.to_parquet(out/'potential_entries.parquet', index=False)
        else:
            if canonical(checks) != canonical(json.loads((out/'input_verification.json').read_text())):
                raise ValueError('현재 정책 진입 효용 재개의 시세 검증 변경')
            pd.testing.assert_frame_equal(opportunities, pd.read_parquet(out/'potential_entries.parquet'), check_exact=True)
        events = list(iter_events(bars))
        cutoff = pd.Timestamp(OUTCOME_PERIOD[1], tz='UTC')-pd.Timedelta(days=1)
        risk = EngineConfig(**frozen['risk'])
        records, computed = [], 0
        identity = {**settings, 'opportunities_sha256': sha256(out/'potential_entries.parquet')}
        with OutcomeJournal(out/'outcomes.sqlite', identity) as journal:
            if journal.count() > len(opportunities):
                raise ValueError('현재 정책 진입 효용의 초과 저장 기회')
            if (out/'summary.json').exists() and journal.count() < json.loads((out/'summary.json').read_text())['processed']:
                raise ValueError('현재 정책 진입 효용의 완료 원장 누락')
            for number, opportunity in enumerate(opportunities.to_dict('records')):
                record = journal.read(number, opportunity)
                if record is None:
                    if max_opportunities is not None and computed >= max_opportunities:
                        break
                    start = opportunity['entry_index']
                    if (type(start) is not int or start < 0 or start > len(events)
                        or (start < len(events) and pd.Timestamp(events[start]['time']) != opportunity['decision_time'])):
                        raise ValueError('현재 정책 진입 효용의 다음 시가 연결 오류')
                    if start == len(events) or pd.Timestamp(events[start]['end']) >= cutoff:
                        outcome, fills = {'label_status': 'outside_training_boundary'}, []
                    else:
                        outcome, fills = lifecycle_outcome(events, start, opportunity['order_direction'], risk, policy, cutoff)
                        stop = bars.end.searchsorted(outcome['label_end'], side='right')
                        verify_fill_activity(pd.DataFrame(fills), bars.iloc[start:stop])
                    record = {'outcome': outcome, 'fills': fills}
                    validate_outcome(opportunity, record, cutoff)
                    journal.append(number, opportunity, record)
                    computed += 1
                validate_outcome(opportunity, record, cutoff)
                records.append({**opportunity, **record['outcome']})
                if (number+1) % 100 == 0:
                    print(f'독립 진입 효용 {number+1}/{len(opportunities)}개 완료', flush=True)
            journal.verify()
        ledger = outcome_frame(records)
        ledger.to_parquet(out/'opportunity_ledger.parquet', index=False)
        closed = ledger.loc[ledger.label_status.eq('closed')].reset_index(drop=True) if len(ledger) else ledger.copy()
        closed.to_parquet(out/'training_labels.parquet', index=False)
        intervals, overlap = pd.DataFrame(), {'rows': 0}
        if len(closed):
            _, intervals, overlap = lifecycle_weights(closed)
        intervals.to_parquet(out/'label_intervals.parquet', index=False)
        save_json(out/'support.json', {'overlap': overlap, 'weights_used_for_fitting': False,
            'directions': {str(k): int(v) for k, v in opportunities.order_direction.value_counts().items()},
            'closed_directions': {str(k): int(v) for k, v in closed.order_direction.value_counts().items()} if len(closed) else {},
            'monthly_states': ledger.assign(month=ledger.decision_time.dt.strftime('%Y-%m')).groupby(
                ['month', 'order_direction', 'label_status']).size().rename('rows').reset_index().to_dict('records') if len(ledger) else []})
        complete = len(records) == len(opportunities)
        summary = {'complete': complete, 'signals': counts, 'opportunities': len(opportunities), 'processed': len(records),
            'computed_this_run': computed, 'closed': len(closed), 'cutoff_exclusive': cutoff,
            'statuses': ledger.label_status.value_counts().to_dict() if len(ledger) else {},
            'losing_labels': int(closed.net_bps.lt(0).sum()) if len(closed) else 0,
            'positive_labels': int(closed.net_bps.gt(0).sum()) if len(closed) else 0,
            'losing_labels_removed': False, 'forced_boundary_closes': 0, 'profitability_accepted': False,
            'limit': '기존 모델에도 사용한 내부 적합 구간의 겹친 독립 계좌이며 연속 계좌 성과가 아님'}
        save_json(out/'summary.json', summary)
        save_json(out/'files.json', {n: sha256(out/n) for n in ['manifest.json', 'input_verification.json',
            'potential_entries.parquet', 'opportunity_ledger.parquet', 'training_labels.parquet',
            'label_intervals.parquet', 'support.json', 'summary.json', 'outcomes.sqlite']})
        print(f'진입 효용 처리 {len(records)}/{len(opportunities)}개, 자연 종료 {len(closed)}개, 전체 완료 {complete}', flush=True)
    except Exception as error:
        save_json(out/f'failure-{uuid4().hex}.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
