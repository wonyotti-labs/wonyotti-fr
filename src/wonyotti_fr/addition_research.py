from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .addition_effect import collect_addition_states, paired_addition_outcome
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import iter_events
from .event_research import load_selection
from .minute_data import prepare_minute_period
from .minute_inventory import MinuteInventoryModels


def run_addition_labels(reference: Path, market: Path, features: Path, output: Path) -> Path:
    frozen, policy = load_selection(reference)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('추가 순효과 정답에는 고정 v21 모델이 필요합니다.')
    original = reference / 'candidate-00'
    config = EngineConfig(**frozen['risk'])
    if json.loads((original / 'config.json').read_text()) != frozen['risk']:
        raise ValueError('추가 순효과 정답과 원래 연속 계좌의 위험 설정 불일치')
    out = new_run(output, 'addition-effect-labels', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V22.md')),
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features / 'manifest-5m.json'),
        'training_period': ['2021-01-01', '2022-01-01'], 'development_in_sample': True,
        'baseline_sha256': {n: sha256(original / n) for n in
                            ['config.json', 'equity.parquet', 'fills.parquet', 'trades.parquet', 'final_state.json']}})
    print(f'같은 보유 상태의 추가 순효과 정답: {out}', flush=True)
    try:
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'input_verification.json', checks)
        opportunities, state = collect_addition_states(bars, policy, config, out / 'baseline')
        save_json(out / 'baseline/final_state.json', state)
        for name in ['equity', 'fills', 'trades']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / f'baseline/{name}.parquet'),
                                          pd.read_parquet(original / f'{name}.parquet'), check_exact=True)
        if state != json.loads((original / 'final_state.json').read_text()):
            raise ValueError('추가 기회 수집의 기존 전체 상태 불일치')
        save_json(out / 'opportunity_states.json', opportunities)
        print(f'2021년 전체 출력·상태 일치, 추가 요청 {len(opportunities)}개', flush=True)
        events = list(iter_events(bars))
        original_fills = pd.read_parquet(original / 'fills.parquet').to_dict('records')
        original_trades = pd.read_parquet(original / 'trades.parquet').to_dict('records')
        cutoff = pd.Timestamp('2021-12-31', tz='UTC')
        rows, ledger, matched = [], [], 0
        for index, opportunity in enumerate(opportunities):
            decision = pd.Timestamp(opportunity['decision_time'])
            row = {'opportunity': index, 'decision_time': decision,
                   'position_entry_time': pd.Timestamp(opportunity['position_entry_time']),
                   **dict(zip(MinuteInventoryModels.features, opportunity['features'], strict=True))}
            if decision >= cutoff:
                ledger.append({**row, 'label_status': 'outside_training_boundary', 'label_end': cutoff})
                continue
            outcome, traces = paired_addition_outcome(events, opportunity['start'], opportunity['state'],
                                                       config, policy, cutoff)
            allowed = traces['allow']
            first = opportunity['future_fill_start']
            if original_fills[first:first+len(allowed['fills'])] != allowed['fills']:
                raise ValueError('추가 유지 경로와 기존 포지션의 전체 후속 체결 불일치')
            if allowed['status'] == 'closed':
                if allowed['closed_trades'] != [original_trades[opportunity['trade_index']]]:
                    raise ValueError('추가 유지 경로와 기존 포지션의 종료 손익 불일치')
                matched += 1
            save_json(out / f'pairs/{index:04d}.json', {'opportunity': index, 'outcome': outcome, 'branches': traces})
            ledger.append({**row, **outcome})
            if outcome['label_status'] == 'closed':
                rows.append({**row, **outcome})
            if (index + 1) % 5 == 0:
                pd.DataFrame(ledger).to_parquet(out / 'labels_partial.parquet', index=False)
                print(f'추가 순효과 {index+1}/{len(opportunities)}개 처리', flush=True)
        all_rows = pd.DataFrame(ledger)
        training = pd.DataFrame(rows)
        if len(training) and (training.decision_time.ge(training.label_end).any()
                              or training.label_end.ge(cutoff).any()):
            raise ValueError('추가 순효과 정답의 시간 격리 오류')
        training.to_parquet(out / 'training_labels.parquet', index=False)
        all_rows.to_parquet(out / 'opportunity_ledger.parquet', index=False)
        summary = {'complete': True, 'opportunities': len(opportunities), 'closed': len(training),
            'position_count': int(training.position_entry_time.nunique()) if len(training) else 0,
            'status_counts': all_rows.label_status.value_counts().to_dict() if len(all_rows) else {},
            'negative_labels': int(training.incremental_bps.lt(0).sum()) if len(training) else 0,
            'zero_labels': int(training.incremental_bps.eq(0).sum()) if len(training) else 0,
            'positive_labels': int(training.incremental_bps.gt(0).sum()) if len(training) else 0,
            'baseline_full_outputs_exact': True, 'allowed_closed_paths_exact': matched,
            'cutoff_exclusive': cutoff, 'losses_removed': False, 'profitability_accepted': False}
        save_json(out / 'summary.json', summary)
        save_json(out / 'files.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
            if p.is_file() and 'code_snapshot' not in p.parts})
        (out / 'REPORT.md').write_text('# 같은 보유 상태에서 추가 주문 하나의 순효과\n\n'
            f'기존 연속 계좌 전체 출력·상태 일치. 추가 요청 {len(opportunities)}개와 확정 {len(training)}개 보존. '
            '기존 손익·위험 상태에서 다음 추가만 유지하거나 취소했다. 이후 두 경로는 고정 관리를 따라 '
            '현재 포지션 종료까지 비용·펀딩을 포함한다. 미래 결과는 학습 목표에만 쓰며 판단 입력은 확정 44개 특징이다. '
            '손실·0효과·경계 미확정을 보존한다. 같은 원래 포지션의 경로 중첩과 관리 모델의 2021년 적합 한계가 남는다.\n')
        print(f'확정 정답 {len(training)}개, {summary["position_count"]}개 원래 포지션', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
