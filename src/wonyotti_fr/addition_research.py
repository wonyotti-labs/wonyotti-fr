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

ADDITION_FILES = ['addition_model.json', 'addition_weights.parquet', 'addition_weighting.json']


def copy_minute_parent(reference, out):
    from .inventory_research import INVENTORY_FILES, PARENT_FILES
    from .minute_inventory_research import copy_previous_inventory
    for name in [*PARENT_FILES, 'rate_selection.json', *INVENTORY_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'minute_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
    copy_previous_inventory(reference / 'previous_inventory', out / 'previous_inventory')


def load_addition_selection(selection, frozen):
    from .addition_model import AdditionEffectModel, AdditionEffectPolicy
    from .minute_inventory_research import load_minute_inventory_selection
    parent_path = selection / 'minute_selection.json'
    if parent_path.stat().st_size > 1024**2 or sha256(parent_path) != frozen['minute_selection_sha256']:
        raise ValueError('추가 순효과 정책의 기반 선택 지문 오류')
    parent = json.loads(parent_path.read_text())
    if (parent.get('protocol') != 'minute_inventory_micro_v21' or frozen['protocol'] != 'addition_effect_v22'
        or frozen.get('addition_training_period') != ['2021-01-01', '2022-01-01']
        or type(frozen.get('addition_margin_bps')) is not int or frozen['addition_margin_bps'] != 0 or frozen.get('addition_weighting') != 'equal_original_position_mean_one_v1'
        or any(frozen.get(key) != value for key, value in parent.items() if key not in {'protocol', 'development_metrics'})
        or set(frozen['addition_files_sha256']) != set(ADDITION_FILES)
        or any((selection / name).stat().st_size > 1024**2
               or sha256(selection / name) != frozen['addition_files_sha256'][name] for name in ADDITION_FILES)):
        raise ValueError('추가 순효과 정책의 고정 설정·모델·가중치 지문 오류')
    _, original = load_minute_inventory_selection(selection, parent)
    effect = AdditionEffectModel.from_dict(json.loads((selection / 'addition_model.json').read_text()))
    return frozen, AdditionEffectPolicy(original, effect)


def addition_diagnostics(directory, bars, policy, config):
    import numpy as np

    from .action_research import action_diagnostics
    from .addition_model import AdditionEffectModel, AdditionEffectPolicy
    details = action_diagnostics(directory, bars, config)
    if isinstance(policy, AdditionEffectPolicy) and policy.enabled:
        records = pd.DataFrame(policy.audit)
        curve = pd.read_parquet(directory / 'equity.parquet', columns=['time', 'policy_event'])
        recorded = curve[curve.policy_event.isin(['action_increase', 'action_add_rejected'])]
        if len(records) != len(recorded):
            raise ValueError('추가 순효과 판단과 실행 사건의 건수 불일치')
        if len(records):
            predicted = policy.effect_model.predict(records[AdditionEffectModel.features].to_numpy(dtype=float))
            accepted = np.isfinite(predicted) & (predicted > 0)
            if (not np.allclose(predicted, records.predicted_incremental_bps.to_numpy(dtype=float), atol=1e-10, rtol=0, equal_nan=True)
                or not np.array_equal(accepted, records.accepted)
                or not np.array_equal(accepted, recorded.policy_event.eq('action_increase'))
                or records.decision_time.tolist() != recorded.time.tolist()):
                raise ValueError('추가 순효과 모델의 저장 판단·시각·허용 불일치')
        records.to_parquet(directory / 'addition_decisions.parquet', index=False)
        details['addition_gate'] = {'decisions': len(records), 'accepted': int(records.accepted.sum()) if len(records) else 0,
            'rejected': int((~records.accepted).sum()) if len(records) else 0, 'scores_recomputed': True}
    save_json(directory / 'addition_diagnostics.json', details)
    return details


def run_addition_selection(reference, labels, market, features, confirmation_market, confirmation_features, output):
    from .addition_model import AdditionEffectModel
    from .event_backtest import backtest
    from .inventory_labels import verify_files
    from .reports import table
    parent, original = load_selection(reference)
    metadata = json.loads((labels / 'manifest.json').read_text())['settings']
    verify_files(labels, ['training_labels.parquet', 'opportunity_ledger.parquet', 'summary.json'])
    if (parent['protocol'] != 'minute_inventory_micro_v21'
        or metadata['reference_sha256'] != sha256(reference / 'frozen_selection.json')
        or metadata['market_manifest_sha256'] != sha256(market / 'manifest-1m.json')
        or metadata['feature_manifest_sha256'] != sha256(features / 'manifest-5m.json')
        or metadata['training_period'] != ['2021-01-01', '2022-01-01']
        or json.loads((labels / 'summary.json').read_text())['complete'] is not True):
        raise ValueError('추가 순효과 학습의 모델·시세·정답 지문 오류')
    out = new_run(output, 'addition-effect-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V22.md')), 'candidate_count': 1, 'margin_bps': 0,
        'in_sample_period': ['2021-01-01', '2022-01-01'], 'all_periods_already_observed': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'추가 주문 순효과 학습과 고정 확인: {out}', flush=True)
    try:
        copy_minute_parent(reference, out)
        train = pd.read_parquet(labels / 'training_labels.parquet')
        ledger = pd.read_parquet(labels / 'opportunity_ledger.parquet')
        pd.testing.assert_frame_equal(train.reset_index(drop=True),
            ledger.loc[ledger.label_status.eq('closed'), train.columns].reset_index(drop=True), check_exact=True, check_dtype=False)
        model, weights, support = AdditionEffectModel.fit(train)
        train.to_parquet(out / 'training_used.parquet', index=False)
        train[['decision_time', 'position_entry_time', 'label_end']].assign(weight=weights).to_parquet(out / 'addition_weights.parquet', index=False)
        save_json(out / 'addition_model.json', model.to_dict())
        save_json(out / 'addition_weighting.json', {'algorithm': support['weighting'], 'rows': len(train),
            'positions': support['positions'], 'independent_samples_claimed': False})
        save_json(out / 'training_support.json', support)
        frozen = {**parent, 'protocol': 'addition_effect_v22',
            'minute_selection_sha256': sha256(out / 'minute_selection.json'),
            'addition_files_sha256': {name: sha256(out / name) for name in ADDITION_FILES},
            'addition_margin_bps': 0, 'addition_weighting': support['weighting'],
            'addition_training_period': ['2021-01-01', '2022-01-01']}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        rows = []
        for name, bot, folder in [('addition_effect', policy, 'candidate-00'), ('unfiltered_v21', original, 'control-unfiltered')]:
            metrics = backtest(bars, bot, config, out / folder)
            addition_diagnostics(out / folder, bars, bot, config)
            rows.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', rows)
            print(f'2021 학습 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-unfiltered' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-unfiltered/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('추가 필터 제거와 원래 v21의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **rows[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        addition_diagnostics(out / 'confirmation-2022', bars, policy, config)
        rows.append({'policy': 'addition_effect', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', rows)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 추가 주문 순효과의 고정 확인\n\n' + table(pd.DataFrame(rows)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n2021년 같은 계좌의 추가 유지·취소 차이로 단일 모델을 학습했다. '
            '그해의 재생은 전체 시스템의 학습 적합 진단이다. 정답과 경로 중첩·상태 분포 변화의 한계가 남으며 '
            '이후 시세를 판단 입력에 넣지 않았다. 확인 결과로 모델·문턱을 재선택하지 않았다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
