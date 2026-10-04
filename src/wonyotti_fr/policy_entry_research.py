from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .label_weighting import WEIGHTING
from .lifecycle_edge import LifecycleNetPolicy
from .lifecycle_edge_research import lifecycle_edge_diagnostics
from .minute_data import prepare_minute_period
from .policy_entry import ENTRY_FILES, copy_exit_move_parent, prepare_policy_entry
from .policy_outcomes import OUTCOME_PERIOD
from .reports import table


def run_policy_entry_selection(reference: Path, labels: Path, market: Path, features: Path,
                               confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'exit_move_v54':
        raise ValueError('현재 관리 진입 필터는 고정 v54 정책이 필요합니다.')
    out = new_run(output, 'policy-entry-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference/'frozen_selection.json'),
        'labels_files_sha256': sha256(labels/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V56.md')),
        'candidate_count': 1, 'control_count': 1, 'in_sample_period': OUTCOME_PERIOD,
        'alpha': 100, 'margin_bps': 8, 'sample_weighting': WEIGHTING, 'all_periods_already_observed': True,
        'input_manifests': {str(p/n): sha256(p/n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'현재 관리 손익 기반 진입 필터: {out}', flush=True)
    try:
        copy_exit_move_parent(reference, out)
        prepare_policy_entry(reference, labels, market, features, out)
        frozen = {**parent, 'protocol': 'policy_entry_v56', 'exit_move_selection_sha256': sha256(out/'exit_move_selection.json'),
            'policy_entry_alpha': 100, 'policy_entry_margin_bps': 8, 'policy_entry_training_period': OUTCOME_PERIOD,
            'policy_entry_sample_weighting': WEIGHTING, 'policy_entry_files_sha256': {n: sha256(out/n) for n in ENTRY_FILES}}
        save_json(out/'frozen_selection.json', frozen)
        save_json(out/'frozen_integrity.json', {'frozen_selection_sha256': sha256(out/'frozen_selection.json')})
        _, policy = load_selection(out)
        config = EngineConfig(**parent['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', *OUTCOME_PERIOD, minute_inputs=True)
        save_json(out/'development_input.json', checks)
        rows = []
        for name, bot, folder in [('policy_entry', policy, 'candidate-00'),
                                 ('previous_v54', LifecycleNetPolicy(original, policy.model, enabled=False), 'control-previous')]:
            metrics = backtest(bars, bot, config, out/folder)
            lifecycle_edge_diagnostics(out/folder, bars, bot, config)
            rows.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out/'development.json', rows)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out/'control-previous'/name),
                                          pd.read_parquet(reference/'candidate-00'/name), check_exact=True)
        if json.loads((out/'control-previous/final_state.json').read_text()) != json.loads((reference/'candidate-00/final_state.json').read_text()):
            raise ValueError('필터 제거 대조와 원래 v54 전체 상태 불일치')
        save_json(out/'baseline_parity.json', {'full_outputs_and_state_exact': True, 'filter_disabled_only': True})
        frozen['development_metrics'] = {'candidate': 0, **rows[0]}
        save_json(out/'frozen_selection.json', frozen)
        save_json(out/'frozen_integrity.json', {'frozen_selection_sha256': sha256(out/'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out/'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out/'confirmation-2022')
        lifecycle_edge_diagnostics(out/'confirmation-2022', bars, policy, config)
        rows.append({'policy': 'policy_entry', 'period': '2022', **metrics})
        save_json(out/'comparison.json', rows)
        save_json(out/'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out/'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out/'REPORT.md').write_text('# 현재 관리 손익 기반 신규 진입 필터\n\n'+table(pd.DataFrame(rows)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            +'\n\n현재 v54 전체 관리 결과의 겹친 정답에 같은 가중 표준화·Ridge alpha 100과 8bp 기준을 적용했다. '
            '손실 정답을 보존했으며 필터를 끈 대조는 원래 v54의 전체 출력·상태와 같았다. '
            '모든 기간은 이미 관찰했고 2021년은 내부 적합 진단이다. 독립 초기 계좌 정답은 연속 계좌 성과가 아니다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
