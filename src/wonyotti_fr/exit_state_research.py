from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .exit_state import EXIT_POLICY_FILES, copy_activity_parent, prepare_exit_evidence
from .minute_data import prepare_minute_period
from .reports import table


def run_exit_state_selection(reference: Path, diagnosis: Path, market: Path, features: Path,
                              confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'activity_ablation_v46':
        raise ValueError('직접 청산 비교에는 고정 v46 모델이 필요합니다.')
    original_risk = EngineConfig(**parent['risk'])
    out = new_run(output, 'exit-state-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'diagnosis': str(diagnosis), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V48.md')), 'candidate_count': 1,
        'control_count': 1, 'development_in_sample': True, 'new_models_fitted': False,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'직접 청산 문턱의 자체 상태 실행: {out}', flush=True)
    try:
        copy_activity_parent(reference, out)
        prepare_exit_evidence(reference, diagnosis, out)
        config = original_risk
        frozen = {**parent, 'protocol': 'exit_state_v48',
            'activity_selection_sha256': sha256(out / 'activity_selection.json'),
            'exit_files_sha256': {n: sha256(out / n) for n in EXIT_POLICY_FILES},
            'exit_protocol_sha256': sha256(Path('docs/EXPERIMENT_V48.md'))}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, risk, folder in [('exit_state', policy, config, 'candidate-00'),
                ('previous_v46', original, original_risk, 'control-previous')]:
            metrics = backtest(bars, bot, risk, out / folder)
            action_diagnostics(out / folder, bars, risk)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 청산 배율 대조와 원래 v46의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'exit_state', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 직접 청산 문턱의 자체 상태 실행\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n기존 v46의 모델·진입·관리 문턱을 보존하고 청산 문턱의 배율만 1.0으로 변경했다. '
            '보유 시간 한도·최초 수량·위험·비용을 유지하고 원래 v46 전체 출력과 상태를 대조했다. '
            '2021년은 적합 진단이며 모든 기간을 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
