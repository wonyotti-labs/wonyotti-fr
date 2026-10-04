from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .minute_data import prepare_minute_period
from .net_exit_state import copy_horizon_parent
from .reports import table


def run_net_exit_selection(reference: Path, market: Path, features: Path,
                              confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'horizon_exit_v52':
        raise ValueError('현재 비용 청산에는 고정 v52 모델이 필요합니다.')
    original_risk = EngineConfig(**parent['risk'])
    out = new_run(output, 'net-exit-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V53.md')), 'candidate_count': 1,
        'control_count': 1, 'development_in_sample': True, 'new_models_fitted': False, 'new_model_family': False,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'추가 조기 청산의 현재 비용 조건 비교: {out}', flush=True)
    try:
        copy_horizon_parent(reference, out)
        config = original_risk
        frozen = {**parent, 'protocol': 'net_exit_v53',
            'horizon_exit_selection_sha256': sha256(out / 'horizon_exit_selection.json'),
            'net_exit_rule_sha256': sha256(out / 'net_exit_rule.json')}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, risk, folder in [('net_exit', policy, config, 'candidate-00'),
                ('previous_v52', original, original_risk, 'control-previous')]:
            metrics = backtest(bars, bot, risk, out / folder)
            action_diagnostics(out / folder, bars, risk)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 관리 모형 대조와 원래 v52의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'net_exit', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 추가 조기 청산의 현재 비용 조건 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n기존 v52 모델·문턱을 보존하고 추가 조기 청산에 현재 추정 순손익 조건을 적용했다. '
            '보유 시간 한도·최초 수량·위험·비용을 유지하고 원래 v52 전체 출력과 상태를 대조했다. '
            '2021년은 이미 본 원본 내부 적합 진단이다. 모든 평가 기간을 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
