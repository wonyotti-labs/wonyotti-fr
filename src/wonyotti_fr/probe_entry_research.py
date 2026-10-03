from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .event_backtest import backtest
from .event_research import load_selection
from .minute_data import prepare_minute_period
from .probe_entry import copy_boosted_parent, probe_risks
from .reports import table


def run_probe_entry_selection(reference: Path, market: Path, features: Path,
                              confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'boosted_direction_v28':
        raise ValueError('최초 노출 분리에는 고정 v28 모델이 필요합니다.')
    config, original_config, matched_config = probe_risks(parent)
    out = new_run(output, 'probe-entry-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V29.md')), 'candidate_count': 1,
        'control_count': 2, 'development_in_sample': True, 'new_models_fitted': False,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'최초 노출과 추가 투입의 고정 비교: {out}', flush=True)
    try:
        copy_boosted_parent(reference, out)
        frozen = {**parent, 'protocol': 'probe_entry_v29', 'risk': config.__dict__,
            'boost_selection_sha256': sha256(out / 'boost_selection.json')}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, risk, folder in [('probe_entry', policy, config, 'candidate-00'),
                ('previous_v28', original, original_config, 'control-previous'),
                ('matched_initial_risk', original, matched_config, 'control-matched')]:
            metrics = backtest(bars, bot, risk, out / folder)
            action_diagnostics(out / folder, bars, risk)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 부스팅 방향 대조와 원래 v28의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'probe_entry', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 최초 노출과 추가 투입의 분리 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n처음 투입하는 수량만 전체 할당의 50%에서 12.5%로 줄였다. 활동·방향·관리·축소·전체 한도와 추가 비율은 유지했다. '
            '원래 v28과 최초 계좌 노출이 같은 비례 축소 대조를 보존했다. 2021년은 적합 진단이며 이후 손익으로 비율을 재선택하지 않았다. '
            '최초 노출 비율은 단일 가설이며 원본 레버리지나 최적 위험의 추정치가 아니다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
