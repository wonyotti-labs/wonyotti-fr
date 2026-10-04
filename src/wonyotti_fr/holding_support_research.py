from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .holding_support import copy_first_parent, holding_support
from .inventory_labels import verify_files
from .minute_data import prepare_minute_period
from .reports import table


def prepare_holding_support(diagnosis, reference, out):
    files = json.loads((diagnosis / 'files.json').read_text())
    required = {'training_used.parquet', 'calibration_used.parquet', 'models.json', 'offsets.json', 'summary.json'}
    if (not required <= set(files)
        or any(not (diagnosis / n).resolve().is_relative_to(diagnosis.resolve()) for n in files)):
        raise ValueError('보유 시간 한도의 학습 파일·경로 오류')
    verify_files(diagnosis, list(files))
    evidence = json.loads((reference / 'first_admission.json').read_text())
    if (evidence['settings']['diagnosis_files_sha256'] != sha256(diagnosis / 'files.json')
        or json.loads((diagnosis / 'summary.json').read_text()).get('complete') is not True
        or json.loads((diagnosis / 'models.json').read_text())['histogram'] != json.loads((reference / 'history_manager.json').read_text())
        or json.loads((diagnosis / 'offsets.json').read_text())['histogram'] != json.loads((reference / 'history_offset.json').read_text())):
        raise ValueError('보유 시간 한도의 기존 모형·원본 연결 오류')
    columns = ['end', 'entry_time', 'label_end', 'episode_id', 'log_hold_minutes']
    value = holding_support({name: pd.read_parquet(diagnosis / f'{name}_used.parquet', columns=columns)
                             for name in ['training', 'calibration']})
    value['source'] = {'files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V40.md')),
        'row_files_sha256': {n: files[n] for n in ['training_used.parquet', 'calibration_used.parquet']}}
    save_json(out / 'holding_support.json', value)
    return value


def run_holding_support_selection(reference: Path, diagnosis: Path, market: Path, features: Path,
                              confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'first_state_v39':
        raise ValueError('보유 시간 한도 비교에는 고정 v39 모델이 필요합니다.')
    original_risk = EngineConfig(**parent['risk'])
    out = new_run(output, 'holding-support-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'diagnosis': str(diagnosis), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V40.md')), 'candidate_count': 1,
        'control_count': 1, 'development_in_sample': True, 'new_models_fitted': False,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'관찰한 보유 시간 한도의 고정 비교: {out}', flush=True)
    try:
        copy_first_parent(reference, out)
        support = prepare_holding_support(diagnosis, reference, out)
        config = replace(original_risk, max_hold_bars=support['max_hold_bars'])
        frozen = {**parent, 'protocol': 'holding_support_v40', 'risk': config.__dict__,
            'first_selection_sha256': sha256(out / 'first_selection.json'),
            'holding_support_sha256': sha256(out / 'holding_support.json')}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, risk, folder in [('holding_support', policy, config, 'candidate-00'),
                ('previous_v39', original, original_risk, 'control-previous')]:
            metrics = backtest(bars, bot, risk, out / folder)
            action_diagnostics(out / folder, bars, risk)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 보유 시간 대조와 원래 v39의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'holding_support', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 관찰한 보유 시간 한도의 관리 판단 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n기존 v39의 모든 모델·문턱을 보존하고 학습·보정 행의 최대 보유 시간만 실행 한도로 적용했다. '
            '최대 시간은 정수 분으로 내렸으며 다른 위험·비용은 유지했다. 원래 v39의 전체 출력과 상태를 대조했다. '
            '2021년은 적합 진단이며 모든 기간을 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
