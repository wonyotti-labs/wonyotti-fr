from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .first_state import FIRST_POLICY_FILES, copy_history_parent, validate_first_admission
from .inventory_labels import verify_files
from .minute_data import prepare_minute_period
from .reports import table


def prepare_first_manager(diagnosis, reference, out):
    files = json.loads((diagnosis / 'files.json').read_text())
    required = {'manifest.json', 'summary.json', 'decision.json', 'metrics.json', 'first_thresholds.json',
                'history_manager.json', 'history_offset.json', 'history_thresholds.json'}
    if (not required <= set(files)
        or any(not (diagnosis / n).resolve().is_relative_to(diagnosis.resolve()) for n in files)):
        raise ValueError('첫 자체 체결 관리의 진단 파일·경로 오류')
    verify_files(diagnosis, list(files))
    evidence = {k: json.loads((diagnosis / f'{k}.json').read_text()) for k in ['summary', 'decision', 'metrics']}
    evidence['settings'] = json.loads((diagnosis / 'manifest.json').read_text())['settings']
    evidence['files_sha256'] = sha256(diagnosis / 'files.json')
    validate_first_admission(evidence)
    if (evidence['settings']['selection_sha256'] != sha256(reference / 'frozen_selection.json')
        or any((diagnosis / n).read_bytes() != (reference / n).read_bytes()
               for n in ['history_manager.json', 'history_offset.json', 'history_thresholds.json'])):
        raise ValueError('첫 자체 체결 관리의 기존 모형·문턱 연결 오류')
    (out / 'first_policy_thresholds.json').write_bytes((diagnosis / 'first_thresholds.json').read_bytes())
    save_json(out / 'first_admission.json', evidence)


def run_first_state_selection(reference: Path, diagnosis: Path, market: Path, features: Path,
                              confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'history_state_v36':
        raise ValueError('첫 관리 문턱 비교에는 고정 v36 모델이 필요합니다.')
    config = EngineConfig(**parent['risk'])
    out = new_run(output, 'first-state-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'diagnosis': str(diagnosis), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V39.md')), 'candidate_count': 1,
        'control_count': 1, 'development_in_sample': True, 'new_models_fitted': False,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'첫 자체 체결 문턱의 판단의 고정 비교: {out}', flush=True)
    try:
        copy_history_parent(reference, out)
        prepare_first_manager(diagnosis, reference, out)
        frozen = {**parent, 'protocol': 'first_state_v39',
            'history_selection_sha256': sha256(out / 'history_selection.json'),
            'first_files_sha256': {n: sha256(out / n) for n in FIRST_POLICY_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, risk, folder in [('first_state', policy, config, 'candidate-00'),
                ('previous_v36', original, config, 'control-previous')]:
            metrics = backtest(bars, bot, risk, out / folder)
            action_diagnostics(out / folder, bars, risk)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 현재 상태 대조와 원래 v36의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'first_state', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 첫 자체 체결 문턱의 관리 판단 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n기존 v36의 모든 모델·위험을 보존하고 첫 추가·축소에만 검증된 첫 문턱을 직접 적용했다. '
            '새 학습이나 문턱 재선택 없이 청산과 반복 행동의 문턱·배율을 보존했다. 원래 v36의 전체 출력과 상태를 대조했다. '
            '2021년은 적합 진단이며 모든 기간을 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
