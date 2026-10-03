from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .addition_research import copy_minute_parent
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .minute_data import prepare_minute_period
from .minute_inventory import MinuteInventoryModels, MinuteInventoryPolicy
from .minute_inventory_research import load_minute_inventory_selection
from .minute_management import ACTIONS, purged_window
from .rate_policy import calibrate_rates
from .realized_exit import load_realized_exit_labels
from .reports import table

EXIT_MODEL_FILES = ['realized_exit_model.json', 'realized_exit_calibration.json', 'realized_exit_training.json']


def verify_non_exit_models(manager, thresholds, scales, parent):
    one, two = manager.to_dict(), parent.manager.to_dict()
    if any(one[k] != two[k] for k in ['mean', 'scale', 'features', 'kind', 'format', 'actions']):
        raise ValueError('청산 전용 변경의 원래 표준화·입력 불일치')
    for action in ['reduce', 'increase']:
        index = ACTIONS.index(action)
        if (one['coef'][index] != two['coef'][index] or one['intercept'][index] != two['intercept'][index]
            or thresholds[action] != parent.thresholds[action] or scales[action] != parent.scales[action]):
            raise ValueError('청산 외 모델·문턱·빈도 배율 불일치')


def load_realized_exit_selection(selection, frozen):
    path = selection / 'minute_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['minute_selection_sha256']:
        raise ValueError('실제 청산 정책의 원래 관리 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'realized_exit_v24' or parent.get('protocol') != 'minute_inventory_micro_v21'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['exit_files_sha256']) != set(EXIT_MODEL_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['exit_files_sha256'][n]
               for n in EXIT_MODEL_FILES)):
        raise ValueError('실제 청산 정책의 고정 설정·모델 지문 오류')
    _, original = load_minute_inventory_selection(selection, parent)
    manager = MinuteInventoryModels.from_dict(json.loads((selection / 'realized_exit_model.json').read_text()))
    calibration = json.loads((selection / 'realized_exit_calibration.json').read_text())
    metadata = json.loads((selection / 'realized_exit_training.json').read_text())
    if (manager.kind != 'logistic' or calibration['scales'] != frozen['exit_scales']
        or calibration['thresholds'] != frozen['exit_thresholds']
        or metadata['training_period'] != list(TRAINING_PERIODS[1])
        or metadata['calibration_period'] != list(CALIBRATION_PERIODS[1])
        or metadata['non_exit_models_exact'] is not True):
        raise ValueError('실제 청산 정책의 학습·보정 설정 오류')
    verify_non_exit_models(manager, frozen['exit_thresholds'], frozen['exit_scales'], original)
    return frozen, MinuteInventoryPolicy(original, manager, frozen['exit_thresholds'], original.multiplier,
                                         frozen['exit_scales'], original.size_model)


def run_realized_exit_selection(reference: Path, labels: Path, market: Path, features: Path,
                                 confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    meta = json.loads((labels / 'manifest.json').read_text())['settings']
    prior = json.loads((reference / 'manifest.json').read_text())['settings']
    if parent['protocol'] != 'minute_inventory_micro_v21' or meta['minute_files_sha256'] != prior['files_sha256']:
        raise ValueError('실제 청산 비교의 v21·원래 학습 입력 불일치')
    out = new_run(output, 'realized-exit-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V24.md')), 'candidate_count': 1,
        'training_period': TRAINING_PERIODS[1], 'calibration_period': CALIBRATION_PERIODS[1], 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'실제 보유 종료 청산 학습: {out}', flush=True)
    try:
        copy_minute_parent(reference, out)
        frame, sizes = load_realized_exit_labels(labels)
        train = purged_window(frame, *TRAINING_PERIODS[1])
        calibration = purged_window(frame, *CALIBRATION_PERIODS[1])
        for name, rows in [('training_used', train), ('calibration_used', calibration)]:
            old = pd.read_parquet(reference / f'{name}.parquet')
            unchanged = rows.columns.difference(['y_exit', 'exit_count'])
            pd.testing.assert_frame_equal(rows[unchanged].reset_index(drop=True), old[unchanged].reset_index(drop=True), check_exact=True)
            # 청산 정답만 별도 보존하고 같은 입력 전체는 원래 학습 파일 지문으로 연결한다.
            rows[['end', 'y_exit', 'exit_count']].to_parquet(out / f'{name}.parquet', index=False)
        manager, thresholds, support = MinuteInventoryModels.fit(train, calibration, 'logistic')
        rate = {**calibrate_rates(manager, calibration, CALIBRATION_PERIODS[1]), 'thresholds': thresholds}
        verify_non_exit_models(manager, thresholds, rate['scales'], original)
        save_json(out / 'realized_exit_model.json', manager.to_dict())
        save_json(out / 'realized_exit_calibration.json', rate)
        save_json(out / 'realized_exit_training.json', {'management': support, 'training_period': TRAINING_PERIODS[1],
            'calibration_period': CALIBRATION_PERIODS[1], 'non_exit_models_exact': True, 'non_exit_frames_exact': True,
            'original_frames_sha256': {n: sha256(reference / n) for n in ['training_used.parquet', 'calibration_used.parquet']},
            'reduction_model_sha256': sha256(out / 'reduction_model.json')})
        frozen = {**parent, 'protocol': 'realized_exit_v24', 'minute_selection_sha256': sha256(out / 'minute_selection.json'),
            'exit_files_sha256': {n: sha256(out / n) for n in EXIT_MODEL_FILES}, 'exit_scales': rate['scales'], 'exit_thresholds': thresholds}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'관리 {len(train)}개·보정 {len(calibration)}개, 비청산 모델·입력 정확히 일치', flush=True)
        del frame, sizes, train, calibration, old, rows
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('realized_exit', policy, 'candidate-00'), ('previous_v21', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('이전 청산 대조와 원래 v21의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'realized_exit', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 실제 보유 종료 청산의 고정 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n실제 보유 종료 정답으로 청산 모델만 변경했다. 비청산 모델·입력·진입·수량·위험은 원래 v21과 같다. '
            '2021년은 적합 진단이고 모든 기간은 이미 관찰했다. 실제 종료 체결을 판단 시각으로 해석하지 않는다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
