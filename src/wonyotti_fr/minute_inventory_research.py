from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .inventory_labels import sizing_training, verify_files
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .inventory_research import INVENTORY_FILES, PARENT_FILES, fixed_size_control
from .minute_data import prepare_minute_period
from .minute_inventory import (
    CONTEXT_FEATURES,
    MinuteInventoryModels,
    MinuteInventoryPolicy,
    MinuteReductionModel,
    load_minute_inventory_labels,
)
from .minute_management import purged_window
from .rate_policy import calibrate_rates
from .reports import table

PREVIOUS_FILES = [*PARENT_FILES, 'rate_selection.json', *INVENTORY_FILES, 'frozen_selection.json', 'frozen_integrity.json']


def copy_previous_inventory(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in PREVIOUS_FILES:
        (destination / name).write_bytes((source / name).read_bytes())


def load_minute_inventory_selection(selection, frozen):
    from .event_research import load_selection
    previous_path = selection / 'previous_inventory'
    previous_frozen = previous_path / 'frozen_selection.json'
    if previous_frozen.stat().st_size > 1024**2 or sha256(previous_frozen) != frozen['previous_inventory_sha256']:
        raise ValueError('분봉 관리 정책의 이전 수량 모델 지문 오류')
    if json.loads(previous_frozen.read_text()).get('protocol') != 'minute_inventory_recent_v20':
        raise ValueError('분봉 관리 정책의 이전 모델 계층 오류')
    previous, original = load_selection(previous_path)
    excluded = {'protocol', 'development_metrics', 'inventory_files_sha256', 'inventory_scales', 'inventory_thresholds'}
    if (previous['protocol'] != 'minute_inventory_recent_v20' or frozen['protocol'] != 'minute_inventory_micro_v21'
        or any(frozen.get(key) != value for key, value in previous.items() if key not in excluded)
        or any(sha256(selection / name) != sha256(previous_path / name) for name in [*PARENT_FILES, 'rate_selection.json'])):
        raise ValueError('분봉 관리 정책의 고정 기반 설정 오류')
    if (set(frozen['inventory_files_sha256']) != set(INVENTORY_FILES)
        or any((selection / name).stat().st_size > 1024**2
               or sha256(selection / name) != frozen['inventory_files_sha256'][name] for name in INVENTORY_FILES)):
        raise ValueError('분봉 관리 정책의 모델·보정 지문 오류')
    manager = MinuteInventoryModels.from_dict(json.loads((selection / 'inventory_model.json').read_text()))
    size = MinuteReductionModel.from_dict(json.loads((selection / 'reduction_model.json').read_text()))
    calibration = json.loads((selection / 'inventory_calibration.json').read_text())
    if (manager.kind != 'logistic' or calibration['scales'] != frozen['inventory_scales']
        or calibration['thresholds'] != frozen['inventory_thresholds']):
        raise ValueError('분봉 관리 정책의 모형·보정 설정 오류')
    return frozen, MinuteInventoryPolicy(original, manager, frozen['inventory_thresholds'], original.multiplier,
                                         frozen['inventory_scales'], size)


def sizing_context_training(frame, ledger):
    sized = sizing_training(frame, ledger, TRAINING_PERIODS[1])
    return sized.merge(frame[['end', *CONTEXT_FEATURES]], on='end', validate='one_to_one')


def run_minute_inventory_selection(reference: Path, labels: Path, market: Path, features: Path,
                                   confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    from .event_research import load_selection
    previous, _ = load_selection(reference)
    if previous['protocol'] != 'minute_inventory_recent_v20':
        raise ValueError('분봉 관리 입력 대조에는 고정 v20 모델이 필요합니다.')
    metadata = json.loads((labels / 'manifest.json').read_text())['settings']
    prior = json.loads((reference / 'manifest.json').read_text())['settings']
    if metadata['inventory_files_sha256'] != prior['files_sha256']:
        raise ValueError('분봉 관리와 이전 수량 학습의 원본 정답 불일치')
    verify_files(labels, ['minute_features.parquet', 'summary.json'])
    out = new_run(output, 'minute-inventory-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V21.md')), 'candidate_count': 1, 'control_count': 1,
        'training_period': TRAINING_PERIODS[1], 'calibration_period': CALIBRATION_PERIODS[1], 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'확정 분봉 관리 입력 학습 및 고정 비교: {out}', flush=True)
    try:
        copy_previous_inventory(reference, out / 'previous_inventory')
        for name in [*PARENT_FILES, 'rate_selection.json']:
            (out / name).write_bytes((reference / name).read_bytes())
        save_json(out / 'candidate_plan.json', [{'candidate': 0, 'size_mode': 'learned', 'selection_by_profit': False},
                                               {'control': 'fixed_size', 'selection_by_profit': False}])
        all_events, ledger = load_minute_inventory_labels(labels)
        train = purged_window(all_events, *TRAINING_PERIODS[1])
        calibration = purged_window(all_events, *CALIBRATION_PERIODS[1])
        size_train = sizing_context_training(all_events, ledger)
        for name, frame in [('training_used', train), ('calibration_used', calibration), ('size_training_used', size_train)]:
            frame.to_parquet(out / f'{name}.parquet', index=False)
        manager, thresholds, support = MinuteInventoryModels.fit(train, calibration, 'logistic')
        size, size_support = MinuteReductionModel.fit(size_train, TRAINING_PERIODS[1])
        rate = {**calibrate_rates(manager, calibration, CALIBRATION_PERIODS[1]), 'thresholds': thresholds}
        save_json(out / 'inventory_model.json', manager.to_dict())
        save_json(out / 'reduction_model.json', size.to_dict())
        save_json(out / 'inventory_calibration.json', rate)
        save_json(out / 'training_support.json', {'management': support, 'sizing': size_support,
            'original_actions_unchanged': True, 'future_market_inputs': False})
        frozen = {**previous, 'protocol': 'minute_inventory_micro_v21',
            'previous_inventory_sha256': sha256(out / 'previous_inventory/frozen_selection.json'),
            'inventory_files_sha256': {name: sha256(out / name) for name in INVENTORY_FILES},
            'inventory_scales': rate['scales'], 'inventory_thresholds': thresholds}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'관리 {len(train)}개·크기 {len(size_train)}개·44개 입력 학습 완료', flush=True)
        del all_events, ledger, train, calibration, size_train
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        rows = []
        for name, bot, folder in [('learned_size', policy, 'candidate-00'),
                                  ('fixed_size_control', fixed_size_control(policy), 'control-fixed-size')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            rows.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', rows)
            print(f'2021 학습 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        frozen['development_metrics'] = {'candidate': 0, **rows[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        rows.append({'policy': 'learned_size', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', rows)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 확정 분봉 관리 입력의 고정 비교\n\n' + table(pd.DataFrame(rows)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n학습·보정·모형·진입·위험 설정은 v20과 같고 확정 분봉 입력만 추가했다. '
            '2021년 재생은 학습·보정이 겹친 적합 진단이며 2022년은 그 이후의 고정 정책 확인이다. '
            '모든 기간은 이미 관찰했다. 원본 손실과 부분 체결 근사를 유지하며 고정 축소 대조와 손익으로 재선택하지 않았다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
