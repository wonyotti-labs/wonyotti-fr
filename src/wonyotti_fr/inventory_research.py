from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics, load_action_selection
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .inventory_labels import load_inventory_labels, sizing_training, verify_files
from .inventory_management import (
    CALIBRATION_PERIODS,
    TRAINING_PERIODS,
    InventoryActionModels,
    InventoryRatePolicy,
    ReductionModel,
)
from .minute_data import prepare_minute_period
from .minute_management import purged_window
from .rate_policy import calibrate_rates
from .reports import table

INVENTORY_FILES = ['inventory_model.json', 'reduction_model.json', 'inventory_calibration.json']
PARENT_FILES = ['pullback_selection.json', 'base_selection.json', 'expansion_models.json',
                'action_model.json', 'path_selection.json', 'rate_calibration.json']


def load_inventory_selection(selection, frozen):
    parent_path = selection / 'rate_selection.json'
    if parent_path.stat().st_size > 1024**2 or sha256(parent_path) != frozen['rate_selection_sha256']:
        raise ValueError('수량 관리 정책의 기반 선택 지문 오류')
    parent = json.loads(parent_path.read_text())
    latest = frozen.get('protocol') == 'minute_inventory_recent_v20'
    excluded = {'protocol', 'candidate', 'development_metrics'} | ({'training_period', 'calibration_period'} if latest else set())
    if (frozen.get('protocol') not in {'minute_inventory_v19', 'minute_inventory_recent_v20'} or parent.get('protocol') != 'minute_rate_v14'
        or frozen['candidate'] != 0 or frozen.get('size_mode') != 'learned'
        or any(frozen.get(key) != value for key, value in parent.items()
               if key not in excluded)):
        raise ValueError('수량 관리 정책의 고정 기반 설정 오류')
    if latest and (frozen.get('development_in_sample') is not True
                   or frozen['training_period'] != list(TRAINING_PERIODS[1])
                   or frozen['calibration_period'] != list(CALIBRATION_PERIODS[1])):
        raise ValueError('최신 수량 관리 정책의 학습·보정 기간 오류')
    _, original = load_action_selection(selection, parent)
    if (set(frozen['inventory_files_sha256']) != set(INVENTORY_FILES)
        or any((selection / name).stat().st_size > 1024**2
               or sha256(selection / name) != frozen['inventory_files_sha256'][name] for name in INVENTORY_FILES)):
        raise ValueError('수량 관리 정책의 모델·빈도 보정 지문 오류')
    manager = InventoryActionModels.from_dict(json.loads((selection / 'inventory_model.json').read_text()))
    size = ReductionModel.from_dict(json.loads((selection / 'reduction_model.json').read_text()))
    calibration = json.loads((selection / 'inventory_calibration.json').read_text())
    if (manager.kind != 'logistic' or calibration['scales'] != frozen['inventory_scales']
        or calibration['thresholds'] != frozen['inventory_thresholds']):
        raise ValueError('수량 관리 정책의 고정 모형·빈도 배율 오류')
    return frozen, InventoryRatePolicy(original, manager, frozen['inventory_thresholds'], original.multiplier,
                                      frozen['inventory_scales'], size)


def fixed_size_control(policy):
    return InventoryRatePolicy(policy, policy.manager, policy.thresholds, policy.multiplier, policy.scales)


def run_inventory_selection(reference: Path, labels: Path, market: Path, features: Path,
                            confirmation_market: Path, confirmation_features: Path, output: Path, latest_source: bool = False) -> Path:
    from .event_research import load_selection

    if type(latest_source) is not bool:
        raise ValueError('최신 원본 구간 선택의 형식 오류')
    parent, _ = load_selection(reference)
    if parent.get('protocol') != 'minute_rate_v14':
        raise ValueError('수량 관리 정책은 원래 v14 기반이 필요합니다.')
    verify_files(labels, ['inventory_features.parquet', 'size_ledger.parquet', 'size_training.parquet', 'summary.json'])
    train_period, cal_period = TRAINING_PERIODS[int(latest_source)], CALIBRATION_PERIODS[int(latest_source)]
    out = new_run(output, 'inventory-recent-selection' if latest_source else 'inventory-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path(f'docs/EXPERIMENT_V{20 if latest_source else 19}.md')),
        'training_period': train_period, 'calibration_period': cal_period, 'development_in_sample': latest_source,
        'candidate_count': 1, 'control_count': 1,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'잔여 수량·실제 축소 크기 학습 및 고정 비교: {out}', flush=True)
    try:
        for name in PARENT_FILES:
            (out / name).write_bytes((reference / name).read_bytes())
        (out / 'rate_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
        save_json(out / 'candidate_plan.json', [{'candidate': 0, 'size_mode': 'learned', 'selection_by_profit': False},
                                               {'control': 'fixed_size', 'selection_by_profit': False}])
        all_events, ledger = load_inventory_labels(labels)
        train = purged_window(all_events, *train_period)
        calibration = purged_window(all_events, *cal_period)
        size_train = sizing_training(all_events, ledger, train_period)
        if not latest_source:
            pd.testing.assert_frame_equal(size_train, pd.read_parquet(labels / 'size_training.parquet'), check_exact=True)
        train.to_parquet(out / 'training_used.parquet', index=False)
        calibration.to_parquet(out / 'calibration_used.parquet', index=False)
        size_train.to_parquet(out / 'size_training_used.parquet', index=False)
        manager, thresholds, support = InventoryActionModels.fit(train, calibration, 'logistic')
        size, size_support = ReductionModel.fit(size_train, train_period)
        rate = {**calibrate_rates(manager, calibration, cal_period), 'thresholds': thresholds}
        save_json(out / 'inventory_model.json', manager.to_dict())
        save_json(out / 'reduction_model.json', size.to_dict())
        save_json(out / 'inventory_calibration.json', rate)
        save_json(out / 'training_support.json', {'management': support, 'sizing': size_support,
            'original_actions_unchanged': True, 'future_order_quantities_in_features': False})
        frozen = {**parent, 'protocol': 'minute_inventory_v19', 'candidate': 0, 'size_mode': 'learned',
            'rate_selection_sha256': sha256(out / 'rate_selection.json'),
            'inventory_files_sha256': {name: sha256(out / name) for name in INVENTORY_FILES},
            'inventory_thresholds': thresholds, 'inventory_scales': rate['scales']}
        if latest_source:
            frozen.update(protocol='minute_inventory_recent_v20', training_period=list(train_period),
                          calibration_period=list(cal_period), development_in_sample=True)
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'관리 {len(train)}개·크기 {len(size_train)}개 학습 완료, 빈도 배율 {rate["scales"]}', flush=True)
        del all_events, ledger, train, calibration, size_train
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'development_input.json', checks)
        rows = []
        for name, bot, folder in [('learned_size', policy, 'candidate-00'),
                                  ('fixed_size_control', fixed_size_control(policy), 'control-fixed-size')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            rows.append({'policy': name, 'period': '2021_in_sample' if latest_source else '2021', **metrics})
            save_json(out / 'development.json', rows)
            print(f'2021 {"학습 적합" if latest_source else "개발"} {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        frozen['development_metrics'] = {'candidate': 0, **rows[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        rows.append({'policy': 'learned_size', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', rows)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 잔여 수량과 실제 축소 크기 정책\n\n' + table(pd.DataFrame(rows)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n원래 v14 진입·위험·평탄 청산을 유지하고 과거 최대 보유 대비 잔여 수량을 추가했다. '
            '실제 총 체결 비율을 학습한 단일 후보이며 고정 축소 대조와 손익으로 재선택하지 않았다. '
            + ('2020~2021년 상반기 학습·2021년 하반기 보정으로 2021년 재생은 학습 적합 진단이다. '
               if latest_source else '2019~2020년 상반기 학습·하반기 빈도 보정을 유지했다. ')
            + '포지션·24시간 경계를 유지했다. 부분 체결 총량을 다음 거래 가능 시가의 시장가로 실행하는 근사다. 이미 관찰한 기간의 반복 연구이며 수익성 채택은 후속 평가와 별도다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
