from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .event_features import independent_orders, purged_train
from .expansion_data import source_inputs
from .expansion_research import expansion_gate
from .lifecycle_model import ACTIONS, MANAGEMENT_FEATURES, LifecyclePolicy, ManagementModel
from .minute_data import prepare_minute_period
from .pullback_diagnostics import waiting_diagnostics
from .pullback_research import load_pullback_selection
from .reports import table


def candidate_plan():
    return [{'frequency_factor': factor, 'stop_fraction': stop}
            for factor in [.5, 1.] for stop in [.04, .08]]


def sizing_from_training(orders: pd.DataFrame, train: pd.DataFrame) -> dict:
    supported = orders[orders.target_time.isin(train.target_time.dropna())]
    values = {}
    counts = {}
    for target in ['increase', 'reduce']:
        selected = supported[supported.target.eq(target)]
        ratios = selected.orderqty / selected.before_qty.abs()
        if len(ratios) < 20 or not np.isfinite(ratios).all() or ratios.le(0).any():
            raise ValueError('학습 주문 크기의 지원 표본·유한성 부족')
        values[target] = float(ratios.median())
        counts[target] = len(ratios)
    return {'median_order_over_position': values, 'counts': counts,
            'addition_fraction': float(np.clip(.5 * values['increase'], .05, .25)),
            'reduction_fraction': float(np.clip(values['reduce'], .1, .9)),
            'interpretation': '학습 의도 수량의 고정 중앙값 기준. 원본 계좌 레버리지·상태별 수량 복원 아님'}


def risk_config(previous: dict, sizing: dict, stop: float) -> EngineConfig:
    addition, reduction = sizing['addition_fraction'], sizing['reduction_fraction']
    if (not np.isfinite([addition, reduction]).all() or not .05 <= addition <= .25
        or not .1 <= reduction <= .9 or stop not in (.04, .08)):
        raise ValueError('관리 정책의 추가·축소·손절 설정 오류')
    return replace(EngineConfig(**previous['risk']), max_hold_bars=0, max_adds=5, allow_adverse_add=True,
                   addition_fraction=addition, reduction_fraction=reduction, stop_fraction=stop)


def load_lifecycle_selection(selection: Path, frozen: dict):
    base_path, model_path = selection / 'pullback_selection.json', selection / 'management_model.json'
    if (frozen.get('protocol') != 'lifecycle_v9' or base_path.stat().st_size > 1024**2
        or model_path.stat().st_size > 1024**2 or sha256(base_path) != frozen['pullback_selection_sha256']
        or frozen['model_sha256'] != {'management_model.json': sha256(model_path)}):
        raise ValueError('관리 정책 기반·모델의 크기·지문 오류')
    previous = json.loads(base_path.read_text())
    _, entry = load_pullback_selection(selection, previous)
    if ((entry.offset_bps, entry.ttl_minutes) != (16, 5)
        or frozen['training_period'] != ['2018-03-01', '2021-01-01']
        or frozen['selection_period'] != ['2021-01-01', '2022-01-01']
        or frozen['confirmation_period'] != ['2022-01-01', '2023-01-01']
        or frozen['observed_evaluation_period'] != ['2023-01-01', '2026-01-01']
        or frozen['seen_2026_period'] != ['2026-01-01', '2026-10-01']
        or frozen['evaluation_end_exclusive'] != '2026-10-01' or frozen['unseen_evaluation_available'] is not False
        or frozen['frequency_factor'] not in (.5, 1.)
        or frozen['risk'] != asdict(risk_config(previous, frozen['sizing'], frozen['stop_fraction']))):
        raise ValueError('관리 정책의 고정 기간·설정 오류')
    model = ManagementModel.from_dict(json.loads(model_path.read_text()))
    return frozen, LifecyclePolicy(entry, model, frozen['activity_threshold'])


def lifecycle_diagnostics(directory: Path, bars: pd.DataFrame, config: EngineConfig) -> dict:
    waiting = waiting_diagnostics(directory, bars, config.signal_delay_bars)
    decomposition = decompose_run(directory, config.initial_equity)
    curve = pd.read_parquet(directory / 'equity.parquet', columns=['time', 'policy_event'])
    managed = curve[curve.policy_event.str.startswith('manage_')]
    if (pd.to_datetime(managed.time, utc=True).astype('datetime64[ns, UTC]').array.asi8 % (300 * 10**9)).any():
        raise ValueError('미확정 5분 경계의 관리 판단')
    trades = pd.read_parquet(directory / 'trades.parquet')
    result = {'waiting': waiting, 'decomposition': decomposition, 'management_on_confirmed_boundaries': True,
              'management_events': {str(k): int(v) for k, v in managed.policy_event.value_counts().items()},
              'trades_with_additions': int(trades['adds'].gt(0).sum()) if len(trades) else 0,
              'trades_over_30_minutes': int(trades.hold_bars.gt(30).sum()) if len(trades) else 0}
    save_json(directory / 'lifecycle_diagnostics.json', result)
    return result


def imitation(model: ManagementModel, data: pd.DataFrame, threshold: float) -> dict:
    scores, actions = model.probabilities(data[MANAGEMENT_FEATURES].to_numpy())
    selected = (scores >= threshold) & (actions.max(axis=1) >= .5)
    predicted = np.where(selected, np.asarray(ACTIONS)[actions.argmax(axis=1)], 'hold')
    truth = data.target.replace({'enter_long': 'exit', 'enter_short': 'exit'})
    return {'rows': len(data), 'classification': classification_report(
        truth, predicted, labels=['hold', *ACTIONS], output_dict=True, zero_division=0),
        'predicted_counts': {str(k): int(v) for k, v in pd.Series(predicted).value_counts().items()},
        'interpretation': '원본 상태에서의 모사이며 실제 봇 상태 분포·수익률 평가는 별도'}


def run_lifecycle_selection(reference: Path, audit: Path, study: Path, history: Path, market: Path,
                            features: Path, confirmation_market: Path, confirmation_features: Path,
                            output: Path) -> Path:
    from .event_research import load_selection
    previous, entry = load_selection(reference)
    if previous.get('protocol') != 'pullback_v7' or (entry.offset_bps, entry.ttl_minutes) != (16, 5):
        raise ValueError('고정 v7 후보 2가 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    destination = new_run(output, 'lifecycle-selection', {
        **hashes, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V9.md')),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [
            (market, 'manifest-1m.json'), (features, 'manifest-5m.json'),
            (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]},
        'all_periods_already_observed': True, 'candidate_count': 4, 'training_end_exclusive': '2021-01-01'})
    print(f'관리 정책 선택: {destination}', flush=True)
    (destination / 'pullback_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
    for name in ['base_selection.json', 'expansion_models.json']:
        (destination / name).write_bytes((reference / name).read_bytes())
    save_json(destination / 'candidate_plan.json', candidate_plan())
    try:
        all_train = purged_train(source['events'], source['episodes'], '2021-01-01')
        train = all_train[all_train.direction.ne(0)].copy()
        train.to_parquet(destination / 'management_training.parquet', index=False)
        model, support = ManagementModel.fit(train)
        sizing = sizing_from_training(independent_orders(source['executions'], source['actions']), train)
        save_json(destination / 'management_model.json', model.to_dict())
        save_json(destination / 'training_support.json', support)
        save_json(destination / 'sizing.json', sizing)
        check = source['events']
        check = check[check.usable & check.direction.ne(0) & check.end.ge('2021-01-01') & check.label_end.lt('2022-01-01')]
        print(f'관리 학습 {len(train):,}개 창, 행동 빈도 {support["active_fraction"]:.2%}', flush=True)
        data, verified = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(destination / 'development_input.json', verified)
        rows = []
        for index, candidate in enumerate(candidate_plan()):
            threshold = support['thresholds'][str(candidate['frequency_factor'])]
            config = risk_config(previous, sizing, candidate['stop_fraction'])
            policy = LifecyclePolicy(entry, model, threshold)
            target = destination / f'candidate-{index:02d}'
            metrics = backtest(data, policy, config, target)
            lifecycle_diagnostics(target, data, config)
            rows.append({'candidate': index, **candidate, 'activity_threshold': threshold,
                         'eligible': metrics['closed_trades'] >= 20,
                         'score': metrics['total_return'] - .5 * abs(metrics['max_drawdown']), **metrics})
            save_json(destination / 'development.json', rows)
            print(f'후보 {index}: {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '20거래 적격 후보 없음', 'criteria_relaxed': False})
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        config = risk_config(previous, sizing, winner['stop_fraction'])
        frozen = {'protocol': 'lifecycle_v9', 'candidate': winner['candidate'],
                  **{k: winner[k] for k in ['frequency_factor', 'stop_fraction', 'activity_threshold']},
                  'risk': asdict(config), 'sizing': sizing, 'development_metrics': winner,
                  'pullback_selection_sha256': sha256(destination / 'pullback_selection.json'),
                  'model_sha256': {'management_model.json': sha256(destination / 'management_model.json')},
                  'training_period': ['2018-03-01', '2021-01-01'], 'selection_period': ['2021-01-01', '2022-01-01'],
                  'confirmation_period': ['2022-01-01', '2023-01-01'],
                  'observed_evaluation_period': ['2023-01-01', '2026-01-01'],
                  'seen_2026_period': ['2026-01-01', '2026-10-01'],
                  'evaluation_end_exclusive': '2026-10-01', 'unseen_evaluation_available': False}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, policy = load_lifecycle_selection(destination, frozen)
        save_json(destination / 'imitation_2021.json', imitation(model, check, winner['activity_threshold']))
        del data, source, train, all_train, check
        data, verified = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(destination / 'confirmation_input.json', verified)
        target = destination / 'confirmation-2022'
        metrics = backtest(data, policy, config, target)
        lifecycle_diagnostics(target, data, config)
        gate = expansion_gate(winner, metrics)
        save_json(destination / 'development_confirmation_checks.json', {
            'checks': gate['checks'], 'passed': gate['may_open_new_period'], 'unseen_period_opened': False,
            'confirmation_sha256': sha256(target / 'metrics.json')})
        save_json(destination / 'summary.json', {'complete': True, 'selected': True, 'profitability_accepted': False})
        columns = ['candidate', 'frequency_factor', 'stop_fraction', 'total_return', 'max_drawdown', 'closed_trades']
        (destination / 'REPORT.md').write_text(
            '# 원본 관리 행동을 학습한 후보\n\n' + table(pd.DataFrame(rows)[columns]) + '\n\n'
            f'후보 {winner["candidate"]} 고정. 2022년 확인 {metrics["total_return"]:.2%}·{metrics["closed_trades"]}거래. '
            f'선행 조건 통과 {gate["may_open_new_period"]}. 수익성 채택은 별도다.\n\n'
            '시간 제한 없이 확정 특징·현재 봇 상태에서 추가·축소·청산을 판단한다. '
            '원본 반전은 청산으로 학습했고 반대 진입은 다시 고정 진입 규칙을 거친다. '
            '최초 25%·최대 50% 노출과 위험 한도를 적용하며 실제 계좌 레버리지는 추정하지 않는다. '
            '관리 수량은 학습 주문 비율 중앙값에 의한 고정 기준이다. 원본 크기·청산을 정확히 복제한 것은 아니다. '
            '학습과 모델 숫자·기준·입력 지문을 저장했다. 모든 기간은 이미 관찰했으며 원본 손실을 제거하지 않았다.\n',
            encoding='utf-8')
        print(f'2022년 확인: {metrics["total_return"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return destination
