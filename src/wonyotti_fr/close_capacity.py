from __future__ import annotations

import numpy as np

from .addition_effect import position_weights
from .close_learning import close_metrics
from .common import save_json
from .continuation_inputs import ContinuationCloseModel
from .entry_regression import REGRESSION_SETTINGS

CAPACITY_GRID = [dict(name=name, settings={**REGRESSION_SETTINGS, 'max_depth': depth,
    'max_leaf_nodes': leaves, 'max_iter': iterations}) for name, depth, leaves, iterations in [
        ('depth2_iter64', 2, 4, 64), ('depth2_iter256', 2, 4, 256),
        ('depth4_iter64', 4, 16, 64), ('depth4_iter256', 4, 16, 256)]]


def capacity_class(name):
    matches = [row for row in CAPACITY_GRID if row['name'] == name]
    if len(matches) != 1:
        raise ValueError('청산 복잡도의 허용하지 않은 후보')
    return type('CapacityCloseModel', (ContinuationCloseModel,), {
        'format': 'close_capacity_'+name+'_v1', 'settings': dict(matches[0]['settings'])})


def capacity_from_dict(data):
    for row in CAPACITY_GRID:
        cls = capacity_class(row['name'])
        if data.get('format') == cls.format:
            return cls.from_dict(data)
    raise ValueError('청산 복잡도의 허용하지 않은 모델 형식')


def choose_capacity(metrics):
    names = [row['name'] for row in CAPACITY_GRID]
    if (set(metrics) != set(names)
        or any(not np.isfinite(metrics[n]['weighted_mse']) or metrics[n]['weighted_mse'] < 0 for n in names)):
        raise ValueError('청산 복잡도의 후보·선택 오차 오류')
    # 동일 오차에서는 사전 목록의 앞 후보를 유지한다.
    return min(names, key=lambda n: metrics[n]['weighted_mse'])


def fit_capacity_selection(training, selection, output):
    if (not len(training) or not len(selection)
        or training.label_end.max() >= selection.decision_time.min()
        or set(training.position_entry_time) & set(selection.position_entry_time)):
        raise ValueError('청산 복잡도의 내부 학습·선택 시각 및 포지션 교차')
    rows = {'training': training, 'selection': selection}
    values, weights, predictions = {}, {}, {}
    for name, frame in rows.items():
        frame.to_parquet(output/f'selection_{name}_used.parquet', index=False)
        weights[name] = position_weights(frame)
        values[name] = frame[ContinuationCloseModel.features].to_numpy(dtype=float)
        predictions[name] = frame[['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']].assign(sample_weight=weights[name])
        predictions[name][['decision_time', 'position_entry_time', 'sample_weight']].to_parquet(output/f'selection_{name}_weights.parquet', index=False)
    models, supports, metrics = {}, {}, {k: {} for k in rows}
    target = training.close_advantage_bps.to_numpy()
    for option in CAPACITY_GRID:
        name = option['name']
        model, support = capacity_class(name).fit(values['training'], target, weights['training'], values['selection'])
        models[name], supports[name] = model.to_dict(), support
        for kind in rows:
            prediction = model.predict(values[kind])
            predictions[kind]['predicted_'+name] = prediction
            metrics[kind][name] = close_metrics(predictions[kind], prediction)
    chosen = choose_capacity(metrics['selection'])
    constant = float(np.average(target, weights=weights['training']))
    for kind in rows:
        predictions[kind]['predicted_training_constant'] = constant
        metrics[kind]['training_constant'] = close_metrics(predictions[kind], np.full(len(rows[kind]), constant))
        predictions[kind].to_parquet(output/f'selection_{kind}_predictions.parquet', index=False)
    selection_result = {'candidate': chosen, 'criterion': 'minimum_selection_weighted_mse',
        'tie_break': 'predeclared_grid_order', 'final_diagnosis_used_for_selection': False}
    for name, data in [('selection_models', models), ('selection_support', supports),
        ('selection_metrics', metrics), ('selection', selection_result)]:
        save_json(output/f'{name}.json', data)
    return chosen
