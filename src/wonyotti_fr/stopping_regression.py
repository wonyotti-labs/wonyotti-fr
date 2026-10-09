from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .common import save_json
from .early_stopping_close import EARLY_STOPPING_SETTINGS, validate_early_stopping_training
from .entry_regression import EntryRegressionModel
from .first_close_diagnostics import first_close_positions
from .minute_close_learning import MinuteCloseModel, minute_grid_actions
from .stopping_close import checked_opportunity_indices, stopping_targets


class StoppingRegressionModel(EntryRegressionModel):
    features = MinuteCloseModel.features
    format = 'early_minute_stopping_cash_regression_v1'


def fit_stopping_regression(training, weights, targets, calibration, opportunity_indices, output):
    validate_early_stopping_training(training)
    indices = checked_opportunity_indices(opportunity_indices, len(training))
    pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    expected = stopping_targets(training, targets.teacher_score, indices)
    pd.testing.assert_frame_equal(targets, expected, check_exact=True)
    if targets.target_available_time.ge(pd.Timestamp(EARLY_STOPPING_SETTINGS['label_cutoff'])).any():
        raise ValueError('이후 청산 회귀의 정답 확정 경계 오류')
    y, w = targets.stopping_advantage_bps.to_numpy(dtype=float), weights.sample_weight.to_numpy(dtype=float)
    if (np.count_nonzero(y) < 1000 or min(np.count_nonzero(y > 0), np.count_nonzero(y < 0)) < 64):
        raise ValueError('이후 청산 회귀의 양쪽 손익 지원 부족')
    features = StoppingRegressionModel.features
    model, support = StoppingRegressionModel.fit(training[features].to_numpy(dtype=float), y, w,
        calibration[features].to_numpy(dtype=float))
    prediction = model.predict(training[model.features].to_numpy(dtype=float))
    constant = float(np.average(y, weights=w))
    support = {**support, 'fit_rows': len(training), 'zero_effect_rows': int(np.count_nonzero(y == 0)),
        'positive_rows': int(np.count_nonzero(y > 0)), 'negative_rows': int(np.count_nonzero(y < 0)),
        'positions': int(training.position_entry_time.nunique()), 'fit_time': EARLY_STOPPING_SETTINGS['fit_time'],
        'last_label_end': training.label_end.max(), 'last_target_available_time': targets.target_available_time.max(),
        'training_constant_bps': constant, 'training_weighted_mse': float(np.average((prediction-y)**2, weights=w)),
        'constant_weighted_mse': float(np.average((constant-y)**2, weights=w)),
        'zero_targets_used_with_original_weight': True, 'export_validation_period': 'calibration',
        'diagnosis_used_for_export': False}
    ledger = training[['decision_time', 'position_entry_time', 'close_advantage_bps']].copy()
    ledger['opportunity_index'], ledger['stopping_advantage_bps'], ledger['sample_weight'] = indices, y, w
    ledger['fit_used'], ledger['zero_target'] = True, y == 0
    ledger.to_parquet(output/'regression_training_ledger.parquet', index=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support


def regression_values(values, rows):
    result = np.asarray(values, dtype=float)
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ValueError('이후 청산 회귀의 예측 차원·숫자 오류')
    return result


def regression_policy(frame, predictions):
    values = regression_values(predictions, len(frame))
    action = minute_grid_actions(frame, (values > 0).astype(float), 60)
    part = first_close_positions(frame.assign(_selected=action.astype(float)), '_selected')
    part['first_selected_time'] = pd.to_datetime(part.first_selected_time, utc=True).astype('datetime64[ns, UTC]')
    # 실행 비트와 경제적 예측값을 분리해 bp를 확률처럼 표시하지 않는다.
    part['first_prediction_bps'] = part.first_selected_time.map(pd.Series(values, index=frame.decision_time))
    return action, part
