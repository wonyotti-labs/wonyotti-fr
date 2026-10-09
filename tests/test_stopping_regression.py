import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_early_stopping_close import early_setup
from test_minute_close_learning import minute_cases
from threadpoolctl import threadpool_limits

from wonyotti_fr.stopping_close import stopping_targets
from wonyotti_fr.stopping_regression import (
    StoppingRegressionModel,
    fit_stopping_regression,
    regression_policy,
)


def regression_setup():
    rows, weights, indices = early_setup()
    training = rows['early_training']
    scores = np.where(np.arange(len(training)) % 7 == 0, .7, .3)
    targets = stopping_targets(training, scores, indices)
    return rows, weights, indices, targets


def test_all64_trees_baseline_original_weights_zero_targets_and_bps_export(tmp_path):
    rows, weights, indices, targets = regression_setup()
    training, calibration = rows['early_training'], rows['calibration']
    model, support = fit_stopping_regression(training, weights, targets, calibration, indices, tmp_path)
    x, y, w = training[model.features].to_numpy(), targets.stopping_advantage_bps.to_numpy(), weights.sample_weight.to_numpy()
    with threadpool_limits(limits=1):
        independent = HistGradientBoostingRegressor(**model.settings).fit(x, y, sample_weight=w)
        exported = model.to_dict()
        assert independent.n_iter_ == len(exported['trees']) == 64
        assert exported['baseline'] == independent._baseline_prediction[0, 0] == pytest.approx(np.average(y, weights=w))
        for tree, part in zip(exported['trees'], independent._predictors, strict=True):
            nodes = part[0].nodes
            leaf = nodes['is_leaf'].astype(bool)
            for name, values in {'left': np.where(leaf, -1, nodes['left'].astype(int)), 'right': np.where(leaf, -1, nodes['right'].astype(int)),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)), 'threshold': np.where(leaf, 0., nodes['num_threshold']),
                'value': np.where(leaf, nodes['value'], 0.)}.items():
                np.testing.assert_array_equal(tree[name], values)
        for frame in [training, calibration]:
            values = frame[model.features].to_numpy()
            np.testing.assert_allclose(model.predict(values), independent.predict(values), rtol=0, atol=1e-10)
    saved = pd.read_parquet(tmp_path/'regression_training_ledger.parquet')
    pd.testing.assert_frame_equal(saved[['decision_time', 'position_entry_time', 'close_advantage_bps']],
        training[['decision_time', 'position_entry_time', 'close_advantage_bps']], check_exact=True)
    np.testing.assert_array_equal(saved.opportunity_index, indices)
    np.testing.assert_array_equal(saved.sample_weight, w)
    np.testing.assert_array_equal(saved.stopping_advantage_bps, y)
    assert saved.fit_used.all() and saved.zero_target.any() and saved.loc[saved.zero_target, 'sample_weight'].gt(0).all()
    assert support['rows'] == support['fit_rows'] == len(training)
    assert support['zero_effect_rows'] == int((y == 0).sum())
    assert support['training_constant_bps'] == pytest.approx(np.average(y, weights=w))
    assert support['training_weighted_mse'] == pytest.approx(np.average((model.predict(x)-y)**2, weights=w))
    assert support['zero_targets_used_with_original_weight'] and not support['diagnosis_used_for_export']
    assert support['export_validation_period'] == 'calibration'


@pytest.mark.parametrize('damage', ['weights', 'target', 'cash', 'indices', 'available', 'cutoff', 'features'])
def test_changed_targets_weights_and_boundaries_rejected_before_fit(tmp_path, monkeypatch, damage):
    rows, weights, indices, targets = regression_setup()
    training = rows['early_training'].copy()
    if damage == 'weights':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'target':
        targets.loc[0, 'stopping_advantage_bps'] += 1
    elif damage == 'cash':
        targets.loc[0, 'continuation_policy_cash'] += 1
    elif damage == 'indices':
        indices[1] = indices[0]
    elif damage == 'available':
        targets.loc[0, 'target_available_time'] = pd.Timestamp('2021-07-31', tz='UTC')
    elif damage == 'cutoff':
        training.loc[0, 'label_end'] = pd.Timestamp('2021-07-31', tz='UTC')
    else:
        training.loc[0, 'favorable_move'] = np.nan
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError('잘못된 원장 뒤 회귀 적합 호출')

    monkeypatch.setattr(StoppingRegressionModel, 'fit', forbidden)
    with pytest.raises((ValueError, AssertionError)):
        fit_stopping_regression(training, weights, targets, rows['calibration'], indices, tmp_path)
    assert not calls and not (tmp_path/'model.json').exists()


def test_future_calibration_labels_and_time_units_do_not_change_model(tmp_path):
    rows, weights, indices, targets = regression_setup()
    results = []
    for unit in ['us', 'ns']:
        training, calibration, target, weight = rows['early_training'].copy(), rows['calibration'].copy(), targets.copy(), weights.copy()
        for frame in [training, calibration, target, weight]:
            for name in frame.select_dtypes('datetimetz').columns:
                frame[name] = frame[name].astype(f'datetime64[{unit}, UTC]')
        if unit == 'ns':
            calibration['close_advantage_bps'] *= -100
        folder = tmp_path/unit
        folder.mkdir()
        model, _ = fit_stopping_regression(training, weight, target, calibration, indices, folder)
        results.append(model.to_dict())
    assert results[0] == results[1]


def test_bp_policy_uses_strict_zero_not_probability_boundary_and_preserves_original_exit():
    frame = minute_cases()
    values = np.array([0., .1, -2., 10., -3., 1.5, .5, 2.])
    selected, positions = regression_policy(frame, values)
    np.testing.assert_array_equal(selected, (values > 0) & frame.original_intent.ne('exit').to_numpy())
    for entry, group in frame.groupby('position_entry_time'):
        chosen = group[selected[group.index]]
        part = positions.set_index('position_entry_time').loc[entry]
        assert part.first_prediction_bps == values[chosen.index[0]]
        assert part.first_effect_common_bps == pytest.approx(chosen.close_advantage_pnl.iloc[0]/group.decision_equity.iloc[0]*10000)
    assert 'first_cost_score' not in positions
    for invalid in [np.full(len(frame), np.nan), np.full(len(frame), np.inf), values[:-1]]:
        with pytest.raises(ValueError):
            regression_policy(frame, invalid)
