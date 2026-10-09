import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_first_opportunity_close import setup
from threadpoolctl import threadpool_limits

from wonyotti_fr.first_linear_close import (
    FIRST_LINEAR_SETTINGS,
    FirstLinearCloseModel,
    fit_first_linear,
)
from wonyotti_fr.first_opportunity_close import first_cost_training, first_opportunity_rows


def previous_setup():
    training, weights, calibration = setup()
    first, _ = first_opportunity_rows(training)
    costs, _ = first_cost_training(first.first_target_common_bps)
    keys = ['opportunity_index', 'decision_time', 'position_entry_time', 'label_end', 'direction',
        'decision_equity', 'reference_equity', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']
    return training, weights, calibration, pd.concat([first[keys], costs], axis=1)


def test_equal_first_rows_scaler_all_coefficients_intercept_costs_and_numeric_export(tmp_path):
    training, weights, calibration, previous = previous_setup()
    model, support = fit_first_linear(training, weights, calibration, previous, tmp_path)
    first, _ = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    x, y = first[model.features].to_numpy(), first.first_target_common_bps.to_numpy()
    fit = y != 0
    assert (~fit).any()
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        independent = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(scaler.transform(x)[fit], y[fit] > 0,
            sample_weight=abs(y[fit])/abs(y[fit]).mean())
        data = model.to_dict()
        np.testing.assert_array_equal(data['mean'], scaler.mean_)
        np.testing.assert_array_equal(data['scale'], scaler.scale_)
        np.testing.assert_array_equal(data['coefficient'], independent.coef_[0])
        assert data['intercept'] == independent.intercept_[0]
        assert support['iterations'] == int(independent.n_iter_[0]) < 2000
        np.testing.assert_allclose(data['mean'], x.mean(axis=0), rtol=0, atol=1e-12)
        for frame in [first, validation]:
            values = frame[model.features].to_numpy()
            linear = ((values-np.asarray(data['mean']))/np.asarray(data['scale']))@np.asarray(data['coefficient'])+data['intercept']
            np.testing.assert_allclose(model.probabilities(values)[:, 0], 1/(1+np.exp(-linear)), rtol=0, atol=1e-12)
            np.testing.assert_allclose(model.probabilities(values)[:, 0], independent.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
    assert support['scaler_rows'] == len(first) and support['scaler_zero_cost_rows_included']
    assert support['previous_first_ledger_exact'] and not support['diagnosis_used_for_export']
    assert support['training_constant_score'] == pytest.approx(abs(y[y > 0]).sum()/abs(y).sum())


def test_future_calibration_and_later_training_inputs_leave_scaler_and_model_unchanged(tmp_path):
    training, weights, calibration, previous = previous_setup()
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    first, _ = fit_first_linear(training, weights, calibration, previous, a)
    training.loc[np.arange(len(training)) % 3 != 0, 'favorable_move'] += 5
    calibration['favorable_move'] += .2
    calibration['close_advantage_pnl'] *= -1
    calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
    calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    other, _ = fit_first_linear(training, weights, calibration, previous, b)
    assert other.to_dict() == first.to_dict()


@pytest.mark.parametrize('damage', ['target', 'cost', 'index', 'weight', 'cutoff', 'support', 'feature'])
def test_first_ledger_and_input_damage_rejected_before_linear_fit(tmp_path, monkeypatch, damage):
    training, weights, calibration, previous = previous_setup()
    if damage == 'target':
        previous.loc[0, 'first_target_common_bps'] += 1
    elif damage == 'cost':
        previous.loc[0, 'fit_weight'] *= 2
    elif damage == 'index':
        previous.loc[0, 'opportunity_index'] += 1
    elif damage == 'weight':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'cutoff':
        training.loc[0, 'label_end'] = pd.Timestamp('2021-07-31', tz='UTC')
    elif damage == 'support':
        training['original_intent'] = 'exit'
    else:
        training.loc[3, 'favorable_move'] = np.nan
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        fit_first_linear(training, weights, calibration, previous, tmp_path)
    assert not calls


@pytest.mark.parametrize('warning', [False, True])
def test_nonconvergence_rejected_without_adaptive_refit(monkeypatch, warning):
    training, _, calibration, _ = previous_setup()
    first, _ = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    calls = []

    class FailedLearner:
        def fit(self, *_args, **_kwargs):
            calls.append(True)
            if warning:
                warnings.warn('합성 수렴 실패', ConvergenceWarning, stacklevel=2)
            self.n_iter_ = np.array([2000])
            return self

    monkeypatch.setattr('wonyotti_fr.first_linear_close.LogisticRegression', lambda **_: FailedLearner())
    with pytest.raises((ConvergenceWarning, ValueError)):
        FirstLinearCloseModel.fit(first[FirstLinearCloseModel.features].to_numpy(), first.first_target_common_bps,
            validation[FirstLinearCloseModel.features].to_numpy())
    assert calls == [True]


def test_numeric_format_zero_score_boundary_overflow_and_copy_isolation(tmp_path):
    training, weights, calibration, previous = previous_setup()
    model, _ = fit_first_linear(training, weights, calibration, previous, tmp_path)
    data = model.to_dict()
    first, _ = first_opportunity_rows(calibration)
    for damage in ['shape', 'scale', 'nan', 'boolean', 'settings', 'extra']:
        changed = copy.deepcopy(data)
        if damage == 'shape':
            changed['coefficient'].pop()
        elif damage == 'scale':
            changed['scale'][0] = 0.
        elif damage == 'nan':
            changed['intercept'] = np.nan
        elif damage == 'boolean':
            changed['mean'][0] = True
        elif damage == 'settings':
            changed['settings']['C'] = 1.
        else:
            changed['code'] = 'untrusted'
        with pytest.raises(ValueError):
            FirstLinearCloseModel.from_dict(changed)
    data['coefficient'] = [0.]*len(model.features)
    data['intercept'] = 0.
    fixed = FirstLinearCloseModel.from_dict(data)
    assert (fixed.probabilities(first[model.features].to_numpy()) == .5).all()
    data['intercept'] = 7.
    assert fixed.to_dict()['intercept'] == 0.
    values = first[model.features].to_numpy(copy=True)
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        fixed.probabilities(values)
    overflow = fixed.to_dict()
    feature = model.features.index('favorable_move')
    overflow['coefficient'][feature] = 2.
    overflow['mean'][feature] = 0.
    overflow['scale'][feature] = 1.
    overflowing = FirstLinearCloseModel.from_dict(overflow)
    values = first[model.features].to_numpy(copy=True)
    values[0, feature] = 1e308
    with pytest.raises(ValueError):
        overflowing.probabilities(values)
