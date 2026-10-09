import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_first_linear_close import previous_setup
from threadpoolctl import threadpool_limits

from wonyotti_fr.first_linear_close import (
    FIRST_LINEAR_SETTINGS,
    FirstLinearCloseModel,
    fit_first_linear,
)
from wonyotti_fr.first_opportunity_close import first_opportunity_rows
from wonyotti_fr.weekly_first_linear import fit_weekly_first_linear, weekly_first_boundaries


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    training, weights, calibration, ledger = previous_setup()
    root = tmp_path_factory.mktemp('weekly-first-base')
    model, support = fit_first_linear(training, weights, calibration, ledger, root)
    return training, calibration, ledger, model.to_dict(), support


def test_nine_models_all_coefficients_mature_costs_first_parity_routes_and_empty_weeks(prepared, tmp_path):
    training, calibration, ledger, original_model, original_support = prepared
    scores, constants = fit_weekly_first_linear(*prepared, tmp_path)
    models = json.loads((tmp_path/'weekly_models.json').read_text())
    supports = json.loads((tmp_path/'weekly_support.json').read_text())
    membership = pd.read_parquet(tmp_path/'weekly_training_membership.parquet')
    routing = pd.read_parquet(tmp_path/'prediction_routing.parquet')
    first, _ = first_opportunity_rows(calibration)
    assert len(models) == 9 and models['week-00'] == original_model
    assert len(routing) == len(first) and routing.opportunity_index.is_unique
    assert supports['week-08']['prediction_rows'] == 0
    assert supports['week-08']['eligible_positions'] > len(ledger)
    pool = pd.concat([first_opportunity_rows(training)[0].assign(source_split='early_training'), first.assign(source_split='calibration')], ignore_index=True)
    for number, (start, end) in enumerate(weekly_first_boundaries()):
        key = f'week-{number:02}'
        learned = pool[pool.label_end.lt(start-pd.Timedelta(days=2))].reset_index(drop=True)
        actual = membership[membership.model_key.eq(key)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[['position_entry_time', 'label_end', 'source_split', 'opportunity_index']],
            learned[['position_entry_time', 'label_end', 'source_split', 'opportunity_index']], check_exact=True)
        y = learned.first_target_common_bps.to_numpy()
        cost = abs(y)/abs(y[y != 0]).mean()
        np.testing.assert_array_equal(actual.fit_weight, cost)
        assert actual.position_weight.eq(1.).all()
        x = learned[FirstLinearCloseModel.features].to_numpy()
        with threadpool_limits(limits=1):
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(scaler.transform(x)[y != 0], y[y != 0] > 0, sample_weight=cost[y != 0])
        model = models[key]
        np.testing.assert_array_equal(model['mean'], scaler.mean_)
        np.testing.assert_array_equal(model['scale'], scaler.scale_)
        np.testing.assert_array_equal(model['coefficient'], learner.coef_[0])
        assert model['intercept'] == learner.intercept_[0]
        chosen = first.decision_time.ge(start) & first.decision_time.lt(end)
        part = first.loc[chosen]
        assert not set(learned.position_entry_time) & set(part.position_entry_time)
        assert routing.loc[routing.model_key.eq(key), 'opportunity_index'].tolist() == part.opportunity_index.tolist()
        if len(part):
            expected = learner.predict_proba(scaler.transform(part[FirstLinearCloseModel.features].to_numpy()))[:, 1]
            np.testing.assert_allclose(scores[chosen], expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(constants[chosen], abs(y[y > 0]).sum()/abs(y).sum(), rtol=0, atol=1e-12)
    assert original_support['training_constant_score'] == pytest.approx(constants[0])


def test_exact_two_day_boundary_excluded_until_next_week(prepared, tmp_path):
    training, calibration, ledger, model, support = copy.deepcopy(prepared)
    entry = pd.Timestamp('2021-08-06', tz='UTC')
    calibration.loc[calibration.position_entry_time.eq(entry), 'label_end'] = pd.Timestamp('2021-08-07', tz='UTC')
    fit_weekly_first_linear(training, calibration, ledger, model, support, tmp_path)
    membership = pd.read_parquet(tmp_path/'weekly_training_membership.parquet')
    assert not membership.loc[membership.model_key.eq('week-01'), 'position_entry_time'].eq(entry).any()
    assert membership.loc[membership.model_key.eq('week-02'), 'position_entry_time'].eq(entry).any()


def test_future_changes_preserve_prior_week_models_and_predictions(prepared, tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    original_scores, original_constants = fit_weekly_first_linear(*prepared, a)
    changed = copy.deepcopy(prepared)
    calibration = changed[1]
    cutoff = pd.Timestamp('2021-09-01', tz='UTC')
    future = calibration.decision_time.ge(cutoff)
    calibration.loc[future, 'favorable_move'] += 2
    calibration.loc[future, 'close_advantage_pnl'] *= -3
    calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
    calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    scores, constants = fit_weekly_first_linear(*changed, b)
    original = json.loads((a/'weekly_models.json').read_text())
    altered = json.loads((b/'weekly_models.json').read_text())
    for index in range(5):
        assert original[f'week-{index:02}'] == altered[f'week-{index:02}']
    first, _ = first_opportunity_rows(prepared[1])
    past = first.decision_time.lt(cutoff).to_numpy()
    np.testing.assert_array_equal(scores[past], original_scores[past])
    np.testing.assert_array_equal(constants[past], original_constants[past])
    assert original['week-08'] != altered['week-08']


@pytest.mark.parametrize('damage', ['ledger', 'split_boundary', 'position_end', 'feature', 'order', 'cash'])
def test_invalid_inputs_rejected_before_weekly_fit(prepared, tmp_path, monkeypatch, damage):
    training, calibration, ledger, model, support = copy.deepcopy(prepared)
    if damage == 'ledger':
        ledger.loc[0, 'fit_weight'] *= 2
    elif damage == 'split_boundary':
        calibration['label_end'] += pd.Timedelta(days=100)
    elif damage == 'position_end':
        calibration.loc[3, 'label_end'] += pd.Timedelta(minutes=1)
    elif damage == 'feature':
        calibration.loc[3, 'favorable_move'] = np.nan
    elif damage == 'order':
        calibration = calibration.iloc[::-1]
    else:
        calibration.loc[3, 'close_advantage_pnl'] += 1
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        fit_weekly_first_linear(training, calibration, ledger, model, support, tmp_path)
    assert not calls


def test_first_model_mismatch_rejected_before_later_weeks(prepared, tmp_path):
    training, calibration, ledger, model, support = copy.deepcopy(prepared)
    model['intercept'] += .1
    with pytest.raises(ValueError, match='최초 모델'):
        fit_weekly_first_linear(training, calibration, ledger, model, support, tmp_path)
    assert not (tmp_path/'weekly_models.json').exists()
