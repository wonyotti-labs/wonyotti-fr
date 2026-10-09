import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_first_linear_close import previous_setup
from test_first_opportunity_close import first_examples
from threadpoolctl import threadpool_limits

from wonyotti_fr.expanded_first_linear import expanded_first_inputs, fit_expanded_first_linear
from wonyotti_fr.first_linear_close import FIRST_LINEAR_SETTINGS, FirstLinearCloseModel
from wonyotti_fr.first_opportunity_close import first_opportunity_rows


def added_examples():
    frame = first_examples('2020-04-01', 240, 12)
    added, positions = first_opportunity_rows(frame)
    initial = positions.set_index('position_entry_time')
    added['first_available_time'] = added.position_entry_time.map(initial.first_available_time)
    added['label_status'] = 'closed'
    positions['natural_exit_time'] = positions.position_entry_time+pd.Timedelta(minutes=4)
    positions['natural_exit_reason'] = 'signal_exit'
    positions.loc[positions.has_eligible_opportunity, 'reason'] = 'closed'
    last = added.index[-1]
    added.loc[last, 'label_status'] = 'right_censored'
    added.loc[last, 'first_target_common_bps'] = np.nan
    added.loc[last, 'label_end'] = pd.Timestamp('2020-12-31', tz='UTC')
    mask = positions.position_entry_time.eq(added.loc[last, 'position_entry_time'])
    positions.loc[mask, 'reason'] = 'right_censored'
    positions.loc[mask, 'first_target_common_bps'] = np.nan
    return added, positions


def expanded_setup():
    return (*previous_setup(), *added_examples())


def test_combined_first_sources_all_costs_scaler_coefficients_and_no_eligible_preserved(tmp_path):
    inputs = expanded_setup()
    model, support = fit_expanded_first_linear(*inputs, tmp_path)
    combined = pd.read_parquet(tmp_path/'combined_first_training.parquet')
    costs = pd.read_parquet(tmp_path/'combined_training_costs.parquet')
    membership = pd.read_parquet(tmp_path/'added_training_membership.parquet')
    assert len(combined) == 755 and combined.source_phase.value_counts().to_dict() == {'original_2021': 540, 'expansion_2020': 215}
    assert len(membership) == 240 and membership.fit_eligible.sum() == 215
    assert membership.reason.value_counts().to_dict() == {'closed': 215, 'no_eligible_opportunity': 24, 'right_censored': 1}
    assert combined.position_entry_time.nunique() == len(combined) and combined.decision_time.is_monotonic_increasing
    assert (combined.first_target_common_bps == 0).any() and (combined.first_target_common_bps < 0).any()
    np.testing.assert_array_equal(costs.position_weight, np.ones(len(combined)))
    x, y = combined[model.features].to_numpy(), combined.first_target_common_bps.to_numpy()
    fit = y != 0
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(scaler.transform(x)[fit], y[fit] > 0, sample_weight=abs(y[fit])/abs(y[fit]).mean())
        saved = model.to_dict()
        np.testing.assert_array_equal(saved['mean'], scaler.mean_)
        np.testing.assert_array_equal(saved['scale'], scaler.scale_)
        np.testing.assert_array_equal(saved['coefficient'], learner.coef_[0])
        assert saved['intercept'] == learner.intercept_[0] and support['iterations'] == int(learner.n_iter_[0])
        np.testing.assert_allclose(model.probabilities(x)[:, 0], learner.predict_proba(scaler.transform(x))[:, 1], rtol=0, atol=1e-12)
    assert support['scaler_rows'] == len(combined) and support['old_first_rows_and_costs_exact']
    assert not support['whole_policy_historically_available_claimed'] and not support['diagnosis_used_for_export']
    np.testing.assert_allclose(costs.fit_weight, abs(y)/abs(y[fit]).mean(), rtol=0, atol=1e-12)


def test_future_calibration_later_rows_and_censored_inputs_leave_fit_unchanged(tmp_path):
    inputs = expanded_setup()
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    model, _ = fit_expanded_first_linear(*inputs, a)
    training, weights, calibration, previous, added, positions = inputs
    training.loc[np.arange(len(training)) % 3 != 0, 'favorable_move'] += .5
    calibration['favorable_move'] += .2
    calibration['close_advantage_pnl'] *= -1
    calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
    calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    added.loc[added.label_status.eq('right_censored'), 'favorable_move'] += .4
    other, _ = fit_expanded_first_linear(training, weights, calibration, previous, added, positions, b)
    assert other.to_dict() == model.to_dict()


@pytest.mark.parametrize('damage', ['old_cost', 'old_weight', 'added_target', 'added_cash', 'anchor', 'cutoff', 'exit',
    'duplicate', 'population', 'first_time', 'feature', 'forced_close', 'start'])
def test_added_and_original_damage_rejected_before_fit(tmp_path, monkeypatch, damage):
    training, weights, calibration, previous, added, positions = expanded_setup()
    if damage == 'old_cost':
        previous.loc[0, 'fit_weight'] *= 2
    elif damage == 'old_weight':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'added_target':
        added.loc[0, 'first_target_common_bps'] += 1
    elif damage == 'added_cash':
        added.loc[0, 'close_cash'] += 1
    elif damage == 'anchor':
        added.loc[0, 'reference_equity'] += 1
    elif damage == 'cutoff':
        added.loc[0, 'label_end'] = pd.Timestamp('2020-12-31', tz='UTC')
    elif damage == 'exit':
        added.loc[0, 'original_intent'] = 'exit'
    elif damage == 'duplicate':
        added.loc[1, 'position_entry_time'] = added.loc[0, 'position_entry_time']
    elif damage == 'population':
        positions.loc[positions.has_eligible_opportunity.idxmax(), 'has_eligible_opportunity'] = False
    elif damage == 'first_time':
        added.loc[0, 'first_available_time'] = added.loc[0, 'decision_time']+pd.Timedelta(minutes=1)
    elif damage == 'feature':
        added.loc[0, 'favorable_move'] = np.nan
    elif damage == 'forced_close':
        positions.loc[positions.reason.eq('closed'), 'natural_exit_reason'] = 'end_of_test'
    else:
        positions.loc[0, 'position_entry_time'] = pd.Timestamp('2020-03-31', tz='UTC')
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        fit_expanded_first_linear(training, weights, calibration, previous, added, positions, tmp_path)
    assert not calls


def test_same_inputs_identify_original_first_rows_exactly():
    training, weights, calibration, previous, added, positions = expanded_setup()
    combined, _, population = expanded_first_inputs(training, weights, calibration, previous, added, positions)
    original, expected_population = first_opportunity_rows(training)
    fields = [name for name in combined if name not in ['source_phase', 'source_opportunity_index']]
    actual = combined[combined.source_phase.eq('original_2021')].reset_index(drop=True)
    expected = original[fields].copy()
    for name in ['decision_time', 'position_entry_time', 'label_end']:
        expected[name] = expected[name].astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(actual[fields], expected, check_exact=True)
    pd.testing.assert_frame_equal(population, expected_population, check_exact=True)
