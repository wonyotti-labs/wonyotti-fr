import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_close_threshold import ledger_examples
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.first_opportunity_close import (
    FirstOpportunityCloseModel,
    first_cost_training,
    first_opportunity_policy,
    first_opportunity_rows,
    fit_first_opportunity,
)
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS


def first_examples(start='2021-01-01', positions=600, hours=8):
    template = ledger_examples().iloc[:1]
    frame = pd.concat([template]*positions*3, ignore_index=True)
    group = np.repeat(np.arange(positions), 3)
    step = np.tile(np.arange(3), positions)
    frame['position_entry_time'] = pd.Timestamp(start, tz='UTC')+pd.to_timedelta(group*hours, unit='h')
    frame['decision_time'] = frame.position_entry_time+pd.to_timedelta(step+1, unit='min')
    frame['continue_end'] = frame.position_entry_time+pd.Timedelta(minutes=5)
    frame['label_end'] = frame.continue_end
    frame['direction'] = np.where(group % 2, 1., -1.)
    for name in [key for key in frame if key.startswith('directional_taker_')]:
        frame[name] = frame[name.removeprefix('directional_')]*frame.direction
    frame['favorable_move'] = (group % 7)/100+step/200
    frame['original_intent'] = np.where(group % 10 == 0, 'exit', 'hold')
    frame['decision_equity'] = 10000.-step*10
    frame['continue_cash'] = 10000.
    frame['close_advantage_pnl'] = np.where(step == 0, np.where(group % 3, -2., 7.), 100.+step)
    frame.loc[(group % 53 == 0) & (step == 0), 'close_advantage_pnl'] = 0.
    frame.loc[frame.original_intent.eq('exit'), 'close_advantage_pnl'] = 0.
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    return frame


def setup():
    training, calibration = first_examples(), first_examples('2021-08-02', 40, 24)
    weight = training[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(training))
    return training, weight, calibration


def test_first_rows_equal_position_costs_all64_trees_and_export(tmp_path):
    training, weights, calibration = setup()
    model, support = fit_first_opportunity(training, weights, calibration, tmp_path)
    first, positions = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    assert len(positions) == 600 and len(first) == 540
    assert support['no_eligible_positions'] == 60 and support['zero_effect_positions'] > 0
    ledger = pd.read_parquet(tmp_path/'first_training_ledger.parquet')
    assert ledger.position_weight.eq(1.).all() and ledger.position_entry_time.nunique() == len(ledger)
    np.testing.assert_array_equal(first.opportunity_index, np.array([i*3 for i in range(600) if i % 10]))
    np.testing.assert_array_equal(ledger.first_target_common_bps, first.close_advantage_pnl)
    y, x = ledger.first_target_common_bps.to_numpy(), first[model.features].to_numpy()
    fit = y != 0
    cost = abs(y[fit])/abs(y[fit]).mean()
    with threadpool_limits(limits=1):
        independent = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x[fit], y[fit] > 0, sample_weight=cost)
        saved = model.to_dict()['models'][0]
        assert len(saved['trees']) == independent.n_iter_ == 64
        assert saved['baseline'] == independent._baseline_prediction[0, 0]
        for tree, nodes in zip(saved['trees'], [part[0].nodes for part in independent._predictors], strict=True):
            leaf = nodes['is_leaf'].astype(bool)
            for name, values in {'left': np.where(leaf, -1, nodes['left'].astype(int)),
                'right': np.where(leaf, -1, nodes['right'].astype(int)), 'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)),
                'threshold': np.where(leaf, 0., nodes['num_threshold']), 'value': np.where(leaf, nodes['value'], 0.)}.items():
                np.testing.assert_array_equal(tree[name], values)
        for frame in [first, validation]:
            values = frame[model.features].to_numpy()
            np.testing.assert_allclose(model.probabilities(values)[:, 0], independent.predict_proba(values)[:, 1], rtol=0, atol=1e-12)
    contribution = pd.read_parquet(tmp_path/'first_training_contribution.parquet')
    np.testing.assert_array_equal(contribution.original_weight, weights.sample_weight)
    assert len(contribution) == len(training) and contribution.position_weight.sum() == 540
    assert (contribution.groupby('position_entry_time').position_weight.sum() <= 1).all()
    assert support['training_constant_score'] == pytest.approx(abs(y[y > 0]).sum()/abs(y).sum())


def test_first_rejection_ignores_later_rows_preserves_losses_no_opportunity_and_boundary():
    frame = first_examples(positions=4)
    first, positions = first_opportunity_rows(frame)
    assert len(first) == 3 and len(positions) == 4
    action, result, scores = first_opportunity_policy(frame, [.5, .6, .9])
    assert np.flatnonzero(action).tolist() == [6, 9]
    assert result.first_effect_common_bps.tolist() == [0., 0., -2., 7.]
    assert result.later_selected_opportunities.eq(0).all()
    assert np.flatnonzero(np.isfinite(scores)).tolist() == [3, 6, 9]
    changed = frame.copy()
    later = np.arange(len(frame)) % 3 != 0
    changed.loc[later, 'close_advantage_pnl'] = -1e9
    changed['close_advantage_bps'] = changed.close_advantage_pnl/changed.decision_equity*10000
    np.testing.assert_array_equal(first_opportunity_policy(changed, [.5, .6, .9])[0], action)
    with pytest.raises(ValueError):
        first_opportunity_policy(frame, np.ones(len(frame)))
    with pytest.raises(ValueError):
        first_opportunity_policy(frame, [.5, np.nan, .7])
    frame.loc[3, 'original_intent'] = 'exit'
    first, _ = first_opportunity_rows(frame)
    assert first.opportunity_index.iloc[0] == 4
    assert first.first_target_common_bps.iloc[0] == frame.close_advantage_pnl.iloc[4]
    assert first.first_target_common_bps.iloc[0] != frame.close_advantage_bps.iloc[4]


def test_calibration_targets_and_later_training_features_do_not_change_first_model(tmp_path):
    training, weights, calibration = setup()
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    model, support = fit_first_opportunity(training, weights, calibration, a)
    training.loc[np.arange(len(training)) % 3 != 0, 'favorable_move'] += 100
    calibration['close_advantage_pnl'] *= -4
    calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
    calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    other, other_support = fit_first_opportunity(training, weights, calibration, b)
    assert other.to_dict() == model.to_dict() and support == other_support


@pytest.mark.parametrize('damage', ['weights', 'cutoff', 'support', 'cash', 'feature', 'order', 'direction'])
def test_invalid_first_training_rejected_before_fit(tmp_path, monkeypatch, damage):
    training, weights, calibration = setup()
    if damage == 'weights':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'cutoff':
        training.loc[0, 'label_end'] = pd.Timestamp('2021-07-31', tz='UTC')
    elif damage == 'support':
        training['original_intent'] = 'exit'
    elif damage == 'cash':
        training.loc[3, 'close_advantage_pnl'] += 1
    elif damage == 'feature':
        training.loc[3, 'favorable_move'] = np.nan
    elif damage == 'order':
        training = training.iloc[::-1]
    else:
        training.loc[3, 'direction'] *= -1
    calls = []
    monkeypatch.setattr(FirstOpportunityCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        fit_first_opportunity(training, weights, calibration, tmp_path)
    assert not calls


def test_cost_support_overflow_and_numeric_model_damage(tmp_path):
    for values in [np.ones(499), np.r_[np.ones(600), -np.ones(63)], np.full(600, np.nan), np.r_[np.full(300, 1e308), np.full(300, -1e308)]]:
        with pytest.raises(ValueError):
            first_cost_training(values)
    training, weights, calibration = setup()
    model, _ = fit_first_opportunity(training, weights, calibration, tmp_path)
    changed = copy.deepcopy(model.to_dict())
    changed['models'][0]['trees'][0]['left'][0] = 0
    with pytest.raises(ValueError):
        FirstOpportunityCloseModel.from_dict(changed)
