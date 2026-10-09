import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_close_capacity import internal_frames
from test_close_context_diagnostics import synthetic_context
from test_close_flow_diagnostics import synthetic_flow
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_flow import FlowCloseModel
from wonyotti_fr.close_learning_inputs import close_learning_splits
from wonyotti_fr.close_utility_diagnostics import fit_utility
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.weekly_close import weekly_boundaries, weekly_training_rows
from wonyotti_fr.weekly_flow import fit_flow_week, fit_weekly_flow, weekly_flow_admission


def setup():
    ledger = synthetic_flow(synthetic_context({'all': internal_frames()}, None), None)['all']
    ledger.loc[ledger.index % 13 == 0, ['close_advantage_bps', 'close_advantage_pnl']] = 0.
    rows, _ = close_learning_splits(ledger)
    weights = rows['training'][['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(rows['training']))
    model, support, costs = fit_utility(rows['training'], weights, rows['diagnosis'], model_class=FlowCloseModel)
    return ledger, rows, weights, model, support, costs


def test_all_thirteen_weekly_classifiers_costs_priors_and_routing_match_independent_refits(tmp_path):
    ledger, rows, weights, model, support, costs = setup()
    score, constant = fit_weekly_flow(ledger, rows['diagnosis'], rows['training'], weights, model.to_dict(), costs, support, tmp_path)
    models = json.loads((tmp_path/'weekly_models.json').read_text())
    details = json.loads((tmp_path/'weekly_support.json').read_text())
    membership = pd.read_parquet(tmp_path/'weekly_training_membership.parquet')
    routing = pd.read_parquet(tmp_path/'prediction_routing.parquet')
    assert len(models) == len(details) == 13 and models['week-00'] == model.to_dict()
    pd.testing.assert_frame_equal(routing.drop(columns='model_key'), rows['diagnosis'][['decision_time', 'position_entry_time']], check_exact=True)
    for i, (start, end) in enumerate(weekly_boundaries()):
        key = f'week-{i:02}'
        mask = ledger.label_status.eq('closed') & ledger.position_entry_time.ge(pd.Timestamp('2021-01-01', tz='UTC')) & ledger.label_end.lt(start-pd.Timedelta(days=2))
        train = ledger[mask]
        saved = membership[membership.model_key.eq(key)].reset_index(drop=True)
        np.testing.assert_array_equal(saved.opportunity_index, np.flatnonzero(mask))
        counts = train.position_entry_time.value_counts()
        w = np.array([len(train)/(len(counts)*counts[t]) for t in train.position_entry_time])
        np.testing.assert_allclose(saved.sample_weight, w, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(saved.original_weight, saved.sample_weight)
        y = train.close_advantage_bps.to_numpy()
        raw_cost = saved.sample_weight.to_numpy()*np.abs(y)
        fit = raw_cost > 0
        normalized = raw_cost/raw_cost[fit].mean()
        np.testing.assert_array_equal(saved.cost_weight, raw_cost)
        np.testing.assert_array_equal(saved.fit_weight, normalized)
        np.testing.assert_array_equal(saved.fit_used, fit)
        assert saved.loc[~fit, 'reason'].eq('zero_effect').all()
        prior = float(np.average(y[fit] > 0, weights=normalized[fit]))
        assert details[key]['training_constant_score'] == prior
        selected = rows['diagnosis'].decision_time.ge(start) & rows['diagnosis'].decision_time.lt(end)
        assert routing.loc[selected, 'model_key'].eq(key).all()
        assert not set(train.position_entry_time) & set(rows['diagnosis'].loc[selected, 'position_entry_time'])
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(train[model.features].to_numpy()[fit], y[fit] > 0, sample_weight=normalized[fit])
            binary = models[key]['models'][0]
            assert binary['baseline'] == learner._baseline_prediction[0, 0]
            for exported, trees in zip(binary['trees'], learner._predictors, strict=True):
                nodes = trees[0].nodes
                leaf = nodes['is_leaf'].astype(bool)
                np.testing.assert_array_equal(exported['value'], np.where(leaf, nodes['value'], 0.))
                np.testing.assert_array_equal(exported['feature'], np.where(leaf, -2, nodes['feature_idx'].astype(int)))
                np.testing.assert_array_equal(exported['threshold'], np.where(leaf, 0., nodes['num_threshold']))
            if selected.any():
                expected = learner.predict_proba(rows['diagnosis'].loc[selected, model.features].to_numpy())[:, 1]
                np.testing.assert_allclose(score[selected], expected, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(constant[selected], np.full(int(selected.sum()), prior))
    assert any(d['prediction_rows'] == 0 for d in details.values())
    assert membership.fit_used.eq(False).any()


def test_future_unmatured_inputs_and_targets_cannot_change_current_model_costs_or_scores():
    ledger, rows, _, model, _, _ = setup()
    time = weekly_boundaries()[3][0]
    validation = rows['diagnosis'][model.features].iloc[:10].to_numpy()
    before = fit_flow_week(ledger, time, validation)
    changed = ledger.copy()
    future = ~changed.index.isin(before[1])
    changed.loc[future, 'close_advantage_bps'] *= -100
    changed.loc[future, 'favorable_move'] += 10
    after = fit_flow_week(changed, time, validation)
    pd.testing.assert_frame_equal(before[0], after[0], check_exact=True)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[2], after[2])
    assert before[3].to_dict() == after[3].to_dict() and before[4] == after[4]
    pd.testing.assert_frame_equal(before[5], after[5], check_exact=True)
    np.testing.assert_array_equal(before[3].probabilities(validation), after[3].probabilities(validation))


def test_strict_two_day_maturity_boundary_unfinished_rows_and_unsupported_fit():
    ledger, rows, _, model, _, _ = setup()
    time = weekly_boundaries()[0][0]
    _, indices = weekly_training_rows(ledger, time)
    changed = ledger.copy()
    changed['label_end'] = changed.label_end.astype('datetime64[ns, UTC]')
    index = indices[-1]
    changed.loc[index, 'label_end'] = time-pd.Timedelta(days=2)
    assert index not in weekly_training_rows(changed, time)[1]
    changed.loc[index, 'label_end'] -= pd.Timedelta(nanoseconds=1)
    assert index in weekly_training_rows(changed, time)[1]
    changed.loc[index, 'label_status'] = 'right_censored'
    assert index not in weekly_training_rows(changed, time)[1]
    with pytest.raises(ValueError, match='지원 부족'):
        fit_flow_week(ledger.iloc[:100], time, rows['diagnosis'][model.features].to_numpy())


@pytest.mark.parametrize('damage', ['model', 'costs', 'prior'])
def test_first_week_must_match_original_model_entire_cost_ledger_and_prior(tmp_path, damage):
    ledger, rows, weights, model, support, costs = setup()
    original = model.to_dict()
    if damage == 'model':
        original['models'][0]['baseline'] += 1
    elif damage == 'costs':
        costs.loc[0, 'cost_weight'] += 1
    else:
        support['training_constant_score'] += .1
    with pytest.raises((AssertionError, ValueError)):
        fit_weekly_flow(ledger, rows['diagnosis'], rows['training'], weights, original, costs, support, tmp_path)


def test_all_thirty_two_conditions_keep_old_gates_and_require_static_and_weekly_prior_improvement():
    common = {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.}
    names = ['weekly_flow', 'flow', 'context', 'utility', 'training_constant', 'continuation', 'weekly', 'weekly_constant']
    metrics = {name: {**common, 'weighted_regret_bps': 1.+i} for i, name in enumerate(names)}
    probability = {name: {'rows': 200, 'cost_log_loss': .4+i*.03, 'cost_brier': .1+i*.02} for i, name in enumerate(names)}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 10.-i} for i, name in enumerate(names)}
    interval = {'intervals': {'weekly_flow': {'lower': .1}, 'paired_difference': {'lower': .1}}}
    decision = weekly_flow_admission(metrics, probability, first, interval, interval, interval, interval)
    assert decision['weekly_flow_admitted'] and len(decision['checks']) == 32
    for name in ['flow', 'weekly_constant']:
        for field in ['cost_log_loss', 'cost_brier']:
            changed = copy.deepcopy(probability)
            changed[name][field] = probability['weekly_flow'][field]-.01
            assert not weekly_flow_admission(metrics, changed, first, interval, interval, interval, interval)['weekly_flow_admitted']
        changed = copy.deepcopy(metrics)
        changed[name]['weighted_regret_bps'] = 1.
        assert not weekly_flow_admission(changed, probability, first, interval, interval, interval, interval)['weekly_flow_admitted']
    changed = copy.deepcopy(first)
    changed['flow']['all_position_mean_common_bps'] = 10.
    assert not weekly_flow_admission(metrics, probability, changed, interval, interval, interval, interval)['weekly_flow_admitted']
    changed = copy.deepcopy(interval)
    changed['intervals']['paired_difference']['lower'] = 0.
    assert not weekly_flow_admission(metrics, probability, first, interval, interval, interval, changed)['weekly_flow_admitted']
