import copy

import numpy as np
import pandas as pd
import pytest
from test_managed_first_evaluation import managed_examples

from wonyotti_fr.managed_first_evaluation import evaluate_managed_first
from wonyotti_fr.probability_first_evaluation import (
    PROBABILITY_COMPARISONS,
    PROBABILITY_POLICIES,
    evaluate_probability_first,
    probability_decision,
)


def probability_examples():
    frame, fourteen, scores = managed_examples()
    fifteen = evaluate_managed_first(frame, np.full(len(scores), .6), fourteen)
    return frame, fifteen, scores


def test_sixteen_policies_thirteen_gates_all_saved_scores_recomputed():
    frame, baseline, scores = probability_examples()
    before = copy.deepcopy(baseline)
    result = evaluate_probability_first(frame, scores, baseline)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 13
    assert set(result['metrics']) == set(result['first_metrics']) == set(PROBABILITY_POLICIES)
    pd.testing.assert_frame_equal(result['predictions'][before['predictions'].columns], before['predictions'], check_exact=True)
    for name, values in before['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], values, check_exact=True)
    for field in ['metrics', 'first_metrics', 'first_probability_metrics', 'calibration_decision']:
        assert baseline[field] == before[field]
    assert result['predictions'].probability_first_linear_score.notna().equals(result['predictions'].expanded_first_linear_score.notna())
    assert result['positions']['probability_first_linear'].later_selected_opportunities.eq(0).all()


def test_exact_half_rejects_first_and_keeps_all_positions():
    frame, baseline, scores = probability_examples()
    result = evaluate_probability_first(frame, np.full(len(scores), .5), baseline)
    assert result['first_metrics']['probability_first_linear']['selected_positions'] == 0
    assert result['first_metrics']['probability_first_linear']['positions'] == frame.position_entry_time.nunique()
    assert not result['calibration_decision']['calibration_passed']


def test_every_gate_invalid_values_and_population_mismatch():
    frame, baseline, scores = probability_examples()
    result = evaluate_probability_first(frame, scores, baseline)
    effect = result['first_metrics']['probability_first_linear']['all_position_mean_common_bps']
    for name, key, value in [('probability_first_linear', 'selected_positions', 29), ('probability_first_linear', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', effect) for name in PROBABILITY_COMPARISONS]]:
        changed = copy.deepcopy(result)
        changed['first_metrics'][name][key] = value
        if key == 'selected_positions':
            changed['metrics'][name][key] = value
        assert not probability_decision(changed['metrics'], changed['first_metrics'])['calibration_passed']
    for value in [True, None, np.nan, np.inf]:
        changed = copy.deepcopy(result['first_metrics'])
        changed['probability_first_linear']['all_position_mean_common_bps'] = value
        assert not probability_decision(result['metrics'], changed)['calibration_passed']
    changed = copy.deepcopy(result['metrics'])
    changed['always_first']['rows'] += 1
    with pytest.raises(ValueError):
        probability_decision(changed, result['first_metrics'])


@pytest.mark.parametrize('damage', ['managed_score', 'tree_score', 'expanded_score', 'regression_score', 'regression_constant', 'old_metric', 'old_gate',
    'old_position', 'membership', 'new_score', 'later_score'])
def test_all_old_controls_and_new_score_damage_rejected(damage):
    frame, baseline, scores = probability_examples()
    if damage == 'managed_score':
        baseline['predictions'].loc[3, 'managed_first_linear_score'] = .9
    elif damage == 'tree_score':
        baseline['predictions'].loc[3, 'expanded_first_tree_score'] = .9
    elif damage == 'expanded_score':
        baseline['predictions'].loc[3, 'expanded_first_linear_score'] = .9
    elif damage == 'regression_score':
        baseline['predictions'].loc[3, 'stopping_regression_effect_bps'] += 1000
    elif damage == 'regression_constant':
        baseline['predictions'].loc[3, 'regression_constant_effect_bps'] += 1
    elif damage == 'old_metric':
        baseline['first_probability_metrics']['expanded_first_linear']['cost_log_loss'] += .1
    elif damage == 'old_gate':
        baseline['calibration_decision']['calibration_passed'] = not baseline['calibration_decision']['calibration_passed']
    elif damage == 'old_position':
        baseline['positions']['expanded_first_linear'].loc[1, 'first_effect_pnl'] += 1
    elif damage == 'membership':
        baseline['first_membership'].loc[1, 'first_opportunity_index'] += 1
    elif damage == 'new_score':
        scores[0] = np.nan
    else:
        baseline['predictions'].loc[4, 'expanded_first_linear_score'] = .9
    with pytest.raises((ValueError, AssertionError)):
        evaluate_probability_first(frame, scores, baseline)
