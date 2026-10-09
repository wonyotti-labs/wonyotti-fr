import copy

import numpy as np
import pandas as pd
import pytest
from test_weekly_first_linear_evaluation import weekly_examples

from wonyotti_fr.expanded_first_evaluation import (
    EXPANDED_FIRST_COMPARISONS,
    EXPANDED_FIRST_POLICIES,
    evaluate_expanded_first,
    expanded_first_decision,
)
from wonyotti_fr.weekly_first_linear_evaluation import evaluate_weekly_first_linear


def expanded_examples():
    frame, nine, scores = weekly_examples()
    eleven = evaluate_weekly_first_linear(frame, np.full(len(scores), .6), np.full(len(scores), .6), .516, nine)
    eleven['previous_linear_decision'] = copy.deepcopy(nine['calibration_decision'])
    eleven['previous_first_decision'] = copy.deepcopy(nine['previous_first_decision'])
    eleven['earlier_regression_decision'] = copy.deepcopy(nine['earlier_regression_decision'])
    return frame, eleven, scores


@pytest.mark.parametrize('constant', [.2, .5, .7])
def test_thirteen_policies_ten_gates_prior_controls_and_first_only_scores(constant):
    frame, baseline, scores = expanded_examples()
    before = copy.deepcopy(baseline)
    result = evaluate_expanded_first(frame, scores, constant, .516, baseline)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 10
    assert set(result['metrics']) == set(result['first_metrics']) == set(EXPANDED_FIRST_POLICIES)
    pd.testing.assert_frame_equal(result['predictions'][before['predictions'].columns], before['predictions'], check_exact=True)
    for name, values in before['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], values, check_exact=True)
    for field in ['metrics', 'first_metrics', 'first_probability_metrics', 'calibration_decision']:
        assert baseline[field] == before[field]
    for name in ['expanded_first_linear', 'expanded_first_constant']:
        assert result['predictions'][name+'_score'].notna().equals(result['predictions'].first_linear_score.notna())
        assert result['positions'][name].later_selected_opportunities.eq(0).all()
    if constant == .5:
        assert result['metrics']['expanded_first_constant']['selected_positions'] == 0


def test_each_gate_invalid_value_and_population_mismatch():
    frame, baseline, scores = expanded_examples()
    result = evaluate_expanded_first(frame, scores, .6, .516, baseline)
    effect = result['first_metrics']['expanded_first_linear']['all_position_mean_common_bps']
    changes = [('expanded_first_linear', 'selected_positions', 29), ('expanded_first_linear', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', effect) for name in EXPANDED_FIRST_COMPARISONS]]
    for name, key, value in changes:
        changed = copy.deepcopy(result)
        changed['first_metrics'][name][key] = value
        if key == 'selected_positions':
            changed['metrics'][name][key] = value
        assert not expanded_first_decision(changed['metrics'], changed['first_metrics'])['calibration_passed']
    for value in [True, None, np.nan, np.inf]:
        changed = copy.deepcopy(result['first_metrics'])
        changed['expanded_first_linear']['all_position_mean_common_bps'] = value
        assert not expanded_first_decision(result['metrics'], changed)['calibration_passed']
    changed = copy.deepcopy(result['metrics'])
    changed['always_first']['rows'] += 1
    with pytest.raises(ValueError):
        expanded_first_decision(changed, result['first_metrics'])


@pytest.mark.parametrize('damage', ['weekly_score', 'old_metric', 'old_gate', 'old_position', 'membership', 'new_score', 'constant', 'weekly_constant'])
def test_changed_controls_and_invalid_new_score_rejected(damage):
    frame, baseline, scores = expanded_examples()
    constant = .6
    if damage == 'weekly_score':
        baseline['predictions'].loc[3, 'weekly_first_linear_score'] = .9
    elif damage == 'old_metric':
        baseline['first_probability_metrics']['weekly_first_linear']['cost_log_loss'] += .1
    elif damage == 'old_gate':
        baseline['calibration_decision']['calibration_passed'] = not baseline['calibration_decision']['calibration_passed']
    elif damage == 'old_position':
        baseline['positions']['weekly_first_linear'].loc[1, 'first_effect_pnl'] += 1
    elif damage == 'membership':
        baseline['first_membership'].loc[1, 'first_opportunity_index'] += 1
    elif damage == 'new_score':
        scores[0] = np.nan
    elif damage == 'constant':
        constant = 1.1
    else:
        baseline['predictions'].loc[3, 'weekly_first_constant_score'] = 1.1
    with pytest.raises((ValueError, AssertionError)):
        evaluate_expanded_first(frame, scores, constant, .516, baseline)
