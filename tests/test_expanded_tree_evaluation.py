import copy

import numpy as np
import pandas as pd
import pytest
from test_expanded_first_evaluation import expanded_examples

from wonyotti_fr.expanded_first_evaluation import evaluate_expanded_first
from wonyotti_fr.expanded_tree_evaluation import (
    EXPANDED_TREE_COMPARISONS,
    EXPANDED_TREE_POLICIES,
    evaluate_expanded_tree,
    expanded_tree_decision,
)


def tree_examples():
    frame, eleven, scores = expanded_examples()
    thirteen = evaluate_expanded_first(frame, np.full(len(scores), .6), .6, .516, eleven)
    return frame, thirteen, scores


def test_fourteen_policies_eleven_gates_all_saved_scores_recomputed():
    frame, baseline, scores = tree_examples()
    before = copy.deepcopy(baseline)
    result = evaluate_expanded_tree(frame, scores, baseline)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 11
    assert set(result['metrics']) == set(result['first_metrics']) == set(EXPANDED_TREE_POLICIES)
    pd.testing.assert_frame_equal(result['predictions'][before['predictions'].columns], before['predictions'], check_exact=True)
    for name, values in before['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], values, check_exact=True)
    for field in ['metrics', 'first_metrics', 'first_probability_metrics', 'calibration_decision']:
        assert baseline[field] == before[field]
    assert result['predictions'].expanded_first_tree_score.notna().equals(result['predictions'].expanded_first_linear_score.notna())
    assert result['positions']['expanded_first_tree'].later_selected_opportunities.eq(0).all()


def test_exact_half_rejects_first_and_keeps_all_positions():
    frame, baseline, scores = tree_examples()
    result = evaluate_expanded_tree(frame, np.full(len(scores), .5), baseline)
    assert result['first_metrics']['expanded_first_tree']['selected_positions'] == 0
    assert result['first_metrics']['expanded_first_tree']['positions'] == frame.position_entry_time.nunique()
    assert not result['calibration_decision']['calibration_passed']


def test_every_gate_invalid_values_and_population_mismatch():
    frame, baseline, scores = tree_examples()
    result = evaluate_expanded_tree(frame, scores, baseline)
    effect = result['first_metrics']['expanded_first_tree']['all_position_mean_common_bps']
    for name, key, value in [('expanded_first_tree', 'selected_positions', 29), ('expanded_first_tree', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', effect) for name in EXPANDED_TREE_COMPARISONS]]:
        changed = copy.deepcopy(result)
        changed['first_metrics'][name][key] = value
        if key == 'selected_positions':
            changed['metrics'][name][key] = value
        assert not expanded_tree_decision(changed['metrics'], changed['first_metrics'])['calibration_passed']
    for value in [True, None, np.nan, np.inf]:
        changed = copy.deepcopy(result['first_metrics'])
        changed['expanded_first_tree']['all_position_mean_common_bps'] = value
        assert not expanded_tree_decision(result['metrics'], changed)['calibration_passed']
    changed = copy.deepcopy(result['metrics'])
    changed['always_first']['rows'] += 1
    with pytest.raises(ValueError):
        expanded_tree_decision(changed, result['first_metrics'])


@pytest.mark.parametrize('damage', ['expanded_score', 'regression_score', 'regression_constant', 'old_metric', 'old_gate',
    'old_position', 'membership', 'new_score', 'later_score'])
def test_all_old_controls_and_new_score_damage_rejected(damage):
    frame, baseline, scores = tree_examples()
    if damage == 'expanded_score':
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
        evaluate_expanded_tree(frame, scores, baseline)
