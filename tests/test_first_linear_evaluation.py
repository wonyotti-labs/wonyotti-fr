import copy

import numpy as np
import pandas as pd
import pytest
from test_first_opportunity_evaluation import evaluation_examples

from wonyotti_fr.first_linear_evaluation import (
    LINEAR_COMPARISONS,
    LINEAR_POLICIES,
    evaluate_first_linear,
    linear_calibration_decision,
)
from wonyotti_fr.first_opportunity_evaluation import evaluate_first_opportunity


def linear_examples(constant=.516):
    frame, six, scores = evaluation_examples()
    eight = evaluate_first_opportunity(frame, np.full(len(scores), .6), constant, six)
    eight['earlier_regression_decision'] = copy.deepcopy(six['calibration_decision'])
    return frame, eight, scores


@pytest.mark.parametrize('constant', [.2, .5, .516])
def test_nine_policies_seven_gates_identical_eight_controls_and_one_first_score(constant):
    frame, previous, scores = linear_examples(constant)
    before = copy.deepcopy(previous)
    result = evaluate_first_linear(frame, scores, constant, previous)
    assert set(result['metrics']) == set(result['first_metrics']) == set(LINEAR_POLICIES)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 7
    for name in ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown', 'first_probability_metrics', 'calibration_decision']:
        assert previous[name] == before[name]
    pd.testing.assert_frame_equal(result['predictions'][previous['predictions'].columns], previous['predictions'], check_exact=True)
    for name, part in previous['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], part, check_exact=True)
    assert set(result['first_probability_metrics']) == {'first_opportunity', 'first_constant', 'first_linear'}
    pred = result['predictions']
    assert pred.first_linear_score.notna().equals(pred.first_opportunity_score.notna())
    assert not pred.loc[pred.first_linear_score.isna(), 'selected_first_linear'].any()
    assert result['positions']['first_linear'].later_selected_opportunities.eq(0).all()
    assert result['first_metrics']['first_linear']['all_position_mean_common_bps'] > result['first_metrics']['always_first']['all_position_mean_common_bps']


def test_all_seven_conditions_reject_ties_missing_values_and_wrong_population():
    frame, previous, scores = linear_examples()
    original = evaluate_first_linear(frame, scores, .516, previous)
    changes = [('first_linear', 'selected_positions', 29), ('first_linear', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', original['first_metrics']['first_linear']['all_position_mean_common_bps']) for name in LINEAR_COMPARISONS]]
    for name, key, value in changes:
        result = copy.deepcopy(original)
        result['first_metrics'][name][key] = value
        if key == 'selected_positions':
            result['metrics'][name][key] = value
        assert not linear_calibration_decision(result['metrics'], result['first_metrics'])['calibration_passed']
    for value in [True, None, np.nan, np.inf]:
        first = copy.deepcopy(original['first_metrics'])
        first['first_linear']['all_position_mean_common_bps'] = value
        assert not linear_calibration_decision(original['metrics'], first)['calibration_passed']
    metrics = copy.deepcopy(original['metrics'])
    metrics['always_first']['rows'] += 1
    with pytest.raises(ValueError):
        linear_calibration_decision(metrics, original['first_metrics'])


@pytest.mark.parametrize('damage', ['old_score', 'old_policy', 'first_membership', 'old_gate', 'first_probability', 'new_score', 'constant'])
def test_changed_old_first_policies_and_new_invalid_scores_rejected(damage):
    frame, previous, scores = linear_examples()
    constant = .516
    if damage == 'old_score':
        previous['predictions'].loc[3, 'first_opportunity_score'] = .9
    elif damage == 'old_policy':
        previous['positions']['first_constant'].loc[1, 'first_effect_pnl'] += 1
    elif damage == 'first_membership':
        previous['first_membership'].loc[1, 'first_opportunity_index'] += 1
    elif damage == 'old_gate':
        previous['calibration_decision']['calibration_passed'] = True
    elif damage == 'first_probability':
        previous['first_probability_metrics']['first_opportunity']['cost_log_loss'] += .01
    elif damage == 'new_score':
        scores[0] = np.nan
    else:
        constant = .2
    with pytest.raises((ValueError, AssertionError)):
        evaluate_first_linear(frame, scores, constant, previous)
