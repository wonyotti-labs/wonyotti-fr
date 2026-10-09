import copy

import numpy as np
import pandas as pd
import pytest
from test_first_linear_evaluation import linear_examples

from wonyotti_fr.first_linear_evaluation import evaluate_first_linear
from wonyotti_fr.weekly_first_linear_evaluation import (
    WEEKLY_FIRST_COMPARISONS,
    WEEKLY_FIRST_POLICIES,
    evaluate_weekly_first_linear,
    weekly_first_decision,
)


def weekly_examples():
    frame, eight, scores = linear_examples()
    nine = evaluate_first_linear(frame, np.full(len(scores), .6), .516, eight)
    nine['earlier_regression_decision'] = copy.deepcopy(eight['earlier_regression_decision'])
    nine['previous_first_decision'] = copy.deepcopy(eight['calibration_decision'])
    return frame, nine, scores


@pytest.mark.parametrize('constant', [.2, .5, .7])
def test_eleven_policies_nine_gates_old_controls_and_first_only_scores(constant):
    frame, baseline, scores = weekly_examples()
    before = copy.deepcopy(baseline)
    result = evaluate_weekly_first_linear(frame, scores, np.full(len(scores), constant), .516, baseline)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 9
    assert set(result['metrics']) == set(result['first_metrics']) == set(WEEKLY_FIRST_POLICIES)
    pd.testing.assert_frame_equal(result['predictions'][before['predictions'].columns], before['predictions'], check_exact=True)
    for name, values in before['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], values, check_exact=True)
    for field in ['metrics', 'first_metrics', 'first_probability_metrics', 'calibration_decision']:
        assert baseline[field] == before[field]
    for name in ['weekly_first_linear', 'weekly_first_constant']:
        assert result['predictions'][name+'_score'].notna().equals(result['predictions'].first_linear_score.notna())
        assert result['positions'][name].later_selected_opportunities.eq(0).all()
    if constant == .5:
        assert result['metrics']['weekly_first_constant']['selected_positions'] == 0


def test_every_gate_and_invalid_statistic_or_population():
    frame, baseline, scores = weekly_examples()
    result = evaluate_weekly_first_linear(frame, scores, np.full(len(scores), .6), .516, baseline)
    effect = result['first_metrics']['weekly_first_linear']['all_position_mean_common_bps']
    changes = [('weekly_first_linear', 'selected_positions', 29), ('weekly_first_linear', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', effect) for name in WEEKLY_FIRST_COMPARISONS]]
    for name, key, value in changes:
        changed = copy.deepcopy(result)
        changed['first_metrics'][name][key] = value
        if key == 'selected_positions':
            changed['metrics'][name][key] = value
        assert not weekly_first_decision(changed['metrics'], changed['first_metrics'])['calibration_passed']
    for value in [True, None, np.nan, np.inf]:
        changed = copy.deepcopy(result['first_metrics'])
        changed['weekly_first_linear']['all_position_mean_common_bps'] = value
        assert not weekly_first_decision(result['metrics'], changed)['calibration_passed']
    changed = copy.deepcopy(result['metrics'])
    changed['always_first']['rows'] += 1
    with pytest.raises(ValueError):
        weekly_first_decision(changed, result['first_metrics'])


@pytest.mark.parametrize('damage', ['old_score', 'old_metric', 'old_gate', 'old_position', 'membership', 'new_score', 'weekly_constant'])
def test_changed_controls_and_invalid_new_score_rejected(damage):
    frame, baseline, scores = weekly_examples()
    constants = np.full(len(scores), .6)
    if damage == 'old_score':
        baseline['predictions'].loc[3, 'first_linear_score'] = .9
    elif damage == 'old_metric':
        baseline['first_probability_metrics']['first_linear']['cost_log_loss'] += .1
    elif damage == 'old_gate':
        baseline['calibration_decision']['calibration_passed'] = True
    elif damage == 'old_position':
        baseline['positions']['first_linear'].loc[1, 'first_effect_pnl'] += 1
    elif damage == 'membership':
        baseline['first_membership'].loc[1, 'first_opportunity_index'] += 1
    elif damage == 'new_score':
        scores[0] = np.nan
    else:
        constants[0] = 1.1
    with pytest.raises((ValueError, AssertionError)):
        evaluate_weekly_first_linear(frame, scores, constants, .516, baseline)
