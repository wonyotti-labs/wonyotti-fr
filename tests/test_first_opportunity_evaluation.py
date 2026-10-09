import copy

import numpy as np
import pandas as pd
import pytest
from test_first_opportunity_close import first_examples

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.first_opportunity_close import first_opportunity_rows
from wonyotti_fr.first_opportunity_evaluation import (
    FIRST_COMPARISONS,
    FIRST_POLICIES,
    evaluate_first_opportunity,
    first_calibration_decision,
)
from wonyotti_fr.stopping_regression_evaluation import evaluate_regression_policies


def evaluation_examples():
    frame = first_examples('2021-08-02', 120, 8)
    frame['sample_weight'] = position_weights(frame)
    baseline = evaluate_regression_policies(frame, np.ones(len(frame)), np.full(len(frame), .8), np.full(len(frame), .7), -1.)
    first, _ = first_opportunity_rows(frame)
    scores = np.where(first.first_target_common_bps > 0, .8, .2)
    return frame, baseline, scores


@pytest.mark.parametrize('constant', [.2, .5, .8])
def test_eight_policies_original_controls_nan_later_scores_and_six_gates(constant):
    frame, baseline, scores = evaluation_examples()
    before = copy.deepcopy(baseline)
    result = evaluate_first_opportunity(frame, scores, constant, baseline)
    assert set(result['metrics']) == set(result['first_metrics']) == set(FIRST_POLICIES)
    assert result['calibration_decision']['calibration_passed'] and len(result['calibration_decision']['checks']) == 6
    assert not before['calibration_decision']['calibration_passed']
    for key in ['metrics', 'first_metrics', 'probability_metrics', 'probability_breakdown', 'breakdown', 'first_breakdown', 'calibration_decision']:
        assert baseline[key] == before[key]
    pd.testing.assert_frame_equal(result['predictions'][baseline['predictions'].columns], baseline['predictions'], check_exact=True)
    for name, part in baseline['positions'].items():
        pd.testing.assert_frame_equal(part, result['positions'][name], check_exact=True)
    first, membership = first_opportunity_rows(frame)
    for name in ['first_opportunity', 'first_constant']:
        pred = result['predictions']
        np.testing.assert_array_equal(np.flatnonzero(pred[name+'_score'].notna()), first.opportunity_index)
        assert not pred.loc[~pred.index.isin(first.opportunity_index), 'selected_'+name].any()
        assert result['positions'][name].selected_opportunities.le(1).all()
        assert result['positions'][name].later_selected_opportunities.eq(0).all()
        assert result['first_probability_metrics'][name]['rows'] == len(first)
    assert len(result['first_membership']) == len(membership) == 120
    expected = np.maximum(first.first_target_common_bps, 0).sum()/120
    assert result['first_metrics']['first_opportunity']['all_position_mean_common_bps'] == pytest.approx(expected)
    all_constant = first.first_target_common_bps.sum()/120 if constant > .5 else 0.
    assert result['first_metrics']['first_constant']['all_position_mean_common_bps'] == pytest.approx(all_constant)


def test_each_gate_fails_on_ties_negative_missing_or_insufficient_count():
    frame, baseline, scores = evaluation_examples()
    original = evaluate_first_opportunity(frame, scores, .4, baseline)
    modifications = [('first_opportunity', 'selected_positions', 29), ('first_opportunity', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', original['first_metrics']['first_opportunity']['all_position_mean_common_bps']) for name in FIRST_COMPARISONS]]
    for name, key, value in modifications:
        changed = copy.deepcopy(original)
        changed['first_metrics'][name][key] = value
        if key == 'selected_positions':
            changed['metrics'][name][key] = value
        assert not first_calibration_decision(changed['metrics'], changed['first_metrics'])['calibration_passed']
    for value in [None, np.nan, float('inf'), True]:
        changed = copy.deepcopy(original['first_metrics'])
        changed['first_opportunity']['all_position_mean_common_bps'] = value
        assert not first_calibration_decision(original['metrics'], changed)['calibration_passed']
    changed = copy.deepcopy(original['metrics'])
    changed['always_first']['positions'] += 1
    with pytest.raises(ValueError):
        first_calibration_decision(changed, original['first_metrics'])


@pytest.mark.parametrize('damage', ['old_population', 'old_action', 'old_cash', 'old_decision', 'weight', 'score_length', 'constant'])
def test_old_controls_and_bad_first_inputs_are_rejected(damage):
    frame, baseline, scores = evaluation_examples()
    constant = .4
    if damage == 'old_population':
        baseline['metrics']['always_first']['positions'] += 1
    elif damage == 'old_action':
        baseline['predictions'].loc[3, 'selected_always_first'] = False
    elif damage == 'old_cash':
        baseline['positions']['always_first'].loc[1, 'first_effect_pnl'] += 1
    elif damage == 'old_decision':
        baseline['calibration_decision']['calibration_passed'] = True
    elif damage == 'weight':
        frame.loc[0, 'sample_weight'] *= 2
    elif damage == 'score_length':
        scores = scores[:-1]
    else:
        constant = True
    with pytest.raises((ValueError, AssertionError)):
        evaluate_first_opportunity(frame, scores, constant, baseline)
