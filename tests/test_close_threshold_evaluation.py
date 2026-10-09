import copy

import numpy as np
import pandas as pd
import pytest
from test_close_threshold import selection_metrics
from test_minute_close_learning import minute_cases
from test_retained_weight_close_evaluation import previous_evaluation

from wonyotti_fr.close_threshold import choose_threshold
from wonyotti_fr.close_threshold_evaluation import (
    THRESHOLD_COMPARISONS,
    THRESHOLD_POLICIES,
    THRESHOLD_PROBABILITIES,
    evaluate_threshold_close,
    threshold_admission,
)
from wonyotti_fr.retained_weight_close_evaluation import (
    RETAINED_POLICIES,
    evaluate_retained_weight_close,
)


def calibration_case():
    first = selection_metrics()
    return {'first_metrics': first, 'selection': choose_threshold(first)}


def prior_case(frame):
    values = np.linspace(.1, .9, len(frame))
    previous = previous_evaluation(frame, values, values[::-1], .4)
    return evaluate_retained_weight_close(frame, previous, values, .45)


def test_thirteen_policies_raw_score_first_cash_losses_and_five_identical_bootstraps():
    frame = minute_cases()
    previous = prior_case(frame)
    scores = np.array([.7, 1., .9, .8, .5, .6, .8, .9])
    result = evaluate_threshold_close(frame, previous, scores, .4, calibration_case())
    assert set(result['metrics']) == set(THRESHOLD_POLICIES) and len(result['metrics']) == 13
    assert set(result['probability_metrics']) == THRESHOLD_PROBABILITIES
    pd.testing.assert_frame_equal(result['predictions'][previous['predictions'].columns], previous['predictions'], check_exact=True)
    for key in ['metrics', 'first_metrics', 'probability_metrics']:
        assert {name: result[key][name] for name in previous[key]} == previous[key]
    for key in ['breakdown', 'first_breakdown', 'probability_breakdown']:
        assert result[key][:len(previous[key])] == previous[key]
    for name in RETAINED_POLICIES:
        pd.testing.assert_frame_equal(result['positions'][name], previous['positions'][name], check_exact=True)
    for name, cutoff in [('early_default', .5), ('calibrated', .8), ('always_first', -1.), ('never_extra', 1.)]:
        expected = (scores > cutoff) & frame.original_intent.ne('exit').to_numpy()
        np.testing.assert_array_equal(result['predictions']['selected_'+name], expected)
        for entry, group in frame.groupby('position_entry_time'):
            chosen = group[expected[group.index]]
            first = result['positions'][name].set_index('position_entry_time').loc[entry]
            assert first.first_effect_common_bps == pytest.approx(chosen.close_advantage_pnl.iloc[0]/group.decision_equity.iloc[0]*10000 if len(chosen) else 0.)
            if len(chosen) and name in ['early_default', 'calibrated']:
                assert first.first_cost_score == scores[chosen.index[0]]
    assert result['positions']['calibrated'].first_effect_pnl.tolist() == [-2., -3.]
    assert result['positions']['never_extra'].first_selected_time.isna().all()
    draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    np.testing.assert_array_equal(result['draws'], draws)
    for reference in THRESHOLD_COMPARISONS:
        replica = result['replicates'][reference]
        for i, draw in enumerate(draws):
            selected = [0, 1]*int((draw == 0).sum())
            assert replica.positions.iloc[i] == len(selected)
            for name in ['calibrated', reference]:
                expected = result['positions'][name].first_effect_common_bps.iloc[selected].mean() if selected else np.nan
                assert replica[name].iloc[i] == pytest.approx(expected, nan_ok=True)
        np.testing.assert_allclose(replica.paired_difference, replica.calibrated-replica[reference], equal_nan=True)
        for name in ['calibrated', reference, 'paired_difference']:
            intervals = result['intervals'][reference]['intervals'][name]
            assert intervals['lower'] == pytest.approx(np.quantile(replica[name].dropna(), .025))
            assert intervals['upper'] == pytest.approx(np.quantile(replica[name].dropna(), .975))
    assert len(result['decision']['checks']) == 12 and not result['decision']['threshold_admitted']
    future = frame.copy()
    future[['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed = evaluate_threshold_close(future, prior_case(future), scores, .4, calibration_case())
    pd.testing.assert_frame_equal(result['predictions'].filter(like='selected_'), changed['predictions'].filter(like='selected_'), check_exact=True)


def admission_case():
    metrics = {name: {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.} for name in THRESHOLD_POLICIES}
    probability = {name: {'rows': 200} for name in THRESHOLD_PROBABILITIES}
    first = {name: {'positions': 40, 'selected_positions': 30, 'selected_mean_common_bps': 2.,
        'all_position_mean_common_bps': 1.} for name in THRESHOLD_POLICIES}
    intervals = {name: {'intervals': {'calibrated': {'lower': 1.}, 'paired_difference': {'lower': 1.}}}
        for name in THRESHOLD_COMPARISONS}
    return metrics, probability, first, intervals, calibration_case()['selection']


def test_each_of_twelve_admission_gates_is_required_and_nonfinite_cannot_pass():
    original = admission_case()
    decision = threshold_admission(*original)
    assert decision['threshold_admitted'] and len(decision['checks']) == 12
    for section, key, field, value in [(0, 'calibrated', 'selected', 99), (0, 'calibrated', 'selected_positions', 29),
        (0, 'calibrated', 'selected_weighted_mean_bps', 0.), (0, 'calibrated', 'selected_mean_bps', -1.),
        (2, 'calibrated', 'selected_mean_common_bps', 0.), (2, 'calibrated', 'all_position_mean_common_bps', 0.)]:
        changed = copy.deepcopy(original)
        changed[section][key][field] = value
        if field == 'selected_positions':
            changed[2][key][field] = value
        assert not threshold_admission(*changed)['threshold_admitted']
    for reference, name in [('early_default', 'calibrated'), *[(key, 'paired_difference') for key in THRESHOLD_COMPARISONS[:-1]]]:
        changed = copy.deepcopy(original)
        changed[3][reference]['intervals'][name]['lower'] = 0.
        assert not threshold_admission(*changed)['threshold_admitted']
    changed = copy.deepcopy(original)
    changed[4]['selection_passed'] = False
    assert not threshold_admission(*changed)['threshold_admitted']
    for value in [None, np.nan, np.inf, True, 1+1j]:
        changed = copy.deepcopy(original)
        changed[2]['calibrated']['all_position_mean_common_bps'] = value
        assert not threshold_admission(*changed)['threshold_admitted']
    changed = copy.deepcopy(original)
    del changed[1]['retained_1m']
    with pytest.raises(ValueError):
        threshold_admission(*changed)


def test_selection_failure_or_forged_threshold_and_changed_population_are_rejected():
    frame = minute_cases()
    previous = prior_case(frame)
    score = np.full(len(frame), .8)
    for field, value in [('selection_passed', False), ('selected_threshold', .4)]:
        calibration = calibration_case()
        calibration['selection'][field] = value
        with pytest.raises(ValueError):
            evaluate_threshold_close(frame, previous, score, .4, calibration)
    changed = frame.copy()
    changed.loc[0, 'sample_weight'] *= 2
    with pytest.raises(AssertionError):
        evaluate_threshold_close(changed, previous, score, .4, calibration_case())
    damaged = copy.deepcopy(previous)
    damaged['positions']['legacy_5m'].loc[0, 'reference_equity'] += 1
    with pytest.raises(AssertionError):
        evaluate_threshold_close(frame, damaged, score, .4, calibration_case())
