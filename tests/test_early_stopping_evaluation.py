import copy

import numpy as np
import pandas as pd
import pytest
from test_close_threshold_evaluation import prior_case
from test_minute_close_learning import minute_cases

from wonyotti_fr.early_stopping_evaluation import (
    EARLY_STOPPING_COMPARISONS,
    EARLY_STOPPING_NEW_POLICIES,
    EARLY_STOPPING_POLICIES,
    EARLY_STOPPING_PROBABILITIES,
    calibration_decision,
    early_stopping_admission,
    evaluate_early_stopping_close,
    evaluate_new_policies,
)
from wonyotti_fr.retained_weight_close_evaluation import RETAINED_POLICIES


def calibration_case():
    metrics = {name: {'rows': 200, 'positions': 40, 'selected_positions': 30} for name in EARLY_STOPPING_NEW_POLICIES}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.} for name in EARLY_STOPPING_NEW_POLICIES}
    first['early_stopping']['all_position_mean_common_bps'] = 2.
    return {'metrics': metrics, 'first_metrics': first, 'calibration_decision': calibration_decision(metrics, first)}


def test_four_calibration_requirements_no_ties_fallback_search_or_missing_values():
    original = calibration_case()
    assert original['calibration_decision']['calibration_passed']
    assert len(original['calibration_decision']['checks']) == 4
    for key, field, value in [('early_stopping', 'selected_positions', 29), ('early_stopping', 'all_position_mean_common_bps', 0.),
        ('early_natural', 'all_position_mean_common_bps', 2.), ('always_first', 'all_position_mean_common_bps', 2.)]:
        case = copy.deepcopy(original)
        case['first_metrics'][key][field] = value
        if field == 'selected_positions':
            case['metrics'][key][field] = value
        decision = calibration_decision(case['metrics'], case['first_metrics'])
        assert not decision['calibration_passed'] and decision['threshold'] == .5
        assert not decision['threshold_search'] and not decision['fallback_used'] and decision['reasons']
    for value in [None, np.nan, np.inf, True, 1+1j]:
        case = copy.deepcopy(original)
        case['first_metrics']['early_stopping']['all_position_mean_common_bps'] = value
        assert not calibration_decision(case['metrics'], case['first_metrics'])['calibration_passed']


@pytest.mark.parametrize('constant', [.4, .5, .6])
def test_fixed_scores_original_exit_negative_first_effects_and_constant_policy_identity(constant):
    frame = minute_cases()
    scores = np.array([.5, 1., .9, .8, .5, .6, .8, .9])
    natural = np.full(len(frame), .6)
    result = evaluate_new_policies(frame, scores, natural, constant)
    assert set(result['metrics']) == set(EARLY_STOPPING_NEW_POLICIES)
    for name, values in [('early_stopping', scores), ('early_natural', natural), ('stopping_constant', np.full(len(frame), constant))]:
        action = (values > .5) & frame.original_intent.ne('exit').to_numpy()
        np.testing.assert_array_equal(result['predictions']['selected_'+name], action)
        for entry, group in frame.groupby('position_entry_time'):
            selected = group[action[group.index]]
            part = result['positions'][name].set_index('position_entry_time').loc[entry]
            assert part.first_effect_common_bps == pytest.approx(selected.close_advantage_pnl.iloc[0]/group.decision_equity.iloc[0]*10000 if len(selected) else 0.)
            if len(selected):
                assert part.first_cost_score == values[selected.index[0]]
    assert result['positions']['early_stopping'].first_effect_pnl.tolist() == [-2., 2.]
    counterpart = 'always_first' if constant > .5 else 'never_extra'
    pd.testing.assert_frame_equal(result['positions']['stopping_constant'].drop(columns='first_cost_score'),
        result['positions'][counterpart].drop(columns='first_cost_score'), check_exact=True)
    assert result['positions']['never_extra'].first_selected_time.isna().all()


def test_fourteen_policies_preserve_prior_outputs_and_six_paired_week_resamples():
    frame = minute_cases()
    previous = prior_case(frame)
    scores = np.array([.5, 1., .9, .8, .5, .6, .8, .9])
    natural = np.full(len(frame), .6)
    calibration = calibration_case()
    result = evaluate_early_stopping_close(frame, previous, scores, natural, .4, calibration)
    assert set(result['metrics']) == set(EARLY_STOPPING_POLICIES) and len(result['metrics']) == 14
    assert set(result['probability_metrics']) == EARLY_STOPPING_PROBABILITIES
    pd.testing.assert_frame_equal(result['predictions'][previous['predictions'].columns], previous['predictions'], check_exact=True)
    for key in ['metrics', 'first_metrics', 'probability_metrics']:
        assert {name: result[key][name] for name in previous[key]} == previous[key]
    for key in ['breakdown', 'first_breakdown', 'probability_breakdown']:
        assert result[key][:len(previous[key])] == previous[key]
    for name in RETAINED_POLICIES:
        pd.testing.assert_frame_equal(result['positions'][name], previous['positions'][name], check_exact=True)
    draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    np.testing.assert_array_equal(result['draws'], draws)
    for reference in EARLY_STOPPING_COMPARISONS:
        replicas = result['replicates'][reference]
        for i, draw in enumerate(draws):
            selected = [0, 1]*int((draw == 0).sum())
            assert replicas.positions.iloc[i] == len(selected)
            for name in ['early_stopping', reference]:
                expected = result['positions'][name].first_effect_common_bps.iloc[selected].mean() if selected else np.nan
                assert replicas[name].iloc[i] == pytest.approx(expected, nan_ok=True)
        np.testing.assert_allclose(replicas.paired_difference, replicas.early_stopping-replicas[reference], equal_nan=True)
        for name in ['early_stopping', reference, 'paired_difference']:
            interval = result['intervals'][reference]['intervals'][name]
            assert interval['lower'] == pytest.approx(np.quantile(replicas[name].dropna(), .025))
            assert interval['upper'] == pytest.approx(np.quantile(replicas[name].dropna(), .975))
    assert result['calibration_decision'] == calibration['calibration_decision']
    assert len(result['decision']['checks']) == 12 and not result['decision']['early_stopping_admitted']
    future = frame.copy()
    future[['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed = evaluate_early_stopping_close(future, prior_case(future), scores, natural, .4, calibration)
    assert changed['calibration_decision'] == result['calibration_decision']
    pd.testing.assert_frame_equal(result['predictions'].filter(like='selected_'), changed['predictions'].filter(like='selected_'), check_exact=True)


def admission_case():
    metrics = {name: {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.} for name in EARLY_STOPPING_POLICIES}
    probability = {name: {'rows': 200} for name in EARLY_STOPPING_PROBABILITIES}
    first = {name: {'positions': 40, 'selected_positions': 30, 'selected_mean_common_bps': 2.,
        'all_position_mean_common_bps': 1.} for name in EARLY_STOPPING_POLICIES}
    intervals = {name: {'intervals': {'early_stopping': {'lower': 1.}, 'paired_difference': {'lower': 1.}}}
        for name in EARLY_STOPPING_COMPARISONS}
    return metrics, probability, first, intervals, calibration_case()['calibration_decision']


def test_twelve_execution_gates_reject_each_failure_without_cross_target_probability_requirements():
    original = admission_case()
    assert early_stopping_admission(*original)['early_stopping_admitted']
    for section, field, value in [(0, 'selected', 99), (0, 'selected_positions', 29),
        (0, 'selected_weighted_mean_bps', 0.), (0, 'selected_mean_bps', 0.),
        (2, 'selected_mean_common_bps', 0.), (2, 'all_position_mean_common_bps', 0.)]:
        changed = copy.deepcopy(original)
        changed[section]['early_stopping'][field] = value
        if field == 'selected_positions':
            changed[2]['early_stopping'][field] = value
        assert not early_stopping_admission(*changed)['early_stopping_admitted']
    for reference, name in [('early_natural', 'early_stopping'), *[(key, 'paired_difference') for key in EARLY_STOPPING_COMPARISONS[:4]]]:
        changed = copy.deepcopy(original)
        changed[3][reference]['intervals'][name]['lower'] = 0.
        assert not early_stopping_admission(*changed)['early_stopping_admitted']
    changed = copy.deepcopy(original)
    changed[4]['calibration_passed'] = False
    assert not early_stopping_admission(*changed)['early_stopping_admitted']
    for value in [None, np.nan, np.inf, True, 1+1j]:
        changed = copy.deepcopy(original)
        changed[2]['early_stopping']['all_position_mean_common_bps'] = value
        assert not early_stopping_admission(*changed)['early_stopping_admitted']


def test_failed_or_forged_calibration_and_different_population_cannot_start_final_evaluation():
    frame = minute_cases()
    previous = prior_case(frame)
    score = np.full(len(frame), .6)
    calibration = calibration_case()
    calibration['calibration_decision']['calibration_passed'] = False
    with pytest.raises(ValueError):
        evaluate_early_stopping_close(frame, previous, score, score, .4, calibration)
    changed = frame.copy()
    changed.loc[0, 'sample_weight'] *= 2
    with pytest.raises(AssertionError):
        evaluate_early_stopping_close(changed, previous, score, score, .4, calibration_case())
    for value in [True, np.nan, 0., 1.]:
        with pytest.raises(ValueError):
            evaluate_new_policies(frame, score, score, value)
