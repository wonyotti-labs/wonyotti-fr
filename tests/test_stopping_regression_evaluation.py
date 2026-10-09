import copy

import numpy as np
import pandas as pd
import pytest
from test_close_threshold_evaluation import prior_case
from test_minute_close_learning import minute_cases

from wonyotti_fr.stopping_regression_diagnostics import compare_prior_positions
from wonyotti_fr.stopping_regression_evaluation import (
    REGRESSION_COMPARISONS,
    REGRESSION_NEW_POLICIES,
    REGRESSION_POLICIES,
    REGRESSION_PROBABILITIES,
    evaluate_regression_policies,
    evaluate_stopping_regression,
    regression_admission,
    regression_calibration_decision,
)


def calibration_case():
    metrics = {name: {'rows': 200, 'positions': 40, 'selected_positions': 30} for name in REGRESSION_NEW_POLICIES}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.} for name in REGRESSION_NEW_POLICIES}
    first['stopping_regression']['all_position_mean_common_bps'] = 2.
    return {'metrics': metrics, 'first_metrics': first, 'calibration_decision': regression_calibration_decision(metrics, first)}


def test_five_calibration_requirements_ties_nonfinite_values_and_no_fallback():
    original = calibration_case()
    assert original['calibration_decision']['calibration_passed']
    assert len(original['calibration_decision']['checks']) == 5
    for name, field, value in [('stopping_regression', 'selected_positions', 29),
        ('stopping_regression', 'all_position_mean_common_bps', 0.),
        *[(name, 'all_position_mean_common_bps', 2.) for name in ['early_natural', 'early_stopping', 'always_first']]]:
        case = copy.deepcopy(original)
        case['first_metrics'][name][field] = value
        if field == 'selected_positions':
            case['metrics'][name][field] = value
        decision = regression_calibration_decision(case['metrics'], case['first_metrics'])
        assert not decision['calibration_passed'] and decision['threshold_bps'] == 0.
        assert not decision['fallback_used'] and not decision['threshold_search'] and decision['reasons']
    for value in [None, np.nan, np.inf, True, 1+1j]:
        case = copy.deepcopy(original)
        case['first_metrics']['stopping_regression']['all_position_mean_common_bps'] = value
        assert not regression_calibration_decision(case['metrics'], case['first_metrics'])['calibration_passed']


@pytest.mark.parametrize('constant', [-2., 0., 2.])
def test_bp_and_cost_scores_keep_distinct_boundaries_and_constant_policy_identity(constant):
    frame = minute_cases()
    estimates = np.array([0., .1, -2., 10., -3., 1.5, .5, 2.])
    stopping, natural = np.full(len(frame), .5), np.full(len(frame), .6)
    result = evaluate_regression_policies(frame, estimates, stopping, natural, constant)
    assert set(result['metrics']) == set(REGRESSION_NEW_POLICIES)
    assert set(result['probability_metrics']) == {'early_natural', 'early_stopping'}
    expected = (estimates > 0) & frame.original_intent.ne('exit').to_numpy()
    np.testing.assert_array_equal(result['predictions'].selected_stopping_regression, expected)
    assert not result['predictions'].selected_early_stopping.any()
    assert 'stopping_regression_effect_bps' in result['predictions'] and 'stopping_regression_score' not in result['predictions']
    counterpart = 'always_first' if constant > 0 else 'never_extra'
    pd.testing.assert_frame_equal(result['positions']['regression_constant'].drop(columns='first_prediction_bps'),
        result['positions'][counterpart].drop(columns='first_cost_score'), check_exact=True)
    assert result['positions']['stopping_regression'].first_effect_pnl.tolist() == [20., 2.]
    loss_first = estimates.copy()
    loss_first[2] = .1
    negative = evaluate_regression_policies(frame, loss_first, stopping, natural, constant)
    assert negative['positions']['stopping_regression'].first_effect_pnl.tolist() == [-2., 2.]


def test_fifteen_policies_preserve_previous_and_seven_identical_week_resamples():
    frame = minute_cases()
    previous, calibration = prior_case(frame), calibration_case()
    estimates = np.array([0., .1, -2., 10., -3., 1.5, .5, 2.])
    score = np.full(len(frame), .6)
    result = evaluate_stopping_regression(frame, previous, estimates, score, score, -1., calibration)
    assert len(result['metrics']) == 15 and set(result['metrics']) == set(REGRESSION_POLICIES)
    assert set(result['probability_metrics']) == REGRESSION_PROBABILITIES
    pd.testing.assert_frame_equal(result['predictions'][previous['predictions'].columns], previous['predictions'], check_exact=True)
    for key in ['metrics', 'first_metrics', 'probability_metrics']:
        assert {name: result[key][name] for name in previous[key]} == previous[key]
    for key in ['breakdown', 'first_breakdown', 'probability_breakdown']:
        assert result[key][:len(previous[key])] == previous[key]
    for name, part in previous['positions'].items():
        pd.testing.assert_frame_equal(result['positions'][name], part, check_exact=True)
    draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    np.testing.assert_array_equal(result['draws'], draws)
    for reference in REGRESSION_COMPARISONS:
        replicas = result['replicates'][reference]
        for i, draw in enumerate(draws):
            selected = [0, 1]*int((draw == 0).sum())
            assert replicas.positions.iloc[i] == len(selected)
            for name in ['stopping_regression', reference]:
                expected = result['positions'][name].first_effect_common_bps.iloc[selected].mean() if selected else np.nan
                assert replicas[name].iloc[i] == pytest.approx(expected, nan_ok=True)
        np.testing.assert_allclose(replicas.paired_difference, replicas.stopping_regression-replicas[reference], equal_nan=True)
        for name in ['stopping_regression', reference, 'paired_difference']:
            interval = result['intervals'][reference]['intervals'][name]
            assert interval['lower'] == pytest.approx(np.quantile(replicas[name].dropna(), .025))
            assert interval['upper'] == pytest.approx(np.quantile(replicas[name].dropna(), .975))
    assert len(result['decision']['checks']) == 13 and not result['decision']['stopping_regression_admitted']
    future = frame.copy()
    future[['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed = evaluate_stopping_regression(future, prior_case(future), estimates, score, score, -1., calibration)
    assert changed['calibration_decision'] == result['calibration_decision'] == calibration['calibration_decision']
    pd.testing.assert_frame_equal(result['predictions'].filter(like='selected_'), changed['predictions'].filter(like='selected_'), check_exact=True)


def admission_case():
    metrics = {name: {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.} for name in REGRESSION_POLICIES}
    probability = {name: {'rows': 200} for name in REGRESSION_PROBABILITIES}
    first = {name: {'positions': 40, 'selected_positions': 30, 'selected_mean_common_bps': 2.,
        'all_position_mean_common_bps': 1.} for name in REGRESSION_POLICIES}
    intervals = {name: {'intervals': {'stopping_regression': {'lower': 1.}, 'paired_difference': {'lower': 1.}}} for name in REGRESSION_COMPARISONS}
    return metrics, probability, first, intervals, calibration_case()['calibration_decision']


def test_each_of_thirteen_admission_requirements_rejects_without_probability_gate():
    original = admission_case()
    assert regression_admission(*original)['stopping_regression_admitted']
    for section, field, value in [(0, 'selected', 99), (0, 'selected_positions', 29), (0, 'selected_weighted_mean_bps', 0.),
        (0, 'selected_mean_bps', 0.), (2, 'selected_mean_common_bps', 0.), (2, 'all_position_mean_common_bps', 0.)]:
        changed = copy.deepcopy(original)
        changed[section]['stopping_regression'][field] = value
        if field == 'selected_positions':
            changed[2]['stopping_regression'][field] = value
        assert not regression_admission(*changed)['stopping_regression_admitted']
    for reference, name in [('early_stopping', 'stopping_regression'), *[(key, 'paired_difference') for key in REGRESSION_COMPARISONS[:5]]]:
        changed = copy.deepcopy(original)
        changed[3][reference]['intervals'][name]['lower'] = 0.
        assert not regression_admission(*changed)['stopping_regression_admitted']
    changed = copy.deepcopy(original)
    changed[4]['calibration_passed'] = False
    assert not regression_admission(*changed)['stopping_regression_admitted']


def test_failed_or_forged_calibration_different_population_and_invalid_constants_block_evaluation():
    frame = minute_cases()
    previous, calibration = prior_case(frame), calibration_case()
    score = np.full(len(frame), .6)
    calibration['calibration_decision']['calibration_passed'] = False
    with pytest.raises(ValueError):
        evaluate_stopping_regression(frame, previous, score, score, score, 0., calibration)
    changed = frame.copy()
    changed.loc[0, 'sample_weight'] *= 2
    with pytest.raises(AssertionError):
        evaluate_stopping_regression(changed, previous, score, score, score, 0., calibration_case())
    for value in [True, np.nan, np.inf, 1+1j]:
        with pytest.raises(ValueError):
            evaluate_regression_policies(frame, score, score, score, value)


def test_saved_unselected_time_units_preserve_exact_cash_and_nanosecond_comparison(tmp_path):
    frame = minute_cases()
    score = np.full(len(frame), .5)
    result = evaluate_regression_policies(frame, score, score, score, 0.)
    for name in ['early_natural', 'early_stopping', 'always_first', 'never_extra']:
        current = result['positions'][name]
        path = tmp_path/f'{name}.parquet'
        current.to_parquet(path, index=False)
        saved = pd.read_parquet(path)
        compare_prior_positions(current, saved)
        cash = saved.copy()
        cash.loc[0, 'first_effect_pnl'] += 1
        with pytest.raises(AssertionError):
            compare_prior_positions(current, cash)
        timing = saved.copy()
        timing['first_available_time'] = timing.first_available_time.astype('datetime64[ns, UTC]')
        timing.loc[0, 'first_available_time'] += pd.Timedelta(nanoseconds=1)
        with pytest.raises(AssertionError):
            compare_prior_positions(current, timing)
