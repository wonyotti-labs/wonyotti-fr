import copy

import numpy as np
import pandas as pd
import pytest
from test_minute_close_learning import minute_cases

from wonyotti_fr.minute_close_learning import MINUTE_POLICIES, evaluate_minute_close
from wonyotti_fr.visited_close_evaluation import (
    VISITED_COMPARISONS,
    VISITED_POLICIES,
    evaluate_visited_close,
    visited_close_admission,
)


def test_old_policies_remain_exact_and_new_policies_share_population_cash_and_week_draws():
    frame = minute_cases()
    old = evaluate_minute_close(frame, np.array([.8, 1., .9, .7, .5, .9, .8, .1]),
        np.array([.4, .9, .6, .8, .9, 1., .8, .6]), .5)
    snapshot = copy.deepcopy(old)
    scores = np.array([.6, .7, .9, .8, .5, .6, .8, .9])
    result = evaluate_visited_close(frame, old, scores, .6)
    assert set(result['metrics']) == set(VISITED_POLICIES)
    pd.testing.assert_frame_equal(result['predictions'][old['predictions'].columns], old['predictions'], check_exact=True)
    pd.testing.assert_frame_equal(old['predictions'], snapshot['predictions'], check_exact=True)
    for key in ['metrics', 'first_metrics', 'probability_metrics']:
        for name, value in old[key].items():
            assert result[key][name] == value == snapshot[key][name]
    for key in ['breakdown', 'first_breakdown', 'probability_breakdown']:
        assert result[key][:len(old[key])] == old[key] == snapshot[key]
    for name in MINUTE_POLICIES:
        pd.testing.assert_frame_equal(result['positions'][name], old['positions'][name], check_exact=True)
    new = result['positions']['visited_1m']
    assert new.first_effect_pnl.tolist() == [10., 2.]
    assert result['positions']['visited_constant'].first_effect_pnl.tolist() == [10., -1.]
    assert old['positions']['legacy_5m'].chosen.tolist() == [True, False]
    for name in ['visited_1m', 'visited_constant']:
        assert len(result['positions'][name]) == 2
        assert not result['predictions'].loc[frame.original_intent.eq('exit'), 'selected_'+name].any()
        for entry, group in frame.groupby('position_entry_time'):
            chosen = group[(scores[group.index] > .5) & group.original_intent.ne('exit')] if name == 'visited_1m' else group[group.original_intent.ne('exit')]
            row = result['positions'][name].set_index('position_entry_time').loc[entry]
            assert row.first_selected_time == chosen.decision_time.iloc[0]
            assert row.first_effect_common_bps == pytest.approx(chosen.close_advantage_pnl.iloc[0]/group.decision_equity.iloc[0]*10000)
    expected_draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    np.testing.assert_array_equal(result['draws'], expected_draws)
    for reference in VISITED_COMPARISONS:
        replica = result['replicates'][reference]
        for i, draw in enumerate(expected_draws):
            selected = [0, 1]*int((draw == 0).sum())
            assert replica.positions.iloc[i] == len(selected)
            for name in ['visited_1m', reference]:
                expected = result['positions'][name].first_effect_common_bps.iloc[selected].mean() if selected else np.nan
                assert replica[name].iloc[i] == pytest.approx(expected, nan_ok=True)
        np.testing.assert_allclose(replica.paired_difference, replica.visited_1m-replica[reference], equal_nan=True)
        for name in ['visited_1m', reference, 'paired_difference']:
            interval = result['intervals'][reference]['intervals'][name]
            assert interval['lower'] == pytest.approx(np.quantile(replica[name].dropna(), .025))
            assert interval['upper'] == pytest.approx(np.quantile(replica[name].dropna(), .975))
    assert len(result['decision']['checks']) == 20 and not result['decision']['visited_close_admitted']
    altered = frame.copy()
    altered[['close_advantage_pnl', 'close_advantage_bps']] *= -1
    previous = evaluate_minute_close(altered, old['predictions'].legacy_score, old['predictions'].minute_score, .5)
    changed = evaluate_visited_close(altered, previous, scores, .6)
    pd.testing.assert_frame_equal(result['predictions'].filter(like='selected_'), changed['predictions'].filter(like='selected_'), check_exact=True)
    for name in VISITED_POLICIES:
        pd.testing.assert_series_equal(result['positions'][name].first_selected_time, changed['positions'][name].first_selected_time, check_exact=True)


def admission_case():
    common = {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2., 'weighted_regret_bps': 2.}
    metrics = {name: dict(common) for name in VISITED_POLICIES}
    metrics['visited_1m']['weighted_regret_bps'] = 1.
    probability = {name: {'rows': 200, 'cost_log_loss': .6, 'cost_brier': .3}
        for name in ['legacy_1m', 'minute_1m', 'training_constant', 'visited_1m', 'visited_constant']}
    probability['visited_1m'].update(cost_log_loss=.5, cost_brier=.2)
    first = {name: {'positions': 40, 'selected_positions': 30, 'selected_mean_common_bps': 2., 'all_position_mean_common_bps': 1.}
        for name in VISITED_POLICIES}
    intervals = {key: {'intervals': {'visited_1m': {'lower': .1}, key: {'lower': .1}, 'paired_difference': {'lower': .1}}}
        for key in VISITED_COMPARISONS}
    return metrics, probability, first, intervals


def test_all_twenty_gates_include_prior_minute_model_and_visited_constant_without_missing_value_passes():
    original = admission_case()
    result = visited_close_admission(*original)
    assert result['visited_close_admitted'] and result['candidate'] == 'visited_1m' and len(result['checks']) == 20
    mutations = [(0, 'visited_1m', 'selected', 99), (0, 'visited_1m', 'selected_positions', 29),
        (0, 'visited_1m', 'selected_weighted_mean_bps', 0.), (0, 'visited_1m', 'selected_mean_bps', -1.),
        (2, 'visited_1m', 'selected_mean_common_bps', 0.), (2, 'visited_1m', 'all_position_mean_common_bps', 0.)]
    for reference in ['legacy_1m', 'minute_1m', 'visited_constant']:
        mutations += [(0, reference, 'weighted_regret_bps', 1.), (1, reference, 'cost_log_loss', .5), (1, reference, 'cost_brier', .19)]
    for section, name, field, value in mutations:
        changed = copy.deepcopy(original)
        changed[section][name][field] = value
        if field == 'selected_positions':
            changed[2][name][field] = value
        assert not visited_close_admission(*changed)['visited_close_admitted']
    for reference, name in [('legacy_1m', 'visited_1m'), *[(key, 'paired_difference') for key in ['legacy_1m', 'legacy_5m', 'minute_1m', 'visited_constant']]]:
        changed = copy.deepcopy(original)
        changed[3][reference]['intervals'][name]['lower'] = 0.
        assert not visited_close_admission(*changed)['visited_close_admitted']
    for value in [None, np.nan, np.inf, True, 1+1j]:
        changed = copy.deepcopy(original)
        changed[1]['visited_1m']['cost_log_loss'] = value
        assert not visited_close_admission(*changed)['visited_close_admitted']
    changed = copy.deepcopy(original)
    changed[0]['visited_1m']['selected'] = np.inf
    assert not visited_close_admission(*changed)['visited_close_admitted']
    changed = copy.deepcopy(original)
    changed[0]['legacy_5m']['positions'] = 39
    with pytest.raises(ValueError, match='모집단'):
        visited_close_admission(*changed)
    changed = copy.deepcopy(original)
    del changed[1]['minute_1m']
    with pytest.raises(ValueError, match='대조 누락'):
        visited_close_admission(*changed)


def test_changed_prior_population_weights_and_invalid_new_scores_are_rejected():
    frame = minute_cases()
    score = np.full(len(frame), .6)
    previous = evaluate_minute_close(frame, score, score, .4)
    for constant in [True, np.nan, 0., 1.]:
        with pytest.raises(ValueError):
            evaluate_visited_close(frame, previous, score, constant)
    for value in [np.full(len(frame), np.nan), np.full(len(frame), 1.1), score[:-1]]:
        with pytest.raises(ValueError):
            evaluate_visited_close(frame, previous, value, .4)
    changed = frame.copy()
    changed.loc[0, 'sample_weight'] *= 2
    with pytest.raises(AssertionError):
        evaluate_visited_close(changed, previous, score, .4)
    damaged = copy.deepcopy(previous)
    damaged['positions']['legacy_5m'].loc[0, 'reference_equity'] += 1
    with pytest.raises(AssertionError):
        evaluate_visited_close(frame, damaged, score, .4)
