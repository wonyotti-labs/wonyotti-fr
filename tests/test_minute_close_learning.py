import copy

import numpy as np
import pandas as pd
import pytest
from test_close_utility import examples
from test_first_close_diagnostics import cases

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_flow import FlowCloseModel
from wonyotti_fr.minute_close_learning import (
    MINUTE_COMPARISONS,
    MINUTE_POLICIES,
    MinuteCloseModel,
    evaluate_minute_close,
    minute_close_admission,
    minute_grid_actions,
)


def minute_cases():
    frame = cases()
    frame['decision_time'] = frame.position_entry_time+pd.to_timedelta([1, 2, 5, 6, 1, 2, 3, 4], unit='min')
    frame['sample_weight'] = position_weights(frame)
    return frame


def test_factorial_policies_share_first_minute_anchor_and_keep_positions_without_five_minute_opportunities():
    frame = minute_cases()
    legacy = np.array([.8, 1., .9, .7, .5, .9, .8, .1])
    minute = np.array([.4, .9, .6, .8, .9, 1., .8, .6])
    result = evaluate_minute_close(frame, legacy, minute, .5)
    prediction, positions = result['predictions'], result['positions']
    assert positions['legacy_5m'].chosen.tolist() == [True, False]
    assert positions['legacy_1m'].first_effect_pnl.tolist() == [10., 2.]
    assert positions['legacy_5m'].first_effect_common_bps.tolist() == [-2., 0.]
    assert positions['minute_1m'].first_effect_pnl.tolist() == [-2., -1.]
    assert positions['training_constant'].chosen.tolist() == [False, False]
    for name in MINUTE_POLICIES:
        pd.testing.assert_frame_equal(positions[name][['position_entry_time', 'first_available_time', 'reference_equity']],
            positions['minute_1m'][['position_entry_time', 'first_available_time', 'reference_equity']], check_exact=True)
        assert len(positions[name]) == 2
        assert not prediction.loc[frame.original_intent.eq('exit'), 'selected_'+name].any()
    np.testing.assert_array_equal(prediction.legacy_score, legacy)
    np.testing.assert_array_equal(prediction.minute_score, minute)
    assert not result['decision']['minute_close_admitted'] and len(result['decision']['checks']) == 15
    expected_draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    np.testing.assert_array_equal(result['draws'].to_numpy(), expected_draws)
    for key, names in MINUTE_COMPARISONS.items():
        replicas = result['replicates'][key]
        for i, draw in enumerate(expected_draws):
            selected = [0, 1]*int((draw == 0).sum())
            assert replicas.positions.iloc[i] == len(selected)
            for name in names:
                expected = positions[name].first_effect_common_bps.iloc[selected].mean() if selected else np.nan
                assert replicas[name].iloc[i] == pytest.approx(expected, nan_ok=True)
        np.testing.assert_allclose(replicas.paired_difference, replicas[names[0]]-replicas[names[1]], equal_nan=True)
        for name in [*names, 'paired_difference']:
            low = np.quantile(replicas[name].dropna(), .025)
            assert result['intervals'][key]['intervals'][name]['lower'] == pytest.approx(low)
    altered = frame.copy()
    altered[['close_advantage_pnl', 'close_advantage_bps']] *= -1
    other = evaluate_minute_close(altered, legacy, minute, .5)
    pd.testing.assert_frame_equal(prediction.filter(like='selected_'), other['predictions'].filter(like='selected_'), check_exact=True)
    for name in MINUTE_POLICIES:
        pd.testing.assert_series_equal(positions[name].first_selected_time, other['positions'][name].first_selected_time, check_exact=True)


def test_grid_masks_are_strict_and_reject_invalid_scores_clocks_and_old_weights():
    frame = minute_cases()
    score = np.full(len(frame), .5)
    assert not minute_grid_actions(frame, score, 60).any()
    score[:] = .6
    assert minute_grid_actions(frame, score, 300).tolist() == [False, False, True, False, False, False, False, False]
    for seconds in [True, 60., 0, 120]:
        with pytest.raises(ValueError):
            minute_grid_actions(frame, score, seconds)
    for damaged in [np.full(len(frame), np.nan), np.full(len(frame), 1.1), score[:-1]]:
        with pytest.raises(ValueError):
            minute_grid_actions(frame, damaged, 60)
    for damage in ['duplicate', 'subminute', 'timezone', 'unknown_intent']:
        changed = frame.copy()
        if damage == 'duplicate':
            changed.loc[1, 'decision_time'] = changed.decision_time.iloc[0]
        elif damage == 'subminute':
            changed.loc[0, 'decision_time'] += pd.Timedelta(seconds=1)
        elif damage == 'timezone':
            changed['decision_time'] = changed.decision_time.dt.tz_convert('Asia/Seoul')
        else:
            changed.loc[0, 'original_intent'] = 'other'
        with pytest.raises(ValueError):
            minute_grid_actions(changed, score, 60)
    changed = frame.iloc[:-1].copy()
    with pytest.raises(AssertionError):
        evaluate_minute_close(changed, score[:-1], score[:-1], .4)
    for constant in [True, 0., 1., np.nan]:
        with pytest.raises(ValueError):
            evaluate_minute_close(frame, score, score, constant)


def admission_case():
    common = {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2., 'weighted_regret_bps': 2.}
    metrics = {name: dict(common) for name in MINUTE_POLICIES}
    metrics['minute_1m']['weighted_regret_bps'] = 1.
    probability = {name: {'rows': 200, 'cost_log_loss': .6, 'cost_brier': .3}
        for name in ['minute_1m', 'legacy_1m', 'training_constant']}
    probability['minute_1m'].update(cost_log_loss=.5, cost_brier=.2)
    first = {name: {'positions': 40, 'selected_positions': 30, 'selected_mean_common_bps': 2., 'all_position_mean_common_bps': 1.}
        for name in MINUTE_POLICIES}
    intervals = {key: {'intervals': {**{name: {'lower': .1} for name in names}, 'paired_difference': {'lower': .1}}}
        for key, names in MINUTE_COMPARISONS.items()}
    return metrics, probability, first, intervals


def test_all_fifteen_conditions_are_required_and_missing_statistics_do_not_pass():
    original = admission_case()
    result = minute_close_admission(*original)
    assert result['minute_close_admitted'] and len(result['checks']) == 15 and result['candidate'] == 'minute_1m'
    mutations = [(0, 'minute_1m', 'selected', 99), (0, 'minute_1m', 'selected_positions', 29),
        (0, 'minute_1m', 'selected_weighted_mean_bps', 0.), (0, 'minute_1m', 'selected_mean_bps', -1.),
        (2, 'minute_1m', 'selected_mean_common_bps', 0.), (2, 'minute_1m', 'all_position_mean_common_bps', 0.)]
    for reference in ['legacy_1m', 'training_constant']:
        mutations += [(0, reference, 'weighted_regret_bps', 1.), (1, reference, 'cost_log_loss', .5), (1, reference, 'cost_brier', .19)]
    for section, name, field, value in mutations:
        damaged = copy.deepcopy(original)
        damaged[section][name][field] = value
        if field == 'selected_positions':
            damaged[2][name][field] = value
        assert not minute_close_admission(*damaged)['minute_close_admitted']
    for reference, name in [('legacy_1m', 'minute_1m'), ('legacy_1m', 'paired_difference'), ('legacy_5m', 'paired_difference')]:
        damaged = copy.deepcopy(original)
        damaged[3][reference]['intervals'][name]['lower'] = 0.
        assert not minute_close_admission(*damaged)['minute_close_admitted']
    for value in [None, np.nan, np.inf]:
        damaged = copy.deepcopy(original)
        damaged[1]['minute_1m']['cost_log_loss'] = value
        assert not minute_close_admission(*damaged)['minute_close_admitted']
    damaged = copy.deepcopy(original)
    damaged[0]['legacy_5m']['positions'] = 39
    with pytest.raises(ValueError, match='모집단'):
        minute_close_admission(*damaged)


def test_minute_model_keeps_fixed_learning_and_costs_but_has_distinct_persistent_format():
    x, y, w = examples()
    rng = np.random.default_rng(75)
    x[:, 28] = np.where(np.arange(len(x))%2, 1., -1.)
    raw = rng.uniform(-1, 1, (len(x), 5))
    values = np.c_[x, rng.normal(size=(len(x), 8)), raw, raw*x[:, 28, None]]
    model, support, costs = MinuteCloseModel.fit(values, y, w, values[:100])
    baseline, old_support, old_costs = FlowCloseModel.fit(values, y, w, values[100:200])
    expected = baseline.to_dict()
    expected['format'] = MinuteCloseModel.format
    assert model.to_dict() == expected
    assert {k: v for k, v in support.items() if k != 'export'} == {k: v for k, v in old_support.items() if k != 'export'}
    pd.testing.assert_frame_equal(costs, old_costs, check_exact=True)
    np.testing.assert_array_equal(model.probabilities(values), baseline.probabilities(values))
    with pytest.raises(ValueError):
        FlowCloseModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(MinuteCloseModel.from_dict(model.to_dict()).probabilities(values), model.probabilities(values))
