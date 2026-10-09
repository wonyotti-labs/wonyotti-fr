import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_first_close_diagnostics import financial_examples
from test_minute_close_diagnostics import extend, synthetic_inputs
from test_stopping_close import assert_classifier

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_threshold import (
    THRESHOLDS,
    calibrate_threshold,
    choose_threshold,
    fit_threshold_close,
    threshold_policy,
    threshold_splits,
)


def threshold_examples():
    frame = financial_examples()
    extras = []
    for day in pd.date_range('2021-08-31', '2021-09-29', tz='UTC'):
        part = frame.iloc[:10].copy()
        delta = day-part.position_entry_time.iloc[0]
        for column in ['position_entry_time', 'decision_time', 'label_end', 'continue_end']:
            part[column] += delta
        part['direction'] = 1. if day.day % 2 else -1.
        extras.append(part)
    frame = pd.concat([frame, *extras], ignore_index=True).sort_values('decision_time').reset_index(drop=True)
    group = frame.groupby('position_entry_time').ngroup().to_numpy()
    step = frame.groupby('position_entry_time').cumcount().to_numpy()
    frame['favorable_move'] = step/100
    frame['close_advantage_bps'] = np.where(step == 0, -10., np.where(step == 1, np.where(group % 4 == 0, -8., 4.), 10.))
    frame.loc[frame.index % 113 == 0, 'close_advantage_bps'] = 0.
    frame['close_advantage_pnl'] = frame.close_advantage_bps*frame.decision_equity/10000
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    return frame


def ledger_examples():
    return extend(synthetic_inputs(threshold_examples()))


def test_strict_split_boundaries_whole_positions_indices_exclusions_and_time_units():
    original = ledger_examples()
    extras = []
    for day, end, status in [('2021-07-30T12:00Z', '2021-07-31T00:00Z', 'closed'),
        ('2021-08-01T12:00Z', '2021-08-02T03:00Z', 'closed'),
        ('2021-09-29T12:00Z', '2021-09-30T00:00Z', 'closed'),
        ('2021-10-01T12:00Z', '2021-10-02T03:00Z', 'closed'),
        ('2021-12-30T12:00Z', '2021-12-31T00:00Z', 'censored')]:
        part = original.iloc[:2].copy()
        part['position_entry_time'] = pd.Timestamp(day)
        part['decision_time'] = part.position_entry_time+pd.to_timedelta([1, 2], unit='min')
        part['label_end'] = pd.Timestamp(end)
        part['label_status'] = status
        extras.append(part)
    ledger = pd.concat([original, *extras], ignore_index=True).sort_values('decision_time').reset_index(drop=True)
    outputs = []
    for unit in ['us', 'ns']:
        frame = ledger.copy()
        for column in frame.select_dtypes('datetimetz').columns:
            frame[column] = frame[column].astype(f'datetime64[{unit}, UTC]')
        rows, assignment = threshold_splits(frame)
        np.testing.assert_array_equal(assignment.opportunity_index, np.arange(len(frame)))
        pd.testing.assert_frame_equal(assignment[frame.columns], frame, check_exact=True)
        for part in extras:
            values = assignment.loc[assignment.position_entry_time.eq(part.position_entry_time.iloc[0]), 'split']
            assert len(values) == 2 and values.str.startswith('excluded_').all()
        groups = [set(part.position_entry_time) for part in rows.values()]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        assert rows['early_training'].label_end.max() < pd.Timestamp('2021-07-31', tz='UTC')
        assert rows['calibration'].position_entry_time.min() >= pd.Timestamp('2021-08-02', tz='UTC')
        assert rows['calibration'].label_end.max() < pd.Timestamp('2021-09-30', tz='UTC')
        outputs.append(assignment.split)
    pd.testing.assert_series_equal(*outputs)
    partial = ledger.copy()
    entry = partial.position_entry_time.iloc[0]
    partial.loc[0, 'label_status'] = 'censored'
    _, assignment = threshold_splits(partial)
    assert assignment.loc[partial.position_entry_time.eq(entry), 'split'].str.startswith('excluded_').all()


@pytest.mark.parametrize('damage', ['order', 'duplicate', 'direction', 'label_end', 'timezone', 'nan_feature', 'short_support'])
def test_bad_split_inputs_are_rejected(damage):
    ledger = ledger_examples()
    if damage == 'order':
        ledger = ledger.iloc[::-1]
    elif damage == 'duplicate':
        ledger.loc[1, 'decision_time'] = ledger.decision_time.iloc[0]
    elif damage == 'direction':
        ledger.loc[1, 'direction'] *= -1
    elif damage == 'label_end':
        ledger.loc[0, 'label_end'] += pd.Timedelta(minutes=1)
    elif damage == 'timezone':
        ledger['decision_time'] = ledger.decision_time.dt.tz_localize(None)
    elif damage == 'nan_feature':
        ledger.loc[0, 'favorable_move'] = np.nan
    else:
        ledger = ledger[ledger.decision_time.ge(pd.Timestamp('2021-02-01', tz='UTC'))]
    with pytest.raises(ValueError):
        threshold_splits(ledger)


def test_early_fit_matches_independent_64_trees_weights_zero_cost_and_future_label_invariance(tmp_path):
    rows, _ = threshold_splits(ledger_examples())
    train, calibration = rows['early_training'], rows['calibration']
    model, support = fit_threshold_close(train, calibration, tmp_path)
    weights = pd.read_parquet(tmp_path/'early_training_weights.parquet').sample_weight.to_numpy()
    counts = train.position_entry_time.value_counts().to_dict()
    manual = np.array([1/counts[t] for t in train.position_entry_time])
    np.testing.assert_array_equal(weights, manual/manual.mean())
    costs = pd.read_parquet(tmp_path/'early_training_cost_ledger.parquet')
    y = train.close_advantage_bps.to_numpy()
    np.testing.assert_array_equal(costs.cost_weight, weights*np.abs(y))
    assert costs.loc[y == 0, 'fit_weight'].eq(0).all() and not costs.loc[y == 0, 'fit_used'].any()
    assert (y == 0).any() and support['diagnosis_used_for_export'] is False
    numeric = assert_classifier(model.to_dict(), train[model.features].to_numpy(), y, weights, calibration[model.features].to_numpy())
    np.testing.assert_allclose(numeric, model.probabilities(calibration[model.features].to_numpy())[:, 0], rtol=0, atol=1e-12)
    future = calibration.copy()
    future['close_advantage_bps'] *= -100
    other = tmp_path/'future'
    other.mkdir()
    changed, _ = fit_threshold_close(train, future, other)
    assert changed.to_dict() == model.to_dict() == json.loads((tmp_path/'model.json').read_text())
    frame = calibration.assign(sample_weight=position_weights(calibration))
    result = calibrate_threshold(frame, model.probabilities(frame[model.features].to_numpy())[:, 0])
    assert result['selection']['selection_passed']
    assert result['selection']['selected_threshold'] > .5


def selection_metrics():
    first = {f'threshold_{value:.1f}': {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.}
        for value in THRESHOLDS}
    first['threshold_0.5']['all_position_mean_common_bps'] = 0.
    first['always_first'] = {'positions': 40, 'selected_positions': 40, 'all_position_mean_common_bps': -1.}
    first['never_extra'] = {'positions': 40, 'selected_positions': 0, 'all_position_mean_common_bps': 0.}
    return first


def test_selection_ties_support_negative_and_missing_values_never_fall_back():
    original = selection_metrics()
    selected = choose_threshold(original)
    assert selected['selected_threshold'] == .8
    original['threshold_0.8']['selected_positions'] = 29
    assert choose_threshold(original)['selected_threshold'] == .7
    for value in [None, np.nan, np.inf, True, 0., -1.]:
        altered = copy.deepcopy(original)
        for item in altered.values():
            item['all_position_mean_common_bps'] = value
        result = choose_threshold(altered)
        assert not result['selection_passed'] and result['selected_threshold'] is None and not result['fallback_used']
    for key in ['threshold_0.2', 'always_first', 'never_extra']:
        altered = copy.deepcopy(original)
        del altered[key]
        with pytest.raises(ValueError):
            choose_threshold(altered)


def test_all_thresholds_strict_choices_losses_original_exit_no_choice_and_nullable_times():
    frame = ledger_examples()
    frame = frame.iloc[-40:].reset_index(drop=True).assign(sample_weight=1.)
    scores = np.resize(np.array([.2, .3, .4, .5, .6, .7, .8, .9, .1, 1.]), len(frame))
    frame.loc[0, 'original_intent'] = 'exit'
    result = calibrate_threshold(frame, scores)
    for cutoff in THRESHOLDS:
        key = f'threshold_{cutoff:.1f}'
        expected = (scores > cutoff) & frame.original_intent.ne('exit').to_numpy()
        np.testing.assert_array_equal(result['predictions']['selected_'+key], expected)
        for entry, group in frame.groupby('position_entry_time'):
            eligible = group[expected[group.index]]
            first = result['positions'][key].set_index('position_entry_time').loc[entry]
            assert first.first_effect_common_bps == pytest.approx(eligible.close_advantage_pnl.iloc[0]/group.decision_equity.iloc[0]*10000 if len(eligible) else 0.)
            if len(eligible):
                assert first.first_cost_score == scores[eligible.index[0]]
    never = result['positions']['never_extra']
    assert never.first_selected_time.isna().all() and str(never.first_selected_time.dtype) == 'datetime64[ns, UTC]'
    assert never.first_effect_common_bps.eq(0).all()
    for cutoff in [True, .55, np.nan]:
        with pytest.raises(ValueError):
            threshold_policy(frame, scores, cutoff)
