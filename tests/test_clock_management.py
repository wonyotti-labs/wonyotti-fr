import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.clock_management import (
    CLOCK_FEATURES,
    ClockManagementLinear,
    ClockManagementModel,
    clock_admission,
    clock_values,
    extend_clock_frame,
)
from wonyotti_fr.clock_management_reference import (
    independent_admission,
    independent_clock_models,
    verify_clock_frame,
)
from wonyotti_fr.minute_management import ACTIONS, FEATURES
from wonyotti_fr.paired_management_evaluation import choose_thresholds, evaluate
from wonyotti_fr.paired_management_reference import manual_metrics, manual_threshold
from wonyotti_fr.position_boost_verification import compare_statistics


@pytest.mark.parametrize('unit', ['ns', 'us'])
def test_calendar_boundary_and_negative_epoch_reference(unit):
    times = pd.to_datetime(['1969-12-28T23:59:59.999999Z', '1969-12-29T00:00:00Z',
        '1970-01-01T00:00:00Z', '2021-01-03T23:55:00Z', '2021-01-04T00:00:00Z'], format='ISO8601')
    original = pd.DataFrame({'end': times.astype(f'datetime64[{unit}, UTC]'), 'label': [1, 0, 1, 0, 1]})
    result = extend_clock_frame(original)
    verify_clock_frame(original, result)
    np.testing.assert_allclose(result.loc[1, CLOCK_FEATURES].to_numpy(dtype=float), [0, 1, 0, 1], atol=1e-15)
    np.testing.assert_allclose(result.loc[4, CLOCK_FEATURES].to_numpy(dtype=float), [0, 1, 0, 1], atol=1e-15)
    assert result.loc[0, 'utc_week_cos'] > .9999


def test_nanosecond_part_is_preserved():
    frame = pd.DataFrame({'end': pd.to_datetime(['2021-01-03T23:59:59.123456789Z'])})
    verify_clock_frame(frame, extend_clock_frame(frame))


def test_append_future_and_week_shift_do_not_change_clock_inputs():
    frame = pd.DataFrame({'end': pd.date_range('2021-01-01', periods=10, freq='5min', tz='UTC')})
    before = clock_values(frame.end)
    later = pd.concat([frame, pd.DataFrame({'end': [frame.end.iloc[-1]+pd.Timedelta(days=3)]})], ignore_index=True)
    np.testing.assert_array_equal(before, clock_values(later.end)[:len(frame)])
    np.testing.assert_array_equal(before, clock_values(frame.end+pd.Timedelta(days=7)))


@pytest.mark.parametrize('bad', ['missing', 'naive', 'other_timezone', 'duplicate', 'unsorted', 'empty', 'duplicate_feature', 'no_end'])
def test_invalid_clock_inputs_rejected(bad):
    frame = pd.DataFrame({'end': pd.date_range('2021-01-01', periods=3, freq='5min', tz='UTC')})
    if bad == 'missing':
        frame.loc[1, 'end'] = pd.NaT
    elif bad == 'naive':
        frame.end = frame.end.dt.tz_localize(None)
    elif bad == 'other_timezone':
        frame.end = frame.end.dt.tz_convert('Asia/Seoul')
    elif bad == 'duplicate':
        frame.loc[1, 'end'] = frame.end.iloc[0]
    elif bad == 'unsorted':
        frame = frame.iloc[::-1]
    elif bad == 'empty':
        frame = frame.iloc[:0]
    elif bad == 'duplicate_feature':
        frame['utc_day_sin'] = 0.
    else:
        frame = frame.rename(columns={'end':'wrong'})
    with pytest.raises(ValueError):
        extend_clock_frame(frame)


def synthetic():
    rng = np.random.default_rng(17)
    n = 1500
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame['end'] = pd.date_range('2018-01-01', periods=n, freq='5min', tz='UTC')
    frame['label_end'] = frame.end+pd.Timedelta(minutes=5)
    frame = extend_clock_frame(frame)
    for i, action in enumerate(ACTIONS):
        frame['y_'+action] = (frame.iloc[:, i].to_numpy()+frame.utc_day_sin.to_numpy() > .5).astype(int)
    return frame.iloc[:1100].copy(), frame.iloc[1101:].copy()


def test_full_models_independent_fit_thresholds_requests_and_decision():
    train, cal = synthetic()
    features = ClockManagementModel.features
    x, vx = (np.ascontiguousarray(v[features].to_numpy()) for v in (train, cal))
    y, cy = (v[['y_'+a for a in ACTIONS]].to_numpy() for v in (train, cal))
    tree, _ = ClockManagementModel.fit(x, y, vx)
    linear, _, _ = ClockManagementLinear.fit(train, cal)
    reference = independent_clock_models(tree.to_dict(), linear.to_dict(), x, y, vx)
    predictions = {'clock': tree.probabilities(vx), 'linear': linear.probabilities(vx),
        'baseline': np.tile(y.mean(axis=0), (len(vx), 1)), 'constant': np.tile(y.mean(axis=0), (len(vx), 1))}
    for name, model in [('clock', tree), ('linear', linear)]:
        np.testing.assert_allclose(predictions[name], reference[name], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.vstack([model.probabilities(row[None, :]) for row in vx]), predictions[name], atol=1e-12, rtol=0)
    thresholds, _ = choose_thresholds(cy, predictions)
    metrics, decisions = evaluate(cy, predictions, thresholds)
    for name, scores in predictions.items():
        for i, action in enumerate(ACTIONS):
            threshold, _ = manual_threshold(cy[:, i], scores[:, i], (2., 1., .5)[i])
            assert threshold == thresholds[name][action]
            compare_statistics(metrics[action][name], manual_metrics(cy[:, i], scores[:, i], threshold, (2., 1., .5)[i]))
            np.testing.assert_array_equal(decisions[name]['requests'][:, i], scores[:, i] >= threshold)
    result = clock_admission(metrics)
    assert result == independent_admission(metrics) and len(result['checks']) == 17
    assert not result['profitability_accepted']
    assert not result['management_admitted']
    for model in (tree, linear):
        raw = copy.deepcopy(model.to_dict())
        raw['features'] = raw['features'][:-1]
        with pytest.raises(ValueError):
            type(model).from_dict(raw)


def test_reference_detects_clock_and_original_changes():
    frame = pd.DataFrame({'end': pd.date_range('2021-01-01', periods=3, freq='5min', tz='UTC'), 'y': [0, 1, 0]})
    changed = extend_clock_frame(frame)
    changed.loc[1, 'utc_day_sin'] += .1
    with pytest.raises(AssertionError):
        verify_clock_frame(frame, changed)
    changed = extend_clock_frame(frame)
    changed.loc[1, 'y'] = 0
    with pytest.raises(AssertionError):
        verify_clock_frame(frame, changed)
