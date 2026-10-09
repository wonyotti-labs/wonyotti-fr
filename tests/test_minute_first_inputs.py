import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_managed_first_model import fixed_manager
from test_probability_first_model import probability_source
from threadpoolctl import threadpool_limits

from wonyotti_fr.first_linear_close import FIRST_LINEAR_SETTINGS
from wonyotti_fr.minute_first_fit import fit_minute_first
from wonyotti_fr.minute_first_inputs import (
    MINUTE_FLOW_FEATURES,
    MinuteFirstLinearModel,
    attach_first_minute_flow,
)


def minute_bars(times):
    ends = pd.Series(pd.to_datetime(sorted(set(times)), utc=True), dtype='datetime64[ns, UTC]')
    return pd.DataFrame({'time': ends-pd.Timedelta(minutes=1), 'end': ends, 'open': 100., 'high': 101.,
        'low': 99., 'close': 100., 'volume': 100., 'taker_buy_volume': 50+40*np.sin(np.arange(len(ends))), 'count': 2})


def first_setup(path):
    source = probability_source(path)
    training = pd.read_parquet(source/'managed_first_training.parquet')
    calibration = pd.read_parquet(source/'managed_first_calibration.parquet')
    bars = minute_bars([*training.decision_time, *calibration.decision_time])
    return source, training, calibration, bars


def test_exact_closed_minute_zero_volume_direction_and_future_invariance(tmp_path):
    _, first, _, bars = first_setup(tmp_path/'source')
    chosen = first.iloc[:3].copy()
    bars.loc[0, ['volume', 'taker_buy_volume', 'count']] = 0
    bars.loc[0, ['open', 'high', 'low', 'close']] = 100.
    bars.loc[1, 'taker_buy_volume'] = 0.
    bars.loc[2, 'taker_buy_volume'] = 100.
    result, selected = attach_first_minute_flow(chosen, bars)
    pd.testing.assert_frame_equal(result[chosen.columns], chosen.reset_index(drop=True), check_exact=True)
    np.testing.assert_array_equal(result[MINUTE_FLOW_FEATURES[0]], [0., -1., 1.])
    np.testing.assert_array_equal(result[MINUTE_FLOW_FEATURES[1]], np.array([0., -1., 1.])*chosen.direction)
    np.testing.assert_array_equal(selected.end, chosen.decision_time)
    assert selected.end.sub(selected.time).eq(pd.Timedelta(minutes=1)).all()
    future = bars.copy()
    future.loc[future.end.gt(chosen.decision_time.max()), ['volume', 'taker_buy_volume', 'close']] = np.nan
    other, linked = attach_first_minute_flow(chosen, future)
    pd.testing.assert_frame_equal(other, result, check_exact=True)
    pd.testing.assert_frame_equal(linked, selected, check_exact=True)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'next_minute', 'naive', 'negative_buy', 'excess_buy', 'nan_buy', 'wrong_direction'])
def test_missing_future_duplicate_and_invalid_market_inputs_rejected(tmp_path, damage):
    _, first, _, bars = first_setup(tmp_path/'source')
    first = first.iloc[:3].copy()
    if damage == 'missing':
        bars = bars.iloc[1:].copy()
    elif damage == 'duplicate':
        bars = pd.concat([bars.iloc[:1], bars], ignore_index=True)
    elif damage == 'next_minute':
        bars.loc[0, ['time', 'end']] += pd.Timedelta(minutes=1)
    elif damage == 'naive':
        bars['time'] = bars.time.dt.tz_localize(None)
        bars['end'] = bars.end.dt.tz_localize(None)
    elif damage == 'wrong_direction':
        first.loc[first.index[0], 'direction'] = 0
    else:
        bars.loc[0, 'taker_buy_volume'] = {'negative_buy': -1., 'excess_buy': 101., 'nan_buy': np.nan}[damage]
    with pytest.raises((ValueError, TypeError)):
        attach_first_minute_flow(first, bars)


def test_all79_coefficients_inputs_costs_and_scores_independently_reproduced(tmp_path):
    source, training, calibration, bars = first_setup(tmp_path/'source')
    out = tmp_path/'run'
    out.mkdir()
    model, support = fit_minute_first(source, fixed_manager(), bars, out)
    table = pd.read_parquet(out/'minute_first_training.parquet')
    pd.testing.assert_frame_equal(table[training.columns], training, check_exact=True)
    assert len(model.features) == 79
    x, y = table[model.features].to_numpy(), table.first_target_common_bps.to_numpy()
    used = y != 0
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(scaler.transform(x)[used], y[used] > 0,
            sample_weight=abs(y[used])/abs(y[used]).mean())
        data = model.to_dict()
        for name, values in [('mean', scaler.mean_), ('scale', scaler.scale_), ('coefficient', learner.coef_[0])]:
            np.testing.assert_array_equal(data[name], values)
        assert data['intercept'] == learner.intercept_[0]
        for phase in ['training', 'calibration']:
            frame = pd.read_parquet(out/('minute_first_'+phase+'.parquet'))
            linked = pd.read_parquet(out/('minute_bars_'+phase+'.parquet'))
            selected = bars.set_index('end').loc[frame.decision_time]
            np.testing.assert_array_equal(frame[MINUTE_FLOW_FEATURES[0]], 2*selected.taker_buy_volume.to_numpy()/selected.volume.to_numpy()-1)
            np.testing.assert_array_equal(linked.end, frame.decision_time)
            matrix = frame[model.features].to_numpy()
            np.testing.assert_allclose(model.probabilities(matrix)[:, 0], learner.predict_proba(scaler.transform(matrix))[:, 1], rtol=0, atol=1e-12)
    assert support['eligible_positions'] == len(training)
    future = bars.copy()
    future.loc[future.end.gt(training.decision_time.max()), 'taker_buy_volume'] = 20.
    other_out = tmp_path/'future'
    other_out.mkdir()
    other, _ = fit_minute_first(source, fixed_manager(), future, other_out)
    assert other.to_dict() == model.to_dict()
    invalid = table[model.features].to_numpy(copy=True)
    invalid[0, -1] += .1
    with pytest.raises(ValueError):
        MinuteFirstLinearModel.matrix(invalid)
