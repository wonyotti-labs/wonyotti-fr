import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.label_weighting import lifecycle_weights
from wonyotti_fr.net_edge_model import NetEdgeModel, net_values


def intervals():
    origin = pd.Timestamp('2021-01-01', tz='UTC')
    return pd.DataFrame({'decision_time': origin+pd.to_timedelta([0, 2, 6], unit='min'),
                         'label_end': origin+pd.to_timedelta([4, 6, 8], unit='min'),
                         'net_bps': [-100., 10., 20.]})


def training():
    frame = pd.DataFrame(np.random.default_rng(29).normal(size=(200, len(MARKET_FEATURES))), columns=MARKET_FEATURES)
    frame['decision_time'] = pd.date_range('2021-01-02', periods=200, freq='1D', tz='UTC')
    frame['label_end'] = frame.decision_time+pd.Timedelta(hours=1)
    frame['order_direction'], frame['net_bps'] = np.tile([-1, 1], 100), np.linspace(-20, 40, 200)
    frame['favorable_bps'], frame['wait_minutes'] = np.linspace(16, 50, 200), np.tile([1, 2, 3, 4, 5], 40)
    return frame


def test_overlap_weights_match_hand_calculation_touching_boundaries_and_keep_losses():
    frame = intervals()
    weight, ledger, report = lifecycle_weights(frame)
    np.testing.assert_allclose(ledger.mean_inverse_concurrency, [.75, .75, 1])
    np.testing.assert_allclose(weight, [.9, .9, 1.2])
    assert weight.sum() == pytest.approx(3) and weight[0] > 0
    assert report['max_concurrent_labels'] == 2 and report['covered_minutes'] == 8 and report['sum_label_minutes'] == 10
    frame['net_bps'] *= -10000
    np.testing.assert_array_equal(weight, lifecycle_weights(frame)[0])


@pytest.mark.parametrize('damage', ['future', 'cutoff', 'duplicate', 'order', 'negative', 'seconds', 'naive', 'missing'])
def test_weighting_rejects_future_unordered_and_invalid_boundaries(damage):
    frame = intervals()
    if damage == 'future':
        frame.loc[2, 'label_end'] = pd.Timestamp('2022-01-01', tz='UTC')
    elif damage == 'cutoff':
        frame.loc[2, 'label_end'] = pd.Timestamp('2021-12-31', tz='UTC')
    elif damage == 'duplicate':
        frame.loc[1, 'decision_time'] = frame.loc[0, 'decision_time']
    elif damage == 'order':
        frame = frame.iloc[::-1]
    elif damage == 'negative':
        frame.loc[1, 'label_end'] = frame.loc[1, 'decision_time']
    elif damage == 'seconds':
        frame.loc[0, 'label_end'] += pd.Timedelta(seconds=1)
    elif damage == 'naive':
        frame['decision_time'] = frame.decision_time.dt.tz_localize(None)
    else:
        frame.loc[0, 'label_end'] = pd.NaT
    with pytest.raises(ValueError, match='시각'):
        lifecycle_weights(frame)


def test_weighted_scaler_and_ridge_match_independent_normal_equation():
    frame = training()
    weights = np.linspace(.1, 2., len(frame))
    weights /= weights.mean()
    model, support = NetEdgeModel.fit(frame, 100, sample_weight=weights)
    values = net_values(frame[MARKET_FEATURES], frame.order_direction, frame.favorable_bps, frame.wait_minutes)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values-mean)**2, axis=0, weights=weights))
    scale[scale == 0] = 1.
    x = np.column_stack([(values-mean)/scale, np.ones(len(frame))])
    penalty = np.eye(x.shape[1])*100
    penalty[-1, -1] = 0.
    expected = np.linalg.solve(x.T @ (weights[:, None]*x)+penalty, x.T @ (weights*frame.net_bps.to_numpy()))
    np.testing.assert_allclose(model.data['mean'], mean, atol=1e-13)
    np.testing.assert_allclose(model.data['scale'], scale, atol=1e-13)
    np.testing.assert_allclose(model.predict(values), x@expected, atol=1e-10)
    assert support['rows'] == 200 and support['weighted_target_mean_bps'] == pytest.approx(np.average(frame.net_bps, weights=weights))
    ordinary, _ = NetEdgeModel.fit(frame, 100)
    explicit, _ = NetEdgeModel.fit(frame, 100, sample_weight=None)
    assert ordinary.to_dict() == explicit.to_dict()
    unit, _ = NetEdgeModel.fit(frame, 100, sample_weight=np.ones(len(frame)))
    np.testing.assert_allclose(ordinary.predict(values), unit.predict(values), atol=1e-10)


@pytest.mark.parametrize('weights', [[1]*199, [0]*200, [-1]*200, [np.inf]*200, [np.nan]*200])
def test_invalid_weights_cannot_drop_losing_rows(weights):
    with pytest.raises(ValueError, match='가중치'):
        NetEdgeModel.fit(training(), 100, sample_weight=weights)
