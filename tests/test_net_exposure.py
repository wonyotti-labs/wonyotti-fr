import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.net_exposure_evaluation import exposure_metrics
from wonyotti_fr.net_exposure_model import NetExposureModel, fit_ridge
from wonyotti_fr.net_exposure_reference import (
    independent_metrics,
    independent_model_check,
    independent_target_check,
    require_exact_metrics,
)
from wonyotti_fr.net_exposure_targets import net_exposure_targets


def target_fixture():
    source = pd.DataFrame({'end': pd.date_range('2020-03-01', periods=4, freq='5min', tz='UTC')})
    source['label_end'] = source.end+pd.Timedelta(minutes=5)
    source['episode_id'] = [1, 1, 1, 2]
    source['position_target_episode_id'] = [1, 1, 2, 2]
    for field in MARKET_FEATURES:
        source[field] = 0.
    rows = []
    for boundary, name in [('current', 'end'), ('target', 'label_end')]:
        for i, row in enumerate(source.itertuples(index=False)):
            delta = [-100000000, 200000000, 0, 50000000][i]
            rows.append({'part': 'training', 'boundary': boundary, 'time': getattr(row, name),
                'source_end': row.end, 'source_episode_id': row.episode_id,
                'source_target_episode_id': row.position_target_episode_id,
                'all_prices_have_observed_trade': True, 'external_uncertain_count': 0,
                'unsupported_reasons': '', 'equity_min_satoshi': '100000000', 'equity_max_satoshi': '100000000',
                'delta_min_satoshi': str(delta), 'delta_max_satoshi': str(delta),
                'stress_equity_min_satoshi': '99000000',
                'btc_partial_leverage_min': delta/100000000-.1, 'btc_partial_leverage_max': delta/100000000+.1})
    return source, pd.DataFrame(rows)


def test_target_is_signed_net_exposure_and_preserves_all_rows():
    source, comparisons = target_fixture()
    got = net_exposure_targets(source, comparisons, 'training')
    assert got.net_exposure.tolist() == [-1., 2., 0., .5]
    assert got.net_exposure_supported.all() and len(got) == len(source)
    assert got.current_net_exposure.tolist() == got.net_exposure.tolist()
    pd.testing.assert_frame_equal(got[source.columns], source, check_exact=True)
    assert independent_target_check(source, got, comparisons, 'training') == 4


@pytest.mark.parametrize('mode,reason', [('price', 'unobserved_price_trade'),
    ('external', 'uncertain_external_transfer'), ('valuation', 'unsupported_valuation'),
    ('equity', 'nonpositive_equity_lower_bound')])
def test_unknown_target_is_retained_and_not_imputed(mode, reason):
    source, comparisons = target_fixture()
    if mode == 'price':
        comparisons.loc[4, 'all_prices_have_observed_trade'] = False
    elif mode == 'external':
        comparisons.loc[4, 'external_uncertain_count'] = 1
    elif mode == 'valuation':
        comparisons.loc[4, 'unsupported_reasons'] = 'ETHUSD:price'
    else:
        comparisons.loc[4, 'stress_equity_min_satoshi'] = '-1'
    got = net_exposure_targets(source, comparisons, 'training')
    assert got.net_exposure_reason.iloc[0] == reason
    assert np.isnan(got.net_exposure.iloc[0]) and len(got) == 4
    assert got.current_net_exposure_supported.all()
    assert independent_target_check(source, got, comparisons, 'training') == 4


@pytest.mark.parametrize('mode', ['time', 'source_time', 'episode', 'count', 'window', 'market', 'bounds', 'cash'])
def test_target_join_or_value_inconsistency_rejected(mode):
    source, comparisons = target_fixture()
    if mode == 'time':
        comparisons.loc[4, 'time'] += pd.Timedelta(minutes=5)
    elif mode == 'source_time':
        comparisons.loc[4, 'source_end'] += pd.Timedelta(minutes=5)
    elif mode == 'episode':
        comparisons.loc[4, 'source_target_episode_id'] = 999
    elif mode == 'count':
        comparisons = comparisons.iloc[1:]
    elif mode == 'window':
        source.loc[0, 'label_end'] += pd.Timedelta(minutes=5)
    elif mode == 'market':
        source.loc[0, MARKET_FEATURES[0]] = np.nan
    elif mode == 'bounds':
        comparisons.loc[4, 'btc_partial_leverage_max'] = -2.
    else:
        comparisons.loc[4, 'equity_max_satoshi'] = '200000000'
    with pytest.raises(ValueError):
        net_exposure_targets(source, comparisons, 'training')


@pytest.fixture(scope='module')
def model_case():
    rng = np.random.default_rng(97)
    x = rng.normal(size=(1200, len(MARKET_FEATURES)))
    y = np.where(x[:, 0] > .1, 1., -1.)+.2*x[:, 1]
    vx = rng.normal(size=(100, len(MARKET_FEATURES)))
    model, support = NetExposureModel.fit(x, y, vx)
    return x, y, vx, model, support


def test_histogram_export_and_future_input_invariance(model_case):
    x, y, vx, model, support = model_case
    assert support['training_rows'] == 1200 and support['diagnosis_export_error'] <= 1e-12
    restored = NetExposureModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict(vx), restored.predict(vx))
    changed, _ = NetExposureModel.fit(x, y, vx*9)
    assert changed.to_dict() == model.to_dict()


def test_ridge_training_only_statistics_and_closed_form(model_case):
    x, y, vx, _, _ = model_case
    data, prediction = fit_ridge(x, y, vx)
    z = (x-x.mean(axis=0))/x.std(axis=0)
    coefficients = np.linalg.solve(z.T@z+np.eye(x.shape[1]), z.T@(y-y.mean()))
    independently = ((vx-x.mean(axis=0))/x.std(axis=0))@coefficients+y.mean()
    np.testing.assert_allclose(prediction, independently, atol=1e-12, rtol=0)
    changed, _ = fit_ridge(x, y, vx*9)
    assert changed == data


@pytest.mark.parametrize('mode', ['format', 'features', 'settings', 'baseline', 'trees', 'cycle', 'unreachable', 'depth', 'nan', 'bool'])
def test_model_shape_and_numeric_tampering_rejected(model_case, mode):
    data = model_case[3].to_dict()
    if mode == 'format':
        data['format'] = 'position_target_histogram_v1'
    elif mode == 'features':
        data['features'] = list(reversed(MARKET_FEATURES))
    elif mode == 'settings':
        data['settings']['max_iter'] = 65
    elif mode == 'baseline':
        data['baseline'] = float('inf')
    elif mode == 'trees':
        data['trees'].pop()
    elif mode == 'cycle':
        data['trees'][0]['left'][0] = 0
    elif mode == 'unreachable':
        data['trees'][0]['left'][0] = data['trees'][0]['right'][0]
    elif mode == 'depth':
        data['trees'][0]['feature'][0] = 14
    elif mode == 'nan':
        data['trees'][0]['threshold'][0] = float('nan')
    else:
        data['trees'][0]['left'][0] = True
    with pytest.raises(ValueError):
        NetExposureModel.from_dict(data)


@pytest.mark.parametrize('mode', ['short', 'one_side', 'nan', 'shape'])
def test_training_support_rejected(model_case, mode):
    x, y, vx, _, _ = copy.deepcopy(model_case)
    if mode == 'short':
        x, y = x[:900], y[:900]
    elif mode == 'one_side':
        y = np.abs(y)
    elif mode == 'nan':
        y[0] = np.nan
    else:
        x = x[:, :-1]
    with pytest.raises(ValueError):
        NetExposureModel.fit(x, y, vx)


def test_regression_and_interval_metrics_have_expected_denominators():
    result = exposure_metrics([-1, 0, 2], [-.5, .5, 1.5], [-1.1, -.1, 1.9], [-.9, .1, 2.1])
    assert result['mse'] == .25 and result['mae'] == .5
    assert result['direction_accuracy'] == 2/3
    assert result['long_recall'] == result['short_recall'] == 1
    assert result['interval_mse'] == pytest.approx(.16)
    assert result['independent_trade_count'] is None
    require_exact_metrics(independent_metrics([-1, 0, 2], [-.5, .5, 1.5], [-1.1, -.1, 1.9], [-.9, .1, 2.1]), result)


def test_independent_full_refit_and_closed_form_prediction(model_case):
    x, y, vx, model, _ = model_case
    ridge, prediction = fit_ridge(x, y, vx)
    expected = independent_model_check(model.to_dict(), ridge, x, y, vx)
    np.testing.assert_allclose(expected['ridge'], prediction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(expected['histogram'], model.predict(vx), atol=1e-12, rtol=0)
    bad = model.to_dict()
    bad['trees'][0]['value'][-1] += .01
    with pytest.raises(ValueError):
        independent_model_check(bad, ridge, x, y, vx)
    ridge['coef'][0] += .01
    with pytest.raises(ValueError):
        independent_model_check(model.to_dict(), ridge, x, y, vx)


@pytest.mark.parametrize('field,value', [('net_exposure', 9.), ('net_exposure_supported', False),
    ('net_exposure_reason', 'deleted'), ('current_net_exposure', 9.), ('net_exposure_min', -9.),
    ('end', pd.Timestamp('2020-03-02', tz='UTC'))])
def test_independent_target_tampering_rejected(field, value):
    source, comparisons = target_fixture()
    got = net_exposure_targets(source, comparisons, 'training')
    got.loc[0, field] = value
    with pytest.raises((ValueError, AssertionError)):
        independent_target_check(source, got, comparisons, 'training')


def test_current_original_state_does_not_enter_market_inputs():
    source, comparisons = target_fixture()
    a = net_exposure_targets(source, comparisons, 'training')
    comparisons.loc[comparisons.boundary.eq('current'), ['delta_min_satoshi', 'delta_max_satoshi']] = '0'
    comparisons.loc[comparisons.boundary.eq('current'), ['btc_partial_leverage_min', 'btc_partial_leverage_max']] = 0.
    b = net_exposure_targets(source, comparisons, 'training')
    pd.testing.assert_frame_equal(a[MARKET_FEATURES], b[MARKET_FEATURES])
    np.testing.assert_array_equal(a.net_exposure, b.net_exposure)


@pytest.mark.parametrize('key', ['rows', 'mse', 'long_recall', 'interval_mae'])
def test_independent_metrics_tampering_rejected(key):
    a = exposure_metrics([-1., 2.], [-.5, 1.], [-1.1, 1.9], [-.9, 2.1])
    b = independent_metrics([-1., 2.], [-.5, 1.], [-1.1, 1.9], [-.9, 2.1])
    a[key] += .1
    with pytest.raises(ValueError):
        require_exact_metrics(b, a)
