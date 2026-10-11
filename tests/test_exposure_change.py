import numpy as np
import pytest
from test_net_exposure import target_fixture

from wonyotti_fr.exposure_change_model import EXPOSURE_FEATURES, ExposureChangeModel, fit_ridge
from wonyotti_fr.exposure_change_targets import change_targets
from wonyotti_fr.net_exposure_reference import independent_model_check
from wonyotti_fr.net_exposure_targets import net_exposure_targets


def test_signed_changes_and_complete_current_target_support():
    source, comparisons = target_fixture()
    base = net_exposure_targets(source, comparisons, 'training')
    base['current_net_exposure'] = [-.5, 1.5, .1, .5]
    base['current_net_exposure_min'] = base.current_net_exposure-.2
    base['current_net_exposure_max'] = base.current_net_exposure+.2
    got = change_targets(base)
    np.testing.assert_allclose(got.exposure_change, [-.5, .5, -.1, 0])
    assert len(got) == len(base) and got.change_supported.all()
    base.loc[0, ['current_net_exposure_supported', 'current_net_exposure_reason']] = [False, 'unobserved_price_trade']
    base.loc[1, ['net_exposure_supported', 'net_exposure_reason']] = [False, 'uncertain_external_transfer']
    restricted = change_targets(base)
    assert restricted.change_supported.tolist() == [False, False, True, True]
    assert restricted.change_reason.iloc[0] == 'current:unobserved_price_trade'
    assert restricted.change_reason.iloc[1] == 'target:uncertain_external_transfer'
    assert restricted.exposure_change.iloc[:2].isna().all()


@pytest.fixture(scope='module')
def trained():
    rng = np.random.default_rng(98)
    x = rng.normal(size=(1200, len(EXPOSURE_FEATURES)))
    y = np.where(x[:, 0] > 0, .015, -.015)-.002*x[:, -1]
    vx = rng.normal(size=(120, len(EXPOSURE_FEATURES)))
    model, _ = ExposureChangeModel.fit(x, y, vx)
    return x, y, vx, model


def test_batched_and_single_own_state_inference_match(trained):
    _, _, vx, model = trained
    values = [model.predict_one(row[:-1], float(row[-1])) for row in vx]
    np.testing.assert_allclose(values, model.predict(vx), atol=1e-15, rtol=0)
    restored = ExposureChangeModel.from_dict(model.to_dict())
    assert restored.to_dict()['features'][-1] == 'current_net_exposure'
    assert restored.predict_one(vx[0, :-1], float(vx[0, -1])) == values[0]


def test_full_independent_refit_and_future_invariance(trained):
    x, y, vx, model = trained
    ridge, rp = fit_ridge(x, y, vx)
    expected = independent_model_check(model.to_dict(), ridge, x, y, vx)
    np.testing.assert_allclose(expected['histogram'], model.predict(vx), atol=1e-12, rtol=0)
    np.testing.assert_allclose(expected['ridge'], rp, atol=1e-12, rtol=0)
    altered, _ = ExposureChangeModel.fit(x, y, vx*11)
    assert altered.to_dict() == model.to_dict()


@pytest.mark.parametrize('mode', ['format', 'features', 'cycle', 'nan'])
def test_changed_model_identity_and_structure_rejected(trained, mode):
    data = trained[3].to_dict()
    if mode == 'format':
        data['format'] = 'account_btc_net_exposure_histogram_v1'
    elif mode == 'features':
        data['features'].pop()
    elif mode == 'cycle':
        data['trees'][0]['left'][0] = 0
    else:
        data['trees'][0]['value'][-1] = float('nan')
    with pytest.raises(ValueError):
        ExposureChangeModel.from_dict(data)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '0'])
def test_invalid_own_account_exposure_rejected(trained, bad):
    with pytest.raises(ValueError):
        trained[3].predict_one([0.]*14, bad)


@pytest.mark.parametrize('bad', [np.zeros((14, 1)), np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)])
def test_invalid_single_market_vector_rejected(trained, bad):
    with pytest.raises(ValueError):
        trained[3].predict_one(bad, 0.)
