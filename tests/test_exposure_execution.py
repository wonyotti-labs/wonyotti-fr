import copy

import numpy as np
import pandas as pd
import pytest
from test_exposure_engine import bar, config

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.exposure_change_model import ExposureChangeModel
from wonyotti_fr.exposure_engine import ExposureEngine
from wonyotti_fr.exposure_policy import ExposurePolicy
from wonyotti_fr.exposure_reference import ReferenceAccount, reference_policy, require_near
from wonyotti_fr.net_exposure_model import SETTINGS, NetExposureModel


def model(kind):
    cls = ExposureChangeModel if kind == 'change' else NetExposureModel
    feature = 14 if kind == 'change' else 0
    tree = {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [feature, -2, -2],
        'threshold': [0., 0., 0.], 'value': [0., -.001, .001]}
    return cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': SETTINGS,
        'baseline': .02, 'trees': [tree for _ in range(64)]})


@pytest.mark.parametrize('kind', ['change', 'absolute', 'constant'])
def test_own_account_policy_ignores_original_state_and_future_fields(kind):
    learner = None if kind == 'constant' else model(kind)
    policy = ExposurePolicy(kind, learner, constant_exposure=.25 if kind == 'constant' else None)
    event = bar(4)
    event.update(features=[0.]*len(MARKET_FEATURES), feature_end=event['end'])
    own = {'equity': 10000., 'quantity': 20., 'net_exposure': -999.}
    wanted = reference_policy(kind, None if learner is None else learner.to_dict(), event, 8000., 20., constant_exposure=.25 if kind == 'constant' else None)
    assert policy(event, own) == pytest.approx(wanted)
    mutated = copy.deepcopy(event)
    mutated.update(original_current_net_exposure=-99., next_net_exposure=99., future_close=100000.)
    assert policy(mutated, own) == policy(event, own)
    assert policy(bar(3), own) is None
    if kind == 'change':
        assert policy(event, {'equity': 10000., 'quantity': -20.}) != policy(event, own)
    if kind == 'constant':
        assert policy(event, own) == .25


@pytest.mark.parametrize('kind', ['change', 'absolute'])
def test_policy_time_and_invalid_feature_checks(kind):
    policy = ExposurePolicy(kind, model(kind))
    event = bar(4)
    event['features'] = [0.]*14
    with pytest.raises(ValueError, match='시각'):
        policy(event, {'equity': 10000., 'quantity': 0.})
    event['feature_end'] = event['end']
    event['features'][0] = None
    assert policy(event, {'equity': 10000., 'quantity': 0.}) is None
    event['features'] = [0.]*13
    with pytest.raises(ValueError, match='차원'):
        policy(event, {'equity': 10000., 'quantity': 0.})


@pytest.mark.parametrize('kind', ['change', 'absolute'])
def test_cached_policy_matches_batch_and_independent_tree_walk(kind):
    learner = model(kind)
    policy = ExposurePolicy(kind, learner, constant_exposure=.25 if kind == 'constant' else None)
    random = np.random.default_rng(9801)
    market = random.normal(size=(1000, 14))
    current = random.uniform(-.5, .5, 1000)
    values = np.column_stack([market, current]) if kind == 'change' else market
    expected = learner.predict(values)+(current if kind == 'change' else 0)
    for row, exposure, value in zip(market, current, expected, strict=True):
        event = bar(4)
        event.update(features=row.tolist(), feature_end=event['end'])
        own = {'equity': 10000., 'quantity': exposure*100}
        assert policy(event, own) == pytest.approx(np.clip(value, -.5, .5), abs=1e-14)
        assert reference_policy(kind, learner.to_dict(), event, 10000.-exposure*10000., exposure*100) == pytest.approx(policy(event, own))


@pytest.mark.parametrize('delay', [0, 1])
@pytest.mark.parametrize('seed', [12, 987])
def test_independent_full_replay_random_prices_funding_risk_and_partial_trades(delay, seed):
    settings = config(signal_delay_bars=delay, stop_fraction=.04, max_hold_bars=111,
        cooldown_bars=15, daily_loss_limit=.03, max_drawdown=.25)
    engine, reference = ExposureEngine(settings), ReferenceAccount(settings)
    random = np.random.default_rng(seed)
    price = 100.
    for i in range(3000):
        price *= float(np.exp(random.normal(0, .005)))
        close = price*float(np.exp(random.normal(0, .003)))
        volume = 0 if i % 41 == 0 and i != 2999 else 1
        event = bar(i, price=price, high=max(price, close)*1.001, low=min(price, close)*.999,
            closing=close, funding=.0001 if i % 480 == 0 else 0., count=volume)
        if not volume:
            event.update(high=price, low=price, close=price)
        value = float(random.choice([-.5, -.2, 0., .1, .4])) if i % 5 == 0 else None
        actual = engine.step(event, lambda b, s, value=value: value, final=i == 2999)
        expected = reference.step(event, lambda b, cash, q, value=value: value, final=i == 2999)
        assert abs(actual.pop('accounting_residual')) < 1e-7
        require_near(actual, expected, str(i))
    assert engine.state['closed_net'] == pytest.approx(reference.cash-10000., abs=1e-7)
    assert reference.trades == engine.state['closed_trades']


def test_daily_loss_resets_at_new_utc_day_before_funding():
    settings = config(daily_loss_limit=.01)
    engine, reference = ExposureEngine(settings), ReferenceAccount(settings)
    for i in range(1444):
        event = bar(i, price=100. if i < 2 else 95., funding=.03 if i == 1440 else 0.)
        result = engine.step(event, lambda b, s: .5)
        expected = reference.step(event, lambda b, cash, q: .5)
        result.pop('accounting_residual')
        require_near(result, expected, str(i))
        if i == 2:
            assert result['daily_halted'] and not result['halted'] and result['quantity'] == 0
        if 2 < i < 1440:
            assert result['quantity'] == 0
        if i == 1440:
            assert not result['daily_halted'] and result['quantity'] == 0
        if i == 1441:
            assert result['quantity'] > 0


def test_independent_comparison_rejects_missing_altered_and_extra_records():
    for actual in [{'x': 1.1}, {'x': 1., 'extra': 1}, {'x': None}, {}]:
        with pytest.raises(ValueError):
            require_near(actual, {'x': 1.})
    with pytest.raises(ValueError):
        require_near([], [{'x': 1.}])


def test_completed_engine_cannot_be_advanced():
    engine = ExposureEngine(config())
    engine.step(bar(0), lambda b, s: None, final=True)
    with pytest.raises(ValueError):
        engine.step(bar(1), lambda b, s: None)
    assert pd.Timestamp(engine.state['last_end']) == pd.Timestamp(bar(0)['end'])


@pytest.mark.parametrize('value', [None, True, '.25', float('nan'), float('inf'), -.51, .51])
def test_training_constant_must_be_explicit_finite_and_bounded(value):
    with pytest.raises(ValueError, match='상수'):
        ExposurePolicy('constant', constant_exposure=value)
    with pytest.raises(ValueError, match='상수'):
        reference_policy('constant', None, bar(4), 10000., 0., constant_exposure=value)


@pytest.mark.parametrize('value', [-.5, 0., .25, .5])
def test_training_constant_is_external_configuration(value):
    policy = ExposurePolicy('constant', constant_exposure=value)
    assert policy(bar(4), {'equity': 10000., 'quantity': 0.}) == value
    assert reference_policy('constant', None, bar(4), 10000., 0., constant_exposure=value) == value
    assert policy.to_dict() == {'kind': 'constant', 'model': None, 'constant_exposure': value}


def test_nonconstant_policy_rejects_an_unused_training_constant():
    with pytest.raises(ValueError, match='비상수'):
        ExposurePolicy('change', model('change'), constant_exposure=.25)
