import numpy as np

from wonyotti_fr.event_backtest import EventPolicy
from wonyotti_fr.event_features import MARKET_FEATURES, STATE_FEATURES
from wonyotti_fr.event_model import EventModel


def model(features, classes, intercept):
    n = len(features)
    return EventModel(features, classes, np.zeros(n), np.ones(n), np.zeros((len(classes), n)), np.array(intercept))


def test_management_excludes_same_side_entry_without_probability_inflation():
    entry = model(MARKET_FEATURES, ['enter_long', 'enter_short', 'hold'], [0, 0, 0])
    management = model(MARKET_FEATURES + STATE_FEATURES, ['enter_long', 'exit', 'hold'], [10, 5, 0])
    policy = EventPolicy(entry, management, 0.5, 0.5)
    state = {'halted': False, 'direction': 1, 'favorable_move': 0.01, 'hold_bars': 5, 'adds': 0}
    event = {'features': [0] * len(MARKET_FEATURES)}
    assert policy(event, state) == 'hold'
    management.intercept = np.array([0, 10, 0])
    assert policy(event, state) == 'exit'
    event['features'][0] = None
    assert policy(event, state) == 'hold'


def test_period_loader_preserves_time_units_and_rejects_missing_candles(monkeypatch, tmp_path):
    import pandas as pd
    import pytest

    from wonyotti_fr.event_backtest import prepare_period

    times = pd.date_range('2020-01-01', periods=600, freq='5min', tz='UTC').as_unit('us')
    data = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5),
                         'open': 100, 'high': 101, 'low': 99, 'close': 100, 'volume': 1})
    funding = pd.DataFrame({'time': pd.to_datetime(['2020-01-02T00:00:00.001Z']), 'rate': [0.0001]})
    monkeypatch.setattr('wonyotti_fr.event_backtest.load_market', lambda *_: (data, funding))
    monkeypatch.setattr('wonyotti_fr.event_backtest.check_funding_coverage', lambda *_: None)
    result = prepare_period(tmp_path, 'BTCUSDT', '2020-01-02', '2020-01-03')
    assert len(result) == 288 and result.funding_rate.iloc[0] == 0.0001
    data.drop(index=350, inplace=True)
    with pytest.raises(ValueError, match='시세가 빠졌거나'):
        prepare_period(tmp_path, 'BTCUSDT', '2020-01-02', '2020-01-03')
