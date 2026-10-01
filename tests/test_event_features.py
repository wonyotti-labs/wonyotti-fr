import numpy as np
import pandas as pd

from wonyotti_fr.event_features import (
    MARKET_FEATURES,
    event_features,
    independent_orders,
    purged_train,
    training_events,
)


def bars():
    times = pd.date_range('2019-01-01', periods=650, freq='5min', tz='UTC')
    prices = 100 + np.sin(np.arange(650) / 10)
    return pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5),
                         'close': prices, 'high': prices + 1, 'low': prices - 1,
                         'volume': np.arange(650) + 1})


def source():
    t = bars().end.iloc[310]
    times = pd.to_datetime([t, t + pd.Timedelta(seconds=10), t + pd.Timedelta(hours=1)], utc=True)
    executions = pd.DataFrame({'time': times, 'symbol': 'XBTUSD', 'exectype': 'Trade',
                               'order_key': ['a', 'a', 'b'], 'orderqty': [10, 10, 10],
                               'side': ['Buy', 'Buy', 'Sell']})
    actions = pd.DataFrame({'time': times, 'order_key': ['a', 'a', 'b'],
                            'action': ['open', 'increase', 'close'], 'before_qty': [0, 4, 10],
                            'after_qty': [4, 10, 0], 'episode_id': [1, 1, 1]})
    events = pd.DataFrame({'time': times, 'order_key': ['a', 'a', 'b'],
                           'exectype': 'Trade', 'quantity': [4, 6, 10],
                           'cost_satoshi': [-4000000, -6000000, 10000000]})
    return executions, actions, events


def test_features_cannot_change_when_future_prices_change():
    original = bars()
    changed = original.copy()
    changed.loc[500:, ['close', 'high', 'low', 'volume']] *= 2
    pd.testing.assert_frame_equal(event_features(original).iloc[:500], event_features(changed).iloc[:500])
    gap = original.drop(index=400).reset_index(drop=True)
    assert event_features(gap).loc[400:, MARKET_FEATURES].isna().all().all()


def test_first_order_label_uses_submitted_quantity_and_excludes_partial_fills():
    executions, actions, _ = source()
    orders = independent_orders(executions, actions)
    assert orders.target.tolist() == ['enter_long', 'exit']
    assert orders.order_key.tolist() == ['a', 'b']


def test_exact_boundary_trade_is_target_and_never_prior_position():
    executions, actions, events = source()
    data, diagnostics = training_events(bars(), actions, events, executions)
    row = data.iloc[310]
    assert row.direction == 0 and row.target == 'enter_long'
    assert row.orders_in_window == 1 and row.usable
    following = data.iloc[311]
    assert following.direction == 1 and following.adds == 0
    assert following.average_entry == 100
    assert diagnostics['independent_orders'] == 2


def test_purge_removes_complete_boundary_episode_including_open_episode():
    frame = pd.DataFrame({'usable': True, 'label_end': pd.to_datetime(['2019-12-20'] * 3, utc=True),
                          'episode_id': [1, 2, 3], 'target_episode_id': [0, 0, 0]})
    episodes = pd.DataFrame({'episode_id': [1, 2, 3],
                             'entry_time': pd.to_datetime(['2019-12-10'] * 3, utc=True),
                             'exit_time': pd.to_datetime(['2020-01-02', None, '2019-12-22'], utc=True)})
    assert purged_train(frame, episodes, '2020-01-01').episode_id.tolist() == [3]
