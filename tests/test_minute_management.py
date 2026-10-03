import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.minute_management import (
    FEATURES,
    management_events,
    management_orders,
    management_values,
)


def inputs():
    times = pd.date_range('2020-01-01', periods=1800, freq='min', tz='UTC').as_unit('ns')
    minute = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=1),
                           'close': 100 + np.sin(np.arange(len(times)) / 23)})
    five = minute.groupby(minute.end.dt.ceil('5min')).agg(close=('close', 'last')).reset_index()
    five['time'] = five.end - pd.Timedelta(minutes=5)
    five['open'], five['high'], five['low'], five['volume'] = five.close, five.close + 1, five.close - 1, 10.
    start = times[1500]
    states = pd.DataFrame({'state_time': [times[1000], start], 'direction': [1, -1],
                          'average_entry': [100., 102.], 'entry_time': [times[1000], start],
                          'adds': [0, 0], 'episode_id': [1, 2]})
    orders = pd.DataFrame({'order_key': ['a', 'b', 'c', 'd', 'e'],
                           'target_time': [start, start + pd.Timedelta(seconds=10), start + pd.Timedelta(seconds=20),
                                           start + pd.Timedelta(seconds=30), start + pd.Timedelta(minutes=1)],
                           'before_qty': [10, -10, 10, 0, -10], 'target_episode_id': [2, 2, 1, 2, 2],
                           'before_episode_id': [1, 2, 1, 0, 2],
                           'target': ['enter_short', 'increase', 'reduce', 'enter_short', 'increase']})
    return minute, five, states, orders


def test_all_management_actions_and_exact_boundary_state_are_preserved():
    minute, five, states, orders = inputs()
    frame, ledger, summary = management_events(minute, five, states, orders, minute.end.iloc[-1])
    boundary = orders.target_time.iloc[0]
    row = frame[frame.end.eq(boundary)].iloc[0]
    assert row.direction == 1 and row.y_exit == row.y_reduce == 1
    assert row.exit_count == row.reduce_count == 1 and row.y_increase == 0
    assert frame.loc[frame.end.eq(boundary + pd.Timedelta(minutes=1)), 'y_increase'].item() == 1
    assert ledger.reason.tolist() == ['linked', 'different_direction', 'linked', 'new_entry', 'linked']
    assert summary['multiple_action_windows'] == 1
    assert sum(summary['reasons'].values()) == len(orders)


def test_order_multiplicity_is_counted_without_overwriting_first_action():
    minute, five, states, orders = inputs()
    extra = orders.iloc[[2]].assign(order_key='f', target_time=orders.target_time.iloc[2] + pd.Timedelta(seconds=1))
    orders = pd.concat([orders, extra]).sort_values('target_time')
    frame, _, summary = management_events(minute, five, states, orders, minute.end.iloc[-1])
    row = frame[frame.end.eq(orders.target_time.iloc[0])].iloc[0]
    assert row.reduce_count == 2 and row.y_reduce == 1
    assert summary['action_order_counts']['reduce'] == 2 and summary['action_window_counts']['reduce'] == 1


def test_future_price_and_state_changes_do_not_change_past_features_or_labels():
    minute, five, states, orders = inputs()
    before, _, _ = management_events(minute, five, states, orders, minute.end.iloc[-1])
    boundary = minute.end.iloc[1650]
    changed, changed_five = minute.copy(), five.copy()
    changed.loc[changed.end.ge(boundary), 'close'] *= 2
    changed_five.loc[changed_five.end.ge(boundary), ['open', 'high', 'low', 'close']] *= 2
    after, _, _ = management_events(changed, changed_five, states, orders, minute.end.iloc[-1])
    pd.testing.assert_frame_equal(before[before.label_end.lt(boundary)], after[after.label_end.lt(boundary)], check_exact=True)


def test_missing_minutes_and_duplicate_orders_are_rejected():
    minute, five, states, orders = inputs()
    with pytest.raises(ValueError, match='연속성'):
        management_events(minute.drop(index=1500), five, states, orders, minute.end.iloc[-1])
    with pytest.raises(ValueError, match='중복'):
        management_events(minute, five, states, pd.concat([orders, orders.iloc[[-1]]]), minute.end.iloc[-1])


def test_online_values_match_training_feature_order():
    minute, five, states, orders = inputs()
    frame, _, _ = management_events(minute, five, states, orders, minute.end.iloc[-1])
    row = frame[frame.usable].iloc[-1]
    actual = management_values(row[FEATURES[:14]], row.direction, row.favorable_move,
                               (row.end - row.entry_time).total_seconds() / 60, row['adds'])
    np.testing.assert_allclose(actual, row[FEATURES].to_numpy(dtype=float))


def test_reverse_order_keeps_previous_episode_as_management_target():
    times = pd.date_range('2020-01-01', periods=3, freq='s', tz='UTC').as_unit('ns')
    actions = pd.DataFrame({'time': times, 'order_key': ['a', 'b', 'c'],
                            'action': ['open', 'reverse', 'reduce'], 'before_qty': [0, 10, -5], 'episode_id': [1, 2, 2]})
    executions = pd.DataFrame({'time': times, 'order_key': ['a', 'b', 'c'], 'symbol': 'XBTUSD',
                               'exectype': 'Trade', 'orderqty': [10, 15, 2], 'side': ['Buy', 'Sell', 'Buy']})
    result = management_orders(executions, actions)
    assert result.before_episode_id.tolist() == [0, 1, 2]
    assert result.target_episode_id.tolist() == [1, 2, 2]
