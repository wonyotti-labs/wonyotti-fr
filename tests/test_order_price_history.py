import copy

import numpy as np
import pandas as pd
import pytest
from test_order_history import example

from wonyotti_fr.order_history import HISTORY_FEATURES, attach_order_history
from wonyotti_fr.order_price_history import PRICE_FEATURES, attach_order_prices, price_support
from wonyotti_fr.order_price_reference import (
    verify_execution_prices,
    verify_order_prices,
    verify_price_support,
)
from wonyotti_fr.order_price_research import checked_price_reference


def source(short=False):
    frame, actions = example()
    actions['price'] = [100., 105., 400., 108., 0., 101., 100., 110., 100.]
    frame['close'] = [110., 120., 90., 100., 100., 80.]
    frame['direction'] = -1 if short else 1
    frame['y_increase'] = [0, 1, 0, 1, 0, 1]
    frame['y_reduce'] = [1, 0, 1, 0, 1, 0]
    if short:
        actions[['before_qty', 'after_qty']] *= -1
    return attach_order_history(frame, actions)[0], actions


@pytest.mark.parametrize('short', [False, True])
def test_price_boundary_partial_fill_episode_and_direction(short):
    frame, actions = source(short)
    got, links, ledger = attach_order_prices(frame, actions)
    verify_order_prices(frame, actions, got, links, ledger)
    assert ledger.order_key.tolist() == ['a', 'b', 'c', 'f']
    assert links.increase_action_row.tolist() == [-1, 1, 1, 1, -1, 8]
    assert links.reduce_action_row.tolist() == [-1, -1, 3, 5, -1, -1]
    assert links.increase_price.tolist() == [0, 105, 105, 105, 0, 100]
    expected = np.log(120) - np.log(105)
    assert got.loc[20, PRICE_FEATURES[0]] == expected
    assert got.loc[20, PRICE_FEATURES[1]] == (-expected if short else expected)
    assert got.loc[50, PRICE_FEATURES].eq(0).all()
    pd.testing.assert_frame_equal(got[frame.columns], frame, check_exact=True)
    support = price_support(got)
    verify_price_support(got, support)
    assert support['increase']['first']['references']['increase']['present'] == 0


def test_future_and_later_partial_prices_cannot_change_past():
    frame, actions = source()
    expected = attach_order_prices(frame, actions)[0]
    later = actions.copy()
    later.loc[2, 'price'] = 999999.
    pd.testing.assert_frame_equal(attach_order_prices(frame, later)[0], expected, check_exact=True)
    future = actions.copy()
    future.loc[5, 'price'] = 300.
    changed = attach_order_prices(frame, future)[0]
    pd.testing.assert_frame_equal(changed.iloc[:3], expected.iloc[:3], check_exact=True)
    assert changed.iloc[3][PRICE_FEATURES[2]] != expected.iloc[3][PRICE_FEATURES[2]]
    frame.loc[60, 'close'] = 500.
    changed = attach_order_prices(frame, actions)[0]
    pd.testing.assert_frame_equal(changed.iloc[:-1], expected.iloc[:-1], check_exact=True)


def test_same_timestamp_keeps_observed_order_and_strict_nanosecond_boundary():
    frame, actions = source()
    actions.loc[2, 'time'] = actions.loc[1, 'time']
    actions.loc[2, 'order_key'] = 'a2'
    frame = frame.drop(columns=HISTORY_FEATURES)
    frame.loc[10, 'end'] += pd.Timedelta(nanoseconds=1)
    frame = attach_order_history(frame, actions)[0]
    got, links, ledger = attach_order_prices(frame, actions)
    assert links.loc[10, 'increase_price'] == 400
    verify_order_prices(frame, actions, got, links, ledger)
    actions['time'] = actions.time.astype('datetime64[us, UTC]')
    pd.testing.assert_frame_equal(attach_order_prices(frame, actions)[0], got, check_exact=True)


@pytest.mark.parametrize('damage', ['zero', 'negative', 'nan', 'inf', 'close', 'direction', 'episode', 'exists', 'time', 'naive', 'order', 'quantity', 'duplicate_features'])
def test_invalid_price_state_is_rejected(damage):
    frame, actions = source()
    if damage in {'zero', 'negative', 'nan', 'inf'}:
        actions.loc[1, 'price'] = {'zero': 0, 'negative': -1, 'nan': np.nan, 'inf': np.inf}[damage]
    elif damage == 'close':
        frame.loc[20, 'close'] = 0
    elif damage == 'direction':
        frame.loc[20, 'direction'] = -1
    elif damage == 'episode':
        frame.loc[20, 'episode_id'] = 0
    elif damage == 'exists':
        frame.loc[20, 'past_increase_exists'] = 0
    elif damage == 'time':
        actions.loc[1, 'time'] = pd.NaT
    elif damage == 'naive':
        frame['end'] = frame.end.dt.tz_localize(None)
    elif damage == 'order':
        actions.loc[1, 'order_key'] = None
    elif damage == 'quantity':
        actions.loc[1, 'after_qty'] += 1
    else:
        frame[PRICE_FEATURES[0]] = 0
    with pytest.raises((ValueError, AssertionError)):
        attach_order_prices(frame, actions)


@pytest.mark.parametrize('damage', ['feature', 'reference', 'time', 'price', 'ledger', 'support'])
def test_independent_checker_rejects_output_damage(damage):
    frame, actions = source()
    got, links, ledger = attach_order_prices(frame, actions)
    if damage == 'feature':
        got.loc[20, PRICE_FEATURES[0]] += .1
    elif damage == 'reference':
        links.loc[20, 'increase_action_row'] += 1
    elif damage == 'time':
        links.loc[20, 'increase_time'] += pd.Timedelta(seconds=1)
    elif damage == 'price':
        links.loc[20, 'increase_price'] += 1
    elif damage == 'ledger':
        ledger.loc[0, 'price'] += 1
    else:
        support = price_support(got)
        changed = copy.deepcopy(support)
        changed['increase']['quiet']['references']['increase']['present'] += 1
        with pytest.raises(ValueError, match='불일치'):
            verify_price_support(got, changed)
        return
    with pytest.raises(AssertionError):
        verify_order_prices(frame, actions, got, links, ledger)


def price_sources():
    start = pd.Timestamp('2020-01-01', tz='UTC')
    raw = pd.DataFrame({'time': start + pd.to_timedelta([0, 1, 1, 2, 3, 4], unit='min'),
        'order_key': ['open', 'a', 'a', 'a', 'b', 'funding'], 'exectype': ['Trade']*5 + ['Funding'],
        'side': ['Buy']*4 + ['Sell', 'Buy'], 'ordtype': ['Limit']*6,
        'lastqty': [100, 10, 20, 10, 20, 120], 'lastpx': [100., 110., 140., 150., 120., 125.],
        'source_row': [2, 3, 4, 5, 6, 7], 'execid': list('abcdef'), 'symbol': ['XBTUSD']*6})
    events = raw.iloc[[0, 1, 3, 4, 5]][['time', 'order_key', 'exectype', 'side', 'ordtype', 'source_row']].reset_index(drop=True)
    events['quantity'], events['price'], events['fills'] = [100, 30, 10, 20, 120], [100., 130., 150., 120., 125.], [1, 2, 1, 1, 1]
    actions = events[['time', 'order_key', 'price']].copy()
    actions['action'] = ['open', 'increase', 'increase', 'reduce', 'funding']
    actions['before_qty'], actions['after_qty'] = [0, 100, 130, 140, 120], [100, 130, 140, 120, 120]
    return raw, events, actions


def test_reaggregation_uses_only_simultaneous_group_and_preserves_event_order():
    raw, events, actions = price_sources()
    evidence = verify_execution_prices(raw, events, actions)
    assert evidence['source_executions'] == 6 and evidence['events'] == 5
    assert evidence['trade_events'] == 4


@pytest.mark.parametrize('damage', ['raw_price', 'raw_time', 'event_price', 'quantity', 'fills', 'source_order', 'actions_price', 'actions_time', 'actions_quantity', 'duplicate', 'missing'])
def test_provenance_damage_is_rejected(damage):
    raw, events, actions = price_sources()
    if damage == 'raw_price':
        raw.loc[1, 'lastpx'] += 1
    elif damage == 'raw_time':
        raw.loc[1, 'time'] += pd.Timedelta(seconds=1)
    elif damage == 'event_price':
        events.loc[1, 'price'] += 1
    elif damage == 'quantity':
        events.loc[1, 'quantity'] += 1
    elif damage == 'fills':
        events.loc[1, 'fills'] += 1
    elif damage == 'source_order':
        events.loc[1, 'source_row'] += 1
    elif damage == 'actions_price':
        actions.loc[1, 'price'] += 1
    elif damage == 'actions_time':
        actions.loc[1, 'time'] += pd.Timedelta(seconds=1)
    elif damage == 'actions_quantity':
        actions.loc[1, 'after_qty'] += 1
    elif damage == 'duplicate':
        events = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    else:
        raw = raw.iloc[1:]
    with pytest.raises((ValueError, AssertionError)):
        verify_execution_prices(raw, events, actions)


def test_proof_digest_and_symlink_rejected(tmp_path):
    proof = tmp_path/'proof.json'
    proof.write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        checked_price_reference(proof, '0'*64)
    link = tmp_path/'link.json'
    link.symlink_to(proof)
    with pytest.raises(ValueError, match='지문'):
        checked_price_reference(link, '0'*64)
