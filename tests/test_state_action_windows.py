import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.audit import aggregate_events
from wonyotti_fr.entry_boundary_reference import independent_ledger
from wonyotti_fr.entry_boundary_support import transition_ledger
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.reconstruct import reconstruct
from wonyotti_fr.state_action_windows import action_windows, requested_orders, summarize_windows
from wonyotti_fr.state_action_windows_reference import (
    independent_summary,
    verify_requests,
    verify_windows,
)


def dataset():
    first = pd.Timestamp('2020-01-03', tz='UTC')
    events = [(-60, 'special:fund0', 'Funding', 0, 0), (0, 'a', 'Buy', 10, 20), (60, 'a', 'Buy', 10, 20),
        (60, 'b', 'Sell', 5, 5), (90, 'c', 'Buy', 3, 3), (120, 'd', 'Sell', 18, 18),
        (150, 'e', 'Sell', 4, 4), (180, 'f', 'Buy', 4, 4), (300, 'g', 'Buy', 10, 10),
        (600, 'h', 'Sell', 2, 12), (600, 'i', 'Buy', 2, 2), (660, 'j', 'Sell', 12, 12),
        (690, 'k', 'Buy', 1, 1), (900, 'special:close', 'Buy', 1, 1),
        (1250, 'special:fund1', 'Funding', 0, 0)]
    rows = []
    for i, (second, order, side, qty, requested) in enumerate(events):
        rows.append({'time': first+pd.Timedelta(seconds=second), 'order_key': order, 'execid': f'e{i}',
            'exectype': 'Funding' if side == 'Funding' else 'Trade', 'side': 'Sell' if side == 'Funding' else side,
            'ordtype': 'Limit', 'lastqty': qty, 'orderqty': requested,
            'execcost': qty*(-1000000 if side == 'Buy' else 1000000), 'execcomm': 1,
            'source_row': i+2, 'lastliquidityind': 'RemovedLiquidity', 'lastpx': 100., 'symbol': 'XBTUSD'})
    raw = pd.DataFrame(rows)
    grouped = aggregate_events(raw, 'XBTUSD')
    episodes, actions, _ = reconstruct(grouped)
    ledger = transition_ledger(grouped, actions)
    features = pd.DataFrame({'end': pd.date_range(first-pd.Timedelta(minutes=5), periods=7, freq='5min')})
    for i, name in enumerate(MARKET_FEATURES):
        features[name] = np.arange(len(features), dtype=float)+i/10
    return raw, grouped, actions, episodes, ledger, features


def build():
    raw, events, actions, episodes, ledger, features = dataset()
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    return raw, events, actions, episodes, ledger, features, orders, windows, filled, requested


def test_complete_independent_raw_orders_states_links_counts_and_summaries():
    raw, events, actions, episodes, ledger, features, orders, windows, filled, requested = build()
    reference = independent_ledger(events, actions)
    pd.testing.assert_frame_equal(reference, ledger, check_exact=True)
    assert verify_requests(raw, reference, orders)['first_orders_verified'] == 11
    assert verify_windows(features, reference, orders, episodes, windows, filled, requested)['windows_verified'] == 7
    assert summarize_windows(windows, filled, requested) == independent_summary(windows, filled, requested)
    assert len(filled) == len(events) and len(requested) == 11


def test_strict_boundary_both_flat_directions_partial_fills_and_special_close():
    *_, windows, filled, requested = build()
    flat, held, special = windows.iloc[1], windows.iloc[3], windows.iloc[4]
    assert flat.state_quantity == 0 and flat.y_requested_enter_long and flat.y_requested_enter_short
    assert held.state_quantity == 10 and held.event_reduce_count == 2
    assert held.filled_reduce_count == 1 and held.requested_exit_count == 2
    assert held.requested_increase_count == 1 and held.requested_reduce_count == 0
    assert special.filled_exit_count == 1 and special.requested_exit_count == 0
    assert filled.loc[filled.order_key.eq('a'), 'enter_long'].tolist() == [True, False]
    assert requested.loc[requested.order_key.eq('k'), 'label_scope'].item() == 'new_or_different_episode'
    mismatch = requested.loc[requested.order_key.eq('h')].iloc[0]
    assert mismatch.requested_action == 'reverse' and mismatch.first_filled_action == 'reduce'
    summary = summarize_windows(windows, filled, requested)
    assert summary['all']['coexisting_management']['requested'] == 1
    assert summary['ledgers']['requested']['later_management_orders_in_window'] == 2
    assert summary['requested_vs_first_fill_action_mismatches'] == 1


def test_missing_end_market_and_unknown_or_after_record_windows_retained():
    raw, _, _, episodes, ledger, features = dataset()
    features = features.drop(index=2).reset_index(drop=True)
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert windows.support_reason.tolist() == ['outside_source_coverage', 'missing_end_market',
        'included', 'included', 'outside_source_coverage', 'outside_source_coverage']
    assert filled.loc[filled.order_key.eq('g'), 'label_scope'].item() == 'outside_windows'
    assert pd.isna(windows.state_quantity.iloc[0])
    verify_windows(features, ledger, orders, episodes, windows, filled, requested)


@pytest.mark.parametrize('column', ['start', 'end'])
def test_nonfinite_market_is_reason_not_filled_zero(column):
    raw, _, _, episodes, ledger, features = dataset()
    features.loc[1 if column == 'start' else 2, MARKET_FEATURES[0]] = np.nan
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert windows.support_reason.iloc[1] == f'nonfinite_{column}_market'
    verify_windows(features, ledger, orders, episodes, windows, filled, requested)


@pytest.mark.parametrize('closed', [True, False])
def test_current_and_related_crossing_episode_purge_with_unclosed_tail(closed):
    raw, _, _, episodes, ledger, features = dataset()
    # 해당 창에서 새로 진입하는 에피소드도 기간 밖까지 이어지면 배제한다.
    episodes.loc[episodes.episode_id.eq(1), 'closed'] = closed
    episodes.loc[episodes.episode_id.eq(1), 'exit_time'] = pd.Timestamp('2021-01-01', tz='UTC') if closed else pd.Timestamp('2020-01-04', tz='UTC')
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert windows.support_reason.iloc[1] == 'right_episode_boundary'
    assert windows.support_reason.iloc[2] == 'included'
    verify_windows(features, ledger, orders, episodes, windows, filled, requested)


def test_left_episode_and_embargo_edges():
    raw, _, _, episodes, ledger, features = dataset()
    episodes.loc[episodes.episode_id.eq(3), 'entry_time'] = pd.Timestamp('2017-12-31', tz='UTC')
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert windows.support_reason.iloc[2:4].tolist() == ['left_episode_boundary']*2
    verify_windows(features, ledger, orders, episodes, windows, filled, requested)
    for frame, names in [(raw, ['time']), (ledger, ['time']), (features, ['end']), (episodes, ['entry_time', 'exit_time'])]:
        for name in names:
            frame[name] -= pd.Timestamp('2020-01-03')-pd.Timestamp('2018-01-01')
    windows, filled, requested = action_windows(features, ledger, requested_orders(raw, ledger), episodes)
    assert windows.support_reason.iloc[1] == 'left_embargo'


@pytest.mark.parametrize('bad', ['fractional', 'negative', 'zero', 'infinite', 'overflow', 'side', 'duplicate', 'missing_time', 'unsorted', 'missing_order', 'ledger_quantity', 'ledger_episode', 'ledger_time'])
def test_invalid_order_or_prior_state_rejected(bad):
    raw, _, _, _, ledger, _ = dataset()
    if bad in ('fractional', 'negative', 'zero', 'infinite', 'overflow'):
        raw.orderqty = raw.orderqty.astype(float)
        raw.loc[1, 'orderqty'] = {'fractional': 1.5, 'negative': -1, 'zero': 0, 'infinite': np.inf, 'overflow': float(2**63)}[bad]
    elif bad == 'side':
        raw.loc[1, 'side'] = 'Unknown'
    elif bad == 'duplicate':
        raw.loc[1, 'execid'] = raw.execid.iloc[0]
    elif bad == 'missing_time':
        raw.loc[1, 'time'] = pd.NaT
    elif bad == 'unsorted':
        raw = raw.iloc[::-1]
    elif bad == 'missing_order':
        raw.loc[1, 'order_key'] = None
    elif bad == 'ledger_quantity':
        ledger.loc[2, 'before_qty'] += 1
    elif bad == 'ledger_episode':
        ledger.loc[2, 'before_episode_id'] += 1
    else:
        ledger.loc[1, 'time'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        requested_orders(raw, ledger)


@pytest.mark.parametrize('bad', ['duplicate_time', 'unaligned', 'episode_unknown', 'closed_exit_missing', 'episode_duplicate', 'duplicate_orders'])
def test_invalid_windows_rejected(bad):
    raw, _, _, episodes, ledger, features = dataset()
    orders = requested_orders(raw, ledger)
    if bad == 'duplicate_time':
        features.loc[1, 'end'] = features.end.iloc[0]
    elif bad == 'unaligned':
        features.end += pd.Timedelta(seconds=1)
    elif bad == 'episode_unknown':
        episodes = episodes.iloc[1:]
    elif bad == 'closed_exit_missing':
        episodes.loc[0, 'exit_time'] = pd.NaT
    elif bad == 'episode_duplicate':
        episodes.loc[1, 'episode_id'] = episodes.episode_id.iloc[0]
    else:
        orders.loc[1, 'order_key'] = orders.order_key.iloc[0]
    with pytest.raises(ValueError):
        action_windows(features, ledger, orders, episodes)


@pytest.mark.parametrize('bad', ['quantity', 'market', 'support', 'count', 'scope', 'request'])
def test_independent_verification_detects_tampering(bad):
    raw, _, _, episodes, ledger, features, orders, windows, filled, requested = build()
    if bad == 'quantity':
        windows.loc[3, 'state_quantity'] += 1
    elif bad == 'market':
        windows.loc[3, 'start_market_'+MARKET_FEATURES[0]] += 1
    elif bad == 'support':
        windows.loc[3, 'support_reason'] = 'included_wrong'
    elif bad == 'count':
        windows.loc[3, 'requested_exit_count'] = 0
    elif bad == 'scope':
        filled.loc[1, 'label_scope'] = 'same_episode'
    else:
        orders.loc[0, 'requested_quantity'] += 1
    with pytest.raises((AssertionError, ValueError)):
        verify_requests(raw, ledger, orders)
        verify_windows(features, ledger, orders, episodes, windows, filled, requested)


def test_future_market_or_later_orders_do_not_change_prior_state_and_label_fields():
    raw, _, _, episodes, ledger, features, orders, before, _, _ = build()
    features.loc[features.index >= 5, MARKET_FEATURES] += 100
    raw.loc[raw.order_key.eq('k'), 'orderqty'] = 3
    after, _, _ = action_windows(features, ledger, requested_orders(raw, ledger), episodes)
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3], check_exact=True)
    assert not orders.empty


def test_no_normal_orders_preserves_empty_requested_schema():
    raw, _, _, episodes, ledger, features = dataset()
    raw.order_key = 'special:'+raw.order_key
    ledger.order_key = 'special:'+ledger.order_key
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert orders.empty and requested.empty and len(filled) == len(ledger)
    verify_requests(raw, ledger, orders)
    verify_windows(features, ledger, orders, episodes, windows, filled, requested)
    assert summarize_windows(windows, filled, requested) == independent_summary(windows, filled, requested)


def test_one_nanosecond_before_boundary_is_state_and_exact_boundary_is_label():
    raw, _, _, _, _, features = dataset()
    raw.time = raw.time.astype('datetime64[ns, UTC]')
    raw.loc[raw.order_key.eq('h'), 'time'] -= pd.Timedelta(nanoseconds=1)
    events = aggregate_events(raw, 'XBTUSD')
    episodes, actions, _ = reconstruct(events)
    ledger = transition_ledger(events, actions)
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    assert windows.state_quantity.iloc[3] == 8
    assert windows.state_time.iloc[3] == features.end.iloc[3]-pd.Timedelta(nanoseconds=1)
    assert requested.loc[requested.order_key.eq('h'), 'window_index'].item() == 2
    assert requested.loc[requested.order_key.eq('i'), 'window_index'].item() == 3
    verify_requests(raw, ledger, orders)
    verify_windows(features, independent_ledger(events, actions), orders, episodes, windows, filled, requested)
