import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.audit import aggregate_events
from wonyotti_fr.entry_boundary_reference import (
    independent_ledger,
    independent_summary,
    verify_aggregation,
    verify_entries,
)
from wonyotti_fr.entry_boundary_support import entry_support, summarize_support, transition_ledger
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.reconstruct import reconstruct


def fixture():
    events = [(10, 'a', 'Buy', 10), (20, 'a', 'Buy', 5), (30, 'b', 'Sell', 7), (40, 'c', 'Sell', 8),
        (60, 'd', 'Buy', 3), (90, 'e', 'Sell', 8), (290, 'f', 'Buy', 5),
        (295, 'special:funding', 'Funding', 0), (299, 'g', 'Sell', 3), (300, 'h', 'Buy', 3),
        (300, 'i', 'Sell', 7), (330, 'j', 'Buy', 7), (660, 'k', 'Buy', 1), (670, 'k', 'Buy', 1), (670, 'k', 'Buy', 2)]
    first = pd.Timestamp('2020-01-03', tz='UTC')
    rows = []
    for i, (second, order, side, quantity) in enumerate(events):
        rows.append({'time': first+pd.Timedelta(seconds=second), 'order_key': order, 'execid': f'e{i}',
            'exectype': 'Funding' if side == 'Funding' else 'Trade', 'side': 'Sell' if side == 'Funding' else side,
            'ordtype': 'Limit', 'lastqty': quantity, 'execcost': -quantity*1000000 if side == 'Buy' else quantity*1000000,
            'execcomm': 1, 'source_row': i+2, 'lastliquidityind': 'RemovedLiquidity', 'lastpx': 100., 'symbol': 'XBTUSD'})
    raw = pd.DataFrame(rows)
    grouped = aggregate_events(raw, 'XBTUSD')
    _, actions, _ = reconstruct(grouped)
    return raw, grouped, actions


def inputs():
    raw, grouped, actions = fixture()
    ledger = transition_ledger(grouped, actions)
    first = pd.Timestamp('2020-01-03', tz='UTC')
    features = pd.DataFrame({'end': pd.date_range(first, periods=3, freq='5min')})
    for key in MARKET_FEATURES:
        features[key] = 0.
    source = pd.DataFrame({'end': pd.to_datetime(['2020-01-03T00:05Z', '2020-01-03T00:10Z']),
        'alignment_supported': [False, True], 'direction': [-1, 0], 'xbtusd_execution_change': [0., .2]})
    return raw, grouped, actions, ledger, features, {'training': source, 'diagnosis': source.iloc[:0].copy()}


def test_full_raw_integer_grouping_and_all_states_exact():
    raw, events, actions, ledger, features, sources = inputs()
    proof = verify_aggregation(raw, events)
    assert proof['raw_xbtusd_rows'] == 15 and proof['event_groups'] == 14
    reference = independent_ledger(events, actions)
    pd.testing.assert_frame_equal(ledger, reference, check_exact=True)
    saved = entry_support(ledger, features, sources)
    assert verify_entries(reference, features, sources, saved)['all_entries_verified'] == 5
    assert summarize_support(ledger, saved) == independent_summary(reference, saved)


def test_zero_gap_exact_boundaries_funding_and_unknown_initial_state():
    _, _, _, ledger, features, sources = inputs()
    entries = entry_support(ledger, features, sources)
    assert entries.prior_flat_band.tolist() == ['unknown', 'up_to_60s', 'up_to_60s', 'zero', 'over_300s']
    assert entries.prior_flat_ns.iloc[1:].tolist() == [20*10**9, 9*10**9, 0, 330*10**9]
    assert entries.boundary_60_flat.tolist() == [False, True, False, False, True]
    assert entries.boundary_300_flat.tolist() == [False, False, False, False, True]
    assert entries.boundary_60_observed.tolist() == [False, True, True, True, True]
    assert entries.source_reason.tolist() == ['outside_original_source']*3+['unsupported_original_window', 'jointly_supported']
    assert entries.source_window_flat.tolist() == [False]*4+[True]
    assert ledger.loc[ledger.order_key.eq('k'), 'first_trade_of_order'].tolist() == [True, False]
    assert summarize_support(ledger, entries)['opposite_trade_sides_same_time_groups'] == 1


@pytest.mark.parametrize('bad', ['quantity', 'action', 'episode', 'order', 'time'])
def test_cached_actions_tampering_rejected(bad):
    _, events, actions = fixture()
    if bad == 'quantity':
        actions.loc[1, 'after_qty'] += 1
    elif bad == 'action':
        actions.loc[1, 'action'] = 'open'
    elif bad == 'episode':
        actions.loc[1, 'episode_id'] += 1
    elif bad == 'order':
        actions.loc[1, 'order_key'] = 'wrong'
    else:
        actions.loc[1, 'time'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        transition_ledger(events, actions)


@pytest.mark.parametrize('bad', ['fractional', 'negative', 'zero_trade', 'unknown_type', 'unknown_side', 'unsorted', 'duplicate', 'timezone', 'missing'])
def test_invalid_event_inputs_rejected(bad):
    _, events, actions = fixture()
    if bad == 'fractional':
        events.quantity = events.quantity.astype(float)+.5
    elif bad == 'negative':
        events.loc[0, 'quantity'] = -1
    elif bad == 'zero_trade':
        events.loc[0, 'quantity'] = 0
    elif bad == 'unknown_type':
        events.loc[0, 'exectype'] = 'Settlement'
    elif bad == 'unknown_side':
        events.loc[0, 'side'] = 'unknown'
    elif bad == 'unsorted':
        events = events.iloc[::-1]
    elif bad == 'duplicate':
        events.loc[1, ['time', 'order_key', 'exectype', 'side', 'ordtype']] = events.loc[0, ['time', 'order_key', 'exectype', 'side', 'ordtype']].to_numpy()
    elif bad == 'timezone':
        events.time = events.time.dt.tz_convert('Asia/Seoul')
    else:
        events = events.drop(columns='quantity')
    with pytest.raises(ValueError):
        transition_ledger(events, actions)


@pytest.mark.parametrize('bad', ['raw_fraction', 'duplicate_execid', 'quantity', 'cost', 'fee', 'fills', 'source_row'])
def test_raw_aggregation_tampering_rejected(bad):
    raw, events, _ = fixture()
    if bad == 'raw_fraction':
        raw.lastqty = raw.lastqty.astype(float)+.5
    elif bad == 'duplicate_execid':
        raw.loc[1, 'execid'] = raw.execid.iloc[0]
    else:
        field = {'cost': 'cost_satoshi', 'fee': 'fee_satoshi'}.get(bad, bad)
        events.loc[0, field] += 1
    with pytest.raises(ValueError):
        verify_aggregation(raw, events)


@pytest.mark.parametrize('mode', ['missing', 'nonfinite', 'embargo'])
def test_feature_support_does_not_fill_or_drop_entries(mode):
    _, events, actions, ledger, features, sources = inputs()
    if mode == 'missing':
        features = features.iloc[1:].copy()
    elif mode == 'nonfinite':
        features.loc[0, MARKET_FEATURES[0]] = np.nan
    else:
        offset = pd.Timedelta(days=2)
        for frame, column in [(events, 'time'), (actions, 'time'), (features, 'end')]:
            frame[column] -= offset
        for frame in sources.values():
            frame['end'] -= offset
        ledger = transition_ledger(events, actions)
    result = entry_support(ledger, features, sources)
    assert len(result) == 5
    expected = {'missing': 'missing_feature_boundary', 'nonfinite': 'nonfinite_market', 'embargo': 'year_embargo'}[mode]
    assert result.market_reason.iloc[0] == expected
    verify_entries(independent_ledger(events, actions), features, sources, result)


def test_source_prestate_mismatch_rejected():
    _, _, _, ledger, features, sources = inputs()
    sources['training'].loc[0, 'direction'] = 1
    with pytest.raises(ValueError):
        entry_support(ledger, features, sources)


def test_future_events_do_not_change_prior_states_or_first_order_flags():
    raw, events, actions = fixture()
    before = transition_ledger(events, actions)
    future = raw.tail(1).copy()
    future['time'] += pd.Timedelta(days=2)
    future['execid'] = 'future'
    future['source_row'] += 100
    extended = aggregate_events(pd.concat([raw, future], ignore_index=True), 'XBTUSD')
    _, later_actions, _ = reconstruct(extended)
    after = transition_ledger(extended, later_actions)
    pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True), check_exact=True)


def test_independent_entry_connection_detects_rewritten_support():
    _, _, _, ledger, features, sources = inputs()
    saved = entry_support(ledger, features, sources)
    saved.loc[0, 'market_reason'] = 'year_embargo'
    with pytest.raises(ValueError):
        verify_entries(ledger, features, sources, saved)


@pytest.mark.parametrize(('gap', 'band'), [(0, 'zero'), (1, 'up_to_60s'), (60*10**9, 'up_to_60s'),
    (60*10**9+1, 'up_to_300s'), (300*10**9, 'up_to_300s'), (300*10**9+1, 'over_300s')])
def test_flat_duration_nanosecond_cutoffs_and_close_exactly_at_boundary(gap, band):
    raw, _, _ = fixture()
    rows = raw.iloc[:3].copy()
    first = pd.Timestamp('2020-01-03', tz='UTC')
    rows['time'] = pd.to_datetime([first+pd.Timedelta(seconds=1), first+pd.Timedelta(minutes=5),
        first+pd.Timedelta(minutes=5, nanoseconds=gap)], utc=True)
    rows['order_key'] = ['one', 'two', 'three']
    rows['side'] = ['Buy', 'Sell', 'Sell']
    rows['lastqty'] = 10
    rows['execcost'] = [-10000000, 10000000, 10000000]
    events = aggregate_events(rows, 'XBTUSD')
    _, actions, _ = reconstruct(events)
    result = transition_ledger(events, actions)
    assert result.prior_flat_band.iloc[-1] == band and result.prior_flat_ns.iloc[-1] == gap
    if gap < 60*10**9:
        assert not result.boundary_60_flat.iloc[-1]
        assert not result.boundary_300_flat.iloc[-1]
    pd.testing.assert_frame_equal(result, independent_ledger(events, actions), check_exact=True)


def test_no_open_events_still_preserves_funding_and_empty_support():
    raw, _, _ = fixture()
    raw = raw.loc[raw.exectype.eq('Funding')].copy()
    events = aggregate_events(raw, 'XBTUSD')
    _, actions, _ = reconstruct(events)
    ledger = transition_ledger(events, actions)
    _, _, _, _, features, sources = inputs()
    entries = entry_support(ledger, features, sources)
    assert entries.empty
    verify_entries(independent_ledger(events, actions), features, sources, entries)
    assert summarize_support(ledger, entries) == independent_summary(ledger, entries)


def test_funding_position_sign_does_not_change_inventory():
    raw, _, _ = fixture()
    raw.loc[raw.exectype.eq('Funding'), 'lastqty'] = -10
    events = aggregate_events(raw, 'XBTUSD')
    _, actions, _ = reconstruct(events)
    result = transition_ledger(events, actions)
    funding = result.loc[result.action.eq('funding')].iloc[0]
    assert funding.before_qty == funding.after_qty == 0
    pd.testing.assert_frame_equal(result, independent_ledger(events, actions), check_exact=True)
