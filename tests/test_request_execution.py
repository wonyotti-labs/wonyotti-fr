import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.audit import aggregate_events
from wonyotti_fr.entry_boundary_support import transition_ledger
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.reconstruct import reconstruct
from wonyotti_fr.request_execution import (
    MODELS,
    binary_metrics,
    link_terminations,
    order_observations,
    summarize_execution,
    window_comparison,
)
from wonyotti_fr.request_execution_reference import (
    compare_summary,
    independent_summary,
    verify_orders,
    verify_windows,
)
from wonyotti_fr.state_action_windows import action_windows, requested_orders


def inputs(unit='ns'):
    start = pd.Timestamp('2020-01-03', tz='UTC')
    trades = [(0, 'a', 'Buy', 10, 10), (10, 'c', 'Sell', 3, 15), (20, 'e', 'Buy', 4, 4),
        (300, 'c', 'Sell', 7, 15), (301, 'd', 'Sell', 4, 4),
        (330, 'f', 'Buy', 10, 10), (350, 'g', 'Sell', 2, 10), (600, 'g', 'Sell', 8, 10),
        (620, 'z', 'Buy', 10, 10), (630, 'h', 'Sell', 2, 2), (640, 'h', 'Sell', 8, 10),
        (660, 'j', 'Buy', 10, 20), (670, 'k', 'Sell', 20, 20), (680, 'j', 'Buy', 10, 20),
        (690, 'n', 'Buy', 10, 10), (700, 'special:close', 'Sell', 10, 10),
        (720, 'm', 'Buy', 10, 10), (730, 'p', 'Sell', 1, 10)]
    rows, cumulative = [], {}
    for i, (second, key, side, qty, requested) in enumerate(trades):
        cumulative[key] = cumulative.get(key, 0)+qty
        rows.append({'time': start+pd.Timedelta(seconds=second), 'symbol': 'XBTUSD', 'order_key': key,
            'execid': str(i), 'side': side, 'exectype': 'Trade', 'ordtype': 'Limit', 'lastqty': qty,
            'orderqty': requested, 'cumqty': cumulative[key], 'ordstatus': 'Filled' if cumulative[key] == requested else 'PartiallyFilled',
            'lastliquidityind': 'AddedLiquidity' if i % 2 else 'RemovedLiquidity',
            'lastpx': 100., 'execcost': qty*(-1000000 if side == 'Buy' else 1000000), 'execcomm': 1, 'source_row': i+2})
    raw = pd.DataFrame(rows)
    raw.time = raw.time.astype(f'datetime64[{unit}, UTC]')
    events = aggregate_events(raw, 'XBTUSD')
    episodes, actions, _ = reconstruct(events)
    ledger = transition_ledger(events, actions)
    features = pd.DataFrame({'end': pd.date_range(start, periods=4, freq='5min')})
    for name in MARKET_FEATURES:
        features[name] = 0.
    requested = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, requested, episodes)
    paired = windows.copy()
    paired['paired_part'] = ['training', 'calibration', 'diagnosis', 'diagnosis']
    paired['paired_usable'] = [False, True, True, False]
    predictions = paired.loc[paired.paired_usable, ['end', 'state_episode_id', 'paired_part']].copy()
    predictions['all_y_exit'] = paired.loc[paired.paired_usable, 'y_requested_exit']
    for i, name in enumerate(MODELS):
        predictions[f'{name}_exit_request'] = [i % 2 == 0, True]
    return raw, ledger, requested, windows, filled, paired, predictions


def build(unit='ns'):
    raw, ledger, requested, windows, filled, paired, predictions = inputs(unit)
    observations = order_observations(raw, ledger, requested)
    requests, ended = link_terminations(ledger, observations)
    comparison = window_comparison(windows, requested, filled, paired, predictions)
    return raw, ledger, requests, ended, windows, requested, filled, comparison


@pytest.mark.parametrize('unit', ['ns', 'us'])
def test_full_orders_terminations_labels_metrics_and_independent_summary(unit):
    raw, ledger, requests, ended, windows, requested, filled, comparison = build(unit)
    assert verify_orders(raw, ledger, requests, ended)['orders_verified'] == 13
    assert verify_windows(windows, requested, filled, comparison)['windows_verified'] == 4
    compare_summary(independent_summary(requests, ended, comparison), summarize_execution(requests, ended, comparison))
    lookup = requests.set_index('order_key')
    assert lookup.loc['c', 'end_owner'] == 'other_order'
    assert lookup.loc['c', 'termination_delay_ns'] == 291*10**9
    assert lookup.loc['c', 'fill_span_ns'] == 290*10**9
    assert not lookup.loc['c', 'termination_in_first_window']
    assert lookup.loc['g', 'end_owner'] == 'same_order'
    assert not lookup.loc['g', 'termination_in_first_window']
    assert lookup.loc['p', 'end_owner'] == 'not_observed'
    assert pd.isna(lookup.loc['p', 'termination_delay_ns'])
    assert lookup.loc['p', 'observed_to_first_requested'] == .1
    assert lookup.loc['h', 'quantity_versions'] == 2
    origins = ended.set_index('order_key').origin
    assert origins['h'] == 'other_action_same_episode'
    assert origins['j'] == 'different_episode'
    assert origins['special:close'] == 'special_order'
    assert comparison.loc[2, 'y_filled_exit'] and not comparison.loc[2, 'y_requested_exit']


def test_same_timestamp_keeps_declared_order_without_negative_delay():
    raw, *_ = inputs()
    raw.loc[raw.order_key.isin(['k', 'j']) & raw.time.ge(pd.Timestamp('2020-01-03 00:11:10Z')), 'time'] = pd.Timestamp('2020-01-03 00:11:10Z')
    events = aggregate_events(raw, 'XBTUSD')
    _, actions, _ = reconstruct(events)
    ledger = transition_ledger(events, actions)
    requests, ended = link_terminations(ledger, order_observations(raw, ledger, requested_orders(raw, ledger)))
    verify_orders(raw, ledger, requests, ended)
    assert (requests.termination_delay_ns.dropna() >= 0).all()


@pytest.mark.parametrize('column,value', [('lastqty', -1), ('lastqty', 0), ('orderqty', 0), ('cumqty', -1)])
def test_invalid_quantities_rejected(column, value):
    raw, ledger, requested, *_ = inputs()
    raw.loc[1, column] = value
    with pytest.raises((ValueError, AssertionError)):
        order_observations(raw, ledger, requested)


def test_same_order_side_change_rejected():
    raw, ledger, requested, *_ = inputs()
    raw.loc[3, 'side'] = 'Buy'
    with pytest.raises(ValueError, match='방향 변경'):
        order_observations(raw, ledger, requested)


def test_cumulative_mismatch_and_unknown_liquidity_preserved():
    raw, ledger, requested, *_ = inputs()
    raw.loc[1, 'cumqty'] += 1
    raw.loc[1, 'lastliquidityind'] = 'Unknown'
    requests, ended = link_terminations(ledger, order_observations(raw, ledger, requested))
    verify_orders(raw, ledger, requests, ended)
    row = requests.loc[requests.order_key.eq('c')].iloc[0]
    assert row.cumulative_mismatch_fills == 1 and row.other_liquidity_fills == 1


@pytest.mark.parametrize('change', ['paired_label', 'prediction_label', 'prediction_state', 'prediction_part', 'prediction_missing', 'prediction_float', 'event_label'])
def test_modified_label_support_or_prediction_rejected(change):
    _, _, requested, windows, filled, paired, predictions = inputs()
    if change == 'paired_label':
        paired.loc[1, 'y_requested_exit'] = not paired.loc[1, 'y_requested_exit']
    elif change == 'prediction_label':
        predictions.loc[predictions.index[0], 'all_y_exit'] = not predictions.all_y_exit.iloc[0]
    elif change == 'prediction_state':
        predictions.loc[predictions.index[0], 'state_episode_id'] += 1
    elif change == 'prediction_part':
        predictions.loc[predictions.index[0], 'paired_part'] = 'diagnosis'
    elif change == 'prediction_missing':
        predictions = predictions.iloc[1:]
    elif change == 'prediction_float':
        predictions['all_exit_request'] = predictions.all_exit_request.astype(float)
    else:
        filled.loc[filled.exit, 'exit'] = False
    with pytest.raises(ValueError):
        window_comparison(windows, requested, filled, paired, predictions)


def test_reference_catches_quantity_and_termination_mutation():
    raw, ledger, requests, ended, *_ = build()
    requests.loc[requests.order_key.eq('g'), 'termination_delay_ns'] += 1
    with pytest.raises(AssertionError):
        verify_orders(raw, ledger, requests, ended)


def test_metrics_empty_truth_and_zero_prediction_are_explicit():
    assert binary_metrics([False], [False]) == {'positive': 0, 'requests': 0, 'tp': 0, 'fp': 0,
        'fn': 0, 'precision': None, 'recall': None, 'f2': None}
    assert binary_metrics([True, False], [True, True])['f2'] == 5/6
    with pytest.raises(AssertionError):
        compare_summary({'requests': 3}, {'requests': 2})


def test_observed_quantity_ledger_mismatch_rejected():
    raw, ledger, requested, *_ = inputs()
    raw.loc[3, 'lastqty'] += 1
    with pytest.raises(ValueError, match='원본 체결량'):
        order_observations(raw, ledger, requested)


def test_fractional_quantity_rejected():
    raw, ledger, requested, *_ = inputs()
    raw['lastqty'] = raw.lastqty.astype(float)
    raw.loc[3, 'lastqty'] = 1.5
    with pytest.raises(ValueError):
        order_observations(raw, ledger, requested)


def test_later_observed_fills_never_change_first_request_label():
    raw, ledger, requested, *_ = inputs()
    before = requested.set_index('order_key').requested_action
    raw.loc[3, 'orderqty'] = 100
    observations = order_observations(raw, ledger, requested)
    assert observations.set_index('order_key').requested_action.equals(before)
    assert observations.set_index('order_key').loc['c', 'quantity_versions'] == 2
    assert np.isfinite(observations.observed_to_first_requested).all()
