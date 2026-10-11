import copy
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_exposure_components import prepared, source_rows

from wonyotti_fr.contract_components import decompose_contracts, summarize_contracts
from wonyotti_fr.contract_components_reference import reference_summary, verify_contracts
from wonyotti_fr.exposure_components import account_snapshots, decompose, price_lookup
from wonyotti_fr.exposure_components_reference import reference_prices, reference_snapshots
from wonyotti_fr.sparse_exposure_reference import verify_metrics


def original():
    events, wallet, points, prices, activity, specs, snapshots, _, _, _, rows = prepared()
    source = decompose(rows, snapshots, price_lookup(prices, activity), specs)
    return source, events, wallet, prices, activity, specs


def replay(inputs):
    source, events, wallet, prices, activity, specs = inputs
    actual, detail = decompose_contracts(source, events, wallet, price_lookup(prices, activity), specs)
    proof = verify_contracts(source, actual, detail, events, wallet, prices, activity, specs)
    verify_metrics(summarize_contracts(actual), reference_summary(actual))
    return actual, detail, proof


def window(actions):
    start = pd.Timestamp('2020-01-04', tz='UTC')
    points = pd.date_range(start, periods=3, freq='5min').as_unit('ns')
    specs = {s: {'supported': True, 'inverse': s != 'ETHUSD', 'quote': 'USD',
        'multiplier_satoshi': '100' if s == 'ETHUSD' else '100000000',
        'first_time': (start-pd.Timedelta(minutes=2)).isoformat()}
        for s in ['XBTUSD', 'XBTH20', 'ETHUSD']}

    def event(time, symbol, kind, qty, side, cost, fee):
        return {'time': time, 'symbol': symbol, 'exectype': kind, 'lastqty': qty,
            'side': side, 'execcost': cost, 'execcomm': fee, 'settlcurrency': 'XBt'}

    records = [event(start-pd.Timedelta(minutes=1), 'XBTUSD', 'Trade', 50, 'Buy', -50000000, 0)]
    records += [event(start+pd.Timedelta(seconds=seconds), *args) for seconds, *args in actions]
    records.append(event(points[-1], 'XBTUSD', 'Funding', 0, 'Buy', 0, 0))
    events = pd.DataFrame(records).sort_values('time', kind='stable').reset_index(drop=True)
    wallet = pd.DataFrame([{'date': '2019-12-28', 'currency': 'XBt', 'transactstatus': 'Completed',
        'transacttype': 'Deposit', 'amount': 100000000, 'posted_amount': 100000000}])
    prices = pd.DataFrame([{'symbol': s, 'end': t, 'close': 10. if s == 'ETHUSD' else 100., 'support': 'available'}
        for s in specs for t in points])
    activity = prices[['symbol', 'end']].assign(observed_trade=True)
    refs = reference_snapshots(events, wallet, points)
    book, observed = reference_prices(prices, activity)
    rows = source_rows(refs, points, book, observed, specs)
    source = decompose(rows, account_snapshots(events, wallet, points), price_lookup(prices, activity), specs)
    return source, events, wallet, prices, activity, specs


def test_whole_account_contracts_match_independent_rational_replay_and_statistics():
    inputs = original()
    actual, detail, proof = replay(inputs)
    assert proof['supported_rows_compared'] == len(inputs[0])
    assert proof['contract_rows_compared'] == len(detail) == 4
    assert proof['exact_rational_additivity']
    assert actual.iloc[0].contract_count == 0 and actual.iloc[0].contract_float_error == 0
    assert actual.iloc[4].contract_count == 0 and actual.iloc[4].funding_change != 0
    pd.testing.assert_frame_equal(actual[inputs[0].columns], inputs[0], check_exact=True)


def test_exact_inverse_contract_change_and_time_boundaries():
    inputs = window([(0, 'XBTUSD', 'Trade', 10, 'Buy', -10000000, 0),
        (300, 'XBTUSD', 'Trade', 10, 'Sell', 10000000, 0),
        (600, 'XBTUSD', 'Trade', 5, 'Buy', -5000000, 0)])
    actual, detail, _ = replay(inputs)
    np.testing.assert_array_equal(actual.xbtusd_event_count, [1, 1])
    np.testing.assert_allclose(actual.xbtusd_execution_change, [.1, -.1], rtol=0, atol=1e-16)
    assert detail.net_quantity.tolist() == [10, -10]


def test_other_contract_profit_can_reverse_total_without_reversing_btc_order():
    inputs = window([(0, 'XBTUSD', 'Trade', 1, 'Buy', -1000000, 0),
        (1, 'ETHUSD', 'Trade', 100, 'Buy', 100000, 0),
        (2, 'ETHUSD', 'Trade', 100, 'Sell', -4100000, 0)])
    actual, detail, _ = replay(inputs)
    row = actual.iloc[0]
    assert row.xbtusd_execution_change == pytest.approx(float(Fraction(1, 104)))
    assert row.other_contract_execution_change == pytest.approx(float(-Fraction(2, 104)))
    assert row.execution_change < 0 < row.xbtusd_execution_change
    other = detail.loc[detail.symbol.eq('ETHUSD')].iloc[0]
    assert other.net_quantity == 0 and other.cost_satoshi == -4000000
    assert Decimal(other.equity_change_satoshi) == Decimal(other.delta_change_satoshi) == 4000000


def test_roundtrip_needs_no_price_or_spec_when_net_quantity_is_zero():
    source, events, wallet, prices, activity, specs = window([
        (1, 'ETHUSD', 'Trade', 100, 'Buy', 100000, 0),
        (2, 'ETHUSD', 'Trade', 100, 'Sell', -200000, 0)])
    prices = prices.loc[prices.symbol.ne('ETHUSD')]
    activity = activity.loc[activity.symbol.ne('ETHUSD')]
    del specs['ETHUSD']
    actual, detail, _ = replay((source, events, wallet, prices, activity, specs))
    assert actual.iloc[0].other_contract_execution_change != 0
    assert detail.iloc[0].net_quantity == 0


def test_offsetting_contracts_keep_both_contributions_and_order_independence():
    inputs = window([(0, 'XBTUSD', 'Trade', 5, 'Buy', -5000000, 0),
        (0, 'XBTH20', 'Trade', 5, 'Sell', 5000000, 0)])
    actual, detail, _ = replay(inputs)
    assert actual.iloc[0].execution_change == 0
    assert actual.iloc[0].xbtusd_execution_change == -actual.iloc[0].dated_btc_execution_change == .05
    changed = list(inputs)
    changed[1] = inputs[1].sort_values(['time', 'symbol'], ascending=[True, False], kind='stable').reset_index(drop=True)
    second, rows, _ = replay(changed)
    pd.testing.assert_frame_equal(actual, second, check_exact=True)
    pd.testing.assert_frame_equal(detail, rows, check_exact=True)


def test_funding_commissions_rebates_are_not_principal_or_double_charged():
    source, events, wallet, prices, activity, specs = window([
        (0, 'XBTUSD', 'Funding', 0, 'Buy', 999999999, 1000),
        (1, 'XBTUSD', 'Trade', 1, 'Buy', -1000000, -100),
        (301, 'XBTUSD', 'Funding', 0, 'Buy', -777, -300)])
    actual, _, _ = replay((source, events, wallet, prices, activity, specs))
    assert actual.iloc[0].xbtusd_execution_change == pytest.approx(1000000/99999100)
    assert actual.iloc[1].contract_count == 0 and actual.iloc[1].execution_change == 0


def test_future_events_do_not_change_previous_contract_components():
    inputs = list(original())
    inputs[0] = inputs[0].iloc[:3].copy()
    first, a, _ = replay(inputs)
    inputs[1] = inputs[1].copy()
    future = inputs[1].time.ge(inputs[0].label_end.max())
    inputs[1].loc[future, 'execcomm'] += 4321
    second, b, _ = replay(inputs)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_all_unsupported_rows_and_contract_records_remain_without_imputed_values():
    inputs = list(original())
    inputs[0] = inputs[0].copy()
    inputs[0]['decomposition_supported'] = False
    inputs[0]['decomposition_reason'] = 'source:missing_price'
    actual, detail, proof = replay(inputs)
    assert len(actual) == len(inputs[0]) and proof['supported_rows_compared'] == 0
    assert actual.xbtusd_execution_change.isna().all() and detail.contribution.isna().all()


@pytest.mark.parametrize('seed', [7, 59, 301])
def test_random_mixed_positions_recurring_prices_and_rational_identity(seed):
    random = np.random.default_rng(seed)
    actions = []
    for i in range(100):
        symbol = str(random.choice(['XBTUSD', 'XBTH20', 'ETHUSD']))
        amount, sign = int(random.integers(1, 10)), int(random.choice([-1, 1]))
        unit = 1000 if symbol == 'ETHUSD' else -1000000
        cost = round(unit*amount*sign*(1+float(random.normal(0, .01))))
        actions.append((i*6, symbol, 'Settlement' if i % 11 == 0 else 'Trade',
            amount, 'Buy' if sign > 0 else 'Sell', cost, int(random.integers(-100, 1000))))
    source, events, wallet, prices, activity, specs = window(actions)
    prices['close'] *= np.exp(random.normal(0, .02, len(prices)))
    points = pd.DatetimeIndex(sorted(set(source.end) | set(source.label_end)))
    refs = reference_snapshots(events, wallet, points)
    book, observed = reference_prices(prices, activity)
    rows = source_rows(refs, points, book, observed, specs)
    source = decompose(rows, account_snapshots(events, wallet, points), price_lookup(prices, activity), specs)
    actual, _, proof = replay((source, events, wallet, prices, activity, specs))
    assert proof['supported_rows_compared'] == 2 and actual.contract_count.eq(3).all()


@pytest.mark.parametrize('field', ['cost_satoshi', 'net_quantity', 'trade_count', 'label_end',
    'group', 'contribution', 'equity_change_satoshi', 'delta_change_satoshi', 'supported'])
def test_independent_replay_rejects_contract_record_tampering(field):
    source, events, wallet, prices, activity, specs = inputs = original()
    actual, detail, _ = replay(inputs)
    bad = detail.copy()
    if field == 'label_end':
        bad.loc[0, field] += pd.Timedelta(minutes=5)
    elif field == 'group':
        bad.loc[0, field] = 'other_contract'
    elif field == 'supported':
        bad.loc[0, field] = False
    elif field in ['contribution', 'equity_change_satoshi', 'delta_change_satoshi']:
        bad.loc[0, field] = str(Decimal(bad.loc[0, field])+1)
    else:
        bad.loc[0, field] += 1
    with pytest.raises(ValueError):
        verify_contracts(source, actual, bad, events, wallet, prices, activity, specs)


@pytest.mark.parametrize('mode', ['missing', 'extra', 'duplicate'])
def test_independent_replay_rejects_missing_extra_and_duplicate_rows(mode):
    source, events, wallet, prices, activity, specs = inputs = original()
    actual, detail, _ = replay(inputs)
    if mode == 'missing':
        detail = detail.iloc[1:]
    else:
        extra = detail.iloc[:1].copy()
        if mode == 'extra':
            extra['symbol'] = 'OTHER'
        detail = pd.concat([detail, extra], ignore_index=True)
    with pytest.raises(ValueError):
        verify_contracts(source, actual, detail, events, wallet, prices, activity, specs)


@pytest.mark.parametrize('mode', ['missing_price', 'unobserved', 'unsupported', 'nonpositive',
    'missing_first_time', 'wrong_count', 'noninteger', 'duplicate_time', 'overwrite'])
def test_source_and_valuation_integrity_fail_closed(mode):
    source, events, wallet, prices, activity, specs = original()
    specs = copy.deepcopy(specs)
    if mode == 'missing_price':
        prices = prices.loc[prices.symbol.ne('XBTUSD')]
    elif mode == 'unobserved':
        activity.loc[activity.symbol.eq('XBTUSD'), 'observed_trade'] = False
    elif mode == 'unsupported':
        specs['XBTUSD']['supported'] = False
    elif mode == 'missing_first_time':
        del specs['XBTUSD']['first_time']
    elif mode == 'nonpositive':
        source.loc[1, 'fees_equity_satoshi'] = '0'
    elif mode == 'wrong_count':
        source.loc[0, 'trade_count'] = 1
    elif mode == 'noninteger':
        events['lastqty'] = events.lastqty.astype(float)
    elif mode == 'duplicate_time':
        source.loc[1, 'end'] = source.loc[0, 'end']
    else:
        source['xbtusd_execution_change'] = 0
    with pytest.raises(ValueError):
        decompose_contracts(source, events, wallet, price_lookup(prices, activity), specs)
