from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.collateral_direction import direction_ledger, summarize_rows
from wonyotti_fr.collateral_reference import independent_ledger


def events(rows=None):
    if rows is None:
        rows = [('XBTUSD', 'Trade', 'Buy', 100, -1000, 0),
                ('XBTUSD', 'Trade', 'Sell', 60, 600, 0),
                ('XBTUSD', 'Trade', 'Sell', 90, 900, 0),
                ('XBTUSD', 'Funding', 'Sell', 50, 999999, 25),
                ('XBTUSD', 'Trade', 'Sell', 200, 2000, 0),
                ('XBTUSD', 'Trade', 'Buy', 250, -2500, 0)]
    f = pd.DataFrame(rows, columns=['symbol', 'exectype', 'side', 'lastqty', 'execcost', 'execcomm'])
    f['time'] = pd.date_range('2020-01-02', periods=len(f), freq='min', tz='UTC')
    f['settlcurrency'] = 'XBt'
    return f


def wallet(rows=None):
    if rows is None:
        rows = [('2020-01-01', 'Deposit', 'Completed', 2000)]
    f = pd.DataFrame(rows, columns=['date', 'transacttype', 'transactstatus', 'amount'])
    f['posted_amount'] = f.amount.where(f.transactstatus.eq('Completed'), 0)
    f['currency'] = 'XBt'
    return f


def points(e):
    return pd.DatetimeIndex([e.time.iloc[0], *[v+pd.Timedelta(1, 'ns') for v in e.time]])


def test_long_short_reversal_funding_and_close():
    e, w = events(), wallet()
    actual = direction_ledger(e, w, points(e))
    assert actual.xbt_quantity.tolist() == [0, 100, 40, -50, -50, -250, 0]
    assert actual.delta_min_satoshi.tolist() == [2000, 3000, 2400, 1500, 1475, -525, 1975]
    assert actual.account_direction.tolist() == ['long']*5+['short', 'long']
    pd.testing.assert_frame_equal(actual, independent_ledger(e, w, points(e)))
    summary = summarize_rows(actual)
    assert summary['different_rows'] == 4
    assert summary['independent_trade_count'] is None


@pytest.mark.parametrize('cost,expected', [(1999, 'long'), (2000, 'flat'), (2001, 'short')])
def test_exact_zero_without_arbitrary_dust_threshold(cost, expected):
    e = events([('XBTUSD', 'Trade', 'Sell', 200, cost, 0)])
    assert direction_ledger(e, wallet(), points(e)).account_direction.iloc[-1] == expected


def test_finite_difference_value_matches_cash_and_signed_basis():
    cash, q, entry = Decimal('1.2'), Decimal('-10000'), Decimal('10000')
    def equity(p):
        return p*(cash+q*(1/entry-1/p))
    derivative = (equity(Decimal('30001'))-equity(Decimal('30000')))
    assert derivative == cash+q/entry == Decimal('.2')


@pytest.mark.parametrize('symbol,cost,close_cost,ending', [
    ('ETHUSD', 500, -600, 3100), ('ETHUSD', 500, 0, 2500), ('XBTZ20', -500, 600, 2900)])
def test_other_contracts_only_supported_when_closed(symbol, cost, close_cost, ending):
    e = events([('XBTUSD', 'Trade', 'Buy', 100, -1000, 0),
                (symbol, 'Trade', 'Buy', 10, cost, 0),
                (symbol, 'Settlement', 'Sell', 10, close_cost, 0)])
    got = direction_ledger(e, wallet(), points(e))
    assert got.account_direction.iloc[2] == 'unsupported_other_contract'
    assert pd.isna(got.delta_min_satoshi.iloc[2])
    assert got.delta_min_satoshi.iloc[-1] == ending
    pd.testing.assert_frame_equal(got, independent_ledger(e, wallet(), points(e)))


@pytest.mark.parametrize('extra_days', [0, 1])
def test_each_transfer_bounded_without_netting_or_double_count(extra_days):
    e = events()
    w = wallet([('2019-12-25', 'Deposit', 'Completed', 2000),
                ('2020-01-02', 'Deposit', 'Completed', 4000),
                ('2020-01-02', 'Withdrawal', 'Completed', -6000),
                ('2020-01-02', 'Withdrawal', 'Canceled', -9000),
                ('2020-01-02', 'RealisedPNL', 'Completed', 999999)])
    got = direction_ledger(e, w, points(e), extra_days=extra_days)
    row = got.iloc[1]
    assert row.delta_min_satoshi == -3000 and row.delta_max_satoshi == 7000
    assert row.external_uncertain_count == 2 and row.account_direction == 'ambiguous'
    pd.testing.assert_frame_equal(got, independent_ledger(e, w, points(e), extra_days=extra_days))


def test_midnight_strict_boundaries_and_complete_window():
    e = events()
    e['time'] = pd.date_range('2020-01-01', periods=6, freq='D', tz='UTC')
    w = wallet([('2020-01-02', 'Deposit', 'Completed', 2000)])
    got = direction_ledger(e, w, pd.DatetimeIndex(e.time))
    assert got.iloc[1].delta_min_satoshi == got.iloc[1].delta_max_satoshi == 1000
    assert got.iloc[2].external_uncertain_count == 0
    assert got.iloc[2].delta_min_satoshi == 2400
    pd.testing.assert_frame_equal(got, independent_ledger(e, w, pd.DatetimeIndex(e.time)))


def test_tied_execution_boundary_and_future_invariance():
    e = events()
    e.loc[1, 'time'] = e.time.iloc[0]
    query = pd.DatetimeIndex([e.time.iloc[0], e.time.iloc[0]+pd.Timedelta(1, 'ns')])
    first = direction_ledger(e, wallet(), query)
    assert first.xbt_quantity.tolist() == [0, 40]
    e.loc[e.time.gt(query[-1]), 'execcost'] *= 7
    pd.testing.assert_frame_equal(first, direction_ledger(e, wallet(), query))
    pd.testing.assert_frame_equal(first, independent_ledger(e, wallet(), query))


def test_future_transfer_does_not_change_earlier_bound():
    e, w = events(), wallet()
    query = points(e)
    prior = direction_ledger(e, w, query, extra_days=1)
    w = pd.concat([w, wallet([('2020-01-20', 'Withdrawal', 'Completed', -1000)])], ignore_index=True)
    pd.testing.assert_frame_equal(prior, direction_ledger(e, w, query, extra_days=1))


@pytest.mark.parametrize('case', ['currency', 'kind', 'fraction', 'negative_qty', 'side', 'nan_time',
                                'unsorted', 'naive', 'bad_cost', 'overflow', 'wallet_status',
                                'wallet_posted', 'wallet_sign', 'wallet_currency', 'wallet_date'])
def test_invalid_input_rejected(case):
    e, w = events(), wallet()
    query = points(e)
    if case == 'currency':
        e.loc[0, 'settlcurrency'] = 'USDT'
    elif case == 'kind':
        e.loc[0, 'exectype'] = 'Unknown'
    elif case == 'fraction':
        e['lastqty'] = e.lastqty.astype(float)+.5
    elif case == 'negative_qty':
        e.loc[0, 'lastqty'] = -1
    elif case == 'side':
        e.loc[0, 'side'] = 'Unknown'
    elif case == 'nan_time':
        e.loc[0, 'time'] = pd.NaT
    elif case == 'unsorted':
        e = e.iloc[::-1]
    elif case == 'naive':
        e['time'] = e.time.dt.tz_localize(None)
    elif case == 'bad_cost':
        e.loc[0, 'execcost'] = 1000
    elif case == 'overflow':
        e.loc[0, 'execcomm'] = np.iinfo(np.int64).max
    elif case == 'wallet_status':
        w.loc[0, 'transactstatus'] = 'Pending'
    elif case == 'wallet_posted':
        w.loc[0, 'posted_amount'] += 1
    elif case == 'wallet_sign':
        w.loc[0, 'amount'] = w.loc[0, 'posted_amount'] = -100
    elif case == 'wallet_currency':
        w.loc[0, 'currency'] = 'USDT'
    elif case == 'wallet_date':
        w.loc[0, 'date'] = ''
    with pytest.raises((ValueError, TypeError)):
        direction_ledger(e, w, query)


def test_indexes_are_not_assumed_contiguous():
    e, w = events(), wallet()
    expected = direction_ledger(e, w, points(e))
    e.index = range(10, 10+len(e))
    w.index = [400]
    pd.testing.assert_frame_equal(expected, direction_ledger(e, w, points(e)))


def test_rebates_and_multiple_open_contracts():
    e = events([('XBTUSD', 'Trade', 'Sell', 200, 2000, -15),
                ('ETHUSD', 'Trade', 'Buy', 10, 500, 2),
                ('XBTZ20', 'Trade', 'Buy', 20, -800, 3),
                ('ETHUSD', 'Settlement', 'Sell', 10, -700, 4),
                ('XBTZ20', 'Settlement', 'Sell', 20, 600, 5)])
    got = direction_ledger(e, wallet(), points(e))
    assert got.other_open_contracts.tolist() == [0, 0, 1, 2, 1, 0]
    assert got.delta_min_satoshi.iloc[-1] == 401
    pd.testing.assert_frame_equal(got, independent_ledger(e, wallet(), points(e)))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fractional_basis_accumulation_and_random_reversals(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(500):
        side = 'Buy' if rng.integers(2) else 'Sell'
        qty, cost = int(rng.integers(1, 50)), int(rng.integers(100, 2000))
        rows.append(('XBTUSD', 'Funding' if i % 17 == 16 else 'Trade', side, qty,
            cost*(-1 if side == 'Buy' else 1), int(rng.integers(-5, 15))))
    e = events(rows)
    pd.testing.assert_frame_equal(direction_ledger(e, wallet(), points(e)),
        independent_ledger(e, wallet(), points(e)))


@pytest.mark.parametrize('case', ['late', 'early', 'unsorted', 'window', 'opening'])
def test_unsupported_queries_rejected(case):
    e, w = events(), wallet()
    query, kwargs = points(e), {}
    if case == 'late':
        query += pd.Timedelta(days=1)
    elif case == 'early':
        query -= pd.Timedelta(days=1)
    elif case == 'unsorted':
        query = query[::-1]
    elif case == 'window':
        kwargs['extra_days'] = 2
    elif case == 'opening':
        kwargs['opening'] = .5
    with pytest.raises(ValueError):
        direction_ledger(e, w, query, **kwargs)
