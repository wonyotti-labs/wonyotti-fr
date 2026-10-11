from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from test_collateral_direction import wallet

from wonyotti_fr.collateral_direction import direction_ledger
from wonyotti_fr.portfolio_valuation import VALUE_COLUMNS, contract_specs, portfolio_ledger
from wonyotti_fr.portfolio_valuation_checks import verify_values
from wonyotti_fr.portfolio_valuation_reference import independent_contract_check, independent_values


def fixture():
    rows = [
        ('XBTUSD', 'Trade', 'Buy', 1000, 10000., -10000000, 20, 'USD'),
        ('ETHUSD', 'Trade', 'Buy', 10, 1000., 1000000, 30, 'USD'),
        ('XRPH21', 'Trade', 'Buy', 100, .00002, 200000, 10, 'XBT'),
        ('XBTU21', 'Trade', 'Sell', 500, 12500., 4000000, 20, 'USD'),
        ('ETHUSD', 'Trade', 'Sell', 6, 1200., -720000, 10, 'USD'),
        ('ETHUSD', 'Trade', 'Sell', 8, 1500., -1200000, 10, 'USD'),
        ('ETHUSD', 'Funding', 'Sell', 4, 1600., 999999999, -15, 'USD'),
        ('XRPH21', 'Settlement', 'Sell', 100, .00003, -300000, 0, 'XBT'),
        ('XBTUSD', 'Trade', 'Sell', 2000, 12500., 16000000, 20, 'USD'),
        ('ETHUSD', 'Trade', 'Buy', 4, 1200., 480000, 10, 'USD'),
        ('XBTU21', 'Settlement', 'Buy', 500, 20000., -2500000, 0, 'USD')]
    e = pd.DataFrame(rows, columns=['symbol', 'exectype', 'side', 'lastqty', 'lastpx', 'execcost', 'execcomm', 'currency'])
    e['settlcurrency'] = 'XBt'
    e['time'] = pd.date_range('2021-01-02', periods=len(e), freq='5min', tz='UTC')
    points = pd.DatetimeIndex([e.time.iloc[0], *[t+pd.Timedelta(1, 'ns') for t in e.time]])
    p = pd.DataFrame([(symbol, t, px, 'available') for symbol, px in
        [('XBTUSD', 20000.), ('XBTU21', 25000.), ('ETHUSD', 1400.), ('XRPH21', .000025)]
        for t in e.time], columns=['symbol', 'end', 'close', 'support'])
    w = wallet([('2020-12-20', 'Deposit', 'Completed', 20000000),
        ('2021-01-02', 'Deposit', 'Completed', 4000000),
        ('2021-01-02', 'Withdrawal', 'Completed', -2000000),
        ('2021-01-02', 'Withdrawal', 'Canceled', -999999999),
        ('2021-01-02', 'RealisedPNL', 'Completed', 999999999)])
    return e, w, points, p


@pytest.mark.parametrize('extra_days', [0, 1])
def test_all_boundaries_cash_basis_finite_difference_and_price_stress(extra_days):
    e, w, q, p = fixture()
    specs = contract_specs(e, e.symbol)
    assert independent_contract_check(e, specs) == 4
    actual = portfolio_ledger(e, w, q, p, specs, extra_days=extra_days)
    expected = independent_values(e, w, q, p, specs, extra_days=extra_days)
    assert verify_values(actual, expected)['rows_verified'] == len(actual)
    for row, ref in zip(actual.to_dict('records'), expected, strict=True):
        assert row['unsupported_reasons'] == ref['unsupported_reasons'] == ''
        for key in VALUE_COLUMNS:
            assert abs(Decimal(row[key])-ref[key]) <= Decimal('0.00000051')
        for key in ['cost_balance_satoshi', 'external_min_satoshi', 'external_max_satoshi', 'external_uncertain_count']:
            assert row[key] == ref[key]
        assert row['equity_min_usd'] == pytest.approx(float(ref['equity_min_usd']))
    assert actual.cost_balance_satoshi.iloc[7]-actual.cost_balance_satoshi.iloc[6] == 15


def test_contract_multiplier_and_historical_inverse_rounding():
    e, _, _, _ = fixture()
    e.loc[0, 'execcost'] -= 499
    specs = contract_specs(e, e.symbol)
    assert Decimal(specs['XBTUSD']['multiplier_satoshi']) == 100000000
    assert Decimal(specs['XBTUSD']['maximum_unit_residual_satoshi']) == Decimal('.499')
    assert Decimal(specs['ETHUSD']['multiplier_satoshi']) == 100
    assert Decimal(specs['XRPH21']['multiplier_satoshi']) == 100000000


def test_v95_exact_xbt_only_delta_and_original_directions():
    e, w, q, p = fixture()
    e = e[e.symbol.eq('XBTUSD')].reset_index(drop=True)
    q = q[q <= e.time.max()+pd.Timedelta(1, 'ns')]
    new = portfolio_ledger(e, w, q, p, contract_specs(e, ['XBTUSD']))
    old = direction_ledger(e, w, q)
    for key in ['xbt_quantity', 'other_open_contracts', 'external_uncertain_count', 'account_direction', 'futures_direction']:
        assert new[key].tolist() == old[key].tolist()
    for key in ['delta_min_satoshi', 'delta_max_satoshi']:
        assert [Decimal(v) for v in new[key]] == old[key].tolist()


def test_quanto_and_btc_quoted_partial_delta_are_different():
    e, w, q, p = fixture()
    got = portfolio_ledger(e, w, q, p, contract_specs(e, e.symbol))
    assert Decimal(got.delta_min_satoshi.iloc[2])-Decimal(got.delta_min_satoshi.iloc[1]) == 399970
    assert Decimal(got.delta_min_satoshi.iloc[3])-Decimal(got.delta_min_satoshi.iloc[2]) == -200010


@pytest.mark.parametrize('mode', ['missing', 'invalid', 'future', 'stale', 'contract'])
def test_unsupported_open_contract_is_preserved(mode):
    e, w, q, p = fixture()
    specs = contract_specs(e, e.symbol)
    q = q[3:4]
    mask = p.symbol.eq('ETHUSD') & p.end.eq(q[0].floor('5min'))
    if mode == 'missing':
        p = p[~mask]
    elif mode == 'invalid':
        p.loc[mask, ['close', 'support']] = [np.nan, 'invalid_close']
    elif mode == 'future':
        p = p[~mask]
        assert (p.end > q[0]).any()
    elif mode == 'stale':
        p = p[~(p.symbol.eq('ETHUSD') & p.end.ge(q[0].floor('5min')))]
    else:
        specs['ETHUSD'] = {'supported': False}
    got = portfolio_ledger(e, w, q, p, specs).iloc[0]
    assert got.account_direction == 'unsupported' and got.open_contracts == 3
    assert got.delta_min_satoshi is None and 'ETHUSD' in got.unsupported_reasons


def test_future_events_transfers_and_prices_do_not_change_earlier_value():
    e, w, q, p = fixture()
    specs = contract_specs(e, e.symbol)
    q = q[:4]
    old = portfolio_ledger(e, w, q, p, specs)
    e.loc[e.time.gt(q[-1]), 'execcost'] *= 7
    p.loc[p.end.gt(q[-1]), 'close'] *= 9
    w = pd.concat([w, wallet([('2022-01-01', 'Deposit', 'Completed', 999999)])], ignore_index=True)
    pd.testing.assert_frame_equal(old, portfolio_ledger(e, w, q, p, specs), check_exact=True)


def test_nonpositive_equity_blocks_leverage_without_dropping_row():
    e, _, q, p = fixture()
    got = portfolio_ledger(e, wallet([('2020-12-20', 'Deposit', 'Completed', 0)]), q, p, contract_specs(e, e.symbol))
    assert len(got) == len(q) and np.isnan(got.btc_partial_leverage_min.iloc[0])
    assert got.account_direction.iloc[0] == 'flat'


@pytest.mark.parametrize('mode', ['currency', 'unit', 'costsign', 'inverse_rounding'])
def test_contract_inconsistency_rejected(mode):
    e, _, _, _ = fixture()
    if mode == 'currency':
        e.loc[4, 'settlcurrency'] = 'USDT'
    elif mode == 'unit':
        e.loc[4, 'execcost'] *= 2
    elif mode == 'costsign':
        e.loc[4, 'execcost'] *= -1
    else:
        e.loc[0, 'execcost'] -= 501
    with pytest.raises(ValueError):
        contract_specs(e, e.symbol)


@pytest.mark.parametrize('mode', ['duplicate', 'naive', 'offgrid', 'negative', 'nan', 'status'])
def test_invalid_price_source_rejected(mode):
    e, w, q, p = fixture()
    if mode == 'duplicate':
        p = pd.concat([p, p.iloc[[0]]])
    elif mode == 'naive':
        p['end'] = p.end.dt.tz_localize(None)
    elif mode == 'offgrid':
        p.loc[0, 'end'] += pd.Timedelta(1, 's')
    elif mode == 'negative':
        p.loc[0, 'close'] = -1
    elif mode == 'nan':
        p.loc[0, 'close'] = np.nan
    else:
        p.loc[0, 'support'] = 'invented'
    with pytest.raises(ValueError):
        portfolio_ledger(e, w, q, p, contract_specs(e, e.symbol))


@pytest.mark.parametrize('field,value', [('equity_min_satoshi', '1'), ('delta_min_satoshi', '1'),
    ('stress_delta_max_satoshi', '1'), ('account_direction', 'short'), ('btc_partial_leverage_min', 999.),
    ('cost_balance_satoshi', 999), ('external_uncertain_count', 999), ('unsupported_reasons', 'hidden'),
    ('equity_min_usd', 999.)])
def test_independent_reviewer_rejects_value_and_scope_tampering(field, value):
    e, w, q, p = fixture()
    specs = contract_specs(e, e.symbol)
    actual = portfolio_ledger(e, w, q, p, specs)
    ref = independent_values(e, w, q, p, specs)
    actual.loc[3, field] = value
    with pytest.raises(ValueError):
        verify_values(actual, ref)


@pytest.mark.parametrize('field,value', [('multiplier_satoshi', '101'), ('quote', 'XBT'),
    ('inverse', True), ('fills_checked', 999), ('maximum_unit_residual_satoshi', '1'),
    ('first_time', '2020-01-01T00:00:00+00:00')])
def test_independent_contract_reviewer_rejects_metadata_tampering(field, value):
    e, _, _, _ = fixture()
    specs = contract_specs(e, e.symbol)
    specs['ETHUSD'][field] = value
    with pytest.raises(ValueError):
        independent_contract_check(e, specs)
