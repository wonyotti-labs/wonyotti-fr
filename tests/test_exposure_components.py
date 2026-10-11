import copy
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.exposure_components import (
    COMPONENTS,
    account_snapshots,
    decompose,
    price_lookup,
    summarize_components,
)
from wonyotti_fr.exposure_components_reference import (
    reference_prices,
    reference_snapshots,
    valuation,
    verify_decomposition,
)


def fixture():
    start = pd.Timestamp('2020-01-04', tz='UTC')
    points = pd.date_range(start, periods=8, freq='5min').as_unit('ns')
    specs = {'XBTUSD': {'supported': True, 'inverse': True, 'quote': 'USD', 'multiplier_satoshi': '100000000', 'first_time': (start-pd.Timedelta(minutes=5)).isoformat()},
        'ETHUSD': {'supported': True, 'inverse': False, 'quote': 'USD', 'multiplier_satoshi': '100', 'first_time': (start-pd.Timedelta(minutes=4)).isoformat()},
        'ALTBTC': {'supported': True, 'inverse': False, 'quote': 'XBT', 'multiplier_satoshi': '100000000', 'first_time': (start-pd.Timedelta(minutes=3)).isoformat()}}
    records = []

    def event(at, symbol, kind, quantity, side, cost, fee):
        records.append({'time': at, 'symbol': symbol, 'exectype': kind, 'lastqty': quantity, 'side': side,
            'execcost': cost, 'execcomm': fee, 'settlcurrency': 'XBt'})

    event(start-pd.Timedelta(minutes=5), 'XBTUSD', 'Trade', 50, 'Buy', -50000000, 0)
    event(start-pd.Timedelta(minutes=4), 'ETHUSD', 'Trade', 2, 'Buy', 2000, -10)
    event(start-pd.Timedelta(minutes=3), 'ALTBTC', 'Trade', 3, 'Buy', 3000000, 15)
    event(points[1], 'XBTUSD', 'Funding', 0, 'Buy', 999999999, 2000)
    event(points[1]+pd.Timedelta(minutes=1), 'XBTUSD', 'Trade', 20, 'Sell', 18000000, 1000)
    event(points[2]+pd.Timedelta(minutes=1), 'XBTUSD', 'Trade', 70, 'Sell', 63000000, 4000)
    event(points[3]+pd.Timedelta(minutes=1), 'ETHUSD', 'Settlement', 2, 'Sell', -2200, 25)
    event(points[4]+pd.Timedelta(minutes=1), 'ALTBTC', 'Funding', 0, 'Sell', -900, -400)
    event(points[5]+pd.Timedelta(minutes=1), 'ALTBTC', 'Trade', 3, 'Sell', -4000000, -100)
    event(points[-1], 'XBTUSD', 'Funding', 0, 'Sell', 99, 1000)
    events = pd.DataFrame(records)
    wallet = pd.DataFrame([{'date': '2019-12-29', 'currency': 'XBt', 'transactstatus': 'Completed',
        'transacttype': 'Deposit', 'amount': 200000000, 'posted_amount': 200000000}])
    prices = pd.DataFrame([{'symbol': symbol, 'end': point, 'close': price*(1+i*.01), 'support': 'available'}
        for i, point in enumerate(points) for symbol, price in [('XBTUSD', 100.), ('ETHUSD', 10.), ('ALTBTC', .01)]])
    activity = prices[['symbol', 'end']].assign(observed_trade=True)
    return events, wallet, points, prices, activity, specs


def source_rows(reference, points, prices, observed, specs):
    rows = []
    for first, last in zip(points[:-1], points[1:], strict=True):
        states = []
        for at in [first, last]:
            state = reference[at]
            calculated, missing = valuation(state['cash']+state['external_min'], state['positions'], at, prices, observed, specs)
            assert not missing
            states.append(float(calculated[-1]))
        rows.append({'end': first, 'label_end': last, 'change_supported': True, 'change_reason': 'included',
            'current_net_exposure': states[0], 'net_exposure': states[1], 'exposure_change': states[1]-states[0]})
    return pd.DataFrame(rows)


def prepared():
    events, wallet, points, prices, activity, specs = fixture()
    snapshots = account_snapshots(events, wallet, points)
    reference = reference_snapshots(events, wallet, points)
    independent_prices, observed = reference_prices(prices, activity)
    source = source_rows(reference, points, independent_prices, observed, specs)
    return events, wallet, points, prices, activity, specs, snapshots, reference, independent_prices, observed, source


def test_full_decomposition_cash_basis_funding_rebates_reversal_and_settlement():
    _, _, _, prices, activity, specs, snapshots, reference, independent_prices, observed, source = prepared()
    saved = decompose(source, snapshots, price_lookup(prices, activity), specs)
    proof = verify_decomposition(source, saved, reference, independent_prices, observed, specs)
    assert proof['supported_rows_compared'] == len(source)
    assert saved.iloc[0].trade_count == saved.iloc[0].funding_count == 0
    assert saved.iloc[1].trade_count == saved.iloc[1].funding_count == 1
    assert saved.iloc[3].settlement_count == 1
    assert saved.iloc[4].funding_change != 0
    assert saved.iloc[0].execution_change == 0 and saved.iloc[0].mark_change != 0
    assert np.allclose(saved[COMPONENTS].sum(axis=1), saved.exposure_change, atol=1e-12)


def test_strict_current_and_next_boundaries_and_funding_cost_not_principal():
    events, wallet, points, *_ = fixture()
    actual = account_snapshots(events, wallet, points)
    changed = events.copy()
    changed.loc[changed.exectype.eq('Funding'), 'execcost'] *= -999
    assert account_snapshots(changed, wallet, points) == actual
    assert actual[points[1]]['funding_count'] == 0
    assert actual[points[2]]['funding_count'] == 1
    assert actual[points[-1]]['funding_count'] == 2


@pytest.mark.parametrize('mode', ['missing_price', 'unobserved', 'unsupported_contract', 'nonpositive_stage', 'source_unsupported', 'external_uncertain'])
def test_all_unsupported_rows_preserved_and_independently_verified(mode):
    _, _, points, prices, activity, specs, snapshots, reference, _, _, source = prepared()
    if mode == 'missing_price':
        prices = prices.loc[~(prices.symbol.eq('ETHUSD') & prices.end.eq(points[4]))]
    elif mode == 'unobserved':
        activity.loc[activity.symbol.eq('ETHUSD') & activity.end.eq(points[4]), 'observed_trade'] = False
    elif mode == 'unsupported_contract':
        specs['ETHUSD']['supported'] = False
    elif mode == 'nonpositive_stage':
        prices.loc[prices.symbol.eq('XBTUSD') & prices.end.eq(points[1]), 'close'] = .01
    elif mode == 'source_unsupported':
        source.loc[0, ['change_supported', 'change_reason']] = [False, 'unobserved_price_trade']
    else:
        snapshots[points[1]]['external_uncertain'] = 1
        reference[points[1]]['external_uncertain'] = 1
    book, observed = reference_prices(prices, activity)
    saved = decompose(source, snapshots, price_lookup(prices, activity), specs)
    proof = verify_decomposition(source, saved, reference, book, observed, specs)
    assert len(saved) == len(source) and proof['supported_rows_compared'] < len(source)
    assert saved.loc[~saved.decomposition_supported, COMPONENTS].isna().all().all()


def test_future_event_changes_cannot_change_prior_decomposition():
    events, wallet, points, prices, activity, specs, snapshots, _, _, _, source = prepared()
    saved = decompose(source.iloc[:3], snapshots, price_lookup(prices, activity), specs)
    changed = events.copy()
    changed.loc[changed.time.ge(points[3]), 'execcomm'] += 10
    later = account_snapshots(changed, wallet, points)
    pd.testing.assert_frame_equal(saved, decompose(source.iloc[:3], later, price_lookup(prices, activity), specs), check_exact=True)


@pytest.mark.parametrize('key', ['initial_beta', 'marks_equity_satoshi', 'execution_change', 'trade_count', 'decomposition_reason'])
def test_independent_verifier_rejects_changed_stage_and_counts(key):
    _, _, _, prices, activity, specs, snapshots, reference, book, observed, source = prepared()
    saved = decompose(source, snapshots, price_lookup(prices, activity), specs)
    saved.loc[0, key] = '123' if key in ['marks_equity_satoshi', 'decomposition_reason'] else saved.loc[0, key]+1
    with pytest.raises(ValueError):
        verify_decomposition(source, saved, reference, book, observed, specs)


def test_external_transfer_uncertainty_and_exact_completion_boundary():
    events, wallet, points, *_ = fixture()
    wallet.loc[0, 'date'] = '2020-01-02'
    core = account_snapshots(events, wallet, points)
    reference = reference_snapshots(events, wallet, points)
    assert core[points[0]]['external_uncertain'] == reference[points[0]]['external_uncertain'] == 0
    assert core[points[0]]['external_min'] == reference[points[0]]['external_min'] == 200000000
    wallet.loc[0, 'date'] = '2020-01-03'
    core = account_snapshots(events, wallet, points)
    reference = reference_snapshots(events, wallet, points)
    assert core[points[0]]['external_uncertain'] == reference[points[0]]['external_uncertain'] == 1


def test_known_inverse_price_only_beta_change():
    from wonyotti_fr.exposure_components import value
    point = pd.Timestamp('2020-01-01', tz='UTC')
    specs = {'XBTUSD': {'supported': True, 'first_time': '2019-01-01T00:00:00+00:00', 'inverse': True,
        'quote': 'USD', 'multiplier_satoshi': '100000000'}}
    first, _ = value(150000000, {'XBTUSD': 50}, point, {('XBTUSD', point): (Decimal(100), True)}, specs)
    last, _ = value(150000000, {'XBTUSD': 50}, point, {('XBTUSD', point): (Decimal(110), True)}, specs)
    assert first[2] == Decimal('1.5')
    assert float(last[2]) == pytest.approx(150/(150-5000/110))


def test_summary_retains_no_trade_windows_and_population_covariance():
    from wonyotti_fr.exposure_components_reference import independent_statistics
    _, _, _, prices, activity, specs, snapshots, _, _, _, source = prepared()
    saved = decompose(source, snapshots, price_lookup(prices, activity), specs)
    summary = summarize_components(saved)
    independent, covariance = independent_statistics(saved)
    for key, expected in independent.items():
        for stat in ['mean', 'mae', 'rms']:
            assert summary['statistics'][key][stat] == pytest.approx(expected[stat], abs=1e-15)
    np.testing.assert_allclose(summary['population_covariance'], covariance, rtol=1e-12, atol=1e-15)
    assert summary['no_execution_rows'] >= 2 and summary['no_execution_nonzero_total_change_rows'] >= 1


def test_duplicate_query_and_price_rejected():
    events, wallet, points, prices, activity, *_ = fixture()
    with pytest.raises(ValueError):
        account_snapshots(events, wallet, points.insert(0, points[0]))
    with pytest.raises(ValueError):
        price_lookup(pd.concat([prices, prices.iloc[:1]]), activity)
    bad = copy.deepcopy(activity)
    with pytest.raises(ValueError):
        price_lookup(prices, pd.concat([bad, bad.iloc[:1]]))


@pytest.mark.parametrize('seed', [0, 18, 99])
def test_random_whole_account_changes_match_independent_cash_and_basis(seed):
    events, wallet, old_points, _, _, specs = fixture()
    points = pd.date_range(old_points[0], periods=51, freq='5min').as_unit('ns')
    random = np.random.default_rng(seed)
    records = events.iloc[:3].to_dict('records')
    units = {'XBTUSD': -1000000, 'ETHUSD': 1000, 'ALTBTC': 1000000}
    for i in range(50):
        for offset in [0, 2]:
            symbol = str(random.choice(list(specs)))
            side, amount = str(random.choice(['Buy', 'Sell'])), int(random.integers(1, 20))
            kind = str(random.choice(['Trade', 'Settlement', 'Funding']))
            row = {'time': points[i]+pd.Timedelta(minutes=offset), 'symbol': symbol,
                'exectype': kind, 'lastqty': amount if kind != 'Funding' else 0, 'side': side,
                'execcost': units[symbol]*amount*(1 if side == 'Buy' else -1),
                'execcomm': int(random.integers(-100, 1000)), 'settlcurrency': 'XBt'}
            records.append(row)
    records.append({**records[-1], 'time': points[-1]})
    events = pd.DataFrame(records).sort_values('time', kind='stable').reset_index(drop=True)
    wallet.loc[0, ['amount', 'posted_amount']] = 10000000000
    prices = pd.DataFrame([{'symbol': symbol, 'end': point, 'close': price*float(np.exp(random.normal(0, .05))), 'support': 'available'}
        for point in points for symbol, price in [('XBTUSD', 100.), ('ETHUSD', 10.), ('ALTBTC', .01)]])
    activity = prices[['symbol', 'end']].assign(observed_trade=True)
    snapshots = account_snapshots(events, wallet, points)
    reference = reference_snapshots(events, wallet, points)
    book, observed = reference_prices(prices, activity)
    source = source_rows(reference, points, book, observed, specs)
    saved = decompose(source, snapshots, price_lookup(prices, activity), specs)
    proof = verify_decomposition(source, saved, reference, book, observed, specs)
    assert proof['supported_rows_compared'] == 50


@pytest.mark.parametrize('flag', [None, 1, 'True'])
def test_missing_or_non_boolean_price_activity_rejected(flag):
    _, _, _, prices, activity, _ = fixture()
    activity['observed_trade'] = activity.observed_trade.astype(object)
    activity.loc[0, 'observed_trade'] = flag
    with pytest.raises(ValueError, match='활동'):
        price_lookup(prices, activity)
