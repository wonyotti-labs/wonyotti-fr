from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.reconstruct import reconstruct
from wonyotti_fr.structure_study import SCENARIOS, episode_legs, next_prices, replay_episode


def event(minute, side, quantity, unit, fee=0, order=None, kind='Trade'):
    return {'time': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(minutes=minute),
            'side': side, 'quantity': quantity, 'cost_satoshi': (1 if side == 'Sell' else -1) * quantity * unit,
            'fee_satoshi': fee, 'exectype': kind, 'price': 1e8 / unit,
            'order_key': order or str(minute), 'fills': 1, 'maker_fills': int(fee < 0),
            'taker_fills': int(fee >= 0)}


def test_original_reversal_and_partial_fee_accounting():
    source = pd.DataFrame([event(0, 'Buy', 100, 10000, 10),
                           event(1, '', 100, 10000, 3, kind='Funding'),
                           event(2, 'Sell', 150, 8000, -15),
                           event(3, 'Buy', 50, 9000, 5)])
    expected, _, _ = reconstruct(source)
    legs = episode_legs(source)
    assert len(legs) == 5
    results = [replay_episode(part) for _, part in legs.groupby('episode_id')]
    np.testing.assert_allclose([r['net_btc'] for r in results], expected.net_pnl_btc, rtol=0, atol=1e-12)
    assert sum(r['fees_btc'] for r in results) == pytest.approx(0)
    assert sum(r['funding_btc'] for r in results) == pytest.approx(3e-8)


def test_no_adds_preserves_first_order_partial_fills_and_scales_reductions():
    source = pd.DataFrame([event(0, 'Buy', 40, 10000, order='first'),
                           event(1, 'Buy', 60, 10000, order='first'),
                           event(2, 'Buy', 100, 12500),
                           event(3, '', 200, 12500, 10, kind='Funding'),
                           event(4, 'Sell', 100, 8000),
                           event(5, 'Sell', 100, 9000)])
    result = replay_episode(episode_legs(source), no_adds=True)
    expected_gross = (Decimal(50) * (10000 - 8000) + Decimal(50) * (10000 - 9000)) / 10**8
    assert result['first_order_contracts'] == 100
    assert result['gross_btc'] == pytest.approx(float(expected_gross))
    assert result['funding_btc'] == pytest.approx(5e-8)
    assert result['net_btc_per_1000_first_order_contracts'] == pytest.approx(result['net_btc'] * 10)


def test_time_cap_can_cut_profitable_trade_into_loss_and_keeps_prior_funding():
    source = pd.DataFrame([event(0, 'Buy', 100, 10000),
                           event(30, '', 100, 10000, 20, kind='Funding'),
                           event(60, 'Sell', 100, 8000)])
    legs = episode_legs(source)
    original = replay_episode(legs)
    capped = replay_episode(legs, cap_minutes=30, cap_time=source.time.iloc[1], cap_price=9000)
    assert original['net_btc'] > 0 > capped['net_btc']
    assert capped['time_cap_applied']
    assert capped['funding_btc'] == pytest.approx(20e-8)
    price = 9000 * (1 - .0003)
    assert capped['net_btc'] == pytest.approx(100 * (1 / 10000 - 1 / price) - 100 / price * .0005 - 20e-8)


def test_fee_only_changes_fees_and_price_only_preserves_effective_fee_rate():
    legs = episode_legs(pd.DataFrame([event(0, 'Sell', 100, 10000, -10),
                                      event(60, 'Buy', 100, 12000, 30)]))
    legs['next_price'] = [9000, 8000]
    original = replay_episode(legs)
    fee = replay_episode(legs, fee_bps=5)
    assert fee['gross_btc'] == original['gross_btc']
    assert fee['fees_btc'] == pytest.approx((100 * 10000 + 100 * 12000) / 1e8 * .0005)
    price = replay_episode(legs, replace_price=True)
    expected = -10e-8 * (1 / (9000 * .9997)) / .0001 + 30e-8 * (1 / (8000 * 1.0003)) / .00012
    assert price['fees_btc'] == pytest.approx(expected)
    assert len(SCENARIOS) == 8


def test_minute_price_bounds_gaps_and_no_open_episode_success():
    minute = pd.DataFrame({'end': pd.date_range('2020-01-01', periods=4, freq='min', tz='UTC'),
                           'close': [100, 101, 102, 103]})
    request = pd.Series([minute.end.iloc[1]])
    prices, _ = next_prices(request, minute, strict=True)
    assert prices[0] == 102
    prices, _ = next_prices(request, minute, strict=False)
    assert prices[0] == 101
    with pytest.raises(ValueError, match='연속성'):
        next_prices(request, minute.drop(index=2), strict=True)
    with pytest.raises(ValueError, match='부족'):
        next_prices(pd.Series([minute.end.iloc[-1]]), minute, strict=True)
    with pytest.raises(ValueError, match='잔여'):
        replay_episode(episode_legs(pd.DataFrame([event(0, 'Buy', 100, 10000)])))
    with pytest.raises(ValueError, match='펀딩'):
        episode_legs(pd.DataFrame([event(0, '', 100, 10000, kind='Funding')]))
