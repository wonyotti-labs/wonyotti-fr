from decimal import Decimal

import pandas as pd
import pytest

from wonyotti_fr.reconstruct import reconstruct


def event(minute, side, qty, unit, fee=0, kind="Trade", order=None):
    direction = 1 if side == "Buy" else -1
    return {"time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(minutes=minute),
            "side": side, "quantity": qty, "cost_satoshi": -direction * qty * unit,
            "fee_satoshi": fee, "exectype": kind, "price": 1e8 / unit,
            "order_key": order or str(minute), "fills": 1, "maker_fills": int(fee < 0),
            "taker_fills": int(fee >= 0)}


def test_inverse_partial_close_and_funding():
    rows = [event(0, "Buy", 100, 10000, 10), event(1, "", 100, 10000, 3, "Funding"),
            event(2, "Sell", 40, 8000, 4), event(3, "Sell", 60, 9000, -2)]
    episodes, _, summary = reconstruct(pd.DataFrame(rows))
    expected = Decimal(140000 - 10 - 4 + 2 - 3)
    assert Decimal(summary["realized_net_satoshi_decimal"]) == expected
    assert summary["funding_position_mismatch_count"] == 0
    assert episodes.iloc[0].closed


def test_reversal_allocates_fees_and_cost_basis():
    rows = [event(0, "Buy", 100, 10000, 10), event(1, "Sell", 150, 8000, 15),
            event(2, "Buy", 50, 9000, 5)]
    episodes, actions, summary = reconstruct(pd.DataFrame(rows))
    assert list(actions.action) == ["open", "reverse", "close"]
    assert len(episodes) == 2
    assert episodes.fees_btc.sum() == pytest.approx(30 / 1e8)
    assert episodes.net_pnl_btc.sum() == pytest.approx((250000 - 30) / 1e8)
    assert summary["final_position_contracts"] == 0


def test_partial_fills_are_not_additional_entry_orders():
    rows = [event(0, "Buy", 40, 10000, order="a"), event(1, "Buy", 60, 10000, order="a"),
            event(2, "Buy", 50, 10000, order="b"), event(3, "Sell", 150, 10000)]
    episodes, _, _ = reconstruct(pd.DataFrame(rows))
    assert episodes.iloc[0].additional_entry_orders == 1
    assert episodes.iloc[0].max_qty == 150


def test_open_episode_is_not_reported_as_closed():
    episodes, _, summary = reconstruct(pd.DataFrame([event(0, "Sell", 20, 10000, 7)]))
    assert not episodes.iloc[0].closed
    assert summary["final_position_contracts"] == -20
    assert summary["realized_net_btc"] == -7 / 1e8
