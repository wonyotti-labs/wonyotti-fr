import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.simulator import RiskConfig, simulate


def bars(prices):
    result = pd.DataFrame(prices, columns=["open", "high", "low", "close"])
    result["time"] = pd.date_range("2024-01-01", periods=len(result), freq="15min", tz="UTC")
    result["end"] = result.time + pd.Timedelta(minutes=15)
    return result


def empty_funding():
    return pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns, UTC]"), "rate": pd.Series([], dtype=float)})


def no_cost(**kwargs):
    return RiskConfig(fee_bps=0, slippage_bps=0, allocation=1, stop_fraction=0,
                      max_hold_bars=0, daily_loss_limit=1, max_drawdown=1, **kwargs)


def test_signal_executes_on_next_open_and_round_trip_accounting():
    frame = bars([(100, 200, 100, 200), (150, 160, 150, 160), (170, 180, 170, 180)])
    curve, trades, fills, metrics = simulate(frame, np.array([1, 0, 0]), empty_funding(), no_cost())
    assert fills.iloc[0].time == frame.iloc[1].time
    assert trades.iloc[0].entry_price == 150
    assert curve.iloc[-1].equity == pytest.approx(10000 * 170 / 150)
    assert metrics["accounting_residual"] == pytest.approx(0, abs=1e-8)


def test_gap_stop_uses_worse_open_not_stop_price():
    frame = bars([(100, 100, 100, 100), (100, 101, 99, 100), (80, 90, 75, 85)])
    config = RiskConfig(allocation=1, fee_bps=0, slippage_bps=0, stop_fraction=0.1)
    _, trades, _, _ = simulate(frame, np.array([1, 1, 1]), empty_funding(), config)
    assert trades.iloc[0].exit_reason == "gap_stop"
    assert trades.iloc[0].exit_price == 80


def test_short_funding_and_fees_reconcile():
    frame = bars([(100, 100, 100, 100), (100, 100, 100, 100), (90, 90, 90, 90), (90, 90, 90, 90)])
    funding = pd.DataFrame({"time": [frame.iloc[2].time + pd.Timedelta(milliseconds=4)], "rate": [0.01]})
    _, trades, _, metrics = simulate(frame, np.array([-1, -1, 0, 0]), funding,
                                    RiskConfig(stop_fraction=0, daily_loss_limit=1, max_drawdown=1))
    assert trades.iloc[0].funding_cost < 0
    assert trades.fees.sum() > 0
    assert metrics["accounting_residual"] == pytest.approx(0, abs=1e-8)


def test_invalid_risk_is_rejected():
    with pytest.raises(ValueError, match="노출 비율"):
        RiskConfig(allocation=10).validate()


def test_future_signals_do_not_change_past_equity():
    frame = bars([(100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)])
    signals = np.ones(30, dtype=int)
    changed = signals.copy()
    changed[20:] = -1
    left, _, _, _ = simulate(frame, signals, empty_funding(), no_cost())
    right, _, _, _ = simulate(frame, changed, empty_funding(), no_cost())
    np.testing.assert_array_equal(left.equity.iloc[:21], right.equity.iloc[:21])


def test_stop_cooldown_prevents_instant_reentry():
    frame = bars([(100, 100, 100, 100)] + [(100, 100, 80, 100)] * 5)
    _, trades, fills, _ = simulate(frame, np.ones(6, dtype=int), empty_funding(),
                                  RiskConfig(stop_fraction=0.1, cooldown_bars=3, daily_loss_limit=1, max_drawdown=1))
    entries = fills[fills.reason == "entry"].time.to_list()
    assert entries == [frame.iloc[1].time, frame.iloc[5].time]
    assert len(trades) == 2
