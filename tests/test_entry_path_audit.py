import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json
from wonyotti_fr.engine import EngineConfig, PolicyDecision
from wonyotti_fr.entry_path_audit import (
    occupancy_summary,
    verify_holding_intervals,
    verify_occupancy_counts,
)
from wonyotti_fr.entry_path_research import checked_entry_path_reference
from wonyotti_fr.entry_wait_reference import verify_wait_episodes
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.pullback_diagnostics import waiting_diagnostics
from wonyotti_fr.pullback_policy import PullbackPolicy


class Base:
    def prepare(self, frame):
        pass

    def __call__(self, bar, state):
        return "enter_long"


class Managed:
    def __init__(self):
        self.entry = PullbackPolicy(Base(), 16, 5)

    def prepare(self, frame):
        self.entry.prepare(frame)

    def __call__(self, bar, state):
        if state["direction"]:
            return PolicyDecision(
                "hold",
                {"management_after": 1, "management_direction": state["direction"]},
                "action_hold",
            )
        if "management_direction" in state["policy_state"]:
            return PolicyDecision("hold", {}, "cleared")
        return self.entry(bar, state)


def make_run(tmp_path, *, kind="normal", periods=90):
    time = pd.date_range("2020-01-31T23:30Z", periods=periods, freq="min").as_unit("ns")
    close = np.where(np.arange(periods) % 10 == 5, 99.7, 100.0)
    if kind == "cancel":
        close[:] = 100
    bars = pd.DataFrame(
        {
            "time": time,
            "end": time + pd.Timedelta(minutes=1),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "funding_rate": 0.0,
            "volume": 1.0,
            "count": 2,
        }
    )
    bars[MARKET_FEATURES] = 0.0
    if kind == "illiquid":
        bars.loc[6, ["count", "volume"]] = 0
        bars.loc[6, ["open", "high", "low", "close"]] = 99.7
    if kind == "intrabar":
        bars.loc[6, "low"] = 90
    config = EngineConfig(bar_seconds=60, max_hold_bars=8, cooldown_bars=0, max_adds=0)
    folder = tmp_path / "source"
    backtest(bars, Managed(), config, folder)
    waiting_diagnostics(folder, bars, 0, management_state=True)
    curve, fills, trades, episodes = (
        pd.read_parquet(folder / (n + ".parquet"))
        for n in ["equity", "fills", "trades", "waiting_episodes"]
    )
    return folder, bars, curve, fills, trades, episodes


@pytest.mark.parametrize("kind", ["normal", "illiquid", "intrabar", "cancel"])
def test_complete_waits_holding_ordering_months_and_cancellation(tmp_path, kind):
    folder, bars, curve, fills, trades, episodes = make_run(
        tmp_path, kind=kind, periods=7 if kind == "cancel" else 90
    )
    check = verify_wait_episodes(curve, fills, bars, episodes)
    assert check["matched_entry_fills"] == len(trades)
    held = verify_holding_intervals(curve, trades, fills)
    assert held["held_close_minutes"] == int(curve.quantity.ne(0).sum())
    summary = occupancy_summary(curve, episodes)
    verify_occupancy_counts(curve, episodes, summary)
    if kind == "cancel":
        assert episodes.status.tolist() == ["cleared"] and not len(trades)
    else:
        assert set(summary["monthly"]) == {"2020-01", "2020-02"}
        if kind != "intrabar":
            assert check["management_resets"] > 0
    if kind == "illiquid":
        assert episodes.liquidity_wait_minutes.max() == 1
    if kind == "intrabar":
        assert (trades.exit_reason.eq("intrabar_stop") & trades.hold_bars.eq(1)).any()


@pytest.mark.parametrize(
    "kind",
    [
        "missing_start",
        "reference_price",
        "expiry",
        "wait_state",
        "direction",
        "fill",
        "status",
        "early_trigger",
    ],
)
def test_wait_tampering_detected(tmp_path, kind):
    _, bars, curve, fills, _, episodes = make_run(tmp_path)
    first = curve.index[curve.policy_event.eq("armed")][0]
    if kind == "missing_start":
        curve.loc[first, "policy_event"] = "idle"
    elif kind == "reference_price":
        episodes.loc[0, "reference_price"] *= 1.01
    elif kind == "expiry":
        import json

        value = json.loads(curve.policy_state.iloc[first])
        value["expires_at"] = "2020-02-01T12:00:00Z"
        curve.loc[first, "policy_state"] = json.dumps(value)
    elif kind == "wait_state":
        i = curve.index[curve.policy_event.eq("waiting")][0]
        curve.loc[i, "policy_state"] = "{}"
    elif kind == "direction":
        episodes.loc[0, "direction"] = -1
    elif kind == "fill":
        i = fills.index[fills.reason.eq("entry")][0]
        fills.loc[i, "price"] *= 1.01
    elif kind == "status":
        episodes.loc[0, "status"] = "expired"
    else:
        bars.loc[first + 1, "close"] = 100.0
    with pytest.raises((ValueError, AssertionError)):
        verify_wait_episodes(curve, fills, bars, episodes)


@pytest.mark.parametrize("kind", ["hold", "direction", "close_order", "overlap"])
def test_trade_interval_tampering_detected(tmp_path, kind):
    _, _, curve, fills, trades, _ = make_run(tmp_path, kind="intrabar")
    if kind == "hold":
        trades.loc[0, "hold_bars"] += 1
    elif kind == "direction":
        trades.loc[0, "direction"] *= -1
    elif kind == "close_order":
        trades.loc[0, "exit_reason"] = "time_limit"
    else:
        trades = pd.concat([trades.iloc[:1], trades], ignore_index=True)
    with pytest.raises((ValueError, AssertionError)):
        verify_holding_intervals(curve, trades, fills)


def test_partial_position_is_not_counted_as_new_trade(tmp_path):
    _, _, curve, fills, trades, _ = make_run(tmp_path)
    held = curve.index[curve.quantity.ne(0)]
    curve.loc[held[1:3], "quantity"] /= 2
    proof = verify_holding_intervals(curve, trades, fills)
    assert proof["closed_trades"] == len(trades)


def test_modified_monthly_counts_rejected(tmp_path):
    _, _, curve, _, _, episodes = make_run(tmp_path)
    original = occupancy_summary(curve, episodes)
    changed = copy.deepcopy(original)
    changed["monthly"]["2020-02"]["flat_close_minutes"] += 1
    with pytest.raises(AssertionError):
        verify_occupancy_counts(curve, episodes, changed)


def test_untrusted_reference_rejected(tmp_path):
    path = tmp_path / "proof.json"
    save_json(path, {})
    with pytest.raises(ValueError, match="지문"):
        checked_entry_path_reference(path, "0" * 64)


def test_equal_instants_with_microsecond_storage_match_exactly(tmp_path):
    _, bars, curve, fills, _, episodes = make_run(tmp_path)
    before = verify_wait_episodes(curve, fills, bars, episodes)
    for name in ["time", "end"]:
        bars[name] = bars[name].astype("datetime64[us, UTC]")
    after = verify_wait_episodes(curve, fills, bars, episodes)
    assert before == after


def test_public_runner_preserves_source_and_records_failure(tmp_path, monkeypatch):
    import wonyotti_fr.entry_path_research as research
    from wonyotti_fr.common import sha256
    from wonyotti_fr.retained_weight_close_reference import checked_files

    original, bars, _, _, _, _ = make_run(tmp_path)
    case = "sample"
    root = tmp_path / "parent"
    root.mkdir()
    original.rename(root / case)
    bounds = [str(bars.time.min()), str(bars.end.max())]
    spec = {"cases": {case: {"market": "synthetic"}}}
    monkeypatch.setattr(research, "CASES", {case: ("BTCUSDT", bounds)})
    monkeypatch.setattr(research, "checked_entry_path_reference", lambda *a: (root, spec, {}))
    monkeypatch.setattr(research, "load_market", lambda *a: (bars.copy(), pd.DataFrame()))
    initial = {p.name: sha256(p) for p in (root / case).iterdir() if p.is_file()}
    out = research.run_entry_path_audit(tmp_path / "ref", "synthetic", tmp_path / "output")
    checked_files(out)
    checked_files(out / case)
    summary = read_json(out / "summary.json")
    assert summary["complete"] and summary["all_rows"] == len(bars)
    assert not summary["new_accounts_executed"] and not summary["new_models_fitted"]
    assert initial == {p.name: sha256(p) for p in (root / case).iterdir() if p.is_file()}
    bars.loc[6, "close"] = 101.0
    with pytest.raises((ValueError, AssertionError)):
        research.run_entry_path_audit(tmp_path / "ref", "synthetic", tmp_path / "failed")
    assert len(list((tmp_path / "failed").glob("*/failure.json"))) == 1
