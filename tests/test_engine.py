import copy

import pytest

from wonyotti_fr.engine import EngineConfig, TradingEngine


def bar(index, price=100, **extra):
    from datetime import UTC, datetime, timedelta
    return {"time": (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=index * 5)).isoformat(),
            "open": price, "high": price, "low": price, "close": price, "step": index, **extra}


def config(**extra):
    return EngineConfig(**{"stop_fraction": 0, "fee_bps": 0, "slippage_bps": 0,
                           "max_hold_bars": 0, "daily_loss_limit": 1, "max_drawdown": 1,
                           "allow_adverse_add": True, **extra})


def actions(event, state):
    return {0: "enter_long", 1: "increase", 2: "reduce", 3: "exit"}.get(event["step"], "hold")


def test_partial_changes_reconcile_and_only_execute_next_bar():
    engine = TradingEngine(config())
    outputs = [engine.step(bar(i), actions, final=i == 5) for i in range(6)]
    assert not outputs[0]["fills"]
    assert outputs[1]["quantity"] == 25
    assert outputs[2]["quantity"] == 50
    assert outputs[3]["quantity"] == 25
    assert outputs[4]["quantity"] == 0
    assert engine.state["closed_trades"] == 1
    assert engine.state["cash"] == 10000
    assert max(abs(o["accounting_residual"]) for o in outputs) < 1e-8


def test_snapshot_restart_preserves_fees_funding_and_partial_positions():
    original = TradingEngine(config(fee_bps=5, slippage_bps=3))
    for i in range(3):
        original.step(bar(i, price=100 + i), actions)
    resumed = TradingEngine(original.config, original.snapshot())
    for i in range(3, 7):
        event = bar(i, price=100 + i, funding_rate=0.001 if i == 3 else 0)
        left = original.step(event, actions, final=i == 6)
        right = resumed.step(event, actions, final=i == 6)
        assert left == right
    assert original.snapshot() == resumed.snapshot()
    assert original.state["total_funding"] > 0


def test_bad_policy_or_missing_bar_rolls_back_state():
    engine = TradingEngine(config())
    before = engine.snapshot()
    with pytest.raises(ValueError, match="주문 의도"):
        engine.step(bar(0), lambda *_: "unknown")
    assert engine.snapshot() == before
    engine.step(bar(0), actions)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="누락"):
        engine.step(bar(2), actions)
    assert engine.snapshot() == before


def test_adverse_addition_can_be_blocked_without_deleting_loss():
    engine = TradingEngine(config(allow_adverse_add=False))
    engine.step(bar(0), actions)
    engine.step(bar(1), actions)
    result = engine.step(bar(2, price=90), actions)
    assert "adverse_add_blocked" in result["rejected"]
    assert result["equity"] == 9750
    assert result["quantity"] == 25


def test_future_policy_cannot_rewrite_past_state():
    engine = TradingEngine(config())
    for i in range(3):
        engine.step(bar(i), actions)
    before = copy.deepcopy(engine.snapshot())
    alternative = TradingEngine(engine.config, before)
    engine.step(bar(3), lambda *_: "exit")
    alternative.step(bar(3), lambda *_: "increase")
    assert engine.state["cash"] == alternative.state["cash"]
    assert engine.state["quantity"] == alternative.state["quantity"]
    assert engine.state["pending"] != alternative.state["pending"]
