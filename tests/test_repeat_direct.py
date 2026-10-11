from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from test_history_state import history_inputs
from test_management_support import parent
from test_minute_inventory_research import ready_bars

from wonyotti_fr.common import sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.first_exit_verification import verify_execution
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.repeat_direct import ReferenceRepeatPolicy, RepeatDirectPolicy
from wonyotti_fr.repeat_direct_research import CASES, account_admission, holding_details


@pytest.mark.parametrize("action", ["exit", "reduce", "increase"])
@pytest.mark.parametrize("exists", [0, 1])
def test_only_repeated_addition_reduction_threshold_changes(action, exists):
    old = parent()
    new = RepeatDirectPolicy(old)
    state = {"_fill_features": np.array([exists, 0.0, 0.0, exists, 0.0, 0.0])}
    expected = old.action_threshold(action, state)
    if exists and action != "exit":
        expected = old.thresholds[action]
    assert new.action_threshold(action, state) == expected
    assert new.manager is old.manager and new.size_model is old.size_model
    assert new.first_thresholds == old.first_thresholds and new.thresholds == old.thresholds
    assert old.multiplier == new.multiplier == 1.5


@pytest.mark.parametrize("scores", [(0.0, 0.0, 0.6), (0.0, 0.6, 0.0), (0.8, 0.9, 0.9)])
def test_original_first_action_and_executed_repeat_history(scores):
    old = parent(scores)
    new = RepeatDirectPolicy(parent(scores))
    reference = ReferenceRepeatPolicy(parent(scores))
    stored = {}
    for minute in range(1, 6):
        values = {
            "adds": int(minute >= 2 and scores[2] > 0 and scores[0] == 0),
            "fraction": 0.7 if minute >= 2 and scores[1] > 0 and scores[0] == 0 else 1.0,
        }
        bar, state = history_inputs(minute, stored, **values)
        a, b, c = old(bar, state), new(bar, state), reference(bar, state)
        assert b == c
        if minute == 1 or minute < 4 or scores[0] > 0:
            assert a == b
        elif scores[0] == 0:
            assert a.intent == "hold" and b.intent in {"increase", "reduce"}
        stored = a.state
    assert "action_threshold" not in vars(reference.parent)


def test_reference_restores_prior_method_on_failure_and_invalid_flag_rejected():
    p = parent()

    def failure(*_):
        raise RuntimeError("주입 오류")

    p.action_threshold = failure
    with pytest.raises(RuntimeError):
        ReferenceRepeatPolicy(p)(*history_inputs(1))
    assert p.action_threshold is failure
    with pytest.raises(ValueError):
        RepeatDirectPolicy(parent()).action_threshold(
            "increase", {"_fill_features": [2, 0, 0, 0, 0, 0]}
        )
    with pytest.raises(ValueError):
        RepeatDirectPolicy(RepeatDirectPolicy(parent()))


@pytest.mark.parametrize("scores", [(0.0, 0.0, 0.6), (0.0, 0.6, 0.0), (0.8, 0.9, 0.9)])
def test_full_account_reference_replay_with_costs_funding_and_add_limits(tmp_path, scores):
    bars = ready_bars().assign(count=1, volume=1.0)
    bars.loc[bars.index % 10 == 0, "funding_rate"] = 0.0001
    cfg = EngineConfig(
        bar_seconds=60,
        max_hold_bars=20,
        max_adds=2,
        allow_adverse_add=True,
        entry_fraction=0.125,
        addition_fraction=0.25,
        fee_bps=5.0,
        slippage_bps=3.0,
    )
    folder = tmp_path / "candidate"
    backtest(bars, RepeatDirectPolicy(parent(scores)), cfg, folder)
    proof = verify_execution(folder, bars, ReferenceRepeatPolicy(parent(scores)))
    assert proof["all_policy_decisions_fills_trades_equity_and_state_exact"]
    assert proof["cash_fees_funding_and_net_verified"]
    details = holding_details(folder)
    assert details["closed_trades"] == read_json(folder / "metrics.json")["closed_trades"]
    trades = pd.read_parquet(folder / "trades.parquet")
    assert trades.adds.le(2).all()


def test_restart_and_future_change_preserve_prior_account():
    bars = ready_bars().assign(count=1, volume=1.0)
    cfg = EngineConfig(
        bar_seconds=60,
        max_hold_bars=20,
        entry_fraction=0.125,
        addition_fraction=0.25,
        allow_adverse_add=True,
    )

    def run(data, restart):
        policy = RepeatDirectPolicy(parent((0.0, 0.0, 0.6)))
        policy.prepare(data)
        engine = TradingEngine(cfg)
        records = []
        for i, bar in enumerate(iter_events(data)):
            records.append(engine.step(bar, policy, final=i == len(data) - 1))
            if restart and i % 7 == 0:
                engine = TradingEngine(cfg, engine.snapshot())
                policy = RepeatDirectPolicy(parent((0.0, 0.0, 0.6)))
                policy.prepare(data)
        return records, engine.snapshot()

    original = run(bars, False)
    assert original == run(bars, True)
    altered = bars.copy()
    altered.loc[altered.index >= 40, ["open", "high", "low", "close"]] *= 1.2
    assert original[0][:40] == run(altered, False)[0][:40]


def test_zero_size_reduction_and_wait_are_preserved():
    p = parent((0.0, 0.6, 0.0))
    p.size_model.predict = lambda x: np.zeros(len(x))
    old = p(*history_inputs(1))
    new = RepeatDirectPolicy(p)(*history_inputs(1))
    reference = ReferenceRepeatPolicy(p)(*history_inputs(1))
    assert old == new == reference and new.event == "action_zero_reduction"
    assert new.intent == "hold" and "management_after" in new.state


def test_btc_sample_gate_cannot_be_replaced_by_development_or_eth():
    rows = [
        {"case": key, "total_return": 0.1, "closed_trades": 30, "permanent_halt": False}
        for key in CASES
    ]
    rows[3]["closed_trades"] = 29
    assert not account_admission(rows)["cost_and_market_expansion_eligible"]
    rows[3]["closed_trades"] = 30
    rows[0]["total_return"] = -0.1
    rows[4]["total_return"] = -0.1
    summary = account_admission(rows)
    assert summary["cost_and_market_expansion_eligible"] and not summary["profitability_accepted"]
    rows[1]["permanent_halt"] = True
    assert not account_admission(rows)["cost_and_market_expansion_eligible"]
    with pytest.raises(ValueError):
        account_admission(rows[:-1])


def test_public_runner_completes_five_separate_accounts(tmp_path, monkeypatch):
    from wonyotti_fr import repeat_direct_research as research

    bars = ready_bars().assign(count=1, volume=1.0)
    cfg = EngineConfig(
        bar_seconds=60, max_hold_bars=20, entry_fraction=0.125, addition_fraction=0.25
    )
    baseline = tmp_path / "baseline"
    backtest(bars, parent(), cfg, baseline)
    spec = {
        "selection": str(tmp_path / "selection"),
        "cases": {
            name: {
                "symbol": symbol,
                "bounds": bounds,
                "market": "synthetic",
                "features": "synthetic",
                "baseline": str(baseline),
            }
            for name, (symbol, bounds) in CASES.items()
        },
    }
    monkeypatch.setattr(
        research, "checked_repeat_reference", lambda *_: (spec, {"risk": asdict(cfg)})
    )
    monkeypatch.setattr(research, "load_selection", lambda *_: ({}, parent()))
    monkeypatch.setattr(
        research,
        "prepare_minute_period",
        lambda *_args, **_kwargs: (bars.copy(), {"synthetic": True}),
    )
    root = research.run_repeat_direct(tmp_path / "reference.json", "synthetic", tmp_path / "out")
    assert read_json(root / "summary.json")["runs"] == 5
    for name in CASES:
        assert read_json(root / name / "execution_verification.json")[
            "cash_fees_funding_and_net_verified"
        ]
        for file, digest in read_json(root / name / "files.json").items():
            assert sha256(root / name / file) == digest
