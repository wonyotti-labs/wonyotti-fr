import json

import numpy as np
import pandas as pd
import pytest
from test_minute_inventory_research import ready_bars
from test_minute_inventory_research import selection as minute_selection

from wonyotti_fr.addition_model import AdditionEffectModel, AdditionEffectPolicy
from wonyotti_fr.addition_research import ADDITION_FILES, addition_diagnostics, copy_minute_parent
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_inventory import MinuteInventoryModels
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def training():
    rng = np.random.default_rng(28)
    x = rng.normal(size=(144, 44))
    x[:, -1] = 0.0
    times = pd.date_range("2021-01-02", periods=144, freq="2D", tz="UTC")
    frame = pd.DataFrame(x, columns=AdditionEffectModel.features)
    frame["decision_time"] = times
    frame["label_end"] = times + pd.Timedelta(hours=1)
    frame["position_entry_time"] = times[np.arange(144) // 4 * 4]
    frame["label_status"] = "closed"
    frame["decision_equity"] = 10000.0
    frame["incremental_pnl"] = frame["incremental_bps"] = 5 * x[:, 0] - 3 * x[:, 1]
    return frame


def model_constant(value=1.0):
    return AdditionEffectModel.from_dict(
        {
            "format": "addition_effect_ridge_v1",
            "features": AdditionEffectModel.features,
            "alpha": 100,
            "mean": [0.0] * 44,
            "scale": [1.0] * 44,
            "coef": [0.0] * 44,
            "intercept": value,
        }
    )


def selection(root, parent, previous, value=1.0):
    old = minute_selection(parent, previous)
    manager = json.loads((parent / "inventory_model.json").read_text())
    manager["coef"], manager["intercept"] = [[0.0] * 44 for _ in range(3)], [-100.0, -100.0, 0.0]
    save_json(parent / "inventory_model.json", manager)
    calibration = json.loads((parent / "inventory_calibration.json").read_text())
    calibration["scales"] = {"exit": 1.0, "reduce": 1.0, "increase": 1.0}
    save_json(parent / "inventory_calibration.json", calibration)
    old["inventory_scales"] = calibration["scales"]
    old["inventory_files_sha256"] = {n: sha256(parent / n) for n in old["inventory_files_sha256"]}
    save_json(parent / "frozen_selection.json", old)
    save_json(
        parent / "frozen_integrity.json",
        {"frozen_selection_sha256": sha256(parent / "frozen_selection.json")},
    )
    root.mkdir()
    copy_minute_parent(parent, root)
    save_json(root / "addition_model.json", model_constant(value).to_dict())
    pd.DataFrame({"weight": [1.0]}).to_parquet(root / "addition_weights.parquet")
    save_json(
        root / "addition_weighting.json", {"algorithm": "equal_original_position_mean_one_v1"}
    )
    frozen = {
        **old,
        "protocol": "addition_effect_v22",
        "minute_selection_sha256": sha256(root / "minute_selection.json"),
        "addition_files_sha256": {n: sha256(root / n) for n in ADDITION_FILES},
        "addition_margin_bps": 0,
        "addition_weighting": "equal_original_position_mean_one_v1",
        "addition_training_period": ["2021-01-01", "2022-01-01"],
    }
    save_json(root / "frozen_selection.json", frozen)
    save_json(
        root / "frozen_integrity.json",
        {"frozen_selection_sha256": sha256(root / "frozen_selection.json")},
    )
    return frozen


def force_adds(bot):
    data = bot.manager.to_dict()
    data["coef"], data["intercept"] = [[0.0] * 44 for _ in range(3)], [-100.0, -100.0, 0.0]
    bot.manager = MinuteInventoryModels.from_dict(data)
    bot.scales = {"exit": 1.0, "reduce": 1.0, "increase": 1.0}
    return bot


def test_weighted_fit_matches_independent_normal_equations():
    frame = training().drop(index=[0, 1, 2]).reset_index(drop=True)
    model, weights, details = AdditionEffectModel.fit(frame)
    x, y = frame[model.features].to_numpy(), frame.incremental_bps.to_numpy()
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale == 0] = 1
    z = (x - mean) / scale
    intercept = np.average(y, weights=weights)
    coef = np.linalg.solve(
        z.T @ (z * weights[:, None]) + 100 * np.eye(44), z.T @ ((y - intercept) * weights)
    )
    np.testing.assert_allclose(model.predict(x), z @ coef + intercept, atol=1e-10, rtol=0)
    sums = pd.Series(weights).groupby(frame.position_entry_time).sum()
    np.testing.assert_allclose(sums, sums.iloc[0])
    assert details["negative_labels"] == int(frame.incremental_bps.lt(0).sum())
    assert details["export_max_error"] < 1e-10


@pytest.mark.parametrize("damage", ["rows", "positions", "future", "nan_time", "target", "span"])
def test_fit_rejects_support_or_time_or_target_damage(damage):
    frame = training()
    if damage == "rows":
        frame = frame.iloc[:99]
    elif damage == "positions":
        frame["position_entry_time"] = frame.position_entry_time.iloc[0]
    elif damage == "future":
        frame.loc[0, "label_end"] = pd.Timestamp("2022-01-01", tz="UTC")
    elif damage == "nan_time":
        frame.loc[0, "label_end"] = pd.NaT
    elif damage == "target":
        frame.loc[0, "incremental_pnl"] += 1
    else:
        frame["decision_time"] = pd.date_range("2021-01-02", periods=len(frame), freq="h", tz="UTC")
        frame["label_end"] = frame.decision_time + pd.Timedelta(minutes=1)
        frame["position_entry_time"] = frame.decision_time
    with pytest.raises(ValueError):
        AdditionEffectModel.fit(frame)


@pytest.mark.parametrize("value", [1.0, 0.0, -1.0])
def test_gate_preserves_consumed_state_and_acceptance(value, tmp_path):
    root = tmp_path / "parent"
    minute_selection(root, tmp_path / "previous")
    _, original = load_selection(root)
    original = force_adds(original)
    bot = AdditionEffectPolicy(original, model_constant(value))
    engine = TradingEngine(EngineConfig(bar_seconds=60, max_hold_bars=0, max_adds=5))
    decision_count = 0
    for event in iter_events(ready_bars()):

        def decide(bar, view):
            nonlocal decision_count
            expected, actual = original(bar, view), bot(bar, view)
            assert actual.state == expected.state
            if expected.intent == "increase":
                decision_count += 1
                assert actual.intent == ("increase" if value > 0 else "hold")
                assert actual.state["rate_increase"] == 0
            else:
                assert actual == expected
            return actual

        engine.step(event, decide)
    assert decision_count >= 2
    assert all(row["accepted"] == (value > 0) for row in bot.audit)


@pytest.mark.parametrize("damage", ["parent", "model", "weights", "risk", "margin"])
def test_loader_rejects_changed_model_chain(tmp_path, damage):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous")
    assert isinstance(load_selection(root)[1], AdditionEffectPolicy)
    if damage == "parent":
        (root / "minute_selection.json").write_text("{}")
    elif damage == "model":
        (root / "addition_model.json").write_text("{}")
    elif damage == "weights":
        (root / "addition_weights.parquet").write_bytes(b"changed")
    else:
        if damage == "risk":
            frozen["risk"]["max_adds"] = 0
        else:
            frozen["addition_margin_bps"] = 1
        save_json(root / "frozen_selection.json", frozen)
        save_json(
            root / "frozen_integrity.json",
            {"frozen_selection_sha256": sha256(root / "frozen_selection.json")},
        )
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_gate_actual_process_exit_and_restart(tmp_path, value):
    root = tmp_path / "selection"
    selection(root, tmp_path / "parent", tmp_path / "previous", value)
    report = verify_stress(
        list(iter_events(ready_bars())), root, tmp_path / "stress", {}, "addition", True
    )
    assert report["all_passed"] and report["child_process_exit_code"] == 73
    assert report["interruption_kind"] == "addition"


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_gate_delayed_queue_restarts_exactly(tmp_path, value):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous", value)
    _, bot = load_selection(root)
    cfg = EngineConfig(**{**frozen["risk"], "signal_delay_bars": 1})
    engine = TradingEngine(cfg)
    events = list(iter_events(ready_bars()))
    cut = None
    for i, event in enumerate(events):
        result = engine.step(event, bot)
        if result.get("policy_event") in {"action_increase", "action_add_rejected"}:
            cut = i + 1
            break
    assert cut is not None
    resumed = TradingEngine(cfg, engine.snapshot())
    _, fresh = load_selection(root)
    for event in events[cut:]:
        assert engine.step(event, bot) == resumed.step(event, fresh)
    assert engine.snapshot() == resumed.snapshot()


def test_eleven_conditions_preserve_unfiltered_parent(tmp_path, monkeypatch):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous")
    for name in ["manifest-1m.json", "manifest-5m.json"]:
        (tmp_path / name).write_text("{}")

    def prepared(*_, **kwargs):
        assert kwargs == {"minute_inputs": True}
        return ready_bars(), {}

    monkeypatch.setattr("wonyotti_fr.pullback_evaluation.prepare_minute_period", prepared)
    monkeypatch.setattr(
        "wonyotti_fr.pullback_evaluation.block_interval", lambda _: {"synthetic": True}
    )
    out = run_pullback_evaluation(
        root, tmp_path, tmp_path, tmp_path / "runs", "seen_2026", ["BTCUSDT"]
    )
    rows = json.loads((out / "results.json").read_text())
    assert len(rows) == 11 and "unfiltered_v21" in {r["strategy"] for r in rows}
    assert "inventory_fixed_size" not in {r["strategy"] for r in rows}
    assert isinstance(load_selection(out)[1], AdditionEffectPolicy)
    _, original = load_selection(tmp_path / "parent")
    backtest(ready_bars(), original, EngineConfig(**frozen["risk"]), tmp_path / "original")
    for name in ["equity.parquet", "fills.parquet", "trades.parquet"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "original" / name),
            pd.read_parquet(out / "BTCUSDT/unfiltered_v21" / name),
            check_exact=True,
        )
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, "2020-01-01", "2021-01-01")


def test_audit_recomputes_actual_rejected_decisions(tmp_path):
    root = tmp_path / "parent"
    minute_selection(root, tmp_path / "previous")
    _, original = load_selection(root)
    bot = AdditionEffectPolicy(force_adds(original), model_constant(-1.0))
    cfg = EngineConfig(bar_seconds=60, max_hold_bars=0)
    frame = ready_bars().assign(volume=1.0, count=1)
    backtest(frame, bot, cfg, tmp_path / "run")
    audit = addition_diagnostics(tmp_path / "run", frame, bot, cfg)
    assert audit["addition_gate"]["rejected"] >= 2
    assert audit["addition_gate"]["accepted"] == 0


def test_complete_training_pipeline_preserves_parent_and_all_labels(tmp_path, monkeypatch):
    from wonyotti_fr.addition_research import run_addition_selection

    parent = tmp_path / "parent"
    selection(tmp_path / "unused", parent, tmp_path / "previous")
    frozen, original = load_selection(parent)
    frame = ready_bars().assign(volume=1.0, count=1)
    backtest(frame, original, EngineConfig(**frozen["risk"]), parent / "candidate-00")
    for name in ["manifest-1m.json", "manifest-5m.json"]:
        (tmp_path / name).write_text("{}")
    labels = tmp_path / "labels"
    labels.mkdir()
    source = training()
    for name in ["training_labels.parquet", "opportunity_ledger.parquet"]:
        source.to_parquet(labels / name, index=False)
    save_json(labels / "summary.json", {"complete": True})
    save_json(
        labels / "files.json",
        {
            n: sha256(labels / n)
            for n in ["training_labels.parquet", "opportunity_ledger.parquet", "summary.json"]
        },
    )
    save_json(
        labels / "manifest.json",
        {
            "settings": {
                "reference_sha256": sha256(parent / "frozen_selection.json"),
                "market_manifest_sha256": sha256(tmp_path / "manifest-1m.json"),
                "feature_manifest_sha256": sha256(tmp_path / "manifest-5m.json"),
                "training_period": ["2021-01-01", "2022-01-01"],
            }
        },
    )
    monkeypatch.setattr(
        "wonyotti_fr.addition_research.prepare_minute_period", lambda *_, **__: (frame, {})
    )
    out = run_addition_selection(
        parent, labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "runs"
    )
    assert isinstance(load_selection(out)[1], AdditionEffectPolicy)
    assert json.loads((out / "baseline_parity.json").read_text())["full_outputs_and_state_exact"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(out / "training_used.parquet"), source, check_exact=True
    )
    assert json.loads((out / "summary.json").read_text())["selected_by_profit"] is False
    assert (
        json.loads((out / "candidate-00/addition_diagnostics.json").read_text())["addition_gate"][
            "decisions"
        ]
        > 0
    )
