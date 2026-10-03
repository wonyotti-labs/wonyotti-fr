import json

import numpy as np
import pandas as pd
import pytest
from test_minute_inventory_research import ready_bars
from test_minute_inventory_research import selection as minute_selection

from wonyotti_fr.addition_research import copy_minute_parent
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.pullback_policy import PullbackPolicy
from wonyotti_fr.recent_entry import (
    ENTRY_FILES,
    ENTRY_PERIOD,
    RecentEntryPolicy,
    recent_entry_training,
    run_recent_entry_selection,
)


def source_data():
    n = 2800
    end = pd.date_range("2020-01-03", periods=n, freq="6h", tz="UTC")
    rng = np.random.default_rng(23)
    data = pd.DataFrame(rng.normal(size=(n, 14)), columns=MARKET_FEATURES)
    data["end"], data["label_end"] = end, end + pd.Timedelta(minutes=5)
    data["active"] = np.arange(n) % 2
    data["buy"] = (np.arange(n) // 2) % 2
    data["episode_id"] = np.arange(n) // 10 + 1
    data["target_episode_id"] = data.episode_id.where(data.active.eq(1), 0)
    data["target_time"] = (data.end + pd.Timedelta(minutes=1)).where(data.active.eq(1))
    data["usable"] = True
    episodes = (
        data.groupby("episode_id")
        .agg(entry_time=("end", "min"), exit_time=("label_end", "max"))
        .reset_index()
    )
    return data, episodes


def selection(root, parent, previous):
    old = minute_selection(parent, previous)
    root.mkdir()
    copy_minute_parent(parent, root)
    models = json.loads((root / "expansion_models.json").read_text())
    models["direction"]["intercept"] = -2.0
    save_json(root / "recent_entry_models.json", models)
    save_json(
        root / "recent_entry_training.json",
        {"training_period": ENTRY_PERIOD, "activity_quantile": 0.975, "activity_threshold": 0.1},
    )
    frozen = {
        **old,
        "protocol": "recent_entry_v23",
        "minute_selection_sha256": sha256(root / "minute_selection.json"),
        "entry_training_period": ENTRY_PERIOD,
        "entry_activity_quantile": 0.975,
        "entry_activity_threshold": 0.1,
        "entry_direction_threshold": 0.65,
        "entry_kind": "logistic",
        "entry_files_sha256": {n: sha256(root / n) for n in ENTRY_FILES},
    }
    save_json(root / "frozen_selection.json", frozen)
    save_json(
        root / "frozen_integrity.json",
        {"frozen_selection_sha256": sha256(root / "frozen_selection.json")},
    )
    return frozen


def test_both_episode_boundaries_and_embargo_are_preserved():
    data, episodes = source_data()
    episodes.loc[episodes.episode_id.eq(1), "entry_time"] = pd.Timestamp("2019-12-31", tz="UTC")
    episodes.loc[episodes.episode_id.eq(2), "exit_time"] = pd.Timestamp("2022-01-02", tz="UTC")
    data.loc[20, "target_episode_id"] = 1
    data.loc[21, "target_episode_id"] = 2
    data.loc[22, "usable"] = False
    data.loc[23, "end"] = pd.Timestamp("2020-01-01", tz="UTC")
    data.loc[24, "label_end"] = pd.Timestamp("2021-12-31", tz="UTC")
    train, ledger = recent_entry_training(data, episodes)
    assert ledger.loc[:9, "reason"].eq("left_episode_boundary").all()
    assert ledger.loc[10:19, "reason"].eq("right_episode_boundary").all()
    assert ledger.loc[20:24, "reason"].tolist() == [
        "left_episode_boundary",
        "right_episode_boundary",
        "unusable_original_event",
        "before_training_or_left_embargo",
        "after_training_or_right_embargo",
    ]
    pd.testing.assert_frame_equal(train, data.iloc[25:], check_exact=True)
    assert len(ledger) == len(data)


@pytest.mark.parametrize("damage", ["parent", "model", "quantile", "risk", "threshold"])
def test_recent_entry_model_chain_rejects_tampering(tmp_path, damage):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous")
    assert isinstance(load_selection(root)[1], RecentEntryPolicy)
    if damage == "parent":
        (root / "minute_selection.json").write_text("{}")
    elif damage == "model":
        (root / "recent_entry_models.json").write_text("{}")
    else:
        if damage == "quantile":
            frozen["entry_activity_quantile"] = 0.9
        elif damage == "threshold":
            frozen["entry_activity_threshold"] = 0.2
        else:
            frozen["risk"]["max_adds"] = 0
        save_json(root / "frozen_selection.json", frozen)
        save_json(
            root / "frozen_integrity.json",
            {"frozen_selection_sha256": sha256(root / "frozen_selection.json")},
        )
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize("kind", ["waiting", "position"])
def test_recent_entry_actual_crash_and_observed_guard(tmp_path, kind):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous")
    frame = ready_bars()
    for name in ["open", "high", "low", "close"]:
        frame.loc[np.arange(len(frame)) % 5 == 0, name] = 100.2
    report = verify_stress(
        list(iter_events(frame)), root, tmp_path / "stress", {}, kind, True
    )
    assert report["all_passed"] and report["child_process_exit_code"] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, "2020-01-01", "2021-01-01")


def test_eleven_conditions_keep_old_entry_in_every_legacy_control(tmp_path, monkeypatch):
    root = tmp_path / "selection"
    frozen = selection(root, tmp_path / "parent", tmp_path / "previous")
    for name in ["manifest-1m.json", "manifest-5m.json"]:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        "wonyotti_fr.pullback_evaluation.prepare_minute_period", lambda *_, **__: (ready_bars(), {})
    )
    monkeypatch.setattr(
        "wonyotti_fr.pullback_evaluation.block_interval", lambda _: {"synthetic": True}
    )
    out = run_pullback_evaluation(
        root, tmp_path, tmp_path, tmp_path / "runs", "seen_2026", ["BTCUSDT"]
    )
    rows = json.loads((out / "results.json").read_text())
    assert len(rows) == 11 and "previous_v21" in {r["strategy"] for r in rows}
    _, original = load_selection(tmp_path / "parent")
    _, current = load_selection(out)
    assert current.base.direction.data != original.base.direction.data
    assert current.manager.to_dict() == original.manager.to_dict()
    assert current.size_model.to_dict() == original.size_model.to_dict()
    old_risk = json.loads((root / "pullback_selection.json").read_text())["risk"]
    for name, bot, risk in [
        ("previous_v21", original, frozen["risk"]),
        (
            "ungated_v7",
            PullbackPolicy(original.base, original.offset_bps, original.ttl_minutes),
            old_risk,
        ),
    ]:
        target = tmp_path / name
        backtest(ready_bars(), bot, EngineConfig(**risk), target)
        for file in ["equity.parquet", "fills.parquet", "trades.parquet"]:
            pd.testing.assert_frame_equal(
                pd.read_parquet(target / file),
                pd.read_parquet(out / "BTCUSDT" / name / file),
                check_exact=True,
            )
        assert json.loads((target / "final_state.json").read_text()) == json.loads(
            (out / "BTCUSDT" / name / "final_state.json").read_text()
        )


def test_full_recent_training_pipeline_keeps_purged_rows_and_quantile(tmp_path, monkeypatch):
    root = tmp_path / "parent"
    frozen = minute_selection(root, tmp_path / "previous")
    _, original = load_selection(root)
    frame = ready_bars().assign(volume=1.0, count=1)
    backtest(frame, original, EngineConfig(**frozen["risk"]), root / "candidate-00")
    for name in ["manifest-1m.json", "manifest-5m.json"]:
        (tmp_path / name).write_text("{}")
    data, episodes = source_data()
    monkeypatch.setattr(
        "wonyotti_fr.recent_entry.source_inputs",
        lambda *_: ({"episodes": episodes}, {"synthetic": True}),
    )
    monkeypatch.setattr("wonyotti_fr.recent_entry.make_expansion_data", lambda _: data)
    monkeypatch.setattr(
        "wonyotti_fr.recent_entry.prepare_minute_period", lambda *_, **__: (frame, {})
    )
    out = run_recent_entry_selection(
        root,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path / "runs",
    )
    new, policy = load_selection(out)
    pd.testing.assert_frame_equal(
        pd.read_parquet(out / "entry_training_used.parquet"), data, check_exact=True
    )
    threshold = np.quantile(
        policy.base.activity.probabilities(data[MARKET_FEATURES].to_numpy()), 0.975
    )
    assert threshold == new["entry_activity_threshold"]
    assert new["risk"] == frozen["risk"]
    assert json.loads((out / "baseline_parity.json").read_text())["full_outputs_and_state_exact"]
