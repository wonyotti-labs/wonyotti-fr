import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_history_state import history_inputs
from test_management_support import parent, values

from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.history_state import CalibratedHistoryModels
from wonyotti_fr.management_calibration import ManagementOffset
from wonyotti_fr.management_score_audit import (
    boundaries,
    repeat_admission,
    resolve_requests,
    score_frame,
    summarize_scores,
)
from wonyotti_fr.management_score_reference import forest_by_leaf, verify_scores, verify_summary
from wonyotti_fr.management_support import FEATURES, ManagementInputTrace
from wonyotti_fr.minute_inventory import MinuteReductionModel
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels
from wonyotti_fr.sparse_exposure_reference import raw_forest


def test_public_runner_writes_verified_fixed_score_artifact(tmp_path, monkeypatch):
    from wonyotti_fr import management_score_research as research
    from wonyotti_fr.action_model import select_threshold
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.first_management import BETAS
    from wonyotti_fr.holding_stress import read_json

    p, frame = policy(), source()
    scores = p.manager.probabilities(frame[FEATURES].to_numpy())
    common, first = {}, {}
    for i, action in enumerate(["exit", "reduce", "increase"]):
        p.thresholds[action], common[action] = select_threshold(
            frame[f"y_{action}"], scores[:, i], BETAS[action]
        )
        if action != "exit":
            mask = frame[f"past_{action}_exists"] == 0
            p.first_thresholds[action], first[action] = select_threshold(
                frame.loc[mask, f"y_{action}"], scores[mask, i], BETAS[action]
            )
    selection, teacher, previous = (
        tmp_path / name for name in ["selection", "teacher", "previous"]
    )
    for folder in [selection, teacher, previous]:
        folder.mkdir()
    save_json(selection / "history_thresholds.json", {"support": common})
    save_json(selection / "first_policy_thresholds.json", {"support": first})
    for phase in ["training", "calibration", "diagnosis"]:
        frame.to_parquet(teacher / f"{phase}_used.parquet", index=False)
    expected = score_frame(frame, p)
    old = pd.DataFrame(
        {
            "end": frame.end,
            **{
                a + "_histogram_calibrated": scores[:, i]
                for i, a in enumerate(["exit", "reduce", "increase"])
            },
        }
    )
    old.to_parquet(teacher / "predictions.parquet", index=False)
    for case in research.CASES:
        (previous / case).mkdir()
        trace = frame.drop(columns=["y_exit", "y_reduce", "y_increase"]).copy()
        trace["intent"], trace["event"] = expected.original_choice, expected.original_event
        trace.to_parquet(previous / case / "management_inputs.parquet", index=False)
    monkeypatch.setattr(
        research,
        "checked_source",
        lambda *_: (previous, {"selection": str(selection), "teacher": str(teacher)}),
    )
    monkeypatch.setattr(research, "load_selection", lambda *_: ({}, p))
    root = research.run_management_score_audit(
        tmp_path / "synthetic-reference.json", "synthetic", tmp_path / "outputs"
    )
    summary = read_json(root / "summary.json")
    assert summary["complete"] and summary["rows"] == 5 * len(frame)
    assert summary["previous_bot_intents_and_events_exact"] and not summary["new_account_replayed"]
    for name, digest in read_json(root / "files.json").items():
        assert sha256(root / name) == digest


def model():
    tree = {
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "feature": [0, -2, -2],
        "threshold": [0.5, 0.0, 0.0],
        "value": [0.0, -0.01, 0.01],
    }
    raw = OrderHistoryBoostModels.from_dict(
        {
            "format": OrderHistoryBoostModels.format,
            "features": FEATURES,
            "settings": HISTOGRAM_SETTINGS,
            "models": [
                {"action": a, "baseline": b, "trees": [tree] * 64}
                for a, b in zip(["exit", "reduce", "increase"], [-4.0, -2.0, -1.0], strict=True)
            ],
        }
    )
    offset = ManagementOffset.from_dict(
        {
            "format": ManagementOffset.format,
            "actions": raw.actions,
            "slope": 1.0,
            "epsilon": float(np.finfo(float).eps),
            "bounds": [-64.0, 64.0],
            "iterations": 100,
            "offsets": [0.1, -0.2, -0.1],
        }
    )
    return CalibratedHistoryModels(raw, offset)


def policy():
    return SimpleNamespace(
        manager=model(),
        thresholds={"exit": 0.2, "reduce": 0.1, "increase": 0.25},
        first_thresholds={"reduce": 0.1, "increase": 0.25},
        size_model=None,
    )


def source():
    x = values(300)
    x[:, 44] = np.arange(len(x)) % 2
    x[:, 47] = (np.arange(len(x)) // 2) % 2
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["end"] = pd.date_range("2021-07-02", periods=len(x), freq="min", tz="UTC")
    for action in ["exit", "reduce", "increase"]:
        frame[f"y_{action}"] = (np.arange(len(x)) % 5 == 0).astype(int)
    return frame


def test_leaf_reference_matches_scalar_forest_including_equality():
    p = policy()
    x = source()[FEATURES].to_numpy()
    for raw in p.manager.model.data["models"]:
        np.testing.assert_array_equal(forest_by_leaf(raw, x), 1 / (1 + np.exp(-raw_forest(raw, x))))


def test_complete_scores_metrics_and_parquet_roundtrip(tmp_path):
    frame, p = source(), policy()
    frame.loc[0, "ret_5m"] = np.nan
    before = copy.deepcopy(p.manager.model.data)
    scored = score_frame(frame, p)
    verify_scores(frame, scored, p)
    summary = summarize_scores(scored)
    verify_summary(scored, summary)
    assert summary["actions"]["exit"]["all"]["nonfinite_rows"] == 1
    assert all(summary["unchanged_masks"].values())
    path = tmp_path / "scores.parquet"
    scored.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), scored, check_exact=True)
    assert p.manager.model.data == before
    altered = frame.copy()
    altered.loc[100:, "y_reduce"] = 1
    columns = [k for k in scored if not k.startswith("y_")]
    pd.testing.assert_frame_equal(
        scored[columns], score_frame(altered, p)[columns], check_exact=True
    )


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0, None])
def test_actual_parent_intent_and_zero_size_event_are_reproduced(fraction):
    p = parent()
    p.manager = model()
    p.thresholds = {"exit": 1.0, "reduce": 0.01, "increase": 0.5}
    p.first_thresholds = {"reduce": 0.01, "increase": 0.5}
    p.size_model = (
        None
        if fraction is None
        else MinuteReductionModel.from_dict(
            {
                "format": MinuteReductionModel.format,
                "features": MinuteReductionModel.features,
                "alpha": 100.0,
                "mean": [0.0] * 44,
                "scale": [1.0] * 44,
                "coef": [0.0] * 44,
                "intercept": fraction,
            }
        )
    )
    records = []
    ManagementInputTrace(p, records.append)(*history_inputs(1))
    frame = pd.DataFrame(records)
    scored = score_frame(frame, p)
    verify_scores(frame, scored, p)
    assert scored.original_choice.iloc[0] == ("hold" if fraction == 0 else "reduce")
    assert scored.original_event.iloc[0] == records[0]["event"]
    damaged = frame.copy()
    damaged.loc[0, "intent"] = "exit"
    with pytest.raises(ValueError, match="실제 저장"):
        score_frame(damaged, p)


def test_repeat_only_boundary_priority_and_zero_size_wait():
    x = values(3)
    x[:, 44] = 1
    x[:, 47] = 1
    base = {"exit": 0.5, "reduce": 0.2, "increase": 0.2}
    first = {"reduce": 0.1, "increase": 0.1}
    scores = np.array([[0.5, 0.8, 0.8], [0.0, 0.2, 0.3], [0.0, 0.2, 0.0]])
    old = resolve_requests(scores, boundaries(x, base, first, 1.5), np.ones(3, bool), np.zeros(3))
    new = resolve_requests(scores, boundaries(x, base, first, 1.0), np.ones(3, bool), np.zeros(3))
    assert old[1].tolist() == ["exit", "hold", "hold"]
    assert new[1].tolist() == ["exit", "reduce", "reduce"]
    assert old[2].tolist() == new[2].tolist() == ["exit", "hold", "hold"]
    assert new[3][-1] == "action_zero_reduction"
    x[:, [44, 47]] = 0
    np.testing.assert_array_equal(boundaries(x, base, first, 1.5), boundaries(x, base, first, 1.0))


def test_changed_priority_can_suppress_unchanged_first_increase_mask():
    x = values(1)
    x[:, 47] = 1
    p = policy()
    scores = np.array([[0.0, 0.12, 0.3]])
    old = resolve_requests(
        scores, boundaries(x, p.thresholds, p.first_thresholds, 1.5), np.array([True]), np.ones(1)
    )
    new = resolve_requests(
        scores, boundaries(x, p.thresholds, p.first_thresholds, 1.0), np.array([True]), np.ones(1)
    )
    assert old[0][0, 2] == new[0][0, 2]
    assert old[2][0] == "increase" and new[2][0] == "reduce"


@pytest.mark.parametrize("damage", ["probability", "flag", "threshold", "fraction", "partial_nan"])
def test_invalid_probability_state_or_size_rejected(damage):
    x = values(1)
    p = policy()
    scores = np.array([[0.1, 0.2, 0.3]])
    fraction = np.ones(1)
    if damage == "probability":
        scores[0, 0] = 1.1
    if damage == "flag":
        x[0, 44] = 0.5
    if damage == "threshold":
        p.thresholds["reduce"] = np.nan
    if damage == "fraction":
        fraction[0] = -1
    if damage == "partial_nan":
        scores[0, 0] = np.nan
    with pytest.raises(ValueError):
        resolve_requests(
            scores, boundaries(x, p.thresholds, p.first_thresholds, 1.5), np.array([True]), fraction
        )


def test_empty_group_and_unchanged_behavior_do_not_pass_admission():
    frame = source()
    frame[["past_increase_exists", "past_reduce_exists"]] = 0.0
    result = summarize_scores(score_frame(frame, policy()))
    verify_summary(score_frame(frame, policy()), result)
    assert result["actions"]["reduce"]["repeat"]["calibrated"] is None
    decision = repeat_admission(result, {"bot": result})
    assert not decision["further_account_research_eligible"]
    assert not decision["checks"]["reduce"]["support"]


def test_reference_detects_tampered_saved_scores_and_metrics():
    frame, p = source(), policy()
    saved = score_frame(frame, p)
    broken = saved.copy()
    broken.loc[0, "exit_score"] += 0.01
    with pytest.raises(AssertionError):
        verify_scores(frame, broken, p)
    summary = summarize_scores(saved)
    summary["changed_choices"] += 1
    with pytest.raises(AssertionError):
        verify_summary(saved, summary)
