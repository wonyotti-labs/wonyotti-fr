from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.first_state_models import validate_first_rows
from wonyotti_fr.joint_history_models import evaluate_joint, joint_thresholds
from wonyotti_fr.joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from wonyotti_fr.management_capacity import (
    FEATURES,
    GRID,
    LABELS,
    capacity_admission,
    capacity_predictions,
    fit_inner_models,
    fit_outer_models,
    load_capacity_model,
    select_capacity,
    split_inner,
)
from wonyotti_fr.management_capacity_learning import checked_capacity_reference
from wonyotti_fr.management_capacity_reference import (
    verify_capacity_admission,
    verify_capacity_predictions,
    verify_inner_models,
    verify_inner_split,
    verify_outer_models,
    verify_selection,
)
from wonyotti_fr.management_market_model_reference import verify_offsets
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(29)
    rows = {}
    for i, (phase, dates, count) in enumerate([
        ("training", ["2020-01-03", "2020-07-03"], 2400),
        ("calibration", ["2021-01-03"], 1600), ("diagnosis", ["2021-07-03"], 1600),
    ]):
        parts = []
        for j, date in enumerate(dates):
            frame = pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES)
            frame["past_reduce_exists"] = np.arange(count) % 2
            frame["past_increase_exists"] = (np.arange(count)//2) % 2
            frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
            frame["entry_time"] = frame.end.iloc[0]-pd.Timedelta(hours=1)
            frame["label_end"] = frame.end+pd.Timedelta(minutes=1)
            frame["episode_id"] = i*10+j+1
            for label in LABELS:
                frame[label] = (rng.uniform(size=count) < .15 + .35*frame[FEATURES[0]].gt(.5)).astype(int)
            parts.append(frame)
        rows[phase] = pd.concat(parts, ignore_index=True)
    validate_first_rows(rows)
    inner, ledger, support = split_inner(rows["training"])
    models, predictions, fit_support = fit_inner_models(inner)
    selection = select_capacity(inner, predictions)
    outer, offsets, details = fit_outer_models(rows, selection["selected"])
    return rows, inner, ledger, support, models, predictions, selection, outer, offsets


def test_inner_selection_and_all_outer_predictions_independent(study):
    rows, inner, ledger, support, models, predictions, selection, outer, offsets = study
    assert verify_inner_split(rows["training"], inner, ledger, support)["source_rows"] == 4800
    assert verify_inner_models(inner, models, predictions) == 1920
    verify_selection(inner, predictions, selection)
    assert verify_outer_models(rows, outer, selection["selected"])["independent_outer_trees"] >= 384
    constant = rows["calibration"][LABELS].mean().to_numpy()
    calibration = capacity_predictions(rows["calibration"], outer, offsets, constant)
    verify_offsets(calibration, rows["calibration"][LABELS].to_numpy(), offsets)
    thresholds, details = joint_thresholds(rows["calibration"], calibration)
    verify_joint_thresholds(rows["calibration"], calibration, thresholds, details)
    for phase, frame in rows.items():
        scores = capacity_predictions(frame, outer, offsets, constant)
        verify_capacity_predictions(frame, outer, offsets, constant, scores)
        metrics, requests = evaluate_joint(frame, scores, thresholds)
        verify_joint_evaluation(frame, scores, thresholds, metrics, requests)
        if phase == "diagnosis":
            decision = capacity_admission(metrics)
            verify_capacity_admission(metrics, decision)
            assert len(decision["checks"]) == 25


@pytest.mark.parametrize("name", list(GRID))
def test_all_numeric_formats_batch_single_and_legacy_default(study, name):
    model = study[4][name]
    raw = model.to_dict()
    loaded = load_capacity_model(raw)
    assert loaded.to_dict() == raw
    values = study[1]["selection"][FEATURES].iloc[:17].to_numpy()
    np.testing.assert_array_equal(model.probabilities(values), np.vstack([loaded.probabilities(x[None, :]) for x in values]))
    if name == "depth2_iter64":
        raw["format"] = OrderHistoryBoostModels.format
        old = OrderHistoryBoostModels.from_dict(raw)
        np.testing.assert_array_equal(model.probabilities(values), old.probabilities(values))


def test_inner_selection_labels_do_not_change_learned_candidates(study):
    inner = {k: v.copy() for k, v in study[1].items()}
    inner["selection"][LABELS] = 1-inner["selection"][LABELS]
    changed, predictions, _ = fit_inner_models(inner)
    assert {k: v.to_dict() for k, v in changed.items()} == {k: v.to_dict() for k, v in study[4].items()}
    for name in GRID:
        for phase in inner:
            np.testing.assert_array_equal(predictions[name][phase], study[5][name][phase])


def test_2021_labels_cannot_change_inner_choice_or_final_training(study):
    rows = {k: v.copy() for k, v in study[0].items()}
    for phase in ["calibration", "diagnosis"]:
        rows[phase][LABELS] = 1-rows[phase][LABELS]
    inner, _, _ = split_inner(rows["training"])
    candidates, predictions, _ = fit_inner_models(inner)
    selection = select_capacity(inner, predictions)
    models, offsets, _ = fit_outer_models(rows, selection["selected"])
    assert selection == study[6]
    assert {k: v.to_dict() for k, v in candidates.items()} == {k: v.to_dict() for k, v in study[4].items()}
    assert {k: v.to_dict() for k, v in models.items()} == {k: v.to_dict() for k, v in study[7].items()}
    assert offsets["capacity"].to_dict() != study[8]["capacity"].to_dict()


def test_exact_tie_selects_declared_baseline(study):
    inner = study[1]
    p = inner["fit"][LABELS].mean().to_numpy()
    scores = {name: {phase: np.tile(p, (len(frame), 1)) for phase, frame in inner.items()} for name in GRID}
    selected = select_capacity(inner, scores)
    verify_selection(inner, scores, selected)
    assert selected["selected"] == "depth2_iter64"


def test_boundary_crossing_episode_removed_on_both_sides(study):
    frame = study[0]["training"].copy()
    frame.loc[[2399, 2400], "episode_id"] = 99
    frame.loc[[2399, 2400], "entry_time"] = pd.Timestamp("2020-01-02", tz="UTC")
    inner, ledger, support = split_inner(frame)
    verify_inner_split(frame, inner, ledger, support)
    assert support["crossing_episodes"] == 1 and support["excluded_rows"] == 2
    assert ledger.loc[[2399, 2400], "reason"].eq("crossing_episode").all()
    assert all(99 not in set(part.episode_id) for part in inner.values())


def test_strict_label_embargo_and_inclusive_start(study):
    frame = study[0]["training"].copy()
    frame.loc[2398, "end"] = pd.Timestamp("2020-06-29T23:58Z")
    frame.loc[2398, "label_end"] = pd.Timestamp("2020-06-29T23:59Z")
    frame.loc[2399, "end"] = pd.Timestamp("2020-06-29T23:59Z")
    frame.loc[2399, "label_end"] = pd.Timestamp("2020-06-30T00:00Z")
    frame.loc[2400:2401, "episode_id"] = [98, 99]
    frame.loc[2400:2401, "entry_time"] = pd.Timestamp("2020-07-01T00:00Z")
    frame.loc[2400:2401, "end"] = pd.to_datetime(["2020-07-01T23:59Z", "2020-07-02T00:00Z"])
    frame.loc[2400:2401, "label_end"] = frame.loc[2400:2401, "end"]+pd.Timedelta(minutes=1)
    inner, ledger, support = split_inner(frame)
    verify_inner_split(frame, inner, ledger, support)
    assert ledger.loc[[2398, 2399, 2400, 2401], "reason"].tolist() == ["included", "right_embargo", "left_embargo", "included"]


@pytest.mark.parametrize("change", ["future", "nonfinite", "label", "timezone", "order", "entry", "support"])
def test_invalid_training_rejected(study, change):
    frame = study[0]["training"].copy()
    if change == "future":
        frame.loc[len(frame)-1, "label_end"] = pd.Timestamp("2021-01-01", tz="UTC")
    elif change == "nonfinite":
        frame.loc[0, FEATURES[0]] = np.inf
    elif change == "label":
        frame.loc[0, LABELS[0]] = 2
    elif change == "timezone":
        frame["end"] = frame.end.dt.tz_localize(None)
    elif change == "order":
        frame = frame.iloc[::-1]
    elif change == "entry":
        frame.loc[0, "entry_time"] = frame.end.iloc[0]+pd.Timedelta(minutes=1)
    else:
        frame.loc[frame.end.lt(pd.Timestamp("2020-07-01", tz="UTC")), LABELS[0]] = 0
    with pytest.raises(ValueError):
        split_inner(frame)


@pytest.mark.parametrize("change", ["format", "setting", "count", "cycle", "index", "finite", "depth"])
def test_invalid_numeric_model_rejected(study, change):
    raw = study[4]["depth4_iter256"].to_dict()
    tree = raw["models"][0]["trees"][0]
    if change == "format":
        raw["format"] = "unknown"
    elif change == "setting":
        raw["settings"]["max_iter"] = 257
    elif change == "count":
        raw["models"][0]["trees"].pop()
    elif change == "cycle":
        tree["left"][0] = 0
    elif change == "index":
        tree["feature"][0] = len(FEATURES)
    elif change == "depth":
        raw["models"][0]["trees"][0] = {
            "left": [1, 2, 3, 4, 5]+[-1]*6, "right": [6, 7, 8, 9, 10]+[-1]*6,
            "feature": [0]*5+[-2]*6, "threshold": [0.]*11, "value": [0.]*11}
    else:
        tree["value"][0] = np.inf
    with pytest.raises(ValueError):
        load_capacity_model(raw)


@pytest.mark.parametrize("change", ["membership", "support", "score", "choice", "prediction"])
def test_independent_evidence_detects_mutation(study, change):
    rows, inner, ledger, support, models, predictions, selection, _, _ = study
    with pytest.raises((ValueError, AssertionError)):
        if change == "membership":
            ledger = ledger.copy()
            ledger.loc[0, "source_row"] += 1
            verify_inner_split(rows["training"], inner, ledger, support)
        elif change == "support":
            support = deepcopy(support)
            support["fit"]["positive"]["exit"] += 1
            verify_inner_split(rows["training"], inner, ledger, support)
        elif change == "prediction":
            predictions = deepcopy(predictions)
            predictions["depth2_iter64"]["selection"][0, 0] += .01
            verify_selection(inner, predictions, selection)
        else:
            selection = deepcopy(selection)
            if change == "score":
                selection["scores"]["depth2_iter64"] += .01
            else:
                selection["selected"] = next(k for k in GRID if k != selection["selected"])
            verify_selection(inner, predictions, selection)


def test_wrong_proof_digest_rejected(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_capacity_reference(path, "0"*64)
