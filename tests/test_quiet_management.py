from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.quiet_management import (
    CLASSES,
    FEATURES,
    evaluate_quiet,
    fit_quiet_models,
    groups,
    quiet_admission,
    quiet_mask,
    quiet_predictions,
    validate_quiet_rows,
)
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_admission,
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_refits,
    verify_quiet_support,
)


@pytest.fixture(scope="module")
def study():
    rng, rows, originals = np.random.default_rng(126), {}, {}
    for index, (phase, date, count) in enumerate([
        ("training", "2020-01-03", 6000), ("calibration", "2021-01-03", 1800),
        ("diagnosis", "2021-07-03", 1800),
    ]):
        frame = pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES)
        for shift, action in enumerate(["reduce", "increase"]):
            state = (np.arange(count) + shift) % 3
            frame[f"past_{action}_exists"] = (state != 0).astype(float)
            frame[f"past_{action}_log_count_15m"] = np.where(state == 2, np.log1p(2), 0.)
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["entry_time"] = frame.end.iloc[0] - pd.Timedelta(hours=1)
        frame["episode_id"] = index + 1
        original = pd.DataFrame(index=frame.index)
        for action in ["exit", "reduce", "increase"]:
            frame["y_" + action] = (rng.uniform(size=count) < .18 + .25 * frame[FEATURES[0]].gt(.5)).astype(int)
            for name, scale in [("native", .1), ("capacity", .2)]:
                p = 1 / (1 + np.exp(-scale * frame[FEATURES[1]].to_numpy()))
                original[action + "_" + name + "_calibrated"] = p
                original[action + "_" + name + "_calibrated_request"] = p >= .52
        rows[phase], originals[phase] = frame, original
    fitted = fit_quiet_models(rows["training"], rows["calibration"])
    return rows, originals, fitted


def test_quiet_contains_first_and_restart(study):
    rows, _, _ = study
    support = validate_quiet_rows(rows)
    verify_quiet_support(rows, support)
    for action in ["reduce", "increase"]:
        masks = groups(rows["training"], action)
        assert masks["first"].sum() == masks["restart"].sum() == masks["recent"].sum() == 2000
        assert masks["quiet"].sum() == 4000


def test_all_refits_predictions_calibration_and_metrics(study):
    rows, originals, (models, offsets, thresholds, details, constants) = study
    assert verify_quiet_refits(rows, models)["independent_histogram_trees"] == 512
    for phase, frame in rows.items():
        saved = quiet_predictions(frame, originals[phase], models, offsets, thresholds, constants)
        verify_quiet_predictions(frame, originals[phase], saved, models, offsets, thresholds, constants)
        evaluated = evaluate_quiet(saved)
        verify_quiet_evaluation(saved, evaluated)
        if phase == "calibration":
            verify_quiet_calibration(saved, offsets, thresholds, details, constants)
        if phase == "diagnosis":
            unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
            decision = quiet_admission(evaluated, unchanged)
            assert len(decision["checks"]) == 27
            verify_quiet_admission(evaluated, unchanged, decision)
            broken = deepcopy(decision)
            broken["checks"]["exit"] = False
            with pytest.raises(AssertionError):
                verify_quiet_admission(evaluated, unchanged, broken)


def test_future_labels_do_not_enter_training_arguments(study):
    rows, _, fitted = study
    changed = {p: f.copy() for p, f in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_quiet_models(changed["training"], changed["calibration"]) == fitted


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, np.log1p(.5), 1e-10, 99.])
def test_invalid_log_counts_rejected(study, bad):
    rows, _, _ = study
    frame = rows["training"].copy()
    frame.loc[1, "past_reduce_log_count_15m"] = bad
    with pytest.raises(ValueError):
        quiet_mask(frame, "reduce")


@pytest.mark.parametrize("change", ["exists", "first_count", "recent_support", "overlap", "episode", "nonfinite"])
def test_invalid_support_rejected(study, change):
    rows, _, _ = study
    changed = {p: f.copy() for p, f in rows.items()}
    if change == "exists":
        changed["training"]["past_reduce_exists"] = 2
    elif change == "first_count":
        changed["training"].loc[0, "past_reduce_log_count_15m"] = np.log1p(1)
    elif change == "recent_support":
        changed["training"]["past_reduce_log_count_15m"] = 0.
    elif change == "overlap":
        changed["diagnosis"]["end"] = changed["calibration"].end
    elif change == "episode":
        changed["diagnosis"]["episode_id"] = 1
    else:
        changed["training"][FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        validate_quiet_rows(changed)


@pytest.mark.parametrize("column", ["exit_score", "exit_request", "reduce_tree_score", "increase_tree_request",
                                    "reduce_capacity_score", "increase_baseline_request"])
def test_corrupt_predictions_rejected(study, column):
    rows, originals, (models, offsets, thresholds, _, constants) = study
    frame, original = rows["diagnosis"], originals["diagnosis"]
    saved = quiet_predictions(frame, original, models, offsets, thresholds, constants)
    if column.endswith("request"):
        saved.loc[0, column] = not saved.loc[0, column]
    else:
        saved.loc[0, column] += .01
    with pytest.raises(AssertionError):
        verify_quiet_predictions(frame, original, saved, models, offsets, thresholds, constants)


@pytest.mark.parametrize("action", ["reduce", "increase"])
def test_wrong_tree_configuration_rejected(study, action):
    _, _, (models, _, _, _, _) = study
    raw = deepcopy(models[action]["tree"])
    raw["settings"]["max_iter"] = 64
    with pytest.raises(ValueError):
        CLASSES[action].from_dict(raw)


def test_empty_and_zero_positive_subgroups_preserved(study):
    rows, originals, (models, offsets, thresholds, _, constants) = study
    source = rows["diagnosis"]
    saved = quiet_predictions(source, originals["diagnosis"], models, offsets, thresholds, constants)
    part = saved.loc[groups(source, "reduce")["first"]].copy()
    part["y_reduce"] = 0
    metrics = evaluate_quiet(part)
    verify_quiet_evaluation(part, metrics)
    assert metrics["reduce"]["restart"]["tree"] is None
    assert metrics["reduce"]["quiet"]["tree"]["average_precision"] is None
