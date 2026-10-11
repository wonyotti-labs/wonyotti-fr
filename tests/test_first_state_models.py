from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.action_model import select_threshold
from wonyotti_fr.first_state_learning import checked_first_reference, verify_support
from wonyotti_fr.first_state_models import (
    CLASSES,
    FEATURES,
    calibrated_binary,
    evaluate_frame,
    first_mask,
    first_state_admission,
    fit_binary_offset,
    fit_first_models,
    linear_probabilities,
    prediction_frame,
    validate_first_rows,
)
from wonyotti_fr.first_state_reference import (
    refit_first_models,
    verify_first_admission,
    verify_first_calibration,
    verify_first_evaluation,
    verify_first_frame,
)
from wonyotti_fr.management_calibration import ManagementOffset
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(19)
    rows = {}
    for index, (phase, date, count) in enumerate([
        ("training", "2020-01-03", 3000), ("calibration", "2021-01-03", 1200),
        ("diagnosis", "2021-07-03", 1200),
    ]):
        frame = pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES)
        frame["past_reduce_exists"] = np.arange(count) % 2
        frame["past_increase_exists"] = (np.arange(count) // 2) % 2
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["entry_time"] = frame.end.iloc[0] - pd.Timedelta(hours=1)
        frame["episode_id"] = index + 1
        for action in ["exit", "reduce", "increase"]:
            frame["y_" + action] = (rng.uniform(size=count) < .18 + .25 * frame[FEATURES[0]].gt(.5)).astype(int)
        rows[phase] = frame
    fitted = fit_first_models(rows)
    model, _ = OrderHistoryBoostModels.fit(rows["training"][FEATURES],
        rows["training"][["y_exit", "y_reduce", "y_increase"]], rows["calibration"][FEATURES])
    raw = model.probabilities(rows["calibration"][FEATURES])
    offset, _ = ManagementOffset.fit(raw, rows["calibration"][["y_exit", "y_reduce", "y_increase"]])
    manager = SimpleNamespace(probabilities=lambda x: offset.predict(model.probabilities(x)))
    scores = manager.probabilities(rows["calibration"][FEATURES])
    thresholds, first = {}, {}
    for j, (action, beta) in enumerate([("exit", 2.), ("reduce", 1.), ("increase", .5)]):
        y = rows["calibration"]["y_" + action].to_numpy()
        thresholds[action], _ = select_threshold(y, scores[:, j], beta)
        if action != "exit":
            mask = first_mask(rows["calibration"], action)
            first[action], _ = select_threshold(y[mask], scores[mask, j], beta)
    policy = SimpleNamespace(manager=manager, thresholds=thresholds, first_thresholds=first)
    constants = {a: float(rows["calibration"].loc[first_mask(rows["calibration"], a), "y_" + a].mean())
                 for a in ["reduce", "increase"]}
    return rows, fitted, policy, constants


def test_first_models_independent_fit_and_all_predictions(study):
    rows, (models, offsets, thresholds, details, support), policy, constants = study
    verify_support(rows, support)
    assert refit_first_models(rows, models)["independent_histogram_trees"] == 128
    for phase, frame in rows.items():
        scores = prediction_frame(frame, policy, models, offsets, thresholds, constants)
        verify_first_frame(frame, scores, policy, models, offsets, thresholds, constants)
        metrics = evaluate_frame(scores)
        verify_first_evaluation(scores, metrics)
        if phase == "calibration":
            verify_first_calibration(scores, offsets, thresholds, details, policy)
        if phase == "diagnosis":
            unchanged = {"exit": True, "repeat_reduce": True, "repeat_increase": True}
            decision = first_state_admission(metrics, unchanged)
            assert len(decision["checks"]) == 21
            verify_first_admission(metrics, unchanged, decision)


def test_future_labels_do_not_change_fit_calibration_or_thresholds(study):
    rows, fitted, _, _ = study
    changed = {k: v.copy() for k, v in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    new = fit_first_models(changed)
    assert new[:4] == fitted[:4]


@pytest.mark.parametrize("change", ["flag", "label_support", "time_overlap", "episode_overlap", "nonfinite"])
def test_invalid_first_rows_rejected(study, change):
    rows = {k: v.copy() for k, v in study[0].items()}
    if change == "flag":
        rows["training"].loc[0, "past_reduce_exists"] = 2
    elif change == "label_support":
        rows["training"].loc[first_mask(rows["training"], "reduce"), "y_reduce"] = 0
    elif change == "time_overlap":
        rows["training"]["label_end"] = pd.Timestamp("2022-01-01", tz="UTC")
    elif change == "episode_overlap":
        rows["diagnosis"]["episode_id"] = 1
    else:
        rows["training"].loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        validate_first_rows(rows)


@pytest.mark.parametrize("change", ["settings", "scale", "coefficient", "feature"])
def test_damaged_linear_model_rejected(study, change):
    raw = deepcopy(study[1][0]["reduce"]["linear"])
    if change == "settings":
        raw["settings"]["C"] = 1.
    elif change == "scale":
        raw["scale"][0] = 0.
    elif change == "coefficient":
        raw["coef"][0] = np.nan
    else:
        raw["features"] = list(reversed(FEATURES))
    with pytest.raises(ValueError):
        linear_probabilities(raw, study[0]["diagnosis"][FEATURES].to_numpy())


def test_tree_action_and_graph_corruption_rejected(study):
    raw = deepcopy(study[1][0]["reduce"]["tree"])
    with pytest.raises(ValueError):
        CLASSES["increase"].from_dict(raw)
    raw["models"][0]["trees"][0]["left"][0] = 0
    with pytest.raises(ValueError):
        CLASSES["reduce"].from_dict(raw)


@pytest.mark.parametrize("change", ["repeat_request", "first_request", "probability", "exit_request", "label"])
def test_independent_verifier_rejects_changed_predictions(study, change):
    rows, (models, offsets, thresholds, _, _), policy, constants = study
    frame = rows["diagnosis"]
    scores = prediction_frame(frame, policy, models, offsets, thresholds, constants)
    if change == "repeat_request":
        scores.loc[1, "reduce_tree_request"] = not scores.loc[1, "reduce_tree_request"]
    elif change == "first_request":
        scores.loc[0, "reduce_tree_request"] = not scores.loc[0, "reduce_tree_request"]
    elif change == "probability":
        scores.loc[0, "reduce_tree_raw_score"] += .01
    elif change == "exit_request":
        scores.loc[0, "exit_request"] = not scores.loc[0, "exit_request"]
    else:
        scores.loc[0, "y_reduce"] = 1 - scores.loc[0, "y_reduce"]
    with pytest.raises(AssertionError):
        verify_first_frame(frame, scores, policy, models, offsets, thresholds, constants)


def test_zero_positive_and_absent_group_retained(study):
    rows, (models, offsets, thresholds, _, _), policy, constants = study
    frame = rows["diagnosis"]
    saved = prediction_frame(frame, policy, models, offsets, thresholds, constants)
    saved = saved.loc[saved.past_reduce_exists.eq(0)].copy()
    saved["y_reduce"] = 0
    result = evaluate_frame(saved)
    verify_first_evaluation(saved, result)
    assert result["reduce"]["first"]["tree"]["average_precision"] is None
    assert result["reduce"]["repeat"]["tree"] is None


def test_scalar_offset_bounds_and_support():
    p = np.r_[np.zeros(50), np.ones(50), np.full(100, .2)]
    y = np.arange(200) % 2
    offset = fit_binary_offset(p, y)
    assert abs(calibrated_binary(p, offset).mean() - .5) < 1e-12
    for scores, labels in [(p * 2, y), (p, y[:5]), (p, y * 0)]:
        with pytest.raises(ValueError):
            fit_binary_offset(scores, labels)


def test_admission_requires_both_actions_and_preservation(study):
    rows, (models, offsets, thresholds, _, _), policy, constants = study
    metrics = evaluate_frame(prediction_frame(rows["diagnosis"], policy, models, offsets, thresholds, constants))
    unchanged = {"exit": False, "repeat_reduce": True, "repeat_increase": True}
    decision = first_state_admission(metrics, unchanged)
    assert not decision["first_state_models_admitted"]
    verify_first_admission(metrics, unchanged, decision)
    decision["checks"]["exit"] = True
    with pytest.raises(AssertionError):
        verify_first_admission(metrics, unchanged, decision)


def test_wrong_reference_digest_rejected(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_first_reference(path, "0" * 64)
