from copy import deepcopy

import numpy as np
import pytest
from test_first_state_models import study as study

from wonyotti_fr.joint_history_learning import checked_joint_reference
from wonyotti_fr.joint_history_models import (
    EXTRA_FEATURES,
    JointHistoryBoostModels,
    JointHistoryLinearModels,
    attach_joint_history,
    evaluate_joint,
    fit_joint_models,
    joint_admission,
    joint_codes,
    joint_feature_usage,
    joint_predictions,
    joint_requests,
    joint_thresholds,
    verify_joint_inputs,
)
from wonyotti_fr.joint_history_reference import (
    refit_joint_models,
    verify_joint_admission,
    verify_joint_evaluation,
    verify_joint_predictions,
    verify_joint_thresholds,
    verify_joint_usage,
)
from wonyotti_fr.management_market_model_reference import verify_offsets


@pytest.fixture(scope="module")
def joint_study(study):
    rows = study[0]
    expanded = attach_joint_history(rows)
    models, offsets, support = fit_joint_models(rows, expanded)
    y = rows["calibration"][["y_exit", "y_reduce", "y_increase"]].to_numpy(int)
    constant = y.mean(axis=0)
    scores = joint_predictions(rows["calibration"], expanded["calibration"], models, offsets, constant)
    thresholds, details = joint_thresholds(rows["calibration"], scores)
    return rows, expanded, models, offsets, constant, thresholds, details, support


def test_all_joint_inputs_models_predictions_and_groups(joint_study, study):
    rows, expanded, models, offsets, constant, thresholds, details, _ = joint_study
    assert verify_joint_inputs(rows, expanded) == 4 * sum(len(v) for v in rows.values())
    assert refit_joint_models(rows, expanded, models)["independent_histogram_trees"] == 384
    usage = joint_feature_usage(models["joint"])
    verify_joint_usage(models["joint"], usage)
    assert thresholds["first"]["native_calibrated"] == study[2].first_thresholds
    assert thresholds["full"]["native_calibrated"] == study[2].thresholds
    for phase, frame in rows.items():
        scores = joint_predictions(frame, expanded[phase], models, offsets, constant)
        verify_joint_predictions(frame, expanded[phase], models, offsets, constant, scores)
        if phase == "calibration":
            y = frame[["y_exit", "y_reduce", "y_increase"]].to_numpy(int)
            verify_offsets(scores, y, offsets)
            verify_joint_thresholds(frame, scores, thresholds, details)
        result, requests = evaluate_joint(frame, scores, thresholds)
        verify_joint_evaluation(frame, scores, thresholds, result, requests)
        if phase == "diagnosis":
            decision = joint_admission(result)
            assert len(decision["checks"]) == 25
            verify_joint_admission(result, decision)


def test_future_diagnosis_labels_preserve_fits_and_thresholds(joint_study):
    rows, _, models, offsets, constant, thresholds, details, _ = joint_study
    changed = {k: v.copy() for k, v in rows.items()}
    for a in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + a] = 1 - changed["diagnosis"]["y_" + a]
    added = attach_joint_history(changed)
    new, adjusted, _ = fit_joint_models(changed, added)
    assert {k: v.to_dict() for k, v in new.items()} == {k: v.to_dict() for k, v in models.items()}
    assert {k: v.to_dict() for k, v in adjusted.items()} == {k: v.to_dict() for k, v in offsets.items()}
    predictions = joint_predictions(changed["calibration"], added["calibration"], new, adjusted, constant)
    assert joint_thresholds(changed["calibration"], predictions) == (thresholds, details)


@pytest.mark.parametrize("change", ["native_flag", "two_hot", "wrong_state", "label", "row_order", "feature"])
def test_changed_joint_inputs_rejected(joint_study, change):
    rows = {k: v.copy() for k, v in joint_study[0].items()}
    added = {k: v.copy() for k, v in joint_study[1].items()}
    if change == "native_flag":
        rows["training"].loc[0, "past_reduce_exists"] = np.nan
    elif change == "two_hot":
        added["training"].loc[0, EXTRA_FEATURES] = 1.
    elif change == "wrong_state":
        added["training"].loc[0, EXTRA_FEATURES] = [0., 0., 0., 1.]
    elif change == "label":
        added["training"].loc[0, "y_exit"] = 1 - added["training"].loc[0, "y_exit"]
    elif change == "row_order":
        added["training"] = added["training"].iloc[::-1]
    else:
        added["training"].iloc[0, 0] += .1
    with pytest.raises((ValueError, AssertionError)):
        verify_joint_inputs(rows, added)


def test_joint_flags_are_ordered_increase_then_reduce(joint_study):
    frame = joint_study[0]["training"].iloc[:4].copy()
    frame["past_increase_exists"] = [0, 0, 1, 1]
    frame["past_reduce_exists"] = [0, 1, 0, 1]
    np.testing.assert_array_equal(joint_codes(frame), [0, 1, 2, 3])
    requests = joint_requests(frame, np.full((4, 3), .5), {"exit": .6, "reduce": .7, "increase": .8},
                              {"reduce": .3, "increase": .4})
    np.testing.assert_array_equal(requests, [[False, True, True], [False, False, True],
                                           [False, True, False], [False, False, False]])


@pytest.mark.parametrize("change", ["prediction", "request", "metrics", "threshold"])
def test_independent_joint_verifiers_detect_changes(joint_study, change):
    rows, expanded, models, offsets, constant, thresholds, details, _ = joint_study
    frame = rows["calibration"]
    scores = joint_predictions(frame, expanded["calibration"], models, offsets, constant)
    result, requests = evaluate_joint(frame, scores, thresholds)
    with pytest.raises((ValueError, AssertionError)):
        if change == "prediction":
            scores["joint_raw"][0, 0] += .01
            verify_joint_predictions(frame, expanded["calibration"], models, offsets, constant, scores)
        elif change == "request":
            requests["joint_calibrated"][0, 1] = not requests["joint_calibrated"][0, 1]
            verify_joint_evaluation(frame, scores, thresholds, result, requests)
        elif change == "metrics":
            result["exit"]["joint_00"]["joint_calibrated"]["true_positive"] += 1
            verify_joint_evaluation(frame, scores, thresholds, result, requests)
        else:
            changed = deepcopy(thresholds)
            changed["first"]["joint_calibrated"]["reduce"] += .001
            verify_joint_thresholds(frame, scores, changed, details)


def test_zero_positive_and_empty_joint_groups_retained(joint_study):
    rows, expanded, models, offsets, constant, thresholds, _, _ = joint_study
    frame = rows["diagnosis"]
    predictions = joint_predictions(frame, expanded["diagnosis"], models, offsets, constant)
    mask = joint_codes(frame) == 0
    frame = frame.loc[mask].copy()
    frame["y_exit"] = 0
    scores = {name: values[mask] for name, values in predictions.items()}
    metrics, requests = evaluate_joint(frame, scores, thresholds)
    verify_joint_evaluation(frame, scores, thresholds, metrics, requests)
    assert metrics["exit"]["joint_00"]["joint_calibrated"]["average_precision"] is None
    assert metrics["exit"]["joint_11"]["joint_calibrated"] is None


def test_bad_model_and_admission_rejected(joint_study):
    rows, expanded, models, offsets, constant, thresholds, _, _ = joint_study
    raw = models["joint"].to_dict()
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        JointHistoryBoostModels.from_dict(raw)
    raw = models["linear"].to_dict()
    raw["scale"][0] = 0.
    with pytest.raises(ValueError):
        JointHistoryLinearModels.from_dict(raw)
    scores = joint_predictions(rows["diagnosis"], expanded["diagnosis"], models, offsets, constant)
    metrics, _ = evaluate_joint(rows["diagnosis"], scores, thresholds)
    decision = joint_admission(metrics)
    decision["checks"]["exit_all_f_preserved"] = not decision["checks"]["exit_all_f_preserved"]
    with pytest.raises(AssertionError):
        verify_joint_admission(metrics, decision)


def test_wrong_proof_digest_rejected(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_joint_reference(path, "0" * 64)
