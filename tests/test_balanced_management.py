from copy import deepcopy

import numpy as np
import pytest
from test_management_capacity import study as study

from wonyotti_fr.balanced_management import (
    BalancedManagementModels,
    balanced_admission,
    balanced_weights,
    fit_balanced_management,
)
from wonyotti_fr.balanced_management_learning import (
    checked_balanced_reference,
    load_reference_models,
    verify_reference_metrics,
    verify_reference_thresholds,
)
from wonyotti_fr.balanced_management_reference import (
    verify_balanced_admission,
    verify_balanced_predictions,
    verify_balanced_refit,
    verify_weights,
)
from wonyotti_fr.common import save_json
from wonyotti_fr.joint_history_models import evaluate_joint, joint_thresholds
from wonyotti_fr.joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from wonyotti_fr.management_calibration import ManagementOffset
from wonyotti_fr.management_capacity import FEATURES, LABELS, Depth2Iter256, capacity_predictions
from wonyotti_fr.management_market_model_reference import verify_offsets


@pytest.fixture(scope="module")
def balanced(study):
    rows = study[0]
    result = fit_balanced_management(rows)
    models, offsets = dict(study[7]), dict(study[8])
    models["capacity"], _ = Depth2Iter256.fit(rows["training"][FEATURES], rows["training"][LABELS], rows["calibration"][FEATURES])
    offsets["capacity"], _ = ManagementOffset.fit(models["capacity"].probabilities(rows["calibration"][FEATURES]), rows["calibration"][LABELS])
    return rows, result, models, offsets


def test_all_weights_models_and_requests_verified(balanced):
    rows, (model, offset, weights, support), references, reference_offsets = balanced
    assert verify_balanced_refit(rows, model, weights, support)["independent_weighted_trees"] == 768
    models, offsets = {**references, "balanced": model}, {**reference_offsets, "balanced": offset}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    calibration = capacity_predictions(rows["calibration"], models, offsets, constant)
    verify_offsets(calibration, rows["calibration"][LABELS].to_numpy(int), offsets)
    thresholds, details = joint_thresholds(rows["calibration"], calibration)
    verify_joint_thresholds(rows["calibration"], calibration, thresholds, details)
    for phase, frame in rows.items():
        scores = capacity_predictions(frame, models, offsets, constant)
        verify_balanced_predictions(frame, model, offset, scores)
        metrics, requests = evaluate_joint(frame, scores, thresholds)
        verify_joint_evaluation(frame, scores, thresholds, metrics, requests)
        if phase == "diagnosis":
            decision = balanced_admission(metrics)
            verify_balanced_admission(metrics, decision)
            assert len(decision["checks"]) == 25


def test_numeric_bundle_preserves_batch_single_and_copy(balanced):
    rows, (model, _, _, _), _, _ = balanced
    raw = model.to_dict()
    loaded = BalancedManagementModels.from_dict(raw)
    x = rows["diagnosis"][FEATURES].iloc[:11].to_numpy(float)
    np.testing.assert_array_equal(loaded.probabilities(x), np.vstack([model.probabilities(v[None, :]) for v in x]))
    raw["models"]["exit"]["models"][0]["baseline"] += 1
    assert loaded.to_dict() != raw and loaded.to_dict() == model.to_dict()


def test_2021_labels_preserve_weights_and_models(balanced):
    rows, (model, offset, weights, _), _, _ = balanced
    changed = {k: v.copy() for k, v in rows.items()}
    for phase in ["calibration", "diagnosis"]:
        changed[phase][LABELS] = 1-changed[phase][LABELS]
    other, adjusted, new_weights, _ = fit_balanced_management(changed)
    assert other.to_dict() == model.to_dict() and adjusted.to_dict() != offset.to_dict()
    for action in weights:
        np.testing.assert_array_equal(weights[action], new_weights[action])


def test_diagnosis_labels_preserve_offset_and_thresholds(balanced):
    rows, (model, offset, weights, _), references, reference_offsets = balanced
    changed = {k: v.copy() for k, v in rows.items()}
    changed["diagnosis"][LABELS] = 1-changed["diagnosis"][LABELS]
    new, adjusted, new_weights, _ = fit_balanced_management(changed)
    assert new.to_dict() == model.to_dict() and adjusted.to_dict() == offset.to_dict()
    for a in weights:
        np.testing.assert_array_equal(weights[a], new_weights[a])
    constant = rows["calibration"][LABELS].mean().to_numpy()
    a = capacity_predictions(rows["calibration"], {**references,"balanced":model}, {**reference_offsets,"balanced":offset}, constant)
    b = capacity_predictions(changed["calibration"], {**references,"balanced":new}, {**reference_offsets,"balanced":adjusted}, constant)
    assert joint_thresholds(rows["calibration"], a) == joint_thresholds(changed["calibration"], b)


def test_original_models_and_all_metrics_preserved(balanced, tmp_path):
    rows, (model, offset, _, _), refs, offsets = balanced
    save_json(tmp_path / "models.json", {k: v.to_dict() for k, v in refs.items()})
    save_json(tmp_path / "offsets.json", {k: v.to_dict() for k, v in offsets.items()})
    loaded, adjusted = load_reference_models(tmp_path)
    assert {k: v.to_dict() for k, v in loaded.items()} == {k: v.to_dict() for k, v in refs.items()}
    assert {k: v.to_dict() for k, v in adjusted.items()} == {k: v.to_dict() for k, v in offsets.items()}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    base = capacity_predictions(rows["calibration"], refs, offsets, constant)
    original_thresholds, original_details = joint_thresholds(rows["calibration"], base)
    save_json(tmp_path / "thresholds.json", original_thresholds)
    save_json(tmp_path / "threshold_selection.json", original_details)
    models, offsets = {**refs, "balanced": model}, {**offsets, "balanced": offset}
    values = capacity_predictions(rows["calibration"], models, offsets, constant)
    thresholds, details = joint_thresholds(rows["calibration"], values)
    verify_reference_thresholds(tmp_path, thresholds, details)
    current, previous, months, old_months = {}, {}, {}, {}
    for phase, frame in rows.items():
        scores = capacity_predictions(frame, models, offsets, constant)
        current[phase], _ = evaluate_joint(frame, scores, thresholds)
        previous[phase], _ = evaluate_joint(frame, {k:v for k,v in scores.items() if not k.startswith("balanced_")}, original_thresholds)
        months[phase], old_months[phase] = {}, {}
        for month, indices in frame.groupby(frame.end.dt.strftime("%Y-%m")).indices.items():
            part = frame.iloc[indices]
            ps = {k:v[indices] for k,v in scores.items()}
            months[phase][month], _ = evaluate_joint(part, ps, thresholds)
            old_months[phase][month], _ = evaluate_joint(part, {k:v for k,v in ps.items() if not k.startswith("balanced_")}, original_thresholds)
    save_json(tmp_path / "metrics.json", previous)
    save_json(tmp_path / "monthly.json", old_months)
    verify_reference_metrics(tmp_path, current, months)
    changed = deepcopy(thresholds)
    changed["first"]["capacity_calibrated"]["reduce"] += .01
    with pytest.raises(AssertionError):
        verify_reference_thresholds(tmp_path, changed, details)
    months["diagnosis"]["2021-07"]["exit"]["all"]["native_raw"]["rows"] += 1
    with pytest.raises((AssertionError, ValueError)):
        verify_reference_metrics(tmp_path, current, months)


@pytest.mark.parametrize("change", ["unsupported", "nonbinary", "matrix", "effective", "nan"])
def test_invalid_labels_or_support_rejected(change):
    y = np.r_[np.ones(400), np.zeros(1600)]
    if change == "unsupported":
        y[:] = 0
    elif change == "nonbinary":
        y[0] = 2
    elif change == "matrix":
        y = y[:,None]
    elif change == "effective":
        y = np.r_[np.ones(20), np.zeros(1980)]
    else:
        y[0] = np.nan
    with pytest.raises(ValueError):
        balanced_weights(y)


@pytest.mark.parametrize("change", ["doubled", "reordered", "other_action", "support"])
def test_weight_mutations_detected(balanced, change):
    rows, (_, _, weights, support), _, _ = balanced
    y = rows["training"].y_exit.to_numpy()
    w = weights["exit"].copy()
    detail = deepcopy(support["actions"]["exit"]["weights"])
    if change == "doubled":
        w *= 2
    elif change == "reordered":
        w = w[::-1]
    elif change == "other_action":
        w = weights["reduce"]
    else:
        detail["classes"]["1"]["rows"] += 1
    with pytest.raises((ValueError, AssertionError)):
        verify_weights(y, w, detail)


@pytest.mark.parametrize("change", ["format", "rule", "action", "features", "class_weight", "depth"])
def test_invalid_model_bundle_rejected(balanced, change):
    raw = balanced[1][0].to_dict()
    if change == "format":
        raw["format"] = "other"
    elif change == "rule":
        raw["weight_rule"] = "multioutput_product"
    elif change == "action":
        raw["models"]["exit"] = raw["models"]["reduce"]
    elif change == "features":
        raw["features"] = raw["features"][:-1]
    elif change == "class_weight":
        raw["models"]["exit"]["settings"]["class_weight"] = "balanced"
    else:
        raw["models"]["exit"]["settings"]["max_depth"] = 4
    with pytest.raises(ValueError):
        BalancedManagementModels.from_dict(raw)


def test_changed_prediction_and_decision_detected(balanced):
    rows, (model, offset, _, _), refs, offsets = balanced
    constant = rows["calibration"][LABELS].mean().to_numpy()
    models, offsets = {**refs, "balanced": model}, {**offsets, "balanced": offset}
    scores = capacity_predictions(rows["diagnosis"], models, offsets, constant)
    broken = deepcopy(scores)
    broken["balanced_raw"][0,0] += .01
    with pytest.raises(AssertionError):
        verify_balanced_predictions(rows["diagnosis"], model, offset, broken)
    calibration = capacity_predictions(rows["calibration"], models, offsets, constant)
    thresholds, _ = joint_thresholds(rows["calibration"], calibration)
    result, _ = evaluate_joint(rows["diagnosis"], scores, thresholds)
    decision = balanced_admission(result)
    decision["checks"]["exit_ap_native"] = not decision["checks"]["exit_ap_native"]
    with pytest.raises(AssertionError):
        verify_balanced_admission(result, decision)


def test_wrong_proof_digest_rejected(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_balanced_reference(path, "0"*64)
