from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study

from wonyotti_fr.order_price_learning import checked_price_learning_reference
from wonyotti_fr.order_price_model_reference import (
    verify_price_admission,
    verify_price_quiet_predictions,
    verify_price_refits,
)
from wonyotti_fr.order_price_models import (
    CLASSES,
    FEATURES,
    fit_price_models,
    price_admission,
    price_linear_probabilities,
    price_quiet_predictions,
    validate_price_inputs,
)
from wonyotti_fr.quiet_management import evaluate_quiet, fit_quiet_models, quiet_predictions
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_refits,
)
from wonyotti_fr.quiet_state_calibration import fit_state_calibration, state_predictions
from wonyotti_fr.quiet_state_reference import verify_state_calibration, verify_state_predictions


@pytest.fixture(scope="module")
def price_study(study):
    source, native, _ = study
    rows = {p: f.copy() for p, f in source.items()}
    for frame in rows.values():
        frame["direction"] = np.where(frame.direction > 0, 1.0, -1.0)
        for i, action in enumerate(["increase", "reduce"]):
            raw = np.where(
                frame[f"past_{action}_exists"].eq(1), frame.iloc[:, i].to_numpy() * 0.01, 0.0
            )
            frame[f"past_{action}_log_price_move"] = raw
            frame[f"past_{action}_directional_log_price_move"] = raw * frame.direction
    old_fit = fit_quiet_models(rows["training"], rows["calibration"])
    old_global = {
        p: quiet_predictions(f, native[p], *old_fit[:3], old_fit[4]) for p, f in rows.items()
    }
    old_state = fit_state_calibration(old_global["calibration"])
    previous = {p: state_predictions(f, *old_state[:3])[0] for p, f in old_global.items()}
    fitted = fit_price_models(rows["training"], rows["calibration"])
    global_predictions = {
        p: price_quiet_predictions(f, previous[p], *fitted[:3], fitted[4]) for p, f in rows.items()
    }
    state = fit_state_calibration(global_predictions["calibration"])
    return rows, previous, fitted, global_predictions, state, old_fit


def test_independent_refits_calibration_predictions_and_all_gates(price_study):
    rows, previous, fitted, global_predictions, state, old_fit = price_study
    assert verify_price_refits(rows, fitted[0])["independent_histogram_trees"] == 512
    assert verify_quiet_refits(rows, old_fit[0])["independent_histogram_trees"] == 512
    verify_quiet_calibration(global_predictions["calibration"], *fitted[1:])
    verify_state_calibration(global_predictions["calibration"], *state)
    for phase, frame in rows.items():
        global_prediction = global_predictions[phase]
        unchanged = verify_price_quiet_predictions(
            frame, previous[phase], global_prediction, *fitted[:3], fitted[4]
        )
        candidate, control = state_predictions(global_prediction, *state[:3])
        verify_state_predictions(global_prediction, candidate, control, *state[:3])
        metrics = evaluate_quiet(candidate)
        verify_quiet_evaluation(candidate, metrics)
        if phase == "diagnosis":
            original = evaluate_quiet(previous[phase])
            decision = price_admission(metrics, original, unchanged)
            verify_price_admission(metrics, original, unchanged, decision)
            assert len(decision["checks"]) == 37
            for key in decision["checks"]:
                changed = deepcopy(decision)
                changed["checks"][key] = not changed["checks"][key]
                with pytest.raises(AssertionError):
                    verify_price_admission(metrics, original, unchanged, changed)


def test_future_labels_do_not_change_fit_or_state_parameters(price_study):
    rows, _, fitted, global_predictions, state, _ = price_study
    changed = {p: f.copy() for p, f in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_price_models(changed["training"], changed["calibration"]) == fitted
    copied = {p: f.copy() for p, f in global_predictions.items()}
    copied["diagnosis"]["y_increase"] = 1 - copied["diagnosis"]["y_increase"]
    assert fit_state_calibration(copied["calibration"]) == state


@pytest.mark.parametrize(
    "damage", ["action", "features", "settings", "mean", "scale", "coef", "intercept", "format"]
)
def test_invalid_numeric_linear_model_rejected(price_study, damage):
    rows, _, fitted, _, _, _ = price_study
    raw = deepcopy(fitted[0]["reduce"]["linear"])
    if damage == "action":
        raw["action"] = "increase"
    elif damage == "features":
        raw["features"] = raw["features"][::-1]
    elif damage == "settings":
        raw["settings"]["C"] = 1.0
    elif damage == "mean":
        raw["mean"][0] = np.nan
    elif damage == "scale":
        raw["scale"][0] = 0.0
    elif damage == "coef":
        raw["coef"] = raw["coef"][:-1]
    elif damage == "intercept":
        raw["intercept"] = np.inf
    else:
        raw["format"] = "other"
    with pytest.raises(ValueError):
        price_linear_probabilities(raw, rows["diagnosis"][FEATURES].to_numpy(), "reduce")


@pytest.mark.parametrize(
    "damage", ["price_nan", "direction", "absent", "directional", "missing", "count"]
)
def test_invalid_price_inputs_rejected(price_study, damage):
    frame = price_study[0]["training"].copy()
    if damage == "price_nan":
        frame.loc[0, "past_increase_log_price_move"] = np.nan
    elif damage == "direction":
        frame.loc[0, "direction"] = 0
    elif damage == "absent":
        pos = frame.index[frame.past_reduce_exists.eq(0)][0]
        frame.loc[pos, "past_reduce_log_price_move"] = 1.0
        frame.loc[pos, "past_reduce_directional_log_price_move"] = frame.loc[pos, "direction"]
    elif damage == "directional":
        frame.loc[0, "past_increase_directional_log_price_move"] += 1.0
    elif damage == "missing":
        frame = frame.drop(columns=FEATURES[-1])
    else:
        frame.loc[0, "past_increase_log_count_15m"] = -1
    with pytest.raises(ValueError):
        validate_price_inputs(frame)


@pytest.mark.parametrize(
    "damage", ["model_action", "scope", "threshold", "constant", "source_context"]
)
def test_prediction_binding_rejected(price_study, damage):
    rows, previous, fitted, _, _, _ = price_study
    frame, old = rows["diagnosis"].copy(), previous["diagnosis"].copy()
    models, offsets, thresholds, _, constants = deepcopy(fitted)
    if damage == "model_action":
        models["reduce"]["tree"] = models["increase"]["tree"]
    elif damage == "scope":
        models["reduce"]["scope"] = "other"
    elif damage == "threshold":
        thresholds["reduce"]["tree"] = np.nan
    elif damage == "constant":
        constants["reduce"] = 2.0
    else:
        old.loc[0, "episode_id"] += 1
    with pytest.raises((ValueError, AssertionError)):
        price_quiet_predictions(frame, old, models, offsets, thresholds, constants)


@pytest.mark.parametrize(
    "column",
    [
        "exit_score",
        "reduce_baseline_score",
        "reduce_tree_score",
        "increase_linear_raw_score",
        "reduce_constant_request",
        "past_increase_exists",
    ],
)
def test_independent_prediction_checker_rejects_damage(price_study, column):
    rows, previous, fitted, predictions, _, _ = price_study
    saved = predictions["diagnosis"].copy()
    if column.endswith("request"):
        saved.loc[0, column] = not saved.loc[0, column]
    else:
        saved.loc[0, column] += 0.01
    with pytest.raises(AssertionError):
        verify_price_quiet_predictions(
            rows["diagnosis"], previous["diagnosis"], saved, *fitted[:3], fitted[4]
        )


def test_recent_only_preserves_all_predictions(price_study):
    rows, previous, fitted, _, _, _ = price_study
    frame, old = rows["diagnosis"].copy(), previous["diagnosis"].copy()
    for action in ["reduce", "increase"]:
        for value in [frame, old]:
            value[f"past_{action}_exists"] = 1.0
            value[f"past_{action}_log_count_15m"] = np.log(2.0)
    result = price_quiet_predictions(frame, old, *fitted[:3], fitted[4])
    pd.testing.assert_frame_equal(result, old, check_exact=True)
    verify_price_quiet_predictions(frame, old, result, *fitted[:3], fitted[4])


def test_fractional_labels_rejected_before_fit(price_study):
    rows = price_study[0]
    train = rows["training"].copy()
    train["y_reduce"] = train.y_reduce.astype(float)
    train.loc[0, "y_reduce"] = 1.5
    with pytest.raises(ValueError, match="지원"):
        fit_price_models(train, rows["calibration"])


def test_tree_feature_binding_and_reference_digest_rejected(price_study, tmp_path):
    raw = deepcopy(price_study[2][0]["reduce"]["tree"])
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        CLASSES["reduce"].from_dict(raw)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_price_learning_reference(proof, "0" * 64)
