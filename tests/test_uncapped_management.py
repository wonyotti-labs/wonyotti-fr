from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study

from wonyotti_fr.add_count_audit import audit_count_strata
from wonyotti_fr.add_count_reference import verify_count_strata
from wonyotti_fr.quiet_management import evaluate_quiet, fit_quiet_models, quiet_predictions
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_refits,
)
from wonyotti_fr.quiet_state_calibration import fit_state_calibration, state_predictions
from wonyotti_fr.quiet_state_reference import verify_state_calibration, verify_state_predictions
from wonyotti_fr.uncapped_management import (
    BASE_FEATURES,
    CLASSES,
    COMPARISONS,
    FEATURES,
    fit_uncapped_models,
    pre_limit_restart_metrics,
    uncapped_admission,
    uncapped_linear_probabilities,
    uncapped_quiet_predictions,
    validate_uncapped_inputs,
)
from wonyotti_fr.uncapped_management_learning import checked_uncapped_reference
from wonyotti_fr.uncapped_management_reference import (
    verify_pre_limit_restart,
    verify_uncapped_admission,
    verify_uncapped_quiet_predictions,
    verify_uncapped_refits,
)


@pytest.fixture(scope="module")
def uncapped_study(study):
    source, native, _ = study
    rows = {p: f.copy() for p, f in source.items()}
    for frame in rows.values():
        frame["direction"] = np.where(frame.direction > 0, 1.0, -1.0)
        frame["adds"] = np.arange(len(frame)) % 19
        frame["adds_capped"] = np.minimum(frame.adds, 5)
    old_fit = fit_quiet_models(rows["training"], rows["calibration"])
    old_global = {
        p: quiet_predictions(f, native[p], *old_fit[:3], old_fit[4]) for p, f in rows.items()
    }
    old_state = fit_state_calibration(old_global["calibration"])
    previous = {p: state_predictions(f, *old_state[:3])[0] for p, f in old_global.items()}
    fitted = fit_uncapped_models(rows["training"], rows["calibration"])
    global_predictions = {
        p: uncapped_quiet_predictions(f, previous[p], *fitted[:3], fitted[4])
        for p, f in rows.items()
    }
    state = fit_state_calibration(global_predictions["calibration"])
    return rows, previous, fitted, global_predictions, state, old_fit


def test_independent_refits_calibration_predictions_and_all_gates(uncapped_study):
    rows, previous, fitted, global_predictions, state, old_fit = uncapped_study
    assert verify_uncapped_refits(rows, fitted[0])["independent_histogram_trees"] == 512
    assert verify_quiet_refits(rows, old_fit[0])["independent_histogram_trees"] == 512
    verify_quiet_calibration(global_predictions["calibration"], *fitted[1:])
    verify_state_calibration(global_predictions["calibration"], *state)
    for phase, frame in rows.items():
        global_prediction = global_predictions[phase]
        unchanged = verify_uncapped_quiet_predictions(
            frame, previous[phase], global_prediction, *fitted[:3], fitted[4]
        )
        candidate, control = state_predictions(global_prediction, *state[:3])
        verify_state_predictions(global_prediction, candidate, control, *state[:3])
        metrics = evaluate_quiet(candidate)
        verify_quiet_evaluation(candidate, metrics)
        if phase == "diagnosis":
            original = evaluate_quiet(previous[phase])
            predictions = {"v127": previous[phase], "v135": candidate}
            pre_limit = pre_limit_restart_metrics(frame, predictions)
            verify_pre_limit_restart(frame, predictions, pre_limit)
            strata = audit_count_strata(frame, predictions, COMPARISONS)
            verify_count_strata(frame, predictions, strata, COMPARISONS)
            decision = uncapped_admission(metrics, original, unchanged, pre_limit)
            verify_uncapped_admission(metrics, original, unchanged, pre_limit, decision)
            assert len(decision["checks"]) == 40
            for key in decision["checks"]:
                changed = deepcopy(decision)
                changed["checks"][key] = not changed["checks"][key]
                with pytest.raises(AssertionError):
                    verify_uncapped_admission(metrics, original, unchanged, pre_limit, changed)


def test_future_labels_do_not_change_fit_or_state_parameters(uncapped_study):
    rows, _, fitted, global_predictions, state, _ = uncapped_study
    changed = {p: f.copy() for p, f in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_uncapped_models(changed["training"], changed["calibration"]) == fitted
    copied = {p: f.copy() for p, f in global_predictions.items()}
    copied["diagnosis"]["y_increase"] = 1 - copied["diagnosis"]["y_increase"]
    assert fit_state_calibration(copied["calibration"]) == state


@pytest.mark.parametrize(
    "damage", ["action", "features", "settings", "mean", "scale", "coef", "intercept", "format"]
)
def test_invalid_numeric_linear_model_rejected(uncapped_study, damage):
    rows, _, fitted, _, _, _ = uncapped_study
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
        uncapped_linear_probabilities(raw, rows["diagnosis"][FEATURES].to_numpy(), "reduce")


@pytest.mark.parametrize(
    "damage", ["nan", "direction", "negative", "fraction", "large", "capped", "missing", "history"]
)
def test_invalid_count_inputs_rejected(uncapped_study, damage):
    frame = uncapped_study[0]["training"].copy()
    if damage in {"nan", "negative", "fraction", "large"}:
        frame["adds"] = frame.adds.astype(float)
        frame.loc[0, "adds"] = {"nan": np.nan, "negative": -1, "fraction": 0.5, "large": 2**31}[
            damage
        ]
    elif damage == "direction":
        frame.loc[0, "direction"] = 0
    elif damage == "capped":
        frame.loc[8, "adds_capped"] = 4
    elif damage == "missing":
        frame = frame.drop(columns="adds")
    else:
        frame.loc[0, "past_increase_log_count_15m"] = -1
    with pytest.raises(ValueError):
        validate_uncapped_inputs(frame)


@pytest.mark.parametrize(
    "damage", ["model_action", "scope", "threshold", "constant", "source_context"]
)
def test_prediction_binding_rejected(uncapped_study, damage):
    rows, previous, fitted, _, _, _ = uncapped_study
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
        uncapped_quiet_predictions(frame, old, models, offsets, thresholds, constants)


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
def test_independent_prediction_checker_rejects_damage(uncapped_study, column):
    rows, previous, fitted, predictions, _, _ = uncapped_study
    saved = predictions["diagnosis"].copy()
    if column.endswith("request"):
        saved.loc[0, column] = not saved.loc[0, column]
    else:
        saved.loc[0, column] += 0.01
    with pytest.raises(AssertionError):
        verify_uncapped_quiet_predictions(
            rows["diagnosis"], previous["diagnosis"], saved, *fitted[:3], fitted[4]
        )


def test_recent_only_preserves_all_predictions(uncapped_study):
    rows, previous, fitted, _, _, _ = uncapped_study
    frame, old = rows["diagnosis"].copy(), previous["diagnosis"].copy()
    for action in ["reduce", "increase"]:
        for value in [frame, old]:
            value[f"past_{action}_exists"] = 1.0
            value[f"past_{action}_log_count_15m"] = np.log(2.0)
    result = uncapped_quiet_predictions(frame, old, *fitted[:3], fitted[4])
    pd.testing.assert_frame_equal(result, old, check_exact=True)
    verify_uncapped_quiet_predictions(frame, old, result, *fitted[:3], fitted[4])


def test_fractional_labels_rejected_before_fit(uncapped_study):
    rows = uncapped_study[0]
    train = rows["training"].copy()
    train["y_reduce"] = train.y_reduce.astype(float)
    train.loc[0, "y_reduce"] = 1.5
    with pytest.raises(ValueError, match="지원"):
        fit_uncapped_models(train, rows["calibration"])


def test_tree_feature_binding_and_reference_digest_rejected(uncapped_study, tmp_path):
    raw = deepcopy(uncapped_study[2][0]["reduce"]["tree"])
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        CLASSES["reduce"].from_dict(raw)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_uncapped_reference(proof, "0" * 64)


def test_count_replacement_preserves_other_inputs_and_exposes_counts_above_five(uncapped_study):
    rows, _, fitted, _, _, _ = uncapped_study
    frame = rows["diagnosis"]
    index = FEATURES.index("adds")
    assert len(FEATURES) == len(BASE_FEATURES) == 50
    x, old = frame[FEATURES].to_numpy(), frame[BASE_FEATURES].to_numpy()
    np.testing.assert_array_equal(np.delete(x, index, axis=1), np.delete(old, index, axis=1))
    np.testing.assert_array_equal(x[frame.adds.le(5)], old[frame.adds.le(5)])
    raw = deepcopy(fitted[0]["increase"]["linear"])
    raw.update(mean=[0.0] * 50, scale=[1.0] * 50, coef=[0.0] * 50, intercept=0.0)
    raw["coef"][index] = 0.1
    values = np.zeros((2, 50))
    values[:, index] = [5, 19]
    p = uncapped_linear_probabilities(raw, values, "increase")
    assert p[1] > p[0]
    tree = deepcopy(fitted[0]["increase"]["tree"])
    tree["features"][index] = "adds_capped"
    with pytest.raises(ValueError):
        CLASSES["increase"].from_dict(tree)
    raw["features"][index] = "adds_capped"
    with pytest.raises(ValueError):
        uncapped_linear_probabilities(raw, values, "increase")


def test_pre_limit_direct_population_boundary_and_explicit_model_names():
    from test_add_count_audit import population

    frame, old = population()
    predictions = {"v127": old["v127"], "v135": old["v133"]}
    with pytest.raises(ValueError):
        audit_count_strata(frame, predictions)
    strata = audit_count_strata(frame, predictions, COMPARISONS)
    verify_count_strata(frame, predictions, strata, COMPARISONS)
    direct = pre_limit_restart_metrics(frame, predictions)
    verify_pre_limit_restart(frame, predictions, direct)
    assert direct["v127"]["rows"] == direct["v135"]["rows"] == 5
    assert direct["v127"]["positive"] == direct["v135"]["positive"] == 3
    wrong = deepcopy(direct)
    wrong["v135"]["log_loss"] += 0.1
    with pytest.raises(ValueError):
        verify_pre_limit_restart(frame, predictions, wrong)
    for names in [("v127", "v127"), ("v127", ""), ("v127", "v133", "v135")]:
        with pytest.raises(ValueError):
            audit_count_strata(frame, predictions, names)


@pytest.mark.parametrize("damage", ["empty", "support", "mismatch", "nonfinite"])
def test_pre_limit_gate_rejects_invalid_population(uncapped_study, damage):
    rows, previous, _, globals_, state, _ = uncapped_study
    candidate = state_predictions(globals_["diagnosis"], *state[:3])[0]
    pre = pre_limit_restart_metrics(
        rows["diagnosis"], {"v127": previous["diagnosis"], "v135": candidate}
    )
    if damage == "empty":
        pre["v135"] = None
    elif damage == "support":
        pre["v135"]["positive"] = 1
    elif damage == "mismatch":
        pre["v135"]["rows"] += 1
    else:
        pre["v135"]["recall"] = np.nan
    with pytest.raises(ValueError):
        uncapped_admission(
            evaluate_quiet(candidate),
            evaluate_quiet(previous["diagnosis"]),
            {"exit": True, "recent_reduce": True, "recent_increase": True},
            pre,
        )


@pytest.mark.parametrize(
    "damage", ["complete", "inputs", "accounts", "sources", "buckets", "models", "file", "symlink"]
)
def test_prior_count_proof_damage_rejected(tmp_path, monkeypatch, damage):
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.holding_stress import seal

    run = tmp_path / "run"
    run.mkdir()
    summary = dict.fromkeys(
        [
            "complete",
            "independent_verification_complete",
            "all_inputs_and_code_unchanged",
            "input_reconstruction_verified",
            "original_predictions_and_state_metrics_exact",
            "all_buckets_and_months_verified",
        ],
        True,
    )
    summary.update(
        new_models_fitted=False, new_accounts_executed=False, profitability_accepted=False
    )
    if damage == "complete":
        summary["complete"] = False
    elif damage == "inputs":
        summary["input_reconstruction_verified"] = False
    elif damage == "accounts":
        summary["new_accounts_executed"] = True
    settings = {
        "reference": "unused",
        "reference_sha256": "unused",
        "sources": {"teacher": "a"},
        "buckets": ["0", "1", "2", "3", "4", "5", ">5"],
        "models": ["v127", "v133"],
    }
    if damage == "buckets":
        settings["buckets"][-1] = "6"
    elif damage == "models":
        settings["models"][-1] = "other"
    save_json(run / "summary.json", summary)
    save_json(run / "manifest.json", {"settings": settings})
    seal(run)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "run": str(run),
            "complete": True,
            "files_sha256": sha256(run / "files.json"),
            "summary_sha256": sha256(run / "summary.json"),
            "new_models_fitted": False,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    monkeypatch.setattr(
        "wonyotti_fr.uncapped_management_learning.checked_count_reference",
        lambda *args: {"teacher": "b" if damage == "sources" else "a"},
    )
    digest = sha256(proof)
    if damage == "file":
        save_json(run / "summary.json", {})
    elif damage == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(proof)
        proof = link
    with pytest.raises(ValueError):
        checked_uncapped_reference(proof, digest)
