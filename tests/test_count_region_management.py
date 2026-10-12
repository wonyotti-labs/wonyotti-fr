from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study

from wonyotti_fr.add_count_audit import audit_count_strata
from wonyotti_fr.add_count_reference import verify_count_strata
from wonyotti_fr.count_region_learning import checked_count_region_reference
from wonyotti_fr.count_region_management import (
    BASE_FEATURES,
    CLASSES,
    COMPARISONS,
    FEATURES,
    count_region_admission,
    count_region_linear_probabilities,
    count_region_quiet_predictions,
    evaluate_count_regions,
    fit_count_region_models,
    pre_limit_restart_metrics,
    region_masks,
    region_raw_scores,
    validate_count_region_inputs,
    validate_region_support,
)
from wonyotti_fr.count_region_reference import (
    verify_count_region_admission,
    verify_count_region_quiet_predictions,
    verify_count_region_refits,
    verify_count_regions,
    verify_pre_limit_restart,
    verify_region_support,
    verify_routing_ledger,
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
def count_region_study(study):
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
    fitted = fit_count_region_models(rows["training"], rows["calibration"])
    global_predictions = {
        p: count_region_quiet_predictions(f, previous[p], *fitted[:3], fitted[4])
        for p, f in rows.items()
    }
    state = fit_state_calibration(global_predictions["calibration"])
    return rows, previous, fitted, global_predictions, state, old_fit


def test_independent_refits_calibration_predictions_and_all_gates(count_region_study):
    rows, previous, fitted, global_predictions, state, old_fit = count_region_study
    assert verify_count_region_refits(rows, fitted[0])["independent_histogram_trees"] == 512
    assert verify_quiet_refits(rows, old_fit[0])["independent_histogram_trees"] == 512
    verify_quiet_calibration(global_predictions["calibration"], *fitted[1:])
    verify_state_calibration(global_predictions["calibration"], *state)
    for phase, frame in rows.items():
        global_prediction = global_predictions[phase]
        unchanged = verify_count_region_quiet_predictions(
            frame, previous[phase], global_prediction, *fitted[:3], fitted[4]
        )
        candidate, control = state_predictions(global_prediction, *state[:3])
        verify_state_predictions(global_prediction, candidate, control, *state[:3])
        metrics = evaluate_quiet(candidate)
        verify_quiet_evaluation(candidate, metrics)
        if phase == "diagnosis":
            original = evaluate_quiet(previous[phase])
            predictions = {"v127": previous[phase], "v137": candidate}
            pre_limit = pre_limit_restart_metrics(frame, predictions)
            verify_pre_limit_restart(frame, predictions, pre_limit)
            strata = audit_count_strata(frame, predictions, COMPARISONS)
            verify_count_strata(frame, predictions, strata, COMPARISONS)
            decision = count_region_admission(metrics, original, unchanged, pre_limit)
            verify_count_region_admission(metrics, original, unchanged, pre_limit, decision)
            assert len(decision["checks"]) == 40
            for key in decision["checks"]:
                changed = deepcopy(decision)
                changed["checks"][key] = not changed["checks"][key]
                with pytest.raises(AssertionError):
                    verify_count_region_admission(metrics, original, unchanged, pre_limit, changed)


def test_future_labels_do_not_change_fit_or_state_parameters(count_region_study):
    rows, _, fitted, global_predictions, state, _ = count_region_study
    changed = {p: f.copy() for p, f in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_count_region_models(changed["training"], changed["calibration"]) == fitted
    copied = {p: f.copy() for p, f in global_predictions.items()}
    copied["diagnosis"]["y_increase"] = 1 - copied["diagnosis"]["y_increase"]
    assert fit_state_calibration(copied["calibration"]) == state


@pytest.mark.parametrize(
    "damage", ["action", "features", "settings", "mean", "scale", "coef", "intercept", "format"]
)
def test_invalid_numeric_linear_model_rejected(count_region_study, damage):
    rows, _, fitted, _, _, _ = count_region_study
    raw = deepcopy(fitted[0]["reduce"]["below"]["linear"])
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
        count_region_linear_probabilities(raw, rows["diagnosis"][FEATURES].to_numpy(), "reduce", "below")


@pytest.mark.parametrize(
    "damage", ["nan", "direction", "negative", "fraction", "large", "capped", "missing", "history"]
)
def test_invalid_count_inputs_rejected(count_region_study, damage):
    frame = count_region_study[0]["training"].copy()
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
        validate_count_region_inputs(frame)


@pytest.mark.parametrize(
    "damage", ["model_action", "scope", "threshold", "constant", "source_context"]
)
def test_prediction_binding_rejected(count_region_study, damage):
    rows, previous, fitted, _, _, _ = count_region_study
    frame, old = rows["diagnosis"].copy(), previous["diagnosis"].copy()
    models, offsets, thresholds, _, constants = deepcopy(fitted)
    if damage == "model_action":
        models["reduce"]["below"]["tree"] = models["increase"]["below"]["tree"]
    elif damage == "scope":
        models["reduce"]["scope"] = "other"
    elif damage == "threshold":
        thresholds["reduce"]["tree"] = np.nan
    elif damage == "constant":
        constants["reduce"] = 2.0
    else:
        old.loc[0, "episode_id"] += 1
    with pytest.raises((ValueError, AssertionError)):
        count_region_quiet_predictions(frame, old, models, offsets, thresholds, constants)


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
def test_independent_prediction_checker_rejects_damage(count_region_study, column):
    rows, previous, fitted, predictions, _, _ = count_region_study
    saved = predictions["diagnosis"].copy()
    if column.endswith("request"):
        saved.loc[0, column] = not saved.loc[0, column]
    else:
        saved.loc[0, column] += 0.01
    with pytest.raises(AssertionError):
        verify_count_region_quiet_predictions(
            rows["diagnosis"], previous["diagnosis"], saved, *fitted[:3], fitted[4]
        )


def test_recent_only_preserves_all_predictions(count_region_study):
    rows, previous, fitted, _, _, _ = count_region_study
    frame, old = rows["diagnosis"].copy(), previous["diagnosis"].copy()
    for action in ["reduce", "increase"]:
        for value in [frame, old]:
            value[f"past_{action}_exists"] = 1.0
            value[f"past_{action}_log_count_15m"] = np.log(2.0)
    result = count_region_quiet_predictions(frame, old, *fitted[:3], fitted[4])
    pd.testing.assert_frame_equal(result, old, check_exact=True)
    verify_count_region_quiet_predictions(frame, old, result, *fitted[:3], fitted[4])


def test_fractional_labels_rejected_before_fit(count_region_study):
    rows = count_region_study[0]
    train = rows["training"].copy()
    train["y_reduce"] = train.y_reduce.astype(float)
    train.loc[0, "y_reduce"] = 1.5
    with pytest.raises(ValueError, match="지원"):
        fit_count_region_models(train, rows["calibration"])


def test_tree_feature_binding_and_reference_digest_rejected(count_region_study, tmp_path):
    raw = deepcopy(count_region_study[2][0]["reduce"]["below"]["tree"])
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        CLASSES["reduce"]["below"].from_dict(raw)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_count_region_reference(proof, "0" * 64)


def test_original_inputs_fixed_and_branch_boundary_at_five(count_region_study):
    rows, _, fitted, _, _, _ = count_region_study
    assert FEATURES == BASE_FEATURES and len(FEATURES) == 50 and "adds" not in FEATURES
    frame = rows["diagnosis"].iloc[:6].copy()
    frame["adds"] = [0, 4, 5, 6, 19, 104]
    frame["adds_capped"] = np.minimum(frame.adds, 5)
    for action in ["reduce", "increase"]:
        frame["past_" + action + "_exists"] = 1.
        frame["past_" + action + "_log_count_15m"] = 0.
        mask = region_masks(frame, action)
        assert mask["below"].tolist() == [True, True, False, False, False, False]
        np.testing.assert_array_equal(mask["capped"], ~mask["below"])
        model = deepcopy(fitted[0][action])
        for region, intercept in [("below", -1.), ("capped", 1.)]:
            linear = model[region]["linear"]
            linear.update(mean=[0.] * 50, scale=[1.] * 50, coef=[0.] * 50, intercept=intercept)
            assert len(model[region]["tree"]["models"][0]["trees"]) == 128
        scores = region_raw_scores(frame, action, model)
        np.testing.assert_allclose(scores["linear_raw"], 1 / (1 + np.exp(-np.array([-1., -1., 1., 1., 1., 1.]))), atol=0, rtol=0)
        changed = frame.copy()
        changed.loc[changed.adds.ge(5), "adds"] += 100
        for key, expected in region_raw_scores(frame, action, fitted[0][action]).items():
            np.testing.assert_array_equal(region_raw_scores(changed, action, fitted[0][action])[key], expected)


@pytest.mark.parametrize("damage", ["branch", "tree", "linear", "linear_region", "linear_action", "iterations", "missing"])
def test_region_model_binding_rejects_swaps(count_region_study, damage):
    frame = count_region_study[0]["diagnosis"]
    model = deepcopy(count_region_study[2][0]["reduce"])
    if damage == "branch":
        model["below"], model["capped"] = model["capped"], model["below"]
    elif damage in {"tree", "linear"}:
        model["below"][damage] = model["capped"][damage]
    elif damage == "linear_region":
        model["below"]["linear"]["region"] = "capped"
    elif damage == "linear_action":
        model["below"]["linear"]["action"] = "increase"
    elif damage == "iterations":
        model["below"]["tree"]["settings"]["max_iter"] = 256
    else:
        model.pop("capped")
    with pytest.raises(ValueError):
        region_raw_scores(frame, "reduce", model)


def test_regions_support_routing_full_metrics_and_empty_months(count_region_study):
    rows, old, _, predictions, state, _ = count_region_study
    support = validate_region_support(rows)
    for phase, frame in rows.items():
        verify_region_support(frame, support[phase])
        candidate = state_predictions(predictions[phase], *state[:3])[0]
        pair = {"v127": old[phase], "v137": candidate}
        result = evaluate_count_regions(frame, pair)
        verify_count_regions(frame, pair, result)
        ledger = frame[["end", "episode_id"]].copy()
        for action in ["reduce", "increase"]:
            ledger[action + "_region"] = "recent"
            for region, mask in region_masks(frame, action).items():
                ledger.loc[mask, action + "_region"] = region
        verify_routing_ledger(frame, ledger)
        wrong = deepcopy(result)
        wrong["reduce"]["below"]["support"]["positive_episodes"] += 1
        with pytest.raises(ValueError):
            verify_count_regions(frame, pair, wrong)
        wrong = deepcopy(result)
        wrong["increase"]["capped"]["models"]["v137"]["linear_raw"]["log_loss"] += .1
        with pytest.raises(ValueError):
            verify_count_regions(frame, pair, wrong)
        ledger.loc[0, "reduce_region"] = "incorrect"
        with pytest.raises(AssertionError):
            verify_routing_ledger(frame, ledger)
        small = frame.iloc[:1].copy()
        small["adds"] = 0
        small["adds_capped"] = 0
        minimal = {k: v.iloc[:1] for k, v in pair.items()}
        result = evaluate_count_regions(small, minimal)
        verify_count_regions(small, minimal, result)
        for action in ["reduce", "increase"]:
            assert result[action]["capped"]["support"]["rows"] == 0
            assert result[action]["capped"]["models"]["v137"]["tree"] is None


@pytest.mark.parametrize("damage", ["empty_region", "few_rows", "few_positive", "few_negative", "phase"])
def test_region_fit_support_rejected(count_region_study, damage):
    rows = {p: f.copy() for p, f in count_region_study[0].items()}
    if damage == "empty_region":
        rows["training"]["adds"] = 5
        rows["training"]["adds_capped"] = 5
    elif damage == "few_rows":
        rows["training"] = rows["training"].iloc[:1000]
    elif damage in {"few_positive", "few_negative"}:
        rows["calibration"]["y_reduce"] = int(damage == "few_negative")
    else:
        rows["unknown"] = rows.pop("diagnosis")
    with pytest.raises(ValueError):
        validate_region_support(rows)


def test_single_region_prediction_keeps_context_and_all_models_bound(count_region_study):
    rows, previous, fitted, _, _, _ = count_region_study
    frame = rows["diagnosis"]
    for included in [frame.adds.lt(5), frame.adds.ge(5)]:
        part, old = frame.loc[included], previous["diagnosis"].loc[included]
        actual = count_region_quiet_predictions(part, old, *fitted[:3], fitted[4])
        verify_count_region_quiet_predictions(part, old, actual, *fitted[:3], fitted[4])


def test_pre_limit_direct_population_boundary_and_explicit_model_names():
    from test_add_count_audit import population

    frame, old = population()
    predictions = {"v127": old["v127"], "v137": old["v133"]}
    with pytest.raises(ValueError):
        audit_count_strata(frame, predictions)
    strata = audit_count_strata(frame, predictions, COMPARISONS)
    verify_count_strata(frame, predictions, strata, COMPARISONS)
    direct = pre_limit_restart_metrics(frame, predictions)
    verify_pre_limit_restart(frame, predictions, direct)
    assert direct["v127"]["rows"] == direct["v137"]["rows"] == 5
    assert direct["v127"]["positive"] == direct["v137"]["positive"] == 3
    wrong = deepcopy(direct)
    wrong["v137"]["log_loss"] += 0.1
    with pytest.raises(ValueError):
        verify_pre_limit_restart(frame, predictions, wrong)
    for names in [("v127", "v127"), ("v127", ""), ("v127", "v133", "v137")]:
        with pytest.raises(ValueError):
            audit_count_strata(frame, predictions, names)


@pytest.mark.parametrize("damage", ["empty", "support", "mismatch", "nonfinite"])
def test_pre_limit_gate_rejects_invalid_population(count_region_study, damage):
    rows, previous, _, globals_, state, _ = count_region_study
    candidate = state_predictions(globals_["diagnosis"], *state[:3])[0]
    pre = pre_limit_restart_metrics(
        rows["diagnosis"], {"v127": previous["diagnosis"], "v137": candidate}
    )
    if damage == "empty":
        pre["v137"] = None
    elif damage == "support":
        pre["v137"]["positive"] = 1
    elif damage == "mismatch":
        pre["v137"]["rows"] += 1
    else:
        pre["v137"]["recall"] = np.nan
    with pytest.raises(ValueError):
        count_region_admission(
            evaluate_quiet(candidate),
            evaluate_quiet(previous["diagnosis"]),
            {"exit": True, "recent_reduce": True, "recent_increase": True},
            pre,
        )


@pytest.mark.parametrize("damage", ["none", "complete", "inputs", "admitted", "accounts", "sources", "features", "file", "symlink"])
def test_prior_cross_market_proof_binding(tmp_path, monkeypatch, damage):
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.cross_market_models import FEATURES as prior_features
    from wonyotti_fr.holding_stress import seal

    run = tmp_path / "run"
    run.mkdir()
    summary = dict.fromkeys(["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"], True)
    summary.update(cross_market_admitted=False, new_accounts_executed=False, profitability_accepted=False)
    if damage == "complete":
        summary["complete"] = False
    elif damage == "inputs":
        summary["all_inputs_and_code_unchanged"] = False
    elif damage == "admitted":
        summary["cross_market_admitted"] = True
    elif damage == "accounts":
        summary["new_accounts_executed"] = True
    settings = {"reference": "unused", "reference_sha256": "unused", "sources": {"teacher": "a"}, "features": prior_features.copy()}
    if damage == "features":
        settings["features"] = settings["features"][::-1]
    save_json(run / "summary.json", summary)
    save_json(run / "manifest.json", {"settings": settings})
    seal(run)
    proof = tmp_path / "proof.json"
    save_json(proof, {"run": str(run), "complete": True, "files_sha256": sha256(run / "files.json"), "summary_sha256": sha256(run / "summary.json"), "new_accounts_executed": False, "profitability_accepted": False})
    monkeypatch.setattr("wonyotti_fr.count_region_learning.checked_cross_input_reference", lambda *args: {"teacher": "b" if damage == "sources" else "a"})
    digest = sha256(proof)
    if damage == "file":
        save_json(run / "summary.json", {})
    elif damage == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(proof)
        proof = link
    if damage == "none":
        assert checked_count_region_reference(proof, digest) == {"teacher": "a", "cross_candidate": str(run)}
    else:
        with pytest.raises(ValueError):
            checked_count_region_reference(proof, digest)


def test_training_labels_in_one_region_do_not_change_other_region_models(count_region_study):
    rows, _, fitted, _, _, _ = count_region_study
    changed = rows["training"].copy()
    for action in ["reduce", "increase"]:
        mask = region_masks(changed, action)["below"]
        changed.loc[mask, "y_" + action] = 1 - changed.loc[mask, "y_" + action]
    models = fit_count_region_models(changed, rows["calibration"])[0]
    for action in ["reduce", "increase"]:
        assert models[action]["capped"] == fitted[0][action]["capped"]
        assert models[action]["below"] != fitted[0][action]["below"]


def test_calibration_features_do_not_enter_training_or_scaling(count_region_study):
    rows, _, fitted, _, _, _ = count_region_study
    changed = rows["calibration"].copy()
    feature = next(k for k in FEATURES if k.startswith("ret_"))
    changed[feature] += .01
    models = fit_count_region_models(rows["training"], changed)[0]
    assert models == fitted[0]
