from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_forward_management import fitted as fitted
from test_forward_management import parents as parents
from test_forward_management import training as training
from test_forward_management import weights as weights
from threadpoolctl import threadpool_limits

from wonyotti_fr import forward_final_learning as learning
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_state_models import FEATURES
from wonyotti_fr.fixed_bin_learning import pre_limit_metrics
from wonyotti_fr.fixed_bin_management import fixed_bin_admission, model_digest, training_digest
from wonyotti_fr.fixed_bin_reference import verify_fixed_admission
from wonyotti_fr.forward_final_models import (
    calibrate_forward_models,
    fit_forward_models,
    forward_final_admission,
    validate_forward_models,
)
from wonyotti_fr.forward_final_reference import (
    verify_forward_final_admission,
    verify_forward_final_refits,
)
from wonyotti_fr.forward_management import ARMS, FOLDS, WEIGHT_RULE
from wonyotti_fr.holding_stress import read_json, seal
from wonyotti_fr.paired_error_budget import evaluate_paired_budget, fit_paired_budget
from wonyotti_fr.quiet_management import (
    CLASSES,
    evaluate_quiet,
    fit_quiet_models,
    quiet_mask,
    quiet_predictions,
)
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_predictions,
)
from wonyotti_fr.quiet_state_calibration import fit_state_calibration, state_predictions
from wonyotti_fr.quiet_state_reference import verify_state_calibration, verify_state_predictions
from wonyotti_fr.retained_weight_close_reference import checked_files
from wonyotti_fr.tail_weighted_management import tail_admission


@pytest.fixture(scope="module")
def final_study(training, fitted, parents, weights):
    rows = {"training": training}
    native = {}
    for i, (phase, date) in enumerate([("calibration", "2021-01-03"), ("diagnosis", "2021-07-03")]):
        frame = training.iloc[:1800].copy()
        frame["end"] = pd.date_range(date, periods=len(frame), freq="min", tz="UTC")
        frame["entry_time"] = frame.end - pd.Timedelta(hours=1)
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["episode_id"] += (i + 1) * 10000
        rows[phase] = frame
    for phase, frame in rows.items():
        base = pd.DataFrame(index=frame.index)
        for action in ["exit", "reduce", "increase"]:
            for name, scale in [("native", 0.1), ("capacity", 0.2)]:
                p = 1 / (1 + np.exp(-scale * frame[FEATURES[1]].to_numpy()))
                base[action + "_" + name + "_calibrated"] = p
                base[action + "_" + name + "_calibrated_request"] = p >= 0.52
        native[phase] = base
    old = fit_quiet_models(training, rows["calibration"])
    bins = {}
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            part = training.loc[quiet_mask(training, action)]
            tree = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(
                part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
            )
            assert old[0][action]["tree"] == parents[action]["tree"]
            bins[action] = {
                "format": "fixed_unweighted_quiet_bins_v1",
                "features": FEATURES.copy(),
                "action": action,
                "rows": len(part),
                "parent_sha256": model_digest(parents[action]["tree"]),
                "training_sha256": training_digest(training, action),
                "thresholds": [q.tolist() for q in tree._bin_mapper.bin_thresholds_],
            }
    final = fit_forward_models(training, parents, bins, fitted, weights)
    previous_raw = {p: quiet_predictions(f, native[p], *old[:3], old[4]) for p, f in rows.items()}
    previous_state = fit_state_calibration(previous_raw["calibration"])
    previous = {p: state_predictions(v, *previous_state[:3])[0] for p, v in previous_raw.items()}
    new, budgets, prelimits, metrics = {}, {}, {}, {}
    for arm in ARMS:
        global_fit = (
            final[0][arm],
            *calibrate_forward_models(rows["calibration"], final[0][arm], arm),
        )
        raw = {
            p: quiet_predictions(f, native[p], *global_fit[:3], global_fit[4])
            for p, f in rows.items()
        }
        verify_quiet_calibration(raw["calibration"], *global_fit[1:])
        state = fit_state_calibration(raw["calibration"])
        verify_state_calibration(raw["calibration"], *state)
        new[arm] = {}
        for phase in rows:
            verify_quiet_predictions(
                rows[phase], native[phase], raw[phase], *global_fit[:3], global_fit[4]
            )
            candidate, control = state_predictions(raw[phase], *state[:3])
            verify_state_predictions(raw[phase], candidate, control, *state[:3])
            new[arm][phase] = candidate
        pair = {p: {"v127": previous[p], arm: new[arm][p]} for p in rows}
        cuts = fit_paired_budget(rows["calibration"], pair["calibration"], "v127")
        budgets[arm] = evaluate_paired_budget(rows["diagnosis"], pair["diagnosis"], cuts)[0]
        prelimits[arm] = pre_limit_metrics(rows["diagnosis"], pair["diagnosis"])
        metrics[arm] = evaluate_quiet(new[arm]["diagnosis"])
    return rows, bins, final, previous, metrics, prelimits, budgets


def decision_inputs(study):
    _, _, _, previous, metrics, prelimits, budgets = study
    return (
        metrics["v142forward"],
        metrics["v142matched"],
        evaluate_quiet(previous["diagnosis"]),
        {"exit": True, "recent_reduce": True, "recent_increase": True},
        prelimits["v142forward"],
        budgets["v142forward"],
        budgets["v142matched"],
    )


def test_both_fixed_bin_models_linears_and_all_periods(final_study, parents, weights, tmp_path):
    rows, bins, final, *_ = final_study
    proof = verify_forward_final_refits(rows, parents, bins, weights[0], final)
    assert proof["new_final_trees"] == 1024 and proof["new_linear_models"] == 4
    assert proof["original_trees"] == proof["control_trees"] == 512 and proof["boundary_probes"] > 0
    save_json(tmp_path / "final.json", final)
    assert read_json(tmp_path / "final.json") == list(final)
    for arm in ARMS:
        validate_forward_models(final[0][arm], arm)


def test_all_48_conditions_and_every_flipped_flag(final_study):
    inputs = decision_inputs(final_study)
    decision = forward_final_admission(*inputs)
    verify_forward_final_admission(*inputs, decision)
    assert len(decision["checks"]) == 48
    for key in decision["checks"]:
        changed = deepcopy(decision)
        changed["checks"][key] = not changed["checks"][key]
        with pytest.raises(AssertionError):
            verify_forward_final_admission(*inputs, changed)


def test_generic_42_conditions_preserve_legacy_decisions(final_study):
    candidate, _, previous, unchanged, pre, budget, _ = decision_inputs(final_study)
    for name in ["v140", "v141", *ARMS]:
        p = {"v127": pre["v127"], name: pre["v142forward"]}
        b = deepcopy(budget)
        for item in b.values():
            item["models"][name] = item["models"].pop("v142forward")
        decision = fixed_bin_admission(candidate, previous, unchanged, p, b, model_name=name)
        verify_fixed_admission(candidate, previous, unchanged, p, b, decision, model_name=name)
        if name == "v140":
            assert (
                decision["checks"] == tail_admission(candidate, previous, unchanged, p, b)["checks"]
            )
        if name == "v141":
            assert decision == fixed_bin_admission(candidate, previous, unchanged, p, b)


def test_2021_changes_cannot_change_training_or_calibration(final_study, parents, fitted, weights):
    rows, bins, final, *_ = final_study
    altered = {p: f.copy() for p, f in rows.items()}
    for phase in ["calibration", "diagnosis"]:
        altered[phase][FEATURES[0]] += 100
        for action in ["reduce", "increase"]:
            altered[phase]["y_" + action] = 1 - altered[phase]["y_" + action]
    assert fit_forward_models(altered["training"], parents, bins, fitted, weights) == final
    for arm in ARMS:
        before = calibrate_forward_models(rows["calibration"], final[0][arm], arm)
        altered["diagnosis"][FEATURES[0]] += 20
        assert calibrate_forward_models(rows["calibration"], final[0][arm], arm) == before
        with pytest.raises(ValueError):
            calibrate_forward_models(rows["diagnosis"], final[0][arm], arm)
    with pytest.raises(ValueError):
        fit_forward_models(rows["calibration"], parents, bins, fitted, weights)


@pytest.mark.parametrize("damage", ["arm", "rule", "linear", "tree", "scope", "metadata_alias"])
def test_final_model_damage_rejected(final_study, damage):
    models = deepcopy(final_study[2][0]["v142forward"])
    if damage == "arm":
        models["reduce"]["arm"] = "v142matched"
    elif damage == "rule":
        models["reduce"]["weight_rule"] = "old"
    elif damage == "linear":
        models["reduce"]["linear"]["coef"][0] = np.nan
    elif damage == "tree":
        models["reduce"]["tree"]["models"][0]["baseline"] = np.inf
    elif damage == "scope":
        models["reduce"]["scope"] = "all"
    else:
        original = FEATURES.copy()
        models["reduce"]["linear"]["features"].reverse()
        assert FEATURES == original and models["increase"]["linear"]["features"] == original
    with pytest.raises(ValueError):
        validate_forward_models(models, "v142forward")


@pytest.mark.parametrize(
    "damage", ["bin_source", "bin_edge", "weight", "fold", "parent", "units", "boundary"]
)
def test_training_input_connections_rejected(final_study, parents, fitted, weights, damage):
    rows, bins, *_ = deepcopy(final_study)
    parents, auxiliary, weighted = deepcopy(parents), deepcopy(fitted), deepcopy(weights)
    if damage == "bin_source":
        bins["reduce"]["training_sha256"] = "b" * 64
    elif damage == "bin_edge":
        bins["reduce"]["thresholds"][0][0] -= 1
    elif damage == "weight":
        weighted[0]["v142forward"].loc[0, "reduce_weight"] *= 2
    elif damage == "fold":
        auxiliary[1].loc[1800, "reduce_fold"] = "Q4"
    elif damage == "parent":
        parents["reduce"]["tree"]["models"][0]["baseline"] += 1
    elif damage == "units":
        weighted[0]["v142forward"].loc[0, "reduce_units"] = 1
    else:
        weighted[1]["v142forward"]["reduce"]["folds"]["Q2"]["boundary_score"] += 0.1
    with pytest.raises((ValueError, AssertionError)):
        fit_forward_models(rows["training"], parents, bins, auxiliary, weighted)


@pytest.mark.parametrize("damage", ["tree", "linear", "projection", "levels", "control"])
def test_independent_final_refit_detects_corruption(final_study, parents, weights, damage):
    rows, bins, final, *_ = deepcopy(final_study)
    if damage == "tree":
        final[2]["v142matched"]["reduce"]["models"][0]["baseline"] += 0.1
    elif damage == "linear":
        final[0]["v142matched"]["reduce"]["linear"]["coef"][0] += 0.1
    elif damage == "projection":
        final[0]["v142matched"]["reduce"]["tree"]["models"][0]["baseline"] += 0.1
    elif damage == "levels":
        final[3]["reduce"]["v142matched"][0][0] += 0.1
    else:
        final[1]["reduce"]["models"][0]["baseline"] += 0.1
    with pytest.raises(AssertionError):
        verify_forward_final_refits(rows, parents, bins, weights[0], final)


@pytest.mark.parametrize(
    "damage", ["rows", "nonfinite", "budget_support", "budget_baseline", "budget_name"]
)
def test_matched_comparison_context_rejected(final_study, damage):
    inputs = deepcopy(decision_inputs(final_study))
    if damage == "rows":
        inputs[1]["reduce"]["quiet"]["tree"]["rows"] += 1
    elif damage == "nonfinite":
        inputs[1]["reduce"]["quiet"]["tree"]["log_loss"] = np.nan
    elif damage == "budget_support":
        inputs[6]["increase__restart__below"]["support"]["rows"] += 1
    elif damage == "budget_baseline":
        inputs[6]["increase__restart__below"]["models"]["v127"]["status"] = "changed"
    else:
        inputs[6]["increase__restart__below"]["models"]["other"] = inputs[6][
            "increase__restart__below"
        ]["models"].pop("v142matched")
    with pytest.raises(ValueError):
        forward_final_admission(*inputs)


@pytest.mark.parametrize("name", ["v127", "", None, "v142-forward"])
def test_invalid_generic_candidate_name(final_study, name):
    candidate, _, previous, unchanged, pre, budget, _ = decision_inputs(final_study)
    with pytest.raises(ValueError):
        fixed_bin_admission(candidate, previous, unchanged, pre, budget, model_name=name)


@pytest.mark.parametrize(
    "damage", [None, "hash", "exit", "stage", "complete", "source", "arms", "protocol", "final"]
)
def test_auxiliary_stage_proof_required(tmp_path, monkeypatch, damage):
    root = tmp_path / "stage"
    root.mkdir()
    summary = {
        k: True
        for k in [
            "complete",
            "independent_verification_complete",
            "all_inputs_and_code_unchanged",
            "new_models_fitted",
        ]
    }
    flags = {
        k: False
        for k in [
            "experiment_complete",
            "final_models_fitted",
            "new_accounts_executed",
            "profitability_accepted",
        ]
    }
    summary.update(
        flags,
        scope="auxiliary_and_weight_stage",
        auxiliary_trees=1536,
        new_final_trees=0,
        new_linear_models=0,
    )
    if damage == "stage":
        summary["scope"] = "all"
    if damage == "complete":
        summary["independent_verification_complete"] = False
    if damage == "final":
        summary["final_models_fitted"] = True
    settings = {
        "reference": "prior.json",
        "reference_sha256": "a" * 64,
        "sources": {"teacher": "bound"},
        "features": FEATURES,
        "arms": list(ARMS),
        "folds": {k: list(v) for k, v in FOLDS.items()},
        "weight_rule": WEIGHT_RULE,
        "protocol_sha256": sha256(learning.Path("docs/EXPERIMENT_V142.md")),
    }
    if damage == "arms":
        settings["arms"] = ["v142forward"]
    if damage == "protocol":
        settings["protocol_sha256"] = "b" * 64
    save_json(root / "summary.json", summary)
    save_json(root / "manifest.json", {"settings": settings})
    seal(root)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "complete": True,
            "scope": "auxiliary_and_weight_stage",
            "run": str(root),
            "files_sha256": sha256(root / "files.json"),
            "summary_sha256": sha256(root / "summary.json"),
            "new_models_fitted": True,
            **flags,
            "process_exit_code": False if damage == "exit" else 0,
        },
    )
    monkeypatch.setattr(
        learning,
        "checked_forward_reference",
        lambda *_: {"teacher": "wrong" if damage == "source" else "bound"},
    )
    expected = "b" * 64 if damage == "hash" else sha256(proof)
    if damage is None:
        assert learning.checked_forward_weight_reference(proof, expected) == {
            "teacher": "bound",
            "forward_weights": str(root),
        }
    else:
        with pytest.raises(ValueError):
            learning.checked_forward_weight_reference(proof, expected)


def test_evaluation_persists_all_periods_empty_months_and_child_seal(final_study, tmp_path):
    rows, _, final, previous, expected_metrics, _, _ = final_study
    bases = {}
    for phase, frame in rows.items():
        base = pd.DataFrame(index=frame.index)
        for action in ["exit", "reduce", "increase"]:
            for name, scale in [("native", 0.1), ("capacity", 0.2)]:
                p = 1 / (1 + np.exp(-scale * frame[FEATURES[1]].to_numpy()))
                base[action + "_" + name + "_calibrated"] = p
                base[action + "_" + name + "_calibrated_request"] = p >= 0.52
        bases[phase] = base
    cuts = fit_paired_budget(
        rows["calibration"],
        {"v127": previous["calibration"], "v142matched": previous["calibration"]},
        "v127",
    )
    outputs, decision = learning.evaluate_final_arm(
        rows, final[0]["v142matched"], "v142matched", bases, previous, cuts, tmp_path / "arm"
    )
    assert outputs["metrics"]["diagnosis"] == expected_metrics["v142matched"]
    assert {k: len(v) for k, v in outputs["monthly"].items()} == {
        "training": 12,
        "calibration": 6,
        "diagnosis": 6,
    }
    assert outputs["monthly"]["diagnosis"]["2021-12"]["reduce"]["quiet"]["tree"] is None
    assert len(decision["checks"]) == 42
    checked_files(tmp_path / "arm")
