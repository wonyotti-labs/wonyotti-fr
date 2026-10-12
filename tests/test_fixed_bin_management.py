from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study
from test_tail_weighted_management import weighted_study as weighted_study

from wonyotti_fr import fixed_bin_learning as learning
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_state_models import FEATURES
from wonyotti_fr.fixed_bin_learning import checked_fixed_reference
from wonyotti_fr.fixed_bin_management import (
    encode_bins,
    fit_fixed_bins,
    fixed_bin_admission,
    lift_thresholds,
    validate_bins,
)
from wonyotti_fr.fixed_bin_reference import (
    reference_encode,
    verify_fixed_admission,
    verify_fixed_refits,
    verify_projection,
)
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.paired_error_budget import evaluate_paired_budget, fit_paired_budget
from wonyotti_fr.quiet_management import evaluate_quiet, quiet_predictions
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_predictions,
)
from wonyotti_fr.quiet_state_calibration import fit_state_calibration, state_predictions
from wonyotti_fr.quiet_state_reference import verify_state_calibration, verify_state_predictions
from wonyotti_fr.tail_weighted_learning import pre_limit_metrics
from wonyotti_fr.tail_weighted_management import WEIGHT_RULE, calibrate_tail_models


@pytest.fixture(scope="module")
def fixed_study(weighted_study):
    rows, _, old, _, fitted, ledger, details, _, _, _ = weighted_study
    fixed = fit_fixed_bins(rows["training"], old[0], fitted[0], ledger, details)
    return fixed


def fake_bins():
    return {
        "format": "fixed_unweighted_quiet_bins_v1",
        "features": FEATURES.copy(),
        "action": "reduce",
        "rows": 2000,
        "parent_sha256": "a" * 64,
        "training_sha256": "b" * 64,
        "thresholds": [[-1.0, 0.0, 1.0], [], [1.0, float(np.nextafter(1.0, 2.0))]]
        + [[] for _ in FEATURES[3:]],
    }


def test_edges_constants_extremes_and_empty_inputs():
    bins = fake_bins()
    for j, edges in enumerate(bins["thresholds"]):
        q = np.array(edges)
        values = np.r_[
            -np.finfo(float).max,
            np.nextafter(q, -np.inf),
            q,
            np.nextafter(q, np.inf),
            np.finfo(float).max,
        ]
        x = np.zeros((len(values), len(FEATURES)))
        x[:, j] = values
        actual = encode_bins(bins, "reduce", x)
        np.testing.assert_array_equal(actual[:, j], (values[:, None] > q).sum(axis=1))
        np.testing.assert_array_equal(actual, reference_encode(bins, x))
    assert encode_bins(bins, "reduce", np.empty((0, len(FEATURES)))).shape == (0, len(FEATURES))


@pytest.mark.parametrize(
    "damage",
    [
        "action",
        "format",
        "feature_order",
        "field",
        "rows",
        "hash",
        "nan",
        "inf",
        "reverse",
        "duplicate",
        "too_many",
        "bool",
    ],
)
def test_damaged_binning_rejected(damage):
    b = fake_bins()
    if damage == "action":
        b["action"] = "increase"
    elif damage == "format":
        b["format"] = "other"
    elif damage == "feature_order":
        b["features"] = b["features"][::-1]
    elif damage == "field":
        b["future"] = True
    elif damage == "rows":
        b["rows"] = True
    elif damage == "hash":
        b["training_sha256"] = "g" * 64
    elif damage == "nan":
        b["thresholds"][0][0] = np.nan
    elif damage == "inf":
        b["thresholds"][0][-1] = np.inf
    elif damage == "reverse":
        b["thresholds"][0] = [1.0, 0.0]
    elif damage == "duplicate":
        b["thresholds"][0] = [1.0, 1.0]
    elif damage == "too_many":
        b["thresholds"][0] = list(range(255))
    else:
        b["thresholds"][0] = [False, True]
    with pytest.raises(ValueError):
        validate_bins(b, "reduce")


@pytest.mark.parametrize("damage", ["nan", "inf", "shape", "one_dim"])
def test_bad_encoding_input_rejected(damage):
    x = np.zeros((5, len(FEATURES)))
    if damage == "nan":
        x[0, 0] = np.nan
    elif damage == "inf":
        x[0, 0] = np.inf
    elif damage == "shape":
        x = x[:, :-1]
    else:
        x = x[0]
    with pytest.raises(ValueError):
        encode_bins(fake_bins(), "reduce", x)


def test_independent_full_refits_and_three_period_predictions(weighted_study, fixed_study):
    rows, native, old, _, weighted, ledger, details, _, _, _ = weighted_study
    check = verify_fixed_refits(rows, old[0], weighted[0], ledger, details, fixed_study)
    assert check["original_trees"] == check["control_trees"] == check["weighted_trees"] == 512
    assert check["boundary_probes"] > 0 and check["weighted_linear_refits"] == 2
    models, bins, controls, encoded, _ = fixed_study
    fit = [models, *calibrate_tail_models(rows["calibration"], models)]
    calibration = quiet_predictions(rows["calibration"], native["calibration"], *fit[:3], fit[4])
    verify_quiet_calibration(calibration, *fit[1:])
    state = fit_state_calibration(calibration)
    verify_state_calibration(calibration, *state)
    for phase, frame in rows.items():
        raw = quiet_predictions(frame, native[phase], *fit[:3], fit[4])
        verify_quiet_predictions(frame, native[phase], raw, *fit[:3], fit[4])
        candidate, control = state_predictions(raw, *state[:3])
        verify_state_predictions(raw, candidate, control, *state[:3])
    for action in ["reduce", "increase"]:
        assert lift_thresholds(controls[action], bins[action], action) == old[0][action]["tree"]
        assert models[action]["linear"] == weighted[0][action]["linear"]
        verify_projection(encoded[action], models[action]["tree"], bins[action])


def test_42_gates_and_identity(weighted_study, fixed_study):
    rows, native, _, previous, _, _, _, _, _, _ = weighted_study
    fit = [fixed_study[0], *calibrate_tail_models(rows["calibration"], fixed_study[0])]
    raw = {p: quiet_predictions(f, native[p], *fit[:3], fit[4]) for p, f in rows.items()}
    state = fit_state_calibration(raw["calibration"])
    new = {p: state_predictions(f, *state[:3])[0] for p, f in raw.items()}
    pair = {p: {"v127": previous[p], "v141": new[p]} for p in rows}
    cuts = fit_paired_budget(rows["calibration"], pair["calibration"], "v127")
    budget, _ = evaluate_paired_budget(rows["diagnosis"], pair["diagnosis"], cuts)
    pre = pre_limit_metrics(rows["diagnosis"], pair["diagnosis"])
    candidate, old = evaluate_quiet(new["diagnosis"]), evaluate_quiet(previous["diagnosis"])
    fixed = {"exit": True, "recent_reduce": True, "recent_increase": True}
    decision = fixed_bin_admission(candidate, old, fixed, pre, budget)
    verify_fixed_admission(candidate, old, fixed, pre, budget, decision)
    assert len(decision["checks"]) == 42
    for key in decision["checks"]:
        broken = deepcopy(decision)
        broken["checks"][key] = not broken["checks"][key]
        with pytest.raises(AssertionError):
            verify_fixed_admission(candidate, old, fixed, pre, budget, broken)


def test_future_data_cannot_change_training_or_calibration(weighted_study, fixed_study):
    rows, _, old, _, weighted, ledger, details, _, _, _ = weighted_study
    changed = {p: f.copy() for p, f in rows.items()}
    for phase in ["calibration", "diagnosis"]:
        changed[phase]["y_reduce"] = 1 - changed[phase].y_reduce
        changed[phase][FEATURES[0]] += 100
    assert fit_fixed_bins(changed["training"], old[0], weighted[0], ledger, details) == fixed_study
    before = calibrate_tail_models(rows["calibration"], fixed_study[0])
    changed["diagnosis"]["y_increase"] = 1 - changed["diagnosis"].y_increase
    assert calibrate_tail_models(rows["calibration"], fixed_study[0]) == before
    with pytest.raises(ValueError):
        fit_fixed_bins(rows["calibration"], old[0], weighted[0], ledger, details)


@pytest.mark.parametrize("damage", ["weights", "context", "parent", "tree", "projection", "source"])
def test_context_model_and_projection_corruption(weighted_study, fixed_study, damage):
    rows, _, old, _, weighted, ledger, details, _, _, _ = weighted_study
    parents = deepcopy(old[0])
    altered = ledger.copy()
    broken = deepcopy(fixed_study)
    if damage == "weights":
        mask = altered.reduce_weight.notna()
        altered.loc[mask, "reduce_weight"] *= 1.01
    elif damage == "context":
        altered.loc[0, "end"] += pd.Timedelta(minutes=1)
    elif damage == "parent":
        parents["reduce"]["scope"] = "all"
    elif damage == "tree":
        tree = broken[3]["reduce"]["models"][0]["trees"][0]
        i = next(i for i, f in enumerate(tree["feature"]) if f >= 0)
        tree["threshold"][i] = 300
        with pytest.raises(ValueError):
            lift_thresholds(broken[3]["reduce"], broken[1]["reduce"], "reduce")
        return
    elif damage == "projection":
        tree = broken[0]["reduce"]["tree"]["models"][0]["trees"][0]
        i = next(i for i, f in enumerate(tree["feature"]) if f >= 0)
        tree["threshold"][i] += 1
        with pytest.raises(AssertionError):
            verify_projection(broken[3]["reduce"], broken[0]["reduce"]["tree"], broken[1]["reduce"])
        return
    else:
        broken[1]["reduce"]["training_sha256"] = "0" * 64
        with pytest.raises(AssertionError):
            verify_fixed_refits(rows, old[0], weighted[0], ledger, details, broken)
        return
    with pytest.raises((ValueError, AssertionError)):
        fit_fixed_bins(rows["training"], parents, weighted[0], altered, details)


@pytest.mark.parametrize("damage", [None, "incomplete", "accepted", "source", "exit", "hash"])
def test_fixed_source_requires_verified_weighted_parent(tmp_path, monkeypatch, damage):
    root = tmp_path / "parent"
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
    flags = {k: False for k in ["new_accounts_executed", "profitability_accepted"]}
    summary.update(flags, tail_weighted_admitted=False, new_trees=512, new_linear_models=2)
    if damage == "incomplete":
        summary["independent_verification_complete"] = False
    if damage == "accepted":
        summary["tail_weighted_admitted"] = True
    settings = {
        "reference": "prior.json",
        "reference_sha256": "a" * 64,
        "sources": {"teacher": "bound"},
        "features": FEATURES,
        "models": ["v127", "v140"],
        "weight_rule": WEIGHT_RULE,
    }
    save_json(root / "summary.json", summary)
    save_json(root / "manifest.json", {"settings": settings})
    seal(root)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "complete": True,
            "run": str(root),
            "files_sha256": sha256(root / "files.json"),
            "summary_sha256": sha256(root / "summary.json"),
            "process_exit_code": False if damage == "exit" else 0,
            "new_models_fitted": True,
            **flags,
        },
    )
    monkeypatch.setattr(
        learning,
        "checked_tail_reference",
        lambda *_: {"teacher": "wrong" if damage == "source" else "bound"},
    )
    digest = "b" * 64 if damage == "hash" else sha256(proof)
    if damage is None:
        assert checked_fixed_reference(proof, digest)["teacher"] == "bound"
    else:
        with pytest.raises(ValueError):
            checked_fixed_reference(proof, digest)


def test_editing_returned_metadata_does_not_change_other_models(fixed_study):
    bins = fixed_study[1]
    original = FEATURES.copy()
    other = bins["increase"]["features"].copy()
    try:
        bins["reduce"]["features"].reverse()
        assert FEATURES == original
        assert bins["increase"]["features"] == other
        with pytest.raises(ValueError):
            validate_bins(bins["reduce"], "reduce")
    finally:
        FEATURES[:] = original
        bins["reduce"]["features"] = original.copy()
        bins["increase"]["features"] = other.copy()
