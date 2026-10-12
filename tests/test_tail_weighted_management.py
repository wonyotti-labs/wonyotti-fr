from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_error_budget import population
from test_quiet_management import study as study

import wonyotti_fr.tail_weighted_learning as learning
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.error_budget import evaluate_budget, fit_budget_cutoffs
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.paired_error_budget import (
    evaluate_paired_budget,
    fit_paired_budget,
    verify_paired_budget,
    verify_paired_cutoffs,
)
from wonyotti_fr.quiet_management import (
    FEATURES,
    evaluate_quiet,
    fit_quiet_models,
    quiet_predictions,
)
from wonyotti_fr.quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
)
from wonyotti_fr.quiet_state_calibration import fit_state_calibration, state_predictions
from wonyotti_fr.quiet_state_reference import verify_state_calibration, verify_state_predictions
from wonyotti_fr.tail_weighted_learning import checked_tail_reference, pre_limit_metrics
from wonyotti_fr.tail_weighted_management import (
    calibrate_tail_models,
    fit_tail_models,
    tail_admission,
    tail_weights,
    training_weight_groups,
    validate_tail_models,
)
from wonyotti_fr.tail_weighted_reference import (
    verify_tail_admission,
    verify_tail_refits,
    verify_tail_weights,
    verify_weight_groups,
)


@pytest.mark.parametrize("negatives,rank", [(3999, 40), (4000, 40), (4001, 41)])
def test_integer_ceiling_and_positive_scores_do_not_set_boundary(negatives, rank):
    y = np.r_[np.zeros(negatives, int), np.ones(20, int)]
    p = np.r_[np.linspace(0, 1, negatives), np.full(20, 0.5)]
    w, hard, info = tail_weights(y, p)
    assert info["rank"] == rank and hard.sum() == rank
    assert info["boundary_score"] == p[negatives - rank]
    verify_tail_weights(y, p, w, hard, info)
    changed = p.copy()
    changed[y == 1] = np.linspace(0, 1, 20)
    w2, hard2, info2 = tail_weights(y, changed)
    np.testing.assert_array_equal(w, w2)
    np.testing.assert_array_equal(hard, hard2)
    assert info == info2 and (w > 0).all()
    assert np.isclose(w.mean(), 1, atol=1e-12, rtol=0)
    assert np.isclose(w[hard][0] / w[(y == 0) & ~hard][0], 10)


@pytest.mark.parametrize("p", [0.0, 1.0, 0.5])
def test_ties_include_all_negatives_and_leave_no_easy_group(p):
    y = np.r_[np.zeros(2000, int), np.ones(20, int)]
    scores = np.full(len(y), p)
    w, hard, info = tail_weights(y, scores)
    assert hard.sum() == info["boundary_ties"] == 2000
    np.testing.assert_array_equal(w, np.ones(len(y)))
    assert info["statistics"]["easy_negative"]["effective_rows"] is None
    verify_tail_weights(y, scores, w, hard, info)


def test_tie_block_at_boundary_not_partially_selected():
    y = np.r_[np.zeros(6000, int), np.ones(40, int)]
    p = np.r_[np.zeros(5900), np.ones(100), np.full(40, 0.2)]
    w, h, d = tail_weights(y, p)
    assert d["rank"] == 60 and h.sum() == 100
    verify_tail_weights(y, p, w, h, d)


@pytest.mark.parametrize("size", [5000, 200003])
def test_large_weight_sums_and_effective_support(size):
    y = (np.arange(size) % 97 == 0).astype(int)
    p = (np.arange(size) % 311) / 310
    w, h, d = tail_weights(y, p)
    verify_tail_weights(y, p, w, h, d)
    assert d["statistics"]["all"]["effective_rows"] >= 1000


@pytest.mark.parametrize(
    "damage", ["few", "single_class", "nan", "probability", "label", "shape", "ess"]
)
def test_invalid_weight_inputs_rejected(damage):
    y = np.r_[np.zeros(4000), np.ones(20)]
    p = np.linspace(0, 1, len(y))
    if damage == "few":
        y, p = y[:10], p[:10]
    elif damage == "single_class":
        y[:] = 0
    elif damage == "nan":
        p[0] = np.nan
    elif damage == "probability":
        p[0] = 1.1
    elif damage == "label":
        y[0] = 0.5
    elif damage == "shape":
        p = p[:, None]
    else:
        y = np.r_[np.zeros(1000), np.ones(20)]
        p = np.linspace(0, 1, len(y))
    with pytest.raises(ValueError):
        tail_weights(y, p)


@pytest.fixture(scope="module")
def weighted_study(study):
    source, native, _ = study
    rows = {p: f.copy() for p, f in source.items()}
    for frame in rows.values():
        frame["direction"] = np.where(frame.direction > 0, 1.0, -1.0)
        frame["adds"] = np.arange(len(frame)) % 9
        frame["adds_capped"] = np.minimum(frame.adds, 5)
    old_fit = fit_quiet_models(rows["training"], rows["calibration"])
    old_global = {
        p: quiet_predictions(f, native[p], *old_fit[:3], old_fit[4]) for p, f in rows.items()
    }
    old_state = fit_state_calibration(old_global["calibration"])
    previous = {p: state_predictions(f, *old_state[:3])[0] for p, f in old_global.items()}
    models, ledger, details = fit_tail_models(rows["training"], old_fit[0])
    fitted = [models, *calibrate_tail_models(rows["calibration"], models)]
    global_new = {
        p: quiet_predictions(f, native[p], *fitted[:3], fitted[4]) for p, f in rows.items()
    }
    state = fit_state_calibration(global_new["calibration"])
    predictions = {p: state_predictions(f, *state[:3])[0] for p, f in global_new.items()}
    return rows, native, old_fit, previous, fitted, ledger, details, global_new, state, predictions


@pytest.mark.parametrize("phase", ["training", "calibration"])
@pytest.mark.parametrize(
    "damage",
    [
        "fractional_label",
        "feature_nan",
        "reverse",
        "duplicate_time",
        "off_grid",
        "episode",
        "label_boundary",
        "naive_time",
    ],
)
def test_learning_rejects_damaged_context_before_cast(weighted_study, phase, damage):
    frame = weighted_study[0][phase].copy()
    if damage == "fractional_label":
        frame["y_reduce"] = frame.y_reduce.astype(float)
        frame.loc[0, "y_reduce"] = 0.5
    elif damage == "feature_nan":
        frame.loc[0, "direction"] = np.nan
    elif damage == "reverse":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif damage == "duplicate_time":
        frame.loc[1, "end"] = frame.loc[0, "end"]
    elif damage == "off_grid":
        frame.loc[0, "end"] += pd.Timedelta(seconds=1)
    elif damage == "episode":
        frame["episode_id"] = frame.episode_id.astype(float)
        frame.loc[0, "episode_id"] = 0.5
    elif damage == "label_boundary":
        frame.loc[len(frame) - 1, "label_end"] = pd.Timestamp(
            "2021-01-01" if phase == "training" else "2021-07-01", tz="UTC"
        )
    else:
        frame["end"] = frame.end.dt.tz_localize(None)
    with pytest.raises(ValueError):
        if phase == "training":
            fit_tail_models(frame, weighted_study[2][0])
        else:
            calibrate_tail_models(frame, weighted_study[4][0])


def test_actual_weighted_refits_calibration_and_all_predictions(weighted_study):
    rows, native, old_fit, _, fitted, ledger, details, global_new, state, predictions = (
        weighted_study
    )
    verified = verify_tail_refits(rows, old_fit[0], fitted[0], ledger, details)
    assert verified["independent_histogram_trees"] == 512
    assert verified["weighted_linear_refits"] == 2
    wgroups = training_weight_groups(rows["training"], ledger)
    verify_weight_groups(rows["training"], ledger, wgroups)
    verify_quiet_calibration(global_new["calibration"], *fitted[1:])
    verify_state_calibration(global_new["calibration"], *state)
    for phase, frame in rows.items():
        verify_quiet_predictions(frame, native[phase], global_new[phase], *fitted[:3], fitted[4])
        candidate, control = state_predictions(global_new[phase], *state[:3])
        verify_state_predictions(global_new[phase], candidate, control, *state[:3])
        pd.testing.assert_frame_equal(candidate, predictions[phase], check_exact=True)
        verify_quiet_evaluation(candidate, evaluate_quiet(candidate))


def test_budget_and_all_42_conditions_independently(weighted_study):
    rows, _, _, previous, _, _, _, _, _, predictions = weighted_study
    pair = {p: {"v127": previous[p], "v140": predictions[p]} for p in rows}
    cuts = fit_paired_budget(rows["calibration"], pair["calibration"], "v127")
    verify_paired_cutoffs(rows["calibration"], pair["calibration"], "v127", cuts)
    all_results = {}
    for phase, frame in rows.items():
        actual, ledger = evaluate_paired_budget(frame, pair[phase], cuts)
        verify_paired_budget(frame, pair[phase], cuts, actual, ledger)
        all_results[phase] = actual
        assert all("v137" not in key for key in ledger)
    candidate, old = evaluate_quiet(predictions["diagnosis"]), evaluate_quiet(previous["diagnosis"])
    fixed = {"exit": True, "recent_reduce": True, "recent_increase": True}
    pre = pre_limit_metrics(rows["diagnosis"], pair["diagnosis"])
    decision = tail_admission(candidate, old, fixed, pre, all_results["diagnosis"])
    verify_tail_admission(candidate, old, fixed, pre, all_results["diagnosis"], decision)
    assert len(decision["checks"]) == 42
    for key in decision["checks"]:
        broken = deepcopy(decision)
        broken["checks"][key] = not broken["checks"][key]
        with pytest.raises(AssertionError):
            verify_tail_admission(candidate, old, fixed, pre, all_results["diagnosis"], broken)


def test_general_budget_exactly_preserves_v138_and_new_identity(tmp_path):
    frame, pair = population()
    old = fit_budget_cutoffs(frame, pair)
    actual = fit_paired_budget(frame, pair, "v127")
    assert actual == old
    a, al = evaluate_paired_budget(frame, pair, actual)
    b, bl = evaluate_budget(frame, pair, old)
    assert a == b
    pd.testing.assert_frame_equal(al, bl, check_exact=True)
    renamed = {"v127": pair["v127"], "v140": pair["v137"]}
    cuts = fit_paired_budget(frame, renamed, "v127")
    verify_paired_cutoffs(frame, renamed, "v127", cuts)
    stats, ledger = evaluate_paired_budget(frame, renamed, cuts)
    verify_paired_budget(frame, renamed, cuts, stats, ledger)
    ledger.to_parquet(tmp_path / "budget.parquet", index=False)
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "budget.parquet"),
        ledger.reset_index(drop=True),
        check_exact=True,
    )


def test_unsupported_and_empty_budget_groups():
    frame, pair = population()
    frame["adds"] = frame.adds.clip(upper=4)
    frame["adds_capped"] = frame.adds
    pair = {"v127": pair["v127"], "v140": pair["v137"]}
    cuts = fit_paired_budget(frame, pair, "v127")
    for f, p in [(frame, pair), (frame.iloc[:0], {k: v.iloc[:0] for k, v in pair.items()})]:
        stats, ledger = evaluate_paired_budget(f, p, cuts)
        verify_paired_budget(f, p, cuts, stats, ledger)
        for key, value in stats.items():
            if key.endswith("__capped"):
                assert all(v["budget_matched"] is None for v in value["models"].values())


@pytest.mark.parametrize(
    "damage", ["model", "budget", "threshold", "support", "context", "request"]
)
def test_paired_budget_rejects_corrupted_binding(damage):
    frame, pair = population()
    pair = {"v127": pair["v127"], "v140": pair["v137"]}
    cuts = fit_paired_budget(frame, pair, "v127")
    first = next(iter(cuts))
    if damage == "model":
        pair["v137"] = pair.pop("v140")
    elif damage == "budget":
        cuts[first]["budget"] += 1
    elif damage == "threshold":
        cuts[first]["models"]["v140"]["threshold"] = -1
    elif damage == "support":
        cuts[first]["support"]["rows"] += 1
    elif damage == "context":
        pair["v140"].loc[frame.index[0], "episode_id"] += 1
    else:
        pair["v140"]["increase_tree_request"] = 1
    with pytest.raises((ValueError, AssertionError)):
        evaluate_paired_budget(frame, pair, cuts)


@pytest.mark.parametrize("damage", ["rule", "scope", "linear_action", "tree_action"])
def test_weighted_model_metadata_rejected(weighted_study, damage):
    models = deepcopy(weighted_study[4][0])
    if damage == "rule":
        models["reduce"]["weight_rule"] = "unweighted"
    elif damage == "scope":
        models["reduce"]["scope"] = "all"
    elif damage == "linear_action":
        models["reduce"]["linear"]["action"] = "increase"
    else:
        models["reduce"]["tree"] = models["increase"]["tree"]
    with pytest.raises(ValueError):
        validate_tail_models(models)


def test_future_data_not_accepted_by_training_and_no_recursive_parent(weighted_study):
    rows, _, old_fit, _, fitted, ledger, details, _, state, _ = weighted_study
    changed = {p: f.copy() for p, f in rows.items()}
    for phase in ["calibration", "diagnosis"]:
        changed[phase][FEATURES[0]] += 10
        for action in ["reduce", "increase"]:
            changed[phase]["y_" + action] = 1 - changed[phase]["y_" + action]
    models, w, d = fit_tail_models(changed["training"], old_fit[0])
    assert models == fitted[0] and d == details
    pd.testing.assert_frame_equal(w, ledger, check_exact=True)
    assert fit_state_calibration(weighted_study[7]["calibration"]) == state
    with pytest.raises(ValueError):
        fit_tail_models(rows["diagnosis"], old_fit[0])
    with pytest.raises(ValueError):
        fit_tail_models(rows["training"], models)
    with pytest.raises(ValueError):
        calibrate_tail_models(rows["diagnosis"], models)


@pytest.mark.parametrize("damage", ["weight", "flag", "score"])
def test_independent_weights_reject_corruption(damage):
    y = np.r_[np.zeros(4000, int), np.ones(20, int)]
    p = np.linspace(0, 1, len(y))
    w, h, d = tail_weights(y, p)
    if damage == "weight":
        w[0] *= 2
    elif damage == "flag":
        h[0] = True
    else:
        d["boundary_score"] = np.nextafter(d["boundary_score"], np.inf)
    with pytest.raises(AssertionError):
        verify_tail_weights(y, p, w, h, d)


@pytest.mark.parametrize("damage", [None, "incomplete", "accepted", "source", "exit", "hash"])
def test_tail_source_requires_verified_temporal_parent(tmp_path, monkeypatch, damage):
    root = tmp_path / "parent"
    root.mkdir()
    summary = {
        k: True
        for k in ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"]
    }
    flags = {
        k: False for k in ["new_models_fitted", "new_accounts_executed", "profitability_accepted"]
    }
    summary.update(flags, original_candidate_admitted=False)
    if damage == "incomplete":
        summary["independent_verification_complete"] = False
    if damage == "accepted":
        summary["original_candidate_admitted"] = True
    settings = {
        "reference": "prior.json",
        "reference_sha256": "a" * 64,
        "sources": {"teacher": "bound"},
        "horizons": [1, 15],
        "matching": "same_episode_retained_group_forward_half_open_one_to_one",
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
            **flags,
        },
    )
    monkeypatch.setattr(
        learning,
        "checked_temporal_reference",
        lambda *_: (tmp_path / "budget", {"teacher": "wrong" if damage == "source" else "bound"}),
    )
    digest = "b" * 64 if damage == "hash" else sha256(proof)
    if damage is None:
        assert checked_tail_reference(proof, digest)["teacher"] == "bound"
    else:
        with pytest.raises(ValueError):
            checked_tail_reference(proof, digest)
