from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study
from test_quiet_state_calibration import calibrated as calibrated

from wonyotti_fr.management_experts import (
    expert_admission,
    expert_predictions,
    expert_support,
    fit_experts,
    load_expert,
)
from wonyotti_fr.management_experts_learning import checked_expert_reference
from wonyotti_fr.management_experts_reference import (
    verify_expert_admission,
    verify_expert_calibration,
    verify_expert_predictions,
    verify_expert_refits,
    verify_expert_support,
)
from wonyotti_fr.quiet_management import evaluate_quiet, groups
from wonyotti_fr.quiet_management_reference import verify_quiet_evaluation
from wonyotti_fr.quiet_state_calibration import state_predictions


@pytest.fixture(scope="module")
def prepared(study, calibrated):
    rows = study[0]
    original, fitted = calibrated
    previous = {phase: state_predictions(frame, *fitted[:3])[0] for phase, frame in original.items()}
    experts = fit_experts(rows["training"], rows["calibration"])
    return rows, previous, experts


def test_expert_support_and_independent_models(prepared):
    rows, _, (models, _, _, _, _) = prepared
    support = expert_support(rows)
    verify_expert_support(rows, support)
    assert verify_expert_refits(rows, models)["independent_histogram_trees"] == 1024
    for action in ["reduce", "increase"]:
        assert models[action]["first"] != models[action]["restart"]


def test_all_predictions_calibration_metrics_and_admission(prepared):
    rows, previous, (models, offsets, thresholds, details, constants) = prepared
    for phase, source in rows.items():
        saved = expert_predictions(source, previous[phase], models, offsets, thresholds, constants)
        unchanged = verify_expert_predictions(source, previous[phase], saved, models, offsets, thresholds, constants)
        evaluated = evaluate_quiet(saved)
        verify_quiet_evaluation(saved, evaluated)
        if phase == "calibration":
            verify_expert_calibration(saved, offsets, thresholds, details, constants)
        if phase == "diagnosis":
            old = evaluate_quiet(previous[phase])
            decision = expert_admission(evaluated, old, unchanged)
            assert len(decision["checks"]) == 31
            verify_expert_admission(evaluated, old, unchanged, decision)


def test_future_labels_do_not_change_expert_fits(prepared):
    rows, _, fitted = prepared
    changed = {p: f.copy() for p, f in rows.items()}
    for action in ["exit", "reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_experts(changed["training"], changed["calibration"]) == fitted


@pytest.mark.parametrize("kind", ["state", "action", "linear_action", "settings", "extra"])
def test_wrong_expert_binding_rejected(prepared, kind):
    models = deepcopy(prepared[2][0])
    raw = models["reduce"]["first"]
    if kind == "state":
        raw["state"] = "restart"
    elif kind == "action":
        raw["action"] = "increase"
    elif kind == "linear_action":
        raw["linear"]["action"] = "increase"
    elif kind == "settings":
        raw["tree"]["settings"]["max_iter"] = 64
    else:
        raw["ignored"] = True
    with pytest.raises(ValueError):
        load_expert(models, "reduce", "first")


@pytest.mark.parametrize("kind", ["restart_small", "label_support", "overlap", "bad_count", "nonfinite"])
def test_invalid_expert_support_rejected(prepared, kind):
    changed = {p: f.copy() for p, f in prepared[0].items()}
    mask = groups(changed["training"], "reduce")["restart"]
    if kind == "restart_small":
        indices = np.flatnonzero(mask)[50:]
        changed["training"].loc[indices, "past_reduce_log_count_15m"] = np.log1p(1.)
    elif kind == "label_support":
        changed["training"].loc[mask, "y_reduce"] = 0
    elif kind == "overlap":
        changed["diagnosis"]["episode_id"] = 1
    elif kind == "bad_count":
        changed["calibration"]["past_reduce_log_count_15m"] = -.1
    else:
        changed["training"]["ret_5m"] = np.nan
    with pytest.raises(ValueError):
        expert_support(changed)


@pytest.mark.parametrize("column", ["exit_score", "reduce_baseline_request", "reduce_tree_raw_score",
                                    "reduce_tree_score", "increase_linear_score", "increase_constant_score"])
def test_corrupt_expert_predictions_rejected(prepared, column):
    rows, previous, (models, offsets, thresholds, _, constants) = prepared
    source, old = rows["diagnosis"], previous["diagnosis"]
    saved = expert_predictions(source, old, models, offsets, thresholds, constants)
    if column.endswith("request"):
        saved.loc[0, column] = not saved.loc[0, column]
    else:
        saved.loc[0, column] += .01
    with pytest.raises(AssertionError):
        verify_expert_predictions(source, old, saved, models, offsets, thresholds, constants)


def test_recent_rows_stay_exact_and_empty_state_preserved(prepared):
    rows, previous, (models, offsets, thresholds, _, constants) = prepared
    source = rows["diagnosis"]
    mask = groups(source, "reduce")["first"]
    part, old = source.loc[mask], previous["diagnosis"].loc[mask]
    saved = expert_predictions(part, old, models, offsets, thresholds, constants)
    verify_expert_predictions(part, old, saved, models, offsets, thresholds, constants)
    metrics = evaluate_quiet(saved)
    verify_quiet_evaluation(saved, metrics)
    assert metrics["reduce"]["restart"]["tree"] is None
    source = source.copy()
    old = previous["diagnosis"].copy()
    for action in ["reduce", "increase"]:
        for frame in [source, old]:
            frame[f"past_{action}_exists"] = 1.
            frame[f"past_{action}_log_count_15m"] = np.log1p(1.)
    saved = expert_predictions(source, old, models, offsets, thresholds, constants)
    pd.testing.assert_frame_equal(saved, old, check_exact=True)


def test_damaged_state_constant_rejected(prepared):
    rows, previous, (models, offsets, thresholds, details, constants) = prepared
    saved = expert_predictions(rows["calibration"], previous["calibration"], models, offsets, thresholds, constants)
    constants = deepcopy(constants)
    constants["reduce"]["restart"] += .01
    with pytest.raises(AssertionError):
        verify_expert_calibration(saved, offsets, thresholds, details, constants)


def test_bad_reference_digest_rejected(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text('{}')
    with pytest.raises(ValueError):
        checked_expert_reference(path, '0' * 64)
