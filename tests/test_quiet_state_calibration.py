from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_quiet_management import study as study

from wonyotti_fr.quiet_management import evaluate_quiet, groups, quiet_predictions
from wonyotti_fr.quiet_management_reference import verify_quiet_evaluation
from wonyotti_fr.quiet_state_calibration import (
    fit_state_calibration,
    state_admission,
    state_means,
    state_predictions,
    validate_state_support,
)
from wonyotti_fr.quiet_state_learning import checked_state_reference
from wonyotti_fr.quiet_state_reference import (
    verify_state_admission,
    verify_state_calibration,
    verify_state_means,
    verify_state_predictions,
)


@pytest.fixture(scope="module")
def calibrated(study):
    rows, baselines, (models, offsets, thresholds, _, constants) = study
    originals = {phase: quiet_predictions(f, baselines[phase], models, offsets, thresholds, constants)
                 for phase, f in rows.items()}
    fitted = fit_state_calibration(originals["calibration"])
    return originals, fitted


def test_calibration_probabilities_requests_and_decision(calibrated):
    originals, (offsets, thresholds, controls, details, support) = calibrated
    verify_state_calibration(originals["calibration"], offsets, thresholds, controls, details, support)
    for phase, original in originals.items():
        candidate, control = state_predictions(original, offsets, thresholds, controls)
        unchanged = verify_state_predictions(original, candidate, control, offsets, thresholds, controls)
        verify_state_means(original, candidate, state_means(original, candidate))
        for frame in [candidate, control]:
            verify_quiet_evaluation(frame, evaluate_quiet(frame))
        if phase == "diagnosis":
            new, old = evaluate_quiet(candidate), evaluate_quiet(original)
            decision = state_admission(new, old, unchanged)
            assert len(decision["checks"]) == 31
            verify_state_admission(new, old, unchanged, decision)
            changed = deepcopy(decision)
            key = "reduce_shared_log_loss_preserved"
            changed["checks"][key] = not changed["checks"][key]
            with pytest.raises(AssertionError):
                verify_state_admission(new, old, unchanged, changed)


def test_future_labels_do_not_change_offsets_or_thresholds(calibrated):
    originals, fitted = calibrated
    changed = {p: f.copy() for p, f in originals.items()}
    for action in ["reduce", "increase"]:
        changed["diagnosis"]["y_" + action] = 1 - changed["diagnosis"]["y_" + action]
    assert fit_state_calibration(changed["calibration"]) == fitted


@pytest.mark.parametrize("change", ["zero_positive", "zero_negative", "small", "bad_label", "raw_nan", "score_range", "count", "flag"])
def test_state_support_errors_rejected(calibrated, change):
    originals, _ = calibrated
    frame = originals["calibration"].copy()
    mask = groups(frame, "reduce")["restart"]
    if change == "zero_positive":
        frame.loc[mask, "y_reduce"] = 0
    elif change == "zero_negative":
        frame.loc[mask, "y_reduce"] = 1
    elif change == "small":
        frame = frame.iloc[:90]
    elif change == "bad_label":
        frame.loc[mask, "y_reduce"] = 2
    elif change == "raw_nan":
        frame.loc[mask, "reduce_tree_raw_score"] = np.nan
    elif change == "score_range":
        frame.loc[mask, "reduce_linear_score"] = 2.
    elif change == "count":
        frame.loc[mask, "past_reduce_log_count_15m"] = -1.
    else:
        frame.loc[mask, "past_reduce_exists"] = 2.
    with pytest.raises(ValueError):
        validate_state_support(frame)


@pytest.mark.parametrize("column", ["exit_score", "reduce_tree_raw_score", "reduce_tree_score", "reduce_tree_request",
                                    "increase_linear_request", "reduce_baseline_score", "past_reduce_exists"])
def test_corrupt_candidate_rejected(calibrated, column):
    originals, (offsets, thresholds, controls, _, _) = calibrated
    original = originals["diagnosis"]
    candidate, control = state_predictions(original, offsets, thresholds, controls)
    if column.endswith("request"):
        candidate.loc[0, column] = not candidate.loc[0, column]
    else:
        candidate.loc[0, column] += .01
    with pytest.raises(AssertionError):
        verify_state_predictions(original, candidate, control, offsets, thresholds, controls)


def test_recent_and_empty_states_unchanged(calibrated):
    originals, (offsets, thresholds, controls, _, _) = calibrated
    original = originals["diagnosis"].copy()
    original["past_reduce_exists"] = 1.
    original["past_increase_exists"] = 1.
    original["past_reduce_log_count_15m"] = np.log1p(1)
    original["past_increase_log_count_15m"] = np.log1p(1)
    candidate, control = state_predictions(original, offsets, thresholds, controls)
    pd.testing.assert_frame_equal(candidate, original, check_exact=True)
    pd.testing.assert_frame_equal(control, original, check_exact=True)
    verify_state_predictions(original, candidate, control, offsets, thresholds, controls)
    verify_state_means(original, candidate, state_means(original, candidate))


@pytest.mark.parametrize("kind", ["offset", "threshold", "control"])
def test_damaged_calibration_parameters_rejected(calibrated, kind):
    originals, fitted = calibrated
    offsets, thresholds, controls, details, support = deepcopy(fitted)
    target = {"offset": offsets, "threshold": thresholds, "control": controls}[kind]
    target["reduce"]["first"]["tree"] += .1
    with pytest.raises(AssertionError):
        verify_state_calibration(originals["calibration"], offsets, thresholds, controls, details, support)


def test_reference_digest_rejected_before_loading(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text('{}')
    with pytest.raises(ValueError):
        checked_state_reference(path, '0' * 64)
