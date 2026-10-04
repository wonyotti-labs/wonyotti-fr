import json

import numpy as np
import pandas as pd
import pytest
from test_close_utility import examples
from test_continuation_diagnostics import fixture
from test_first_close_margin_diagnostics import margin_fixture
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_utility import UtilityCloseModel
from wonyotti_fr.close_utility_diagnostics import (
    evaluate_utility,
    fit_utility,
    reproduce_margin,
    run_close_utility_diagnosis,
)
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_diagnostics import first_close_positions
from wonyotti_fr.first_close_margin_diagnostics import run_first_close_margin_diagnosis
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis


def test_fit_keeps_all_cost_rows_and_cannot_use_diagnosis_targets_or_changed_features():
    x, y, w = examples()
    frame = pd.DataFrame(x, columns=UtilityCloseModel.features)
    frame["decision_time"] = pd.date_range("2021-01-01", periods=len(x), freq="5min", tz="UTC")
    frame["position_entry_time"] = frame.decision_time - pd.Timedelta(minutes=1)
    frame["close_advantage_bps"] = y
    weights = frame[["decision_time", "position_entry_time"]].assign(sample_weight=w)
    diagnosis = frame.iloc[:200].copy()
    model, support, ledger = fit_utility(frame, weights, diagnosis)
    diagnosis["close_advantage_bps"] = -1e10
    diagnosis["ret_5m"] *= -100
    other, other_support, other_ledger = fit_utility(frame, weights, diagnosis)
    assert model.to_dict() == other.to_dict()
    assert support["training_constant_score"] == other_support["training_constant_score"]
    pd.testing.assert_frame_equal(ledger, other_ledger, check_exact=True)
    pd.testing.assert_frame_equal(
        ledger[["decision_time", "position_entry_time"]],
        weights.drop(columns="sample_weight"),
        check_exact=True,
    )
    np.testing.assert_array_equal(ledger.original_weight, w)
    assert len(ledger) == len(frame) and ledger.reason.eq("zero_effect").sum() == 200
    weights.loc[0, "position_entry_time"] -= pd.Timedelta(minutes=1)
    with pytest.raises(AssertionError):
        fit_utility(frame, weights, diagnosis)


@pytest.mark.parametrize("margin_passed", [False, True])
def test_evaluation_preserves_all_old_scores_and_handles_both_previous_margin_branches(
    tmp_path, margin_passed
):
    cal, _ = margin_fixture(tmp_path / "capacity")
    diagnosis = cal.copy()
    for name in ["position_entry_time", "decision_time", "label_end", "continue_end"]:
        diagnosis[name] += pd.Timedelta(days=92)
    weekly, margin, out = (tmp_path / name for name in ["weekly", "margin", "out"])
    for folder in [weekly, margin, out]:
        folder.mkdir()
    diagnosis.to_parquet(weekly / "diagnosis_used.parquet", index=False)
    keys = [
        "decision_time",
        "position_entry_time",
        "label_end",
        "direction",
        "original_intent",
        "close_advantage_bps",
    ]
    old = diagnosis[keys].assign(sample_weight=1.0)
    for name in [
        "ridge",
        "boosted",
        "constant",
        "economic",
        "continuation",
        "exposure",
        "capacity",
        "weekly",
    ]:
        old["predicted_" + name] = 0.1
    old.to_parquet(weekly / "predictions.parquet", index=False)
    for name in ["continuation", "weekly"]:
        first_close_positions(diagnosis.assign(prediction=0.1), "prediction").to_parquet(
            weekly / f"positions_{name}.parquet", index=False
        )
    save_json(
        margin / "selection.json",
        {"selection_passed": margin_passed, "chosen_margin_bps": 0.5 if margin_passed else None},
    )
    prior = old.copy()
    if margin_passed:
        prior["predicted_half"] = diagnosis.ret_5m
        prior.to_parquet(margin / "predictions.parquet", index=False)
        for name, threshold in [("margin", 0.5), ("half_zero", 0.0)]:
            first_close_positions(
                diagnosis.assign(prediction=diagnosis.ret_5m), "prediction", margin_bps=threshold
            ).to_parquet(margin / f"positions_{name}.parquet", index=False)

    class FixedCostScore:
        features = UtilityCloseModel.features

        def probabilities(self, values):
            return np.where(values[:, 0] > 1.0, 0.8, 0.5)[:, None]

    decision = evaluate_utility(weekly, margin, FixedCostScore(), 0.4, out)
    assert len(decision["checks"]) == 14 and not decision["trading_returns_evaluated"]
    saved = pd.read_parquet(out / "predictions.parquet")
    pd.testing.assert_frame_equal(
        saved.drop(columns=["utility_score", "training_constant_score"]), prior, check_exact=True
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["utility"]["selected"] == 80 and metrics["utility"]["selected_positions"] == 40
    assert metrics["training_constant"]["selected"] == 0
    assert not any("mse" in field for value in metrics.values() for field in value)
    positions = pd.read_parquet(out / "positions_utility.parquet")
    assert positions.first_cost_score.eq(0.8).all() and "first_prediction_bps" not in positions
    np.testing.assert_allclose(positions.first_effect_common_bps, 3.0, rtol=0, atol=1e-12)
    assert (out / "positions_margin.parquet").exists() == margin_passed
    intervals = json.loads((out / "block_intervals.json").read_text())
    assert intervals["seed"] == 63 and intervals["replicates"] == 1000


def test_full_pipeline_preserves_original_margin_outputs_and_rejects_resigned_score_tampering(
    tmp_path, monkeypatch
):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path / "continuation")
    exposure = run_exposure_close_diagnosis(continuation, tmp_path / "exposure")
    capacity = run_close_capacity_diagnosis(exposure, tmp_path / "capacity")
    weekly = run_weekly_close_diagnosis(capacity, tmp_path / "weekly")
    margin = run_first_close_margin_diagnosis(weekly, tmp_path / "margin")
    out = run_close_utility_diagnosis(margin, tmp_path / "utility")
    summary = json.loads((out / "summary.json").read_text())
    assert (
        summary["complete"]
        and summary["all_previous_outputs_reproduced"]
        and summary["zero_effect_rows_preserved"]
    )
    assert not summary["profitability_accepted"] and len(summary["checks"]) == 14
    for name in [
        "training_used",
        "training_weights",
        "diagnosis_used",
        "diagnosis_weights",
        "exclusion_ledger",
    ]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / f"{name}.parquet"),
            pd.read_parquet(weekly / f"{name}.parquet"),
            check_exact=True,
        )
    pd.testing.assert_frame_equal(
        pd.read_parquet(out / "previous_predictions.parquet"),
        pd.read_parquet(margin / "previous_predictions.parquet"),
        check_exact=True,
    )
    original = pd.read_parquet(margin / "calibration_predictions.parquet")
    original.loc[0, "predicted_half"] += 1.0
    original.to_parquet(margin / "calibration_predictions.parquet", index=False)
    hashes = json.loads((margin / "files.json").read_text())
    hashes["calibration_predictions.parquet"] = sha256(margin / "calibration_predictions.parquet")
    save_json(margin / "files.json", hashes)
    with pytest.raises(AssertionError):
        reproduce_margin(margin, tmp_path / "tampered")
