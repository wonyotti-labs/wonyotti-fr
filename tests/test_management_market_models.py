import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json
from wonyotti_fr.management_market import FEATURES, REPLACED, market_support
from wonyotti_fr.management_market_learning import checked_learning_reference
from wonyotti_fr.management_market_model_reference import (
    refit_market_models,
    verify_market_admission,
    verify_market_evaluation,
    verify_market_predictions,
    verify_offsets,
)
from wonyotti_fr.management_market_models import (
    fit_market_models,
    market_model_admission,
    market_predictions,
    validate_paired_rows,
)
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.paired_management_evaluation import choose_thresholds, evaluate


def frames():
    rng = np.random.default_rng(116)
    native, aligned = {}, {}
    for index, (phase, date, count) in enumerate([
        ("training", "2020-01-02", 1200), ("calibration", "2021-01-02", 400),
        ("diagnosis", "2021-07-02", 400),
    ]):
        frame = pd.DataFrame(rng.normal(size=(count, 50)), columns=FEATURES)
        frame["direction"] = np.where(np.arange(count) % 2, 1, -1)
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["entry_time"] = pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=1)
        frame["episode_id"] = index + 1
        for action, column in zip(ACTIONS, [0, 1, 2], strict=True):
            frame["y_" + action] = (frame[FEATURES[column]] + rng.normal(size=count) > 0).astype(int)
        native[phase] = frame
        target = frame.copy()
        target[REPLACED] = target[REPLACED] * 1.2 + rng.normal(scale=.2, size=(count, 36))
        aligned[phase] = target
    return native, aligned


@pytest.fixture(scope="module")
def fitted():
    native, aligned = frames()
    models, offsets, support = fit_market_models(native, aligned)
    return native, aligned, models, offsets, support


def test_independent_refits_scores_offsets_and_thresholds(fitted):
    native, aligned, models, offsets, _ = fitted
    proof = refit_market_models(native, aligned, models)
    assert proof["independent_histogram_trees"] == 384
    cy = native["calibration"][["y_" + a for a in ACTIONS]].to_numpy()
    constant = cy.mean(axis=0)
    scores = market_predictions(native["calibration"], aligned["calibration"], models, offsets,
                                models["native"], offsets["native"], constant)
    verify_market_predictions(native["calibration"], aligned["calibration"], models, offsets,
                              models["native"], offsets["native"], constant, scores)
    verify_offsets(scores, cy, offsets)
    thresholds, details = choose_thresholds(cy, scores)
    metrics, _ = evaluate(cy, scores, thresholds)
    verify_market_evaluation(cy, scores, thresholds, metrics, details)
    decision = market_model_admission(metrics)
    verify_market_admission(metrics, decision)
    scores["aligned_calibrated"][0, 0] += .01
    with pytest.raises(AssertionError):
        verify_market_predictions(native["calibration"], aligned["calibration"], models, offsets,
                                  models["native"], offsets["native"], constant, scores)


def test_diagnosis_changes_do_not_change_training_or_calibration(fitted):
    native, aligned, old_models, old_offsets, _ = fitted
    a, b = copy.deepcopy(native), copy.deepcopy(aligned)
    for rows in [a, b]:
        rows["diagnosis"][REPLACED] *= 3
        for action in ACTIONS:
            rows["diagnosis"]["y_" + action] = 1 - rows["diagnosis"]["y_" + action]
    models, offsets, _ = fit_market_models(a, b)
    for name in models:
        assert models[name].to_dict() == old_models[name].to_dict()
        assert offsets[name].to_dict() == old_offsets[name].to_dict()


@pytest.mark.parametrize("kind", ["nonfinite", "label", "state", "episode", "time", "support", "missing_split"])
def test_invalid_pairs_fail(kind):
    native, aligned = frames()
    if kind == "nonfinite":
        aligned["training"].loc[0, "ret_5m"] = np.nan
    elif kind == "label":
        aligned["training"].loc[0, "y_exit"] = 1 - native["training"].y_exit.iloc[0]
    elif kind == "state":
        aligned["training"].loc[0, "favorable_move"] += 1
    elif kind == "episode":
        for rows in [native, aligned]:
            rows["diagnosis"]["episode_id"] = 1
    elif kind == "time":
        for rows in [native, aligned]:
            rows["training"].loc[0, "label_end"] = rows["calibration"].end.iloc[0]
    elif kind == "support":
        for rows in [native, aligned]:
            rows["calibration"]["y_exit"] = 0
    else:
        del native["diagnosis"]
    with pytest.raises((ValueError, AssertionError)):
        validate_paired_rows(native, aligned)


def gate_metrics():
    return {a: {name: {"rows": 100, "positive": 30, "log_loss": loss, "average_precision": ap}
                for name, loss, ap in [("native_calibrated", .5, .6), ("linear_calibrated", .48, .62),
                                      ("aligned_calibrated", .4, .65), ("aligned_raw", .41, .65),
                                      ("constant", .7, .3)]} for a in ACTIONS}


@pytest.mark.parametrize("case", ["pass", "one_action", "one_percent_boundary", "raw_worse", "constant"])
def test_all_eighteen_admission_conditions(case):
    metrics = gate_metrics()
    if case == "one_action":
        metrics["reduce"]["aligned_calibrated"]["average_precision"] = .1
    elif case == "one_percent_boundary":
        metrics["exit"]["aligned_calibrated"]["log_loss"] = .495
    elif case == "raw_worse":
        metrics["increase"]["aligned_raw"]["log_loss"] = .3
    elif case == "constant":
        metrics["exit"]["constant"]["log_loss"] = .39
    decision = market_model_admission(metrics)
    assert len(decision["checks"]) == 18
    assert decision["market_model_admitted"] == (case == "pass")
    assert not decision["profitability_accepted"]
    verify_market_admission(metrics, decision)


def test_invalid_metrics_and_reference_fail(tmp_path):
    metrics = gate_metrics()
    metrics["exit"]["aligned_calibrated"]["average_precision"] = None
    with pytest.raises(ValueError):
        market_model_admission(metrics)
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_learning_reference(reference, "0" * 64)


def test_public_learning_runner_with_partial_calibration(tmp_path, monkeypatch, fitted):
    import wonyotti_fr.management_market_learning as research
    from wonyotti_fr.retained_weight_close_reference import checked_files

    native, aligned, models, offsets, _ = fitted
    source, teacher = tmp_path / "input", tmp_path / "teacher"
    source.mkdir()
    teacher.mkdir()
    summaries = {}
    for phase in native:
        original, candidate = native[phase], aligned[phase].copy()
        link = pd.DataFrame({"end": original.end, "expected_five_end": original.end.dt.floor("5min"),
            "minute_found": True, "five_found": True, "market_finite": True, "supported": True, "reason": "supported"})
        if phase == "calibration":
            candidate.loc[0, "minute_ret_1m"] = np.nan
            link.loc[0, ["market_finite", "supported"]] = False
            link.loc[0, "reason"] = "nonfinite_market"
        summaries[phase] = market_support(original, candidate, link)
        original.to_parquet(teacher / f"{phase}_used.parquet", index=False)
        candidate[["end", *REPLACED]].to_parquet(source / f"{phase}_market.parquet", index=False)
        link.to_parquet(source / f"{phase}_linkage.parquet", index=False)
    save_json(source / "support.json", summaries)
    save_json(teacher / "models.json", {"histogram": models["native"].to_dict()})
    save_json(teacher / "offsets.json", {"histogram": offsets["native"].to_dict()})
    old = native["diagnosis"][["end", "episode_id"]].copy()
    raw = models["native"].probabilities(native["diagnosis"][FEATURES].to_numpy(float))
    calibrated = offsets["native"].predict(raw)
    for i, a in enumerate(ACTIONS):
        old[a + "_histogram_raw"] = raw[:, i]
        old[a + "_histogram_calibrated"] = calibrated[:, i]
    old.to_parquet(teacher / "predictions.parquet", index=False)
    monkeypatch.setattr(research, "checked_learning_reference", lambda *args: (source, teacher))
    out = research.run_market_learning(tmp_path / "unused.json", "1" * 64, tmp_path / "outputs")
    checked_files(out)
    proof = research.read_json(out / "verification.json")
    assert proof["native_trees_equal_stored"] and not proof["native_offset_equal_stored"]
    assert proof["all_probability_rows_verified"] == 1999
    assert not proof["native_calibration_identical"]
    assert research.read_json(out / "summary.json")["rows"]["calibration"] == 399
    assert len(pd.read_parquet(out / "calibration_membership.parquet")) == 400
