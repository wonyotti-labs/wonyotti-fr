import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json
from wonyotti_fr.management_flow import EXPANDED_FEATURES, EXTRA_FEATURES
from wonyotti_fr.management_flow_learning import checked_flow_learning_reference
from wonyotti_fr.management_flow_model_reference import (
    refit_flow_models,
    verify_flow_admission,
    verify_flow_predictions,
    verify_flow_usage,
)
from wonyotti_fr.management_flow_models import (
    fit_flow_models,
    flow_feature_usage,
    flow_model_admission,
    flow_predictions,
    validate_flow_rows,
)
from wonyotti_fr.management_market import FEATURES
from wonyotti_fr.management_market_model_reference import verify_market_evaluation, verify_offsets
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.paired_management_evaluation import choose_thresholds, evaluate


def frames():
    rng = np.random.default_rng(117)
    native, expanded = {}, {}
    for i, (phase, date, count) in enumerate([
        ("training", "2020-01-02", 1200), ("calibration", "2021-01-02", 400),
        ("diagnosis", "2021-07-02", 400),
    ]):
        frame = pd.DataFrame(rng.normal(size=(count, 50)), columns=FEATURES)
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["direction"] = np.where(np.arange(count)%2, 1, -1)
        frame["label_end"] = frame.end+pd.Timedelta(minutes=1)
        frame["entry_time"] = pd.Timestamp(date, tz="UTC")-pd.Timedelta(hours=1)
        frame["episode_id"] = i+1
        for j, a in enumerate(ACTIONS):
            frame["y_"+a] = (frame[FEATURES[j]]+rng.normal(size=count) > 0).astype(int)
        native[phase] = frame
        expanded[phase] = frame.copy()
        for name in EXTRA_FEATURES:
            if not name.startswith("directional_"):
                expanded[phase][name] = rng.uniform(-1, 1, count)
                expanded[phase]["directional_"+name] = expanded[phase][name]*frame.direction
    return native, expanded


@pytest.fixture(scope="module")
def fitted():
    native, expanded = frames()
    models, offsets, support = fit_flow_models(native, expanded)
    return native, expanded, models, offsets, support


def test_independent_fit_predictions_calibration_and_gates(fitted):
    native, expanded, models, offsets, _ = fitted
    assert refit_flow_models(native, expanded, models)["independent_histogram_trees"] == 384
    cy = native["calibration"][["y_"+a for a in ACTIONS]].to_numpy()
    scores = flow_predictions(native["calibration"], expanded["calibration"], models, offsets, cy.mean(axis=0))
    verify_offsets(scores, cy, offsets)
    verify_flow_predictions(native["calibration"], expanded["calibration"], models, offsets, cy.mean(axis=0), scores)
    thresholds, details = choose_thresholds(cy, scores)
    metrics, _ = evaluate(cy, scores, thresholds)
    verify_market_evaluation(cy, scores, thresholds, metrics, details)
    decision = flow_model_admission(metrics)
    verify_flow_admission(metrics, decision)
    assert len(decision["checks"]) == 18
    usage = flow_feature_usage(models["flow"])
    verify_flow_usage(models["flow"], usage)
    assert set(models["flow"].features) == set(EXPANDED_FEATURES)


def test_diagnosis_features_and_labels_cannot_change_fit(fitted):
    native, expanded, models, offsets, _ = fitted
    a, b = copy.deepcopy(native), copy.deepcopy(expanded)
    for target in [a, b]:
        for action in ACTIONS:
            target["diagnosis"]["y_"+action] = 1-target["diagnosis"]["y_"+action]
    b["diagnosis"][EXTRA_FEATURES] *= -1
    new, adjusted, _ = fit_flow_models(a, b)
    assert {k: v.to_dict() for k, v in models.items()} == {k: v.to_dict() for k, v in new.items()}
    assert {k: v.to_dict() for k, v in offsets.items()} == {k: v.to_dict() for k, v in adjusted.items()}


@pytest.mark.parametrize("kind", ["feature", "label", "nonfinite", "range", "direction", "episode", "phase"])
def test_changed_or_invalid_original_pair_rejected(kind):
    native, expanded = frames()
    if kind == "feature":
        expanded["training"].loc[0, FEATURES[0]] += 1
    elif kind == "label":
        expanded["training"].loc[0, "y_exit"] = 1-expanded["training"].y_exit.iloc[0]
    elif kind == "nonfinite":
        expanded["training"].loc[0, EXTRA_FEATURES[0]] = np.nan
    elif kind == "range":
        expanded["training"].loc[0, EXTRA_FEATURES[0]] = 1.01
    elif kind == "direction":
        expanded["training"].loc[0, "directional_"+EXTRA_FEATURES[0]] += .1
    elif kind == "episode":
        for target in [native, expanded]:
            target["calibration"]["episode_id"] = 1
    else:
        del expanded["calibration"]
    with pytest.raises((AssertionError, ValueError)):
        validate_flow_rows(native, expanded)


def passing_metrics():
    base = {"positive": 50, "rows": 1000, "average_precision": .2, "log_loss": .1}
    groups = {n: dict(base) for n in ["native_calibrated", "linear_calibrated", "flow_calibrated", "flow_raw", "constant"]}
    groups["flow_calibrated"].update(log_loss=.08, average_precision=.3)
    return {a: copy.deepcopy(groups) for a in ACTIONS}


def test_exact_improvement_boundary_and_single_action_failure():
    metrics = passing_metrics()
    good = flow_model_admission(metrics)
    verify_flow_admission(metrics, good)
    assert good["flow_model_admitted"] and all(good["checks"].values())
    metrics["exit"]["flow_calibrated"]["log_loss"] = .099
    bad = flow_model_admission(metrics)
    verify_flow_admission(metrics, bad)
    assert not bad["flow_model_admitted"]
    assert not bad["checks"]["exit_log_loss_native"]
    assert bad["checks"]["reduce_log_loss_native"]


def test_invalid_metric_rejected():
    metrics = passing_metrics()
    metrics["exit"]["flow_calibrated"]["log_loss"] = np.nan
    with pytest.raises(ValueError):
        flow_model_admission(metrics)


def test_reference_hash_rejected(tmp_path):
    path = tmp_path/"proof.json"
    save_json(path, {})
    with pytest.raises(ValueError, match="지문"):
        checked_flow_learning_reference(path, "0"*64)


def test_independent_usage_rejects_changed_count(fitted):
    model = fitted[2]["flow"]
    usage = flow_feature_usage(model)
    usage["exit"][EXTRA_FEATURES[0]] += 1
    with pytest.raises(AssertionError):
        verify_flow_usage(model, usage)


def test_full_public_runner_preserves_native_scores_and_all_rows(tmp_path, monkeypatch, fitted):
    import wonyotti_fr.management_flow_learning as research
    from wonyotti_fr.holding_stress import read_json
    from wonyotti_fr.retained_weight_close_reference import checked_files

    native, expanded, models, offsets, _ = fitted
    source, teacher = tmp_path/"inputs", tmp_path/"teacher"
    source.mkdir()
    teacher.mkdir()
    support = {}
    for phase, frame in native.items():
        frame.to_parquet(teacher/f"{phase}_used.parquet", index=False)
        expanded[phase][["end", *EXTRA_FEATURES]].to_parquet(source/f"{phase}_flow.parquet", index=False)
        pd.DataFrame({"end": frame.end, "source_row": np.arange(len(frame)), "supported": True,
                      "reason": "supported"}).to_parquet(source/f"{phase}_linkage.parquet", index=False)
        support[phase] = {"groups": {"all": {"rows": len(frame)}}}
    save_json(source/"support.json", support)
    save_json(teacher/"models.json", {"histogram": models["native"].to_dict()})
    save_json(teacher/"offsets.json", {"histogram": offsets["native"].to_dict()})
    old = native["diagnosis"][["end", "episode_id"]].copy()
    raw = models["native"].probabilities(native["diagnosis"][FEATURES].to_numpy())
    for suffix, values in [("raw", raw), ("calibrated", offsets["native"].predict(raw))]:
        for j, a in enumerate(ACTIONS):
            old[a+"_histogram_"+suffix] = values[:, j]
    old.to_parquet(teacher/"predictions.parquet", index=False)
    monkeypatch.setattr(research, "checked_flow_learning_reference", lambda *args: (source, teacher))
    result = research.run_flow_learning(tmp_path/"proof", "synthetic", tmp_path/"output")
    checked_files(result)
    summary = read_json(result/"summary.json")
    assert summary["complete"] and summary["independent_verification_complete"]
    assert summary["rows"] == {p: len(v) for p, v in native.items()}
    assert not summary["profitability_accepted"]
    assert read_json(result/"verification.json")["original_trees_and_offsets_exact"]
