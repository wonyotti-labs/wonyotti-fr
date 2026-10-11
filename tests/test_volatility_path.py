from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json
from wonyotti_fr.joint_history_models import evaluate_joint, joint_thresholds
from wonyotti_fr.joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from wonyotti_fr.management_capacity import LABELS, capacity_predictions, fit_outer_models
from wonyotti_fr.management_market_model_reference import verify_offsets
from wonyotti_fr.volatility_path_learning import (
    checked_volatility_reference,
    verify_path_reference_metrics,
    verify_path_reference_thresholds,
)
from wonyotti_fr.volatility_path_models import (
    EXTRA_FEATURES,
    FEATURES,
    NATIVE_FEATURES,
    VolatilityPathBoost,
    VolatilityPathLinear,
    attach_volatility_path,
    fit_volatility_models,
    path_feature_usage,
    volatility_admission,
    volatility_boundaries,
    volatility_groups,
    volatility_path_values,
    volatility_predictions,
)
from wonyotti_fr.volatility_path_reference import (
    verify_path_usage,
    verify_volatility_admission,
    verify_volatility_boundaries,
    verify_volatility_groups,
    verify_volatility_inputs,
    verify_volatility_predictions,
    verify_volatility_refits,
)


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(123)
    rows = {}
    for i, (phase, date, count) in enumerate([
        ("training", "2020-01-03", 3000), ("calibration", "2021-01-03", 1200),
        ("diagnosis", "2021-07-03", 1200),
    ]):
        frame = pd.DataFrame(rng.normal(size=(count, len(NATIVE_FEATURES))), columns=NATIVE_FEATURES)
        frame["direction"] = np.where(np.arange(count) % 2, -1., 1.)
        frame["vol_1d"] = .01+np.abs(rng.normal(size=count))*.01
        frame["favorable_move"] = rng.uniform(-.05, .05, count)
        frame["best_close_move"] = frame.favorable_move+rng.uniform(0, .03, count)
        frame["worst_close_move"] = frame.favorable_move-rng.uniform(0, .03, count)
        frame["giveback_from_best"] = frame.best_close_move-frame.favorable_move
        frame["past_reduce_exists"] = np.arange(count) % 2
        frame["past_increase_exists"] = (np.arange(count)//2) % 2
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["entry_time"] = frame.end.iloc[0]-pd.Timedelta(hours=1)
        frame["label_end"] = frame.end+pd.Timedelta(minutes=1)
        frame["episode_id"] = i+1
        for label in LABELS:
            frame[label] = (rng.uniform(size=count) < .2+.3*frame.ret_5m.gt(.5)).astype(int)
        rows[phase] = frame
    expanded = attach_volatility_path(rows)
    refs, ref_offsets, _ = fit_outer_models(rows, "depth2_iter256")
    models, offsets, _ = fit_volatility_models(expanded)
    models, offsets = {**refs, **models}, {**ref_offsets, **offsets}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    scores = {p: volatility_predictions(f, expanded[p], models, offsets, constant) for p, f in rows.items()}
    thresholds, details = joint_thresholds(rows["calibration"], scores["calibration"])
    return rows, expanded, models, offsets, scores, thresholds, details


def path_rows(d=1., vol=.1, current=.1, best=.2, worst=-.1):
    frame = pd.DataFrame(np.zeros((1, len(NATIVE_FEATURES))), columns=NATIVE_FEATURES)
    for name, value in zip(["direction", "vol_1d", "favorable_move", "best_close_move", "worst_close_move", "giveback_from_best"],
                           [d, vol, current, best, worst, best-current], strict=True):
        frame[name] = value
    return frame


@pytest.mark.parametrize("direction", [-1., 1.])
def test_long_short_price_ratios(direction):
    values = volatility_path_values(path_rows(d=direction))
    prices = [110, 120, 90] if direction == 1 else [90, 80, 110]
    expected = [direction*np.log(price/100)/.1 for price in prices]
    expected += [direction*np.log(prices[1]/prices[0])/.1]
    np.testing.assert_allclose(values[0], expected, atol=1e-14, rtol=0)


def test_zero_floor_flat_and_small_retracement():
    np.testing.assert_array_equal(volatility_path_values(path_rows(vol=0, current=0, best=0, worst=0)), np.zeros((1, 4)))
    a = volatility_path_values(path_rows(vol=0))
    np.testing.assert_array_equal(a, volatility_path_values(path_rows(vol=1e-8)))
    tiny = np.nextafter(.1, np.inf)-.1
    actual = volatility_path_values(path_rows(current=.1, best=.1+tiny))[0, 3]
    assert actual > 0
    np.testing.assert_allclose(actual, tiny/1.1/.1, rtol=1e-14, atol=0)


@pytest.mark.parametrize("field,value", [
    ("direction", 0), ("vol_1d", -.1), ("best_close_move", -.2),
    ("worst_close_move", .3), ("giveback_from_best", -.1),
    ("giveback_from_best", .11), ("worst_close_move", -1.), ("ret_5m", np.nan),
])
def test_invalid_paths_rejected(field, value):
    frame = path_rows()
    frame[field] = value
    with pytest.raises(ValueError):
        volatility_path_values(frame)


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        volatility_path_values(np.zeros(50))
    with pytest.raises(ValueError):
        volatility_path_values(np.zeros((4, 49)))


def test_all_inputs_models_scores_and_groups_verified(study):
    rows, expanded, models, offsets, scores, thresholds, details = study
    support = verify_volatility_inputs(rows, expanded)
    assert sum(v["extra_values"] for v in support.values()) == 21600
    assert verify_volatility_refits(expanded, models)["independent_new_trees"] == 768
    boundaries = volatility_boundaries(rows["training"])
    verify_volatility_boundaries(rows["training"], boundaries)
    verify_offsets(scores["calibration"], rows["calibration"][LABELS].to_numpy(), offsets)
    verify_joint_thresholds(rows["calibration"], scores["calibration"], thresholds, details)
    for phase, frame in rows.items():
        verify_volatility_predictions(expanded[phase], models, offsets, scores[phase])
        metrics, requests = evaluate_joint(frame, scores[phase], thresholds)
        verify_joint_evaluation(frame, scores[phase], thresholds, metrics, requests)
        groups = volatility_groups(frame, boundaries)
        verify_volatility_groups(frame, boundaries, groups)
        for group in range(4):
            indices = np.flatnonzero(groups == group)
            part, ps = frame.iloc[indices], {k: v[indices] for k, v in scores[phase].items()}
            result, decisions = evaluate_joint(part, ps, thresholds)
            verify_joint_evaluation(part, ps, thresholds, result, decisions)
        if phase == "diagnosis":
            decision = volatility_admission(metrics)
            verify_volatility_admission(metrics, decision)
            assert len(decision["checks"]) == 31
    verify_path_usage(models["scaled"], path_feature_usage(models["scaled"]))


@pytest.mark.parametrize("change", ["extra", "label", "order"])
def test_input_mutations_detected(study, change):
    rows, expanded = study[:2]
    changed = {k: v.copy() for k, v in expanded.items()}
    if change == "extra":
        changed["diagnosis"].loc[0, EXTRA_FEATURES[0]] += .01
    elif change == "label":
        changed["diagnosis"].loc[0, "y_exit"] ^= 1
    else:
        changed["diagnosis"] = changed["diagnosis"].iloc[::-1]
    with pytest.raises(AssertionError):
        verify_volatility_inputs(rows, changed)


def test_future_changes_preserve_past_features(study):
    rows, expanded = study[:2]
    changed = {k: v.copy() for k, v in rows.items()}
    changed["diagnosis"]["vol_1d"] *= 2
    changed["diagnosis"][LABELS] = 1-changed["diagnosis"][LABELS]
    after = attach_volatility_path(changed)
    for phase in ["training", "calibration"]:
        pd.testing.assert_frame_equal(expanded[phase], after[phase], check_exact=True)
    with pytest.raises(ValueError, match="중복"):
        attach_volatility_path(expanded)


@pytest.mark.parametrize("both", [False, True])
def test_future_labels_preserve_models_and_train_boundaries(study, both):
    rows, expanded, models, offsets, scores, thresholds, _ = study
    changed = {k: v.copy() for k, v in expanded.items()}
    for phase in (["calibration", "diagnosis"] if both else ["diagnosis"]):
        changed[phase][LABELS] = 1-changed[phase][LABELS]
    fitted, adjusted, _ = fit_volatility_models(changed)
    assert volatility_boundaries(changed["training"]) == volatility_boundaries(rows["training"])
    for name in fitted:
        assert fitted[name].to_dict() == models[name].to_dict()
        assert (adjusted[name].to_dict() != offsets[name].to_dict()) == both
    if not both:
        combined = {**models, **fitted}
        new_offsets = {**offsets, **adjusted}
        constant = rows["calibration"][LABELS].mean().to_numpy()
        cal = volatility_predictions(rows["calibration"], changed["calibration"], combined, new_offsets, constant)
        assert joint_thresholds(rows["calibration"], cal)[0] == thresholds
        for key in cal:
            np.testing.assert_array_equal(cal[key], scores["calibration"][key])


@pytest.mark.parametrize("name,cls", [("scaled", VolatilityPathBoost), ("scaled_linear", VolatilityPathLinear)])
def test_model_roundtrip_and_single_prediction(study, name, cls):
    _, expanded, models = study[:3]
    raw = models[name].to_dict()
    loaded = cls.from_dict(raw)
    assert raw == loaded.to_dict()
    values = expanded["diagnosis"][FEATURES].iloc[:11].to_numpy()
    np.testing.assert_allclose(loaded.probabilities(values), np.vstack([models[name].probabilities(v[None, :]) for v in values]), rtol=0, atol=1e-12)
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        cls.from_dict(raw)
    assert len(loaded.to_dict()["features"]) == 54


def test_changed_tree_settings_rejected(study):
    raw = study[2]["scaled"].to_dict()
    raw["settings"]["max_iter"] = 64
    with pytest.raises(ValueError):
        VolatilityPathBoost.from_dict(raw)


def test_quartile_ties_empty_and_zero_positive_groups(study):
    frame = pd.DataFrame({"vol_1d": [0., 1., 1., 1., 2.]})
    boundaries = volatility_boundaries(frame)
    assert boundaries == [1., 1., 1.]
    groups = volatility_groups(frame, boundaries)
    np.testing.assert_array_equal(groups, [0, 0, 0, 0, 3])
    verify_volatility_boundaries(frame, boundaries)
    verify_volatility_groups(frame, boundaries, groups)
    for indices in [np.array([], dtype=int), np.array([0, 1])]:
        part = study[0]["diagnosis"].iloc[indices].copy()
        part[LABELS] = 0
        scores = {k: v[indices] for k, v in study[4]["diagnosis"].items()}
        result, requests = evaluate_joint(part, scores, study[5])
        verify_joint_evaluation(part, scores, study[5], result, requests)
        value = result["exit"]["all"]["scaled_calibrated"]
        assert value is None if not len(indices) else value["average_precision"] is None


def test_original_thresholds_and_all_months_preserved(study, tmp_path):
    rows, expanded, models, offsets, scores, thresholds, details = study
    refs = {k: models[k] for k in ["native", "capacity", "linear"]}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    original_scores = {p: capacity_predictions(f, refs, offsets, constant) for p, f in rows.items()}
    old_thresholds, old_details = joint_thresholds(rows["calibration"], original_scores["calibration"])
    save_json(tmp_path / "thresholds.json", old_thresholds)
    save_json(tmp_path / "threshold_selection.json", old_details)
    verify_path_reference_thresholds(tmp_path, thresholds, details)
    all_metrics, old_metrics, monthly, old_monthly = {}, {}, {}, {}
    for phase, frame in rows.items():
        for name, value in original_scores[phase].items():
            np.testing.assert_array_equal(value, scores[phase][name])
        all_metrics[phase], _ = evaluate_joint(frame, scores[phase], thresholds)
        old_metrics[phase], _ = evaluate_joint(frame, original_scores[phase], old_thresholds)
        monthly[phase], old_monthly[phase] = {}, {}
        for month, indices in frame.groupby(frame.end.dt.strftime("%Y-%m")).indices.items():
            part = frame.iloc[indices]
            ps = {k: v[indices] for k, v in scores[phase].items()}
            old_ps = {k: v[indices] for k, v in original_scores[phase].items()}
            monthly[phase][month], _ = evaluate_joint(part, ps, thresholds)
            old_monthly[phase][month], _ = evaluate_joint(part, old_ps, old_thresholds)
    save_json(tmp_path / "metrics.json", old_metrics)
    save_json(tmp_path / "monthly.json", old_monthly)
    verify_path_reference_metrics(tmp_path, all_metrics, monthly)
    monthly["diagnosis"]["2021-07"]["exit"]["all"]["native_raw"]["rows"] += 1
    with pytest.raises((ValueError, AssertionError)):
        verify_path_reference_metrics(tmp_path, all_metrics, monthly)


def test_changed_prediction_decision_and_proof_detected(study, tmp_path):
    rows, expanded, models, offsets, scores, thresholds, _ = study
    broken = deepcopy(scores["diagnosis"])
    broken["scaled_raw"][0, 0] += .01
    with pytest.raises(AssertionError):
        verify_volatility_predictions(expanded["diagnosis"], models, offsets, broken)
    metrics, _ = evaluate_joint(rows["diagnosis"], scores["diagnosis"], thresholds)
    decision = volatility_admission(metrics)
    decision["checks"]["exit_capacity_ap_preserved"] ^= True
    with pytest.raises(AssertionError):
        verify_volatility_admission(metrics, decision)
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_volatility_reference(path, "0"*64)
