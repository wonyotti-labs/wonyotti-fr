from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .management_capacity import LABELS
from .management_capacity_reference import partition_forest, verify_tree_model
from .management_market_model_reference import offset_scores
from .minute_management import ACTIONS
from .paired_management_reference import numeric_linear_scores
from .volatility_path_models import EXTRA_FEATURES, FEATURES, NATIVE_FEATURES


def verify_volatility_inputs(native, expanded):
    if set(native) != set(expanded):
        raise ValueError("변동성 경로의 분할 누락")
    result = {}
    for phase, original in native.items():
        frame = expanded[phase]
        pd.testing.assert_frame_equal(original, frame.drop(columns=EXTRA_FEATURES), check_exact=True)
        expected = []
        floor_count = 0
        ratio_min = math.inf
        for row in original[["direction", "vol_1d", "favorable_move", "best_close_move", "worst_close_move", "giveback_from_best"]].itertuples(index=False):
            d, sigma, current, best, worst, giveback = row
            assert d in (-1, 1) and sigma >= 0 and best >= current >= worst and giveback >= 0
            assert math.isclose(giveback, best-current, rel_tol=1e-12, abs_tol=1e-12)
            ratios = [1+d*value for value in [current, best, worst]]
            assert all(v > 0 and math.isfinite(v) for v in ratios)
            ratio_min = min(ratio_min, *ratios)
            denominator = 1e-8 if sigma < 1e-8 else sigma
            floor_count += sigma < 1e-8
            moves = [d*math.log1p(d*value)/denominator for value in [current, best, worst]]
            moves.append(d*math.log1p(d*(best-current)/(1+d*current))/denominator)
            expected.append(moves)
        expected = np.asarray(expected, dtype=float)
        np.testing.assert_allclose(frame[EXTRA_FEATURES].to_numpy(float), expected, atol=1e-12, rtol=1e-12)
        result[phase] = {"rows": len(frame), "extra_values": len(frame)*4, "floored_rows": floor_count,
            "minimum_volatility": float(original.vol_1d.min()), "minimum_price_ratio": ratio_min,
            "ranges": {name: {"minimum": float(expected[:, i].min()), "maximum": float(expected[:, i].max())}
                       for i, name in enumerate(EXTRA_FEATURES)}}
    return result


def verify_volatility_boundaries(training, boundaries):
    ordered = sorted(float(v) for v in training.vol_1d)
    expected = []
    for fraction in [.25, .5, .75]:
        position = fraction*(len(ordered)-1)
        low, high = math.floor(position), math.ceil(position)
        expected.append(ordered[low]+(ordered[high]-ordered[low])*(position-low))
    np.testing.assert_allclose(boundaries, expected, atol=1e-12, rtol=0)


def verify_volatility_groups(frame, boundaries, groups):
    expected = [sum(value > edge for edge in boundaries) for value in frame.vol_1d]
    np.testing.assert_array_equal(groups, expected)
    assert all(v in range(4) for v in expected)


def verify_volatility_refits(expanded, models):
    x, y = expanded["training"][FEATURES].to_numpy(float), expanded["training"][LABELS].to_numpy(int)
    matrices = {phase: frame[FEATURES].to_numpy(float) for phase, frame in expanded.items()}
    predictions = {phase: models["scaled"].probabilities(values) for phase, values in matrices.items()}
    count = verify_tree_model(models["scaled"], "depth2_iter256", x, y, matrices, predictions)
    raw = models["scaled_linear"].to_dict()
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(raw["mean"], scaler.mean_)
        np.testing.assert_array_equal(raw["scale"], scaler.scale_)
        predictions = {phase: models["scaled_linear"].probabilities(values) for phase, values in matrices.items()}
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(raw["coef"][i], learner.coef_[0])
            assert raw["intercept"][i] == float(learner.intercept_[0])
            for phase, values in matrices.items():
                np.testing.assert_allclose(predictions[phase][:, i], learner.predict_proba(scaler.transform(values))[:, 1], atol=1e-12, rtol=0)
    return {"independent_new_trees": count, "new_linear_standardization_and_three_fits_exact": True}


def verify_volatility_predictions(expanded, models, offsets, scores):
    x = expanded[FEATURES].to_numpy(float)
    for name in ["scaled", "scaled_linear"]:
        raw = models[name].to_dict()
        if name == "scaled_linear":
            expected = numeric_linear_scores(raw, x)
        else:
            logits = np.column_stack([partition_forest(binary, x) for binary in raw["models"]])
            expected = 1/(1+np.exp(-np.clip(logits, -700, 700)))
        np.testing.assert_allclose(scores[name+"_raw"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(scores[name+"_calibrated"], offset_scores(scores[name+"_raw"], offsets[name].to_dict()))


def verify_volatility_admission(metrics, decision):
    checks = {}
    for action in ACTIONS:
        groups = metrics[action]
        whole = groups["all"]
        candidate = whole["scaled_calibrated"]
        for label, name in [("native", "native_calibrated"), ("linear", "scaled_linear_calibrated")]:
            old = whole[name]
            gain = (old["log_loss"]-candidate["log_loss"])/old["log_loss"] if old["log_loss"] else 0.
            checks[action+"_log_loss_"+label] = bool(gain > .01+1e-12)
            checks[action+"_ap_"+label] = bool(old["average_precision"]-candidate["average_precision"] <= 1e-12)
        checks[action+"_raw_preserved"] = bool(candidate["log_loss"]-whole["scaled_raw"]["log_loss"] <= 1e-12)
        checks[action+"_constant_beaten"] = bool(whole["constant"]["log_loss"]-candidate["log_loss"] > 1e-12)
        checks[action+"_all_f_preserved"] = bool(candidate["f_beta"] >= whole["native_calibrated"]["f_beta"]-1e-12)
        if action != "exit":
            for key in ["recall", "f_beta"]:
                checks[action+"_first_"+key+"_preserved"] = bool(groups["first"]["scaled_calibrated"][key] >= groups["first"]["native_calibrated"][key]-1e-12)
        checks[action+"_capacity_log_loss_preserved"] = bool(candidate["log_loss"]-whole["capacity_calibrated"]["log_loss"] <= 1e-12)
        checks[action+"_capacity_ap_preserved"] = bool(whole["capacity_calibrated"]["average_precision"]-candidate["average_precision"] <= 1e-12)
    assert len(checks) == 31 and decision["checks"] == checks
    assert decision["volatility_path_admitted"] == all(checks.values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False


def verify_path_usage(model, usage):
    expected = {action: dict.fromkeys(EXTRA_FEATURES, 0) for action in ACTIONS}
    for binary in model.to_dict()["models"]:
        for tree in binary["trees"]:
            for index in tree["feature"]:
                if index >= len(NATIVE_FEATURES):
                    expected[binary["action"]][FEATURES[index]] += 1
    assert usage == expected
