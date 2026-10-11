from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .management_market import FEATURES
from .minute_management import ACTIONS
from .paired_management_reference import manual_metrics, manual_threshold, numeric_linear_scores
from .position_boost_verification import compare_statistics
from .sparse_exposure_reference import exact_trees, raw_forest


def refit_market_models(native, aligned, models):
    y = native["training"][["y_" + a for a in ACTIONS]].to_numpy(int)
    with threadpool_limits(limits=1):
        for name, rows in [("native", native), ("aligned", aligned)]:
            x, cx = (rows[p][FEATURES].to_numpy(float) for p in ["training", "calibration"])
            for i in range(3):
                learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=64, learning_rate=.05,
                    max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                    max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                    class_weight=None).fit(x, y[:, i])
                assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
                exact_trees(models[name].to_dict()["models"][i], learner)
                np.testing.assert_allclose(models[name].probabilities(cx)[:, i], learner.predict_proba(cx)[:, 1], atol=1e-12, rtol=0)
        x, cx = (aligned[p][FEATURES].to_numpy(float) for p in ["training", "calibration"])
        raw = models["linear"].to_dict()
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(scaler.mean_, raw["mean"])
        np.testing.assert_array_equal(scaler.scale_, raw["scale"])
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(learner.coef_[0], raw["coef"][i])
            assert float(learner.intercept_[0]) == raw["intercept"][i]
            np.testing.assert_allclose(models["linear"].probabilities(cx)[:, i], learner.predict_proba(scaler.transform(cx))[:, 1], atol=1e-12, rtol=0)
    return {"independent_histogram_trees": 384, "linear_standardization_and_three_fits_exact": True}


def offset_scores(scores, raw_offset):
    clipped = np.clip(scores, np.finfo(float).eps, 1-np.finfo(float).eps)
    logits = np.log(clipped)-np.log1p(-clipped)+np.asarray(raw_offset["offsets"])
    return 1/(1+np.exp(-np.clip(logits, -700, 700)))


def verify_offsets(calibration, labels, offsets):
    for name, offset in offsets.items():
        raw = calibration[name + "_raw"]
        clipped = np.clip(raw, np.finfo(float).eps, 1-np.finfo(float).eps)
        logits = np.log(clipped)-np.log1p(-clipped)
        expected = []
        for i in range(3):
            lower, upper = -64., 64.
            target = float(labels[:, i].mean())
            for _ in range(100):
                center = (lower+upper)/2
                mean = float((1/(1+np.exp(-np.clip(logits[:, i]+center, -700, 700)))).mean())
                if mean < target:
                    lower = center
                else:
                    upper = center
            expected.append((lower+upper)/2)
        np.testing.assert_allclose(offset.to_dict()["offsets"], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(calibration[name + "_calibrated"].mean(axis=0), labels.mean(axis=0), rtol=0, atol=1e-12)


def verify_market_predictions(native, aligned, models, offsets, stored_model, stored_offset, constant, predictions):
    for name in ["native", "aligned", "linear", "stored", "transported"]:
        frame = native if name in {"native", "stored"} else aligned
        model = models[name] if name in models else stored_model
        x = frame[FEATURES].to_numpy(float)
        raw = model.to_dict()
        if name == "linear":
            expected = numeric_linear_scores(raw, x)
        else:
            logits = np.column_stack([raw_forest(tree, x) for tree in raw["models"]])
            expected = 1/(1+np.exp(-np.clip(logits, -700, 700)))
        np.testing.assert_allclose(predictions[name + "_raw"], expected, rtol=0, atol=1e-12)
        chosen = offsets[name] if name in offsets else stored_offset
        # 문턱 판정은 저장된 확률의 연산 순서를 보존하고 독립 추론과 수치 대조한다.
        adjusted = offset_scores(predictions[name + "_raw"], chosen.to_dict())
        np.testing.assert_array_equal(predictions[name + "_calibrated"], adjusted)
    np.testing.assert_array_equal(predictions["constant"], np.tile(constant, (len(native), 1)))


def verify_market_evaluation(labels, predictions, thresholds, metrics, selection_details=None):
    for name, scores in predictions.items():
        for i, (action, beta) in enumerate(zip(ACTIONS, [2., 1., .5], strict=True)):
            if selection_details is not None:
                boundary, details = manual_threshold(labels[:, i], scores[:, i], beta)
                assert boundary == thresholds[name][action]
                compare_statistics(selection_details[name][action], details)
            expected = manual_metrics(labels[:, i], scores[:, i], thresholds[name][action], beta)
            compare_statistics(metrics[action][name], expected)


def verify_market_admission(metrics, decision):
    checks = {}
    for a in ["exit", "reduce", "increase"]:
        group = metrics[a]
        selected = group["aligned_calibrated"]
        for name in ["native", "linear"]:
            base = group[name+"_calibrated"]
            gain = (base["log_loss"]-selected["log_loss"])/base["log_loss"] if base["log_loss"] else 0.
            checks[f"{a}_log_loss_{name}"] = bool(gain > .01+1e-12)
            checks[f"{a}_ap_{name}"] = bool(base["average_precision"]-selected["average_precision"] <= 1e-12)
        checks[a+"_raw_preserved"] = bool(selected["log_loss"]-group["aligned_raw"]["log_loss"] <= 1e-12)
        checks[a+"_constant_beaten"] = bool(group["constant"]["log_loss"]-selected["log_loss"] > 1e-12)
    assert decision["checks"] == checks and decision["market_model_admitted"] == all(checks.values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
