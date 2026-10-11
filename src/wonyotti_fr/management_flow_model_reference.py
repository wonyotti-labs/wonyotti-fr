from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .management_flow import EXPANDED_FEATURES
from .management_market import FEATURES
from .management_market_model_reference import offset_scores
from .minute_management import ACTIONS
from .paired_management_reference import numeric_linear_scores
from .sparse_exposure_reference import exact_trees, raw_forest


def refit_flow_models(native, expanded, models):
    labels = native["training"][["y_"+a for a in ACTIONS]].to_numpy(int)
    with threadpool_limits(limits=1):
        for name, rows, names in [("native", native, FEATURES), ("flow", expanded, EXPANDED_FEATURES)]:
            x, cx = (rows[p][names].to_numpy(float) for p in ["training", "calibration"])
            for i in range(3):
                learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=64, learning_rate=.05,
                    max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                    max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                    class_weight=None).fit(x, labels[:, i])
                assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
                exact_trees(models[name].to_dict()["models"][i], learner)
                np.testing.assert_allclose(models[name].probabilities(cx)[:, i], learner.predict_proba(cx)[:, 1], atol=1e-12, rtol=0)
        x, cx = (expanded[p][EXPANDED_FEATURES].to_numpy(float) for p in ["training", "calibration"])
        raw = models["linear"].to_dict()
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(raw["mean"], scaler.mean_)
        np.testing.assert_array_equal(raw["scale"], scaler.scale_)
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), labels[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(raw["coef"][i], learner.coef_[0])
            assert raw["intercept"][i] == float(learner.intercept_[0])
            np.testing.assert_allclose(models["linear"].probabilities(cx)[:, i], learner.predict_proba(scaler.transform(cx))[:, 1], atol=1e-12, rtol=0)
    return {"independent_histogram_trees": 384, "linear_standardization_and_three_fits_exact": True}


def verify_flow_predictions(native, expanded, models, offsets, constant, predictions):
    for name in ["native", "flow", "linear"]:
        frame, names = (native, FEATURES) if name == "native" else (expanded, EXPANDED_FEATURES)
        x, raw = frame[names].to_numpy(float), models[name].to_dict()
        if name == "linear":
            expected = numeric_linear_scores(raw, x)
        else:
            logits = np.column_stack([raw_forest(binary, x) for binary in raw["models"]])
            expected = 1/(1+np.exp(-np.clip(logits, -700, 700)))
        np.testing.assert_allclose(predictions[name+"_raw"], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(predictions[name+"_calibrated"],
            offset_scores(predictions[name+"_raw"], offsets[name].to_dict()))
    np.testing.assert_array_equal(predictions["constant"], np.tile(constant, (len(native), 1)))


def verify_flow_admission(metrics, decision):
    checks = {}
    for action in ACTIONS:
        groups = metrics[action]
        chosen = groups["flow_calibrated"]
        for name in ["native", "linear"]:
            old = groups[name+"_calibrated"]
            gain = (old["log_loss"]-chosen["log_loss"])/old["log_loss"] if old["log_loss"] else 0.
            checks[f"{action}_log_loss_{name}"] = bool(gain > .01+1e-12)
            checks[f"{action}_ap_{name}"] = bool(old["average_precision"]-chosen["average_precision"] <= 1e-12)
        checks[action+"_raw_preserved"] = bool(chosen["log_loss"]-groups["flow_raw"]["log_loss"] <= 1e-12)
        checks[action+"_constant_beaten"] = bool(groups["constant"]["log_loss"]-chosen["log_loss"] > 1e-12)
    assert decision["checks"] == checks and decision["flow_model_admitted"] == all(checks.values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False


def verify_flow_usage(model, usage):
    expected = {action: {name: 0 for name in EXPANDED_FEATURES[50:]} for action in ACTIONS}
    for binary in model.to_dict()["models"]:
        for tree in binary["trees"]:
            for number in tree["feature"]:
                if number >= 50:
                    expected[binary["action"]][EXPANDED_FEATURES[number]] += 1
    assert usage == expected
