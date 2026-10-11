from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .extrema_age_models import EXTRA_FEATURES, FEATURES, NATIVE_FEATURES
from .management_capacity import LABELS
from .management_capacity_reference import partition_forest, verify_tree_model
from .management_market_model_reference import offset_scores
from .minute_management import ACTIONS
from .paired_management_reference import numeric_linear_scores


def verify_age_refits(expanded, models):
    x, y = (
        expanded["training"][FEATURES].to_numpy(float),
        expanded["training"][LABELS].to_numpy(int),
    )
    matrices = {phase: frame[FEATURES].to_numpy(float) for phase, frame in expanded.items()}
    predictions = {
        phase: models["aged"].probabilities(values) for phase, values in matrices.items()
    }
    count = verify_tree_model(models["aged"], "depth2_iter256", x, y, matrices, predictions)
    raw = models["aged_linear"].to_dict()
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(raw["mean"], scaler.mean_)
        np.testing.assert_array_equal(raw["scale"], scaler.scale_)
        predictions = {
            phase: models["aged_linear"].probabilities(values) for phase, values in matrices.items()
        }
        for i in range(3):
            learner = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                scaler.transform(x), y[:, i]
            )
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(raw["coef"][i], learner.coef_[0])
            assert raw["intercept"][i] == float(learner.intercept_[0])
            for phase, values in matrices.items():
                np.testing.assert_allclose(
                    predictions[phase][:, i],
                    learner.predict_proba(scaler.transform(values))[:, 1],
                    atol=1e-12,
                    rtol=0,
                )
    return {"independent_new_trees": count, "new_linear_standardization_and_three_fits_exact": True}


def verify_age_predictions(expanded, models, offsets, scores):
    x = expanded[FEATURES].to_numpy(float)
    for name in ["aged", "aged_linear"]:
        raw = models[name].to_dict()
        if name == "aged_linear":
            expected = numeric_linear_scores(raw, x)
        else:
            logits = np.column_stack([partition_forest(binary, x) for binary in raw["models"]])
            expected = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        np.testing.assert_allclose(scores[name + "_raw"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(
            scores[name + "_calibrated"],
            offset_scores(scores[name + "_raw"], offsets[name].to_dict()),
        )


def verify_age_admission(metrics, decision):
    checks = {}
    for action in ACTIONS:
        groups = metrics[action]
        whole = groups["all"]
        candidate = whole["aged_calibrated"]
        for label, name in [("native", "native_calibrated"), ("linear", "aged_linear_calibrated")]:
            old = whole[name]
            gain = (
                (old["log_loss"] - candidate["log_loss"]) / old["log_loss"]
                if old["log_loss"]
                else 0.0
            )
            checks[action + "_log_loss_" + label] = bool(gain > 0.01 + 1e-12)
            checks[action + "_ap_" + label] = bool(
                old["average_precision"] - candidate["average_precision"] <= 1e-12
            )
        checks[action + "_raw_preserved"] = bool(
            candidate["log_loss"] - whole["aged_raw"]["log_loss"] <= 1e-12
        )
        checks[action + "_constant_beaten"] = bool(
            whole["constant"]["log_loss"] - candidate["log_loss"] > 1e-12
        )
        checks[action + "_all_f_preserved"] = bool(
            candidate["f_beta"] >= whole["native_calibrated"]["f_beta"] - 1e-12
        )
        if action != "exit":
            for key in ["recall", "f_beta"]:
                checks[action + "_first_" + key + "_preserved"] = bool(
                    groups["first"]["aged_calibrated"][key]
                    >= groups["first"]["native_calibrated"][key] - 1e-12
                )
        checks[action + "_capacity_log_loss_preserved"] = bool(
            candidate["log_loss"] - whole["capacity_calibrated"]["log_loss"] <= 1e-12
        )
        checks[action + "_capacity_ap_preserved"] = bool(
            whole["capacity_calibrated"]["average_precision"] - candidate["average_precision"]
            <= 1e-12
        )
    assert len(checks) == 31 and decision["checks"] == checks
    assert decision["extrema_age_admitted"] == all(checks.values())
    assert (
        decision["profitability_accepted"] is False
        and decision["trading_returns_evaluated"] is False
    )


def verify_age_usage(model, usage):
    expected = {action: dict.fromkeys(EXTRA_FEATURES, 0) for action in ACTIONS}
    for binary in model.to_dict()["models"]:
        for tree in binary["trees"]:
            for index in tree["feature"]:
                if index >= len(NATIVE_FEATURES):
                    expected[binary["action"]][FEATURES[index]] += 1
    assert usage == expected
