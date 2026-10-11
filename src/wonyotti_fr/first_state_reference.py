from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES
from .paired_management_reference import manual_metrics, manual_threshold
from .position_boost_verification import compare_statistics
from .sparse_exposure_reference import exact_trees, raw_forest


def refit_first_models(rows, models):
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            train = rows["training"]
            part = train.loc[train[f"past_{action}_exists"].eq(0)]
            x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
            learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=64, learning_rate=.05,
                max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                class_weight=None).fit(x, y)
            assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
            exact_trees(models[action]["tree"]["models"][0], learner)
            scaler = StandardScaler().fit(x)
            linear = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
            assert linear.n_iter_.max() < 2000
            saved = models[action]["linear"]
            for key, value in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", linear.coef_[0])]:
                np.testing.assert_array_equal(saved[key], value)
            assert saved["intercept"] == float(linear.intercept_[0])
    return {"independent_histogram_trees": 128, "two_linear_fits_exact": True}


def verify_first_frame(source, saved, policy, models, offsets, thresholds, constants):
    original = policy.manager.probabilities(source[FEATURES].to_numpy(float))
    np.testing.assert_array_equal(saved.exit_score, original[:, 0])
    np.testing.assert_array_equal(saved.exit_request, original[:, 0] >= policy.thresholds["exit"])
    for j, action in enumerate(["reduce", "increase"], start=1):
        first = source[f"past_{action}_exists"].to_numpy() == 0
        x = source.loc[first, FEATURES].to_numpy(float)
        raw = models[action]["tree"]["models"][0]
        logit = raw_forest(raw, x)
        expected = {"tree_raw": 1 / (1 + np.exp(-np.clip(logit, -700, 700)))}
        linear = models[action]["linear"]
        standardized = (x - np.asarray(linear["mean"])) / np.asarray(linear["scale"])
        z = np.sum(standardized * np.asarray(linear["coef"]), axis=1) + linear["intercept"]
        expected["linear_raw"] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        for name, values in expected.items():
            np.testing.assert_allclose(saved.loc[first, action + "_" + name + "_score"], values, rtol=0, atol=1e-12)
        for name in ["tree", "linear"]:
            # 요청 문턱은 독립 추론과 대조한 저장 확률의 연산 순서를 유지한다.
            q = np.clip(saved.loc[first, action + "_" + name + "_raw_score"].to_numpy(),
                        np.finfo(float).eps, 1 - np.finfo(float).eps)
            z = np.log(q) - np.log1p(-q) + offsets[action][name]
            expected[name] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            np.testing.assert_array_equal(saved.loc[first, action + "_" + name + "_score"], expected[name])
        np.testing.assert_array_equal(saved.loc[first, action + "_constant_score"], np.full(first.sum(), constants[action]))
        limit = np.where(first, policy.first_thresholds[action], policy.thresholds[action])
        np.testing.assert_array_equal(saved[action + "_baseline_score"], original[:, j])
        np.testing.assert_array_equal(saved[action + "_baseline_request"], original[:, j] >= limit)
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            scores = saved[action + "_" + name + "_score"].to_numpy()
            requests = saved[action + "_" + name + "_request"].to_numpy()
            np.testing.assert_array_equal(scores[~first], original[~first, j])
            np.testing.assert_array_equal(requests[~first], original[~first, j] >= policy.thresholds[action])
            np.testing.assert_array_equal(requests[first], scores[first] >= thresholds[action][name])
    for key in ["end", "episode_id", "y_exit", "y_reduce", "y_increase", "past_reduce_exists", "past_increase_exists"]:
        np.testing.assert_array_equal(saved[key], source[key])


def verify_first_calibration(saved, offsets, thresholds, details, policy):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        part = saved.loc[saved[f"past_{action}_exists"].eq(0)]
        y = part["y_" + action].to_numpy(int)
        for name in ["tree", "linear"]:
            q = np.clip(part[action + "_" + name + "_raw_score"].to_numpy(),
                        np.finfo(float).eps, 1 - np.finfo(float).eps)
            logits = np.log(q) - np.log1p(-q)
            low, high = -64., 64.
            for _ in range(100):
                center = (low + high) / 2
                mean = np.mean(1 / (1 + np.exp(-np.clip(logits + center, -700, 700))))
                if mean < y.mean():
                    low = center
                else:
                    high = center
            assert abs(offsets[action][name] - (low + high) / 2) < 1e-12
            assert abs(part[action + "_" + name + "_score"].mean() - y.mean()) < 1e-12
        old, _ = manual_threshold(y, part[action + "_baseline_score"].to_numpy(), beta)
        assert old == policy.first_thresholds[action]
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            boundary, expected = manual_threshold(y, part[action + "_" + name + "_score"].to_numpy(), beta)
            assert boundary == thresholds[action][name]
            compare_statistics(details[action][name], expected)


def verify_first_evaluation(saved, evaluation):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        first = saved[f"past_{action}_exists"].eq(0).to_numpy()
        for group, mask in [("all", np.ones(len(saved), dtype=bool)), ("first", first), ("repeat", ~first)]:
            for name in ["baseline", "tree_raw", "linear_raw", "tree", "linear", "constant"]:
                if not mask.any():
                    assert evaluation[action][group][name] is None
                    continue
                y = saved.loc[mask, "y_" + action].to_numpy(int)
                p = saved.loc[mask, action + "_" + name + "_score"].to_numpy()
                requests = saved.loc[mask, action + "_" + name + "_request"].to_numpy(bool)
                m = manual_metrics(y, p, 0., beta)
                tp = int(np.count_nonzero(requests & (y == 1)))
                fp = int(np.count_nonzero(requests & (y == 0)))
                fn = int(np.sum(y)) - tp
                denominator = (1 + beta**2) * tp + beta**2 * fn + fp
                m.update(predicted_positive=int(sum(requests)), true_positive=tp, false_positive=fp,
                    false_negative=fn, precision=tp / (tp + fp) if tp + fp else 0.,
                    recall=tp / (tp + fn) if tp + fn else 0.,
                    f_beta=(1 + beta**2) * tp / denominator if denominator else 0.)
                compare_statistics(evaluation[action][group][name], m)


def verify_first_admission(metrics, unchanged, decision):
    flags = {}
    for action in ["reduce", "increase"]:
        data = metrics[action]["first"]
        for name in ["baseline", "linear"]:
            old, new = data[name], data["tree"]
            flags[action + "_log_loss_" + name] = bool(1 - new["log_loss"] / old["log_loss"] > .01 + 1e-12) if old["log_loss"] else False
            flags[action + "_ap_" + name] = bool(new["average_precision"] >= old["average_precision"] - 1e-12)
        flags[action + "_raw_preserved"] = bool(data["tree"]["log_loss"] <= data["tree_raw"]["log_loss"] + 1e-12)
        flags[action + "_constant_beaten"] = bool(data["tree"]["log_loss"] < data["constant"]["log_loss"] - 1e-12)
        for key in ["recall", "f_beta"]:
            flags[action + "_first_" + key] = bool(data["tree"][key] > data["baseline"][key] + 1e-12)
        flags[action + "_all_f_preserved"] = bool(metrics[action]["all"]["tree"]["f_beta"] >= metrics[action]["all"]["baseline"]["f_beta"] - 1e-12)
    flags.update(unchanged)
    assert decision["checks"] == flags and decision["first_state_models_admitted"] == all(flags.values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
