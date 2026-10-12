from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .add_count_reference import requested_metrics
from .causal_management_reference import verify_causal_admission
from .count_region_management import CLASSES, FEATURES
from .management_capacity_reference import partition_forest
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups


def independent_regions(frame, action):
    quiet = independent_groups(frame, action)["quiet"]
    below = np.array([count < 5 for count in frame.adds], dtype=bool)
    return {"below": quiet & below, "capped": quiet & ~below}


def verify_region_support(frame, saved):
    for action in ["reduce", "increase"]:
        expected = {}
        masks = independent_regions(frame, action)
        assert not (masks["below"] & masks["capped"]).any()
        np.testing.assert_array_equal(masks["below"] | masks["capped"], independent_groups(frame, action)["quiet"])
        for region, mask in masks.items():
            part = frame.iloc[np.flatnonzero(mask)]
            y = part["y_" + action].to_numpy(int)
            expected[region] = {"rows": len(part), "positive": int(sum(y)), "negative": int(len(y) - sum(y)),
                "episodes": len(set(part.episode_id)), "positive_episodes": len(set(part.loc[y == 1, "episode_id"]))}
        compare_statistics(saved[action], expected)


def verify_region_refit(model, x, y, phases):
    learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=128, learning_rate=.05,
        max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
        max_features=1., categorical_features=None, early_stopping=False, random_state=0,
        class_weight=None).fit(x, y)
    raw = model.to_dict()
    assert len(raw["models"]) == 1 and learner.n_iter_ == 128 and not learner.do_early_stopping_
    np.testing.assert_array_equal(learner.classes_, [0, 1])
    binary = raw["models"][0]
    assert len(binary["trees"]) == len(learner._predictors) == 128
    assert binary["baseline"] == float(learner._baseline_prediction[0, 0])
    for stored, trees in zip(binary["trees"], learner._predictors, strict=True):
        assert len(trees) == 1 and len(stored["left"]) == len(trees[0].nodes)
        for j, node in enumerate(trees[0].nodes):
            assert node["depth"] <= 2 and not node["is_categorical"]
            leaf = bool(node["is_leaf"])
            expected = {"left": -1 if leaf else int(node["left"]), "right": -1 if leaf else int(node["right"]),
                "feature": -2 if leaf else int(node["feature_idx"]), "threshold": 0. if leaf else float(node["num_threshold"]),
                "value": float(node["value"]) if leaf else 0.}
            assert all(stored[k][j] == value for k, value in expected.items())
    for values in phases.values():
        p = model.probabilities(values)[:, 0]
        np.testing.assert_allclose(p, learner.predict_proba(values)[:, 1], atol=1e-12, rtol=0)
        z = partition_forest(binary, values)
        np.testing.assert_array_equal(p, 1 / (1 + np.exp(-np.clip(z, -700, 700))))
    return 128


def verify_count_region_refits(rows, models):
    trees = 0
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            assert models[action]["scope"] == "past_action_count_in_15m_equals_zero"
            for region in ["below", "capped"]:
                saved = models[action][region]
                assert saved["region"] == region
                phases = {p: f.loc[independent_regions(f, action)[region], FEATURES].to_numpy(float) for p, f in rows.items()}
                train = rows["training"]
                y = train.loc[independent_regions(train, action)[region], "y_" + action].to_numpy(int)
                model = CLASSES[action][region].from_dict(saved["tree"])
                x = phases["training"]
                trees += verify_region_refit(model, x, y, phases)
                scaler = StandardScaler().fit(x)
                learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
                assert learner.n_iter_.max() < 2000
                linear = saved["linear"]
                assert linear["region"] == region and linear["action"] == action and linear["features"] == FEATURES
                for key, values in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", learner.coef_[0])]:
                    np.testing.assert_array_equal(linear[key], values)
                assert linear["intercept"] == float(learner.intercept_[0])
    assert trees == 512
    return {"independent_histogram_trees": trees, "four_linear_refits_exact": True}


def verify_count_region_quiet_predictions(source, previous, saved, models, offsets, thresholds, constants):
    changed = {a + "_" + name + "_" + suffix for a in ["reduce", "increase"]
               for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"] for suffix in ["score", "request"]}
    fixed = [k for k in previous if k not in changed]
    pd.testing.assert_frame_equal(previous[fixed], saved[fixed], check_exact=True)
    for action in ["reduce", "increase"]:
        quiet = independent_groups(source, action)["quiet"]
        fields = [k for k in changed if k.startswith(action + "_")]
        pd.testing.assert_frame_equal(previous.loc[~quiet, fields], saved.loc[~quiet, fields], check_exact=True)
        if not quiet.any():
            continue
        expected = {}
        for region, mask in independent_regions(source, action).items():
            if not mask.any():
                continue
            x = source.loc[mask, FEATURES].to_numpy(float)
            z = partition_forest(models[action][region]["tree"]["models"][0], x)
            tree_p = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            linear = models[action][region]["linear"]
            z = np.sum((x - linear["mean"]) / linear["scale"] * linear["coef"], axis=1) + linear["intercept"]
            linear_p = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            for name, value in [("tree_raw", tree_p), ("linear_raw", linear_p)]:
                np.testing.assert_allclose(saved.loc[mask, action + "_" + name + "_score"], value, atol=1e-12, rtol=0)
        for name in ["tree", "linear"]:
            p = np.clip(saved.loc[quiet, action + "_" + name + "_raw_score"].to_numpy(), np.finfo(float).eps, 1-np.finfo(float).eps)
            expected[name] = 1 / (1 + np.exp(-np.clip(np.log(p) - np.log1p(-p) + offsets[action][name], -700, 700)))
            np.testing.assert_array_equal(saved.loc[quiet, action + "_" + name + "_score"], expected[name])
        np.testing.assert_array_equal(saved.loc[quiet, action + "_constant_score"], np.full(quiet.sum(), constants[action]))
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            p = saved.loc[quiet, action + "_" + name + "_score"].to_numpy()
            np.testing.assert_array_equal(saved.loc[quiet, action + "_" + name + "_request"], p >= thresholds[action][name])
    return {"exit": True, "recent_reduce": True, "recent_increase": True}


def verify_pre_limit_restart(frame, predictions, saved):
    mask = independent_groups(frame, "increase")["restart"] & (frame.adds.to_numpy() < 5)
    y = frame.loc[mask, "y_increase"].to_numpy(int)
    expected = {name: requested_metrics(y, values.loc[mask, "increase_tree_score"].to_numpy(),
        values.loc[mask, "increase_tree_request"].to_numpy(bool), .5) for name, values in predictions.items()}
    compare_statistics(saved, expected)


def verify_count_region_admission(candidate, previous, unchanged, pre_limit, decision):
    extra = {a + "_count_region_log_loss_gain" for a in ["reduce", "increase"]}
    extra |= {"pre_limit_restart_increase_" + k + "_improved" for k in ["recall", "f_beta"]}
    extra.add("pre_limit_restart_increase_precision_preserved")
    base = {k: v for k, v in decision["checks"].items() if k not in extra}
    verify_causal_admission(candidate, previous, unchanged, {"checks": base, "causal_distribution_admitted": all(base.values()),
        "profitability_accepted": False, "trading_returns_evaluated": False})
    for action in ["reduce", "increase"]:
        denominator = previous[action]["quiet"]["tree"]["log_loss"]
        gain = 1 - candidate[action]["quiet"]["tree"]["log_loss"] / denominator if denominator else 0.
        assert decision["checks"][action + "_count_region_log_loss_gain"] == bool(gain > .01 + 1e-12)
    old, new = pre_limit["v127"], pre_limit["v137"]
    for key in ["recall", "f_beta"]:
        assert decision["checks"]["pre_limit_restart_increase_" + key + "_improved"] == bool(new[key] - old[key] > 1e-12)
    assert decision["checks"]["pre_limit_restart_increase_precision_preserved"] == bool(new["precision"] >= old["precision"] - 1e-12)
    assert len(decision["checks"]) == 40
    assert decision["count_region_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False


def verify_count_regions(frame, predictions, saved):
    expected = {}
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        expected[action] = {}
        for region, mask in independent_regions(frame, action).items():
            part = frame.iloc[np.flatnonzero(mask)]
            y = part["y_" + action].to_numpy(int)
            support = {"rows": len(y), "positive": int(sum(y)), "negative": int(len(y) - sum(y)),
                "episodes": len(set(part.episode_id)), "positive_episodes": len(set(part.loc[y == 1, "episode_id"]))}
            models = {}
            for model, values in predictions.items():
                models[model] = {}
                for name in ["baseline", "capacity", "tree_raw", "linear_raw", "tree", "linear", "constant"]:
                    models[model][name] = requested_metrics(y, values.loc[mask, action + "_" + name + "_score"].to_numpy(),
                        values.loc[mask, action + "_" + name + "_request"].to_numpy(bool), beta)
            expected[action][region] = {"support": support, "models": models}
    compare_statistics(saved, expected)


def verify_routing_ledger(frame, ledger):
    pd.testing.assert_frame_equal(frame[["end", "episode_id"]], ledger[["end", "episode_id"]], check_exact=True)
    for action in ["reduce", "increase"]:
        expected = np.full(len(frame), "recent", dtype=object)
        for region, mask in independent_regions(frame, action).items():
            expected[mask] = region
        np.testing.assert_array_equal(ledger[action + "_region"], expected)
