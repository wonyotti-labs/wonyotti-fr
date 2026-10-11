from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES
from .management_capacity_reference import partition_forest, verify_tree_model
from .management_experts import load_expert
from .paired_management_reference import manual_threshold
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups, verify_quiet_support
from .quiet_state_reference import verify_state_admission


def verify_expert_support(rows, support):
    verify_quiet_support(rows, support)
    for phase, actions in support["states"].items():
        for states in actions.values():
            for state in ["first", "restart"]:
                value = states[state]
                assert value["rows"] >= (1000 if phase == "training" else 100)
                assert min(value["positive"], value["negative"]) >= 20


def verify_expert_refits(rows, models):
    trees = 0
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            for state in ["first", "restart"]:
                phases = {p: f.loc[independent_groups(f, action)[state], FEATURES].to_numpy(float)
                          for p, f in rows.items()}
                train = rows["training"]
                y = train.loc[independent_groups(train, action)[state], "y_" + action].to_numpy(int)
                model = load_expert(models, action, state)
                predictions = {p: model.probabilities(x) for p, x in phases.items()}
                verify_tree_model(model, "depth2_iter256", phases["training"], y[:, None], phases, predictions)
                raw = models[action][state]
                trees += sum(len(binary["trees"]) for binary in raw["tree"]["models"])
                x = phases["training"]
                scaler = StandardScaler().fit(x)
                learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
                assert learner.n_iter_.max() < 2000
                for key, values in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", learner.coef_[0])]:
                    np.testing.assert_array_equal(raw["linear"][key], values)
                assert raw["linear"]["intercept"] == float(learner.intercept_[0])
    assert trees == 1024
    return {"independent_histogram_trees": trees, "four_linear_refits_exact": True}


def verify_expert_predictions(source, previous, saved, models, offsets, thresholds, constants):
    changed = {a + "_" + m + "_" + suffix for a in ["reduce", "increase"]
               for m in ["tree_raw", "linear_raw", "tree", "linear", "constant"] for suffix in ["score", "request"]}
    preserved = [k for k in previous if k not in changed]
    pd.testing.assert_frame_equal(saved[preserved], previous[preserved], check_exact=True)
    common = [k for k in saved if k in source]
    pd.testing.assert_frame_equal(saved[common], source[common], check_exact=True)
    for action in ["reduce", "increase"]:
        masks = independent_groups(source, action)
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            for suffix in ["score", "request"]:
                column = action + "_" + name + "_" + suffix
                np.testing.assert_array_equal(saved.loc[masks["recent"], column], previous.loc[masks["recent"], column])
        for state in ["first", "restart"]:
            mask = masks[state]
            x = source.loc[mask, FEATURES].to_numpy(float)
            raw = models[action][state]
            assert raw["state"] == state and raw["action"] == action
            z = partition_forest(raw["tree"]["models"][0], x)
            tree = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            linear = raw["linear"]
            z = np.sum((x - linear["mean"]) / linear["scale"] * linear["coef"], axis=1) + linear["intercept"]
            linear_raw = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            for name, expected in [("tree_raw", tree), ("linear_raw", linear_raw)]:
                np.testing.assert_allclose(saved.loc[mask, action + "_" + name + "_score"], expected, atol=1e-12, rtol=0)
            for name in ["tree", "linear"]:
                q = np.clip(saved.loc[mask, action + "_" + name + "_raw_score"].to_numpy(), np.finfo(float).eps, 1 - np.finfo(float).eps)
                expected = 1 / (1 + np.exp(-np.clip(np.log(q) - np.log1p(-q) + offsets[action][state][name], -700, 700)))
                np.testing.assert_array_equal(saved.loc[mask, action + "_" + name + "_score"], expected)
            np.testing.assert_array_equal(saved.loc[mask, action + "_constant_score"], np.full(mask.sum(), constants[action][state]))
            for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
                stem = action + "_" + name
                np.testing.assert_array_equal(saved.loc[mask, stem + "_request"],
                    saved.loc[mask, stem + "_score"].to_numpy() >= thresholds[action][state][name])
    return {"exit": True, "recent_reduce": True, "recent_increase": True}


def verify_expert_calibration(saved, offsets, thresholds, details, constants):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        for state in ["first", "restart"]:
            part = saved.loc[independent_groups(saved, action)[state]]
            y = part["y_" + action].to_numpy(int)
            assert constants[action][state] == y.mean()
            for model in ["tree", "linear"]:
                raw = np.clip(part[action + "_" + model + "_raw_score"].to_numpy(), np.finfo(float).eps, 1 - np.finfo(float).eps)
                z = np.log(raw) - np.log1p(-raw)
                low, high = -64., 64.
                for _ in range(100):
                    middle = (low + high) / 2
                    values = 1 / (1 + np.exp(-np.clip(z + middle, -700, 700)))
                    if values.mean() < y.mean():
                        low = middle
                    else:
                        high = middle
                assert abs(offsets[action][state][model] - (low + high) / 2) < 1e-12
                assert abs(part[action + "_" + model + "_score"].mean() - y.mean()) < 1e-12
            for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
                threshold, expected = manual_threshold(y, part[action + "_" + name + "_score"].to_numpy(), beta)
                assert threshold == thresholds[action][state][name]
                compare_statistics(details[action][state][name], expected)


def verify_expert_admission(candidate, previous, unchanged, decision):
    adapted = {k: v for k, v in decision.items() if k != "state_experts_admitted"}
    adapted["state_calibration_admitted"] = decision["state_experts_admitted"]
    verify_state_admission(candidate, previous, unchanged, adapted)
