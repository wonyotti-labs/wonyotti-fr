from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .add_count_reference import requested_metrics
from .causal_management_reference import verify_causal_admission
from .management_capacity_reference import partition_forest, verify_tree_model
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups
from .uncapped_management import CLASSES, FEATURES


def verify_uncapped_refits(rows, models):
    trees = 0
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            phases = {p: f.loc[independent_groups(f, action)["quiet"], FEATURES].to_numpy(float) for p, f in rows.items()}
            train = rows["training"]
            y = train.loc[independent_groups(train, action)["quiet"], "y_" + action].to_numpy(int)
            model = CLASSES[action].from_dict(models[action]["tree"])
            predictions = {p: model.probabilities(x) for p, x in phases.items()}
            verify_tree_model(model, "depth2_iter256", phases["training"], y[:, None], phases, predictions)
            trees += sum(len(v["trees"]) for v in models[action]["tree"]["models"])
            assert models[action]["scope"] == "past_action_count_in_15m_equals_zero"
            x = phases["training"]
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
            assert learner.n_iter_.max() < 2000
            saved = models[action]["linear"]
            for key, values in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", learner.coef_[0])]:
                np.testing.assert_array_equal(saved[key], values)
            assert saved["intercept"] == float(learner.intercept_[0])
    assert trees == 512
    return {"independent_histogram_trees": trees, "two_linear_refits_exact": True}


def verify_uncapped_quiet_predictions(source, previous, saved, models, offsets, thresholds, constants):
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
        x = source.loc[quiet, FEATURES].to_numpy(float)
        z = partition_forest(models[action]["tree"]["models"][0], x)
        expected = {"tree_raw": 1 / (1 + np.exp(-np.clip(z, -700, 700)))}
        linear = models[action]["linear"]
        z = np.sum((x - linear["mean"]) / linear["scale"] * linear["coef"], axis=1) + linear["intercept"]
        expected["linear_raw"] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        for name, value in expected.items():
            np.testing.assert_allclose(saved.loc[quiet, action + "_" + name + "_score"], value, atol=1e-12, rtol=0)
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


def verify_uncapped_admission(candidate, previous, unchanged, pre_limit, decision):
    extra = {a + "_uncapped_log_loss_gain" for a in ["reduce", "increase"]}
    extra |= {"pre_limit_restart_increase_" + k + "_improved" for k in ["recall", "f_beta"]}
    extra.add("pre_limit_restart_increase_precision_preserved")
    base = {k: v for k, v in decision["checks"].items() if k not in extra}
    verify_causal_admission(candidate, previous, unchanged, {"checks": base, "causal_distribution_admitted": all(base.values()),
        "profitability_accepted": False, "trading_returns_evaluated": False})
    for action in ["reduce", "increase"]:
        denominator = previous[action]["quiet"]["tree"]["log_loss"]
        gain = 1 - candidate[action]["quiet"]["tree"]["log_loss"] / denominator if denominator else 0.
        assert decision["checks"][action + "_uncapped_log_loss_gain"] == bool(gain > .01 + 1e-12)
    old, new = pre_limit["v127"], pre_limit["v135"]
    for key in ["recall", "f_beta"]:
        assert decision["checks"]["pre_limit_restart_increase_" + key + "_improved"] == bool(new[key] - old[key] > 1e-12)
    assert decision["checks"]["pre_limit_restart_increase_precision_preserved"] == bool(new["precision"] >= old["precision"] - 1e-12)
    assert len(decision["checks"]) == 40
    assert decision["uncapped_count_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
