from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .causal_management_reference import verify_causal_admission
from .first_state_models import FEATURES
from .management_capacity_reference import partition_forest
from .position_boost_verification import compare_statistics
from .quiet_management import CLASSES
from .quiet_management_reference import independent_groups


def reference_weights(labels, scores):
    negatives = Counter(float(p) for y, p in zip(labels, scores, strict=True) if y == 0)
    n0, running, rank = sum(negatives.values()), 0, 1
    while rank * 100 < n0:
        rank += 1
    q = None
    for value in sorted(negatives, reverse=True):
        running += negatives[value]
        if running >= rank:
            q = value
            break
    assert q is not None
    hard = np.array([y == 0 and float(p) >= q for y, p in zip(labels, scores, strict=True)])
    units = [10 if y == 1 or flag else 1 for y, flag in zip(labels, hard, strict=True)]
    denominator = sum(units)
    weights = np.array([float(Fraction(len(labels) * u, denominator)) for u in units])
    return weights, hard, q, rank, negatives[q], denominator


def reference_weight_statistics(labels, hard, weights):
    parts = {"all": [], "positive": [], "hard_negative": [], "easy_negative": []}
    for y, flag, w in zip(labels, hard, weights, strict=True):
        assert y in (0, 1) and not (y == 1 and flag) and math.isfinite(w) and w > 0
        parts["all"].append(float(w))
        parts["positive" if y == 1 else "hard_negative" if flag else "easy_negative"].append(
            float(w)
        )
    total, result = math.fsum(parts["all"]), {}
    for key, values in parts.items():
        mass = math.fsum(values)
        result[key] = {
            "rows": len(values),
            "sum": mass,
            "mean": mass / len(values) if values else None,
            "effective_rows": mass**2 / math.fsum(x**2 for x in values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "maximum_share": max(values) / total if values else None,
        }
    return result


def verify_tail_weights(labels, scores, weights, hard, detail):
    expected, flags, q, rank, ties, denominator = reference_weights(labels, scores)
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(hard, flags)
    assert detail["boundary_score"] == q
    compare_statistics(
        detail,
        {
            "rule": "training_negative_top_one_percent_ties_high1_low01_mean1",
            "negative_rows": int(sum(np.asarray(labels) == 0)),
            "rank": rank,
            "boundary_score": q,
            "boundary_ties": ties,
            "hard_negative_fraction": float(sum(flags) / sum(np.asarray(labels) == 0)),
            "normalization_units": denominator,
            "statistics": reference_weight_statistics(labels, flags, expected),
        },
    )


def verify_weight_groups(frame, ledger, actual):
    expected = {}
    for action in ["reduce", "increase"]:
        states = independent_groups(frame, action)
        quiet = states["quiet"]
        partitions = {"all_training": quiet}
        for state, mask in states.items():
            partitions[state] = mask & quiet
            for label, counts in [
                ("below", frame.adds.to_numpy() < 5),
                ("capped", frame.adds.to_numpy() >= 5),
            ]:
                partitions[state + "__" + label] = mask & quiet & counts
        months = frame.end.dt.strftime("%Y-%m").to_numpy()
        for month in sorted(set(months)):
            partitions[month] = (months == month) & quiet
        expected[action] = {}
        for group, mask in partitions.items():
            expected[action][group] = reference_weight_statistics(
                frame.loc[mask, "y_" + action].to_numpy(),
                ledger.loc[mask, action + "_hard_negative"].to_numpy(bool),
                ledger.loc[mask, action + "_weight"].to_numpy(float),
            )
    compare_statistics(actual, expected)


def verify_tail_refits(rows, parents, models, ledger, details):
    count = 0
    train = rows["training"]
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            mask = independent_groups(train, action)["quiet"]
            x, y = (
                train.loc[mask, FEATURES].to_numpy(float),
                train.loc[mask, "y_" + action].to_numpy(int),
            )
            raw_parent = parents[action]["tree"]["models"][0]
            raw_scores = 1 / (1 + np.exp(-np.clip(partition_forest(raw_parent, x), -700, 700)))
            np.testing.assert_array_equal(
                ledger.loc[mask, action + "_score"].to_numpy(float), raw_scores
            )
            weights, hard = (
                ledger.loc[mask, action + "_" + suffix].to_numpy(dtype)
                for suffix, dtype in [("weight", float), ("hard_negative", bool)]
            )
            verify_tail_weights(y, raw_scores, weights, hard, details[action])
            independent_w = reference_weights(y, raw_scores)[0]
            learner = HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=256,
                learning_rate=0.05,
                max_leaf_nodes=4,
                max_depth=2,
                min_samples_leaf=64,
                l2_regularization=1.0,
                max_bins=255,
                max_features=1.0,
                categorical_features=None,
                early_stopping=False,
                random_state=0,
                class_weight=None,
            ).fit(x, y, sample_weight=independent_w)
            raw = models[action]["tree"]["models"][0]
            assert raw["baseline"] == float(learner._baseline_prediction[0, 0])
            assert learner.n_iter_ == len(raw["trees"]) == 256 and not learner.do_early_stopping_
            for stored, trees in zip(raw["trees"], learner._predictors, strict=True):
                assert len(trees) == 1 and len(stored["left"]) == len(trees[0].nodes)
                for i, node in enumerate(trees[0].nodes):
                    assert node["depth"] <= 2 and not node["is_categorical"]
                    leaf = bool(node["is_leaf"])
                    expected = {
                        "left": -1 if leaf else int(node["left"]),
                        "right": -1 if leaf else int(node["right"]),
                        "feature": -2 if leaf else int(node["feature_idx"]),
                        "threshold": 0.0 if leaf else float(node["num_threshold"]),
                        "value": float(node["value"]) if leaf else 0.0,
                    }
                    assert all(stored[key][i] == value for key, value in expected.items())
            scaler = StandardScaler().fit(x)
            linear = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                scaler.transform(x), y, sample_weight=independent_w
            )
            assert linear.n_iter_.max() < 2000
            stored = models[action]["linear"]
            for key, values in [
                ("mean", scaler.mean_),
                ("scale", scaler.scale_),
                ("coef", linear.coef_[0]),
            ]:
                np.testing.assert_array_equal(stored[key], values)
            assert stored["intercept"] == float(linear.intercept_[0])
            model = CLASSES[action].from_dict(models[action]["tree"])
            for frame in rows.values():
                values = frame.loc[independent_groups(frame, action)["quiet"], FEATURES].to_numpy(
                    float
                )
                numeric = 1 / (1 + np.exp(-np.clip(partition_forest(raw, values), -700, 700)))
                np.testing.assert_array_equal(numeric, model.probabilities(values)[:, 0])
                np.testing.assert_allclose(
                    numeric, learner.predict_proba(values)[:, 1], atol=1e-12, rtol=0
                )
                z = (
                    np.sum((values - stored["mean"]) / stored["scale"] * stored["coef"], axis=1)
                    + stored["intercept"]
                )
                np.testing.assert_allclose(
                    1 / (1 + np.exp(-np.clip(z, -700, 700))),
                    linear.predict_proba(scaler.transform(values))[:, 1],
                    atol=1e-12,
                    rtol=0,
                )
            count += len(raw["trees"])
    return {
        "independent_histogram_trees": count,
        "weighted_linear_refits": 2,
        "all_weights_parent_scores_and_model_parameters_verified": True,
    }


def verify_tail_admission(candidate, previous, unchanged, pre_limit, budget, decision):
    extra = {a + "_tail_weight_log_loss_gain" for a in ["reduce", "increase"]}
    extra |= {"pre_limit_restart_increase_" + k + "_improved" for k in ["recall", "f_beta"]}
    extra |= {
        "pre_limit_restart_increase_precision_preserved",
        "budget_pre_limit_restart_increase_tp_improved",
        "budget_pre_limit_restart_increase_fp_preserved",
    }
    checks = decision["checks"]
    base = {k: v for k, v in checks.items() if k not in extra}
    verify_causal_admission(
        candidate,
        previous,
        unchanged,
        {
            "checks": base,
            "causal_distribution_admitted": all(base.values()),
            "profitability_accepted": False,
            "trading_returns_evaluated": False,
        },
    )
    for action in ["reduce", "increase"]:
        a, b = candidate[action]["quiet"]["tree"], previous[action]["quiet"]["tree"]
        assert checks[action + "_tail_weight_log_loss_gain"] == bool(
            (1 - a["log_loss"] / b["log_loss"] if b["log_loss"] else 0) > 0.01 + 1e-12
        )
    old, new = pre_limit["v127"], pre_limit["v140"]
    for key in ["recall", "f_beta"]:
        assert checks["pre_limit_restart_increase_" + key + "_improved"] == bool(
            new[key] > old[key] + 1e-12
        )
    assert checks["pre_limit_restart_increase_precision_preserved"] == bool(
        new["precision"] >= old["precision"] - 1e-12
    )
    focal = budget["increase__restart__below"]["models"]
    old, new = (focal[m]["budget_matched"] for m in ["v127", "v140"])
    supported = old is not None and new is not None
    assert checks["budget_pre_limit_restart_increase_tp_improved"] == bool(
        supported and new["true_positive"] > old["true_positive"]
    )
    assert checks["budget_pre_limit_restart_increase_fp_preserved"] == bool(
        supported and new["false_positive"] <= old["false_positive"]
    )
    assert len(checks) == 42 and set(checks) == set(base) | extra
    assert decision["tail_weighted_admitted"] == all(checks.values())
    assert decision["profitability_accepted"] is decision["trading_returns_evaluated"] is False
