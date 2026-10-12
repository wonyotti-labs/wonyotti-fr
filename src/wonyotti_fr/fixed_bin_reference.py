from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .causal_management_reference import verify_causal_admission
from .first_state_models import FEATURES
from .fixed_bin_management import (
    encode_bins,
    lift_thresholds,
    model_digest,
    training_digest,
    validate_bins,
)
from .management_capacity_reference import partition_forest
from .quiet_management import CLASSES
from .quiet_management_reference import independent_groups
from .tail_weighted_reference import verify_tail_weights


def reference_encode(bins, x):
    return np.array(
        [np.digitize(x[:, j], edges, right=True) for j, edges in enumerate(bins["thresholds"])],
        dtype=np.uint8,
    ).T


def verify_binning(bins, action, x, mapper):
    validate_bins(bins, action)
    expected = reference_encode(bins, x)
    np.testing.assert_array_equal(expected, encode_bins(bins, action, x))
    np.testing.assert_array_equal(expected, mapper.transform(x))
    probe_count = 0
    for j, edges in enumerate(bins["thresholds"]):
        np.testing.assert_array_equal(edges, mapper.bin_thresholds_[j])
        q = np.array(edges, dtype=float)
        with np.errstate(over="ignore"):
            probes = np.r_[
                -np.finfo(float).max,
                np.nextafter(q, -np.inf),
                q,
                np.nextafter(q, np.inf),
                np.finfo(float).max,
            ]
        probes = probes[np.isfinite(probes)]
        values = np.zeros((len(probes), len(FEATURES)))
        values[:, j] = probes
        direct = (probes[:, None] > q).sum(axis=1)
        np.testing.assert_array_equal(encode_bins(bins, action, values)[:, j], direct)
        np.testing.assert_array_equal(mapper.transform(values)[:, j], direct)
        probe_count += len(probes)
    return expected, probe_count


def verify_nodes(learner, saved):
    binary = saved["models"][0]
    assert learner.n_iter_ == 256 and len(binary["trees"]) == 256
    assert not learner.do_early_stopping_ and np.array_equal(learner.classes_, [0, 1])
    assert float(learner._baseline_prediction[0, 0]) == binary["baseline"]
    for raw, predictors in zip(binary["trees"], learner._predictors, strict=True):
        assert len(predictors) == 1 and len(raw["feature"]) == len(predictors[0].nodes)
        for i, node in enumerate(predictors[0].nodes):
            assert node["depth"] <= 2 and not node["is_categorical"]
            leaf = bool(node["is_leaf"])
            for name, value in {
                "left": -1 if leaf else int(node["left"]),
                "right": -1 if leaf else int(node["right"]),
                "feature": -2 if leaf else int(node["feature_idx"]),
                "threshold": 0.0 if leaf else float(node["num_threshold"]),
                "value": float(node["value"]) if leaf else 0.0,
            }.items():
                assert raw[name][i] == value


def verify_projection(encoded, lifted, bins):
    a, b = encoded["models"][0], lifted["models"][0]
    assert a["action"] == b["action"] and a["baseline"] == b["baseline"]
    for source, target in zip(a["trees"], b["trees"], strict=True):
        assert all(source[k] == target[k] for k in ["feature", "left", "right", "value"])
        for i, feature in enumerate(source["feature"]):
            if feature < 0:
                assert source["threshold"][i] == target["threshold"][i]
            else:
                levels = np.arange(len(bins["thresholds"][feature]) + 1)
                left = levels[levels <= source["threshold"][i]]
                assert 0 < len(left) < len(levels)
                assert target["threshold"][i] == bins["thresholds"][feature][left[-1]]


def verify_fixed_refits(rows, parents, weighted, ledger, details, fitted):
    models, bins, controls, encoded, levels = fitted
    result = {
        "original_trees": 0,
        "control_trees": 0,
        "weighted_trees": 0,
        "boundary_probes": 0,
        "all_training_maps_weights_and_levels_exact": True,
        "all_period_raw_probabilities_verified": True,
        "weighted_linear_models_preserved": True,
        "weighted_linear_refits": 2,
    }
    train = rows["training"]
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            quiet = independent_groups(train, action)["quiet"]
            x, y = (
                train.loc[quiet, FEATURES].to_numpy(float),
                train.loc[quiet, "y_" + action].to_numpy(int),
            )
            params = dict(
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
            )
            original = HistGradientBoostingClassifier(**params).fit(x, y)
            verify_nodes(original, parents[action]["tree"])
            assert bins[action]["parent_sha256"] == model_digest(parents[action]["tree"])
            assert bins[action]["training_sha256"] == training_digest(train, action)
            assert bins[action]["rows"] == len(x)
            z, probes = verify_binning(bins[action], action, x, original._bin_mapper)
            result["boundary_probes"] += probes
            score = 1 / (
                1
                + np.exp(
                    -np.clip(partition_forest(parents[action]["tree"]["models"][0], x), -700, 700)
                )
            )
            np.testing.assert_array_equal(
                score, ledger.loc[quiet, action + "_score"].to_numpy(float)
            )
            w = ledger.loc[quiet, action + "_weight"].to_numpy(float)
            verify_tail_weights(
                y,
                score,
                w,
                ledger.loc[quiet, action + "_hard_negative"].to_numpy(bool),
                details[action],
            )
            scaler = StandardScaler().fit(x)
            linear = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                scaler.transform(x), y, sample_weight=w
            )
            assert linear.n_iter_.max() < 2000
            stored = weighted[action]["linear"]
            for key, value in [
                ("mean", scaler.mean_),
                ("scale", scaler.scale_),
                ("coef", linear.coef_[0]),
            ]:
                np.testing.assert_array_equal(stored[key], value)
            assert stored["intercept"] == float(linear.intercept_[0])
            learners = {
                "control": HistGradientBoostingClassifier(**params).fit(z, y),
                "weighted": HistGradientBoostingClassifier(**params).fit(z, y, sample_weight=w),
            }
            for name, learner in learners.items():
                saved = controls[action] if name == "control" else encoded[action]
                verify_nodes(learner, saved)
                expected_levels = [q.tolist() for q in learner._bin_mapper.bin_thresholds_]
                assert levels[action][name] == expected_levels
                np.testing.assert_array_equal(learner._bin_mapper.transform(z), z)
            assert levels[action]["control"] == levels[action]["weighted"]
            assert (
                lift_thresholds(controls[action], bins[action], action) == parents[action]["tree"]
            )
            verify_projection(controls[action], parents[action]["tree"], bins[action])
            verify_projection(encoded[action], models[action]["tree"], bins[action])
            assert models[action]["linear"] == weighted[action]["linear"]
            for frame in rows.values():
                values = frame.loc[independent_groups(frame, action)["quiet"], FEATURES].to_numpy(
                    float
                )
                binned = reference_encode(bins[action], values)
                np.testing.assert_array_equal(binned, original._bin_mapper.transform(values))
                for name, raw, expected_raw in [
                    ("control", controls[action], parents[action]["tree"]),
                    ("weighted", encoded[action], models[action]["tree"]),
                ]:
                    numeric = 1 / (
                        1 + np.exp(-np.clip(partition_forest(raw["models"][0], binned), -700, 700))
                    )
                    other = 1 / (
                        1
                        + np.exp(
                            -np.clip(partition_forest(expected_raw["models"][0], values), -700, 700)
                        )
                    )
                    np.testing.assert_array_equal(numeric, other)
                    np.testing.assert_array_equal(
                        other, CLASSES[action].from_dict(expected_raw).probabilities(values)[:, 0]
                    )
                    np.testing.assert_allclose(
                        numeric, learners[name].predict_proba(binned)[:, 1], atol=1e-12, rtol=0
                    )
            for key in ["original_trees", "control_trees", "weighted_trees"]:
                result[key] += 256
    return result


def verify_fixed_admission(
    candidate, previous, unchanged, pre_limit, budget, decision, *, model_name="v141"
):
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
    assert set(pre_limit) == {"v127", model_name}
    old, new = pre_limit["v127"], pre_limit[model_name]
    for key in ["recall", "f_beta"]:
        assert checks["pre_limit_restart_increase_" + key + "_improved"] == bool(
            new[key] > old[key] + 1e-12
        )
    assert checks["pre_limit_restart_increase_precision_preserved"] == bool(
        new["precision"] >= old["precision"] - 1e-12
    )
    focal = budget["increase__restart__below"]["models"]
    assert set(focal) == {"v127", model_name}
    old, new = (focal[m]["budget_matched"] for m in ["v127", model_name])
    supported = old is not None and new is not None
    assert checks["budget_pre_limit_restart_increase_tp_improved"] == bool(
        supported and new["true_positive"] > old["true_positive"]
    )
    assert checks["budget_pre_limit_restart_increase_fp_preserved"] == bool(
        supported and new["false_positive"] <= old["false_positive"]
    )
    assert len(checks) == 42 and set(checks) == set(base) | extra
    assert decision["fixed_bins_admitted"] == all(checks.values())
    assert decision["profitability_accepted"] is decision["trading_returns_evaluated"] is False
