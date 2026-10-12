from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES, linear_probabilities
from .fixed_bin_reference import (
    reference_encode,
    verify_binning,
    verify_fixed_admission,
    verify_nodes,
    verify_projection,
)
from .management_capacity_reference import partition_forest
from .quiet_management import CLASSES
from .quiet_management_reference import independent_groups


def verify_forward_final_refits(rows, parents, bins, ledgers, fitted):
    models, controls, encoded, levels = fitted
    assert set(models) == set(encoded) == {"v142matched", "v142forward"}
    result = {
        "original_trees": 0,
        "control_trees": 0,
        "new_final_trees": 0,
        "new_linear_models": 0,
        "boundary_probes": 0,
        "all_period_probabilities_verified": True,
    }
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
    train = rows["training"]
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            quiet = independent_groups(train, action)["quiet"]
            x, y = (
                train.loc[quiet, FEATURES].to_numpy(float),
                train.loc[quiet, "y_" + action].to_numpy(int),
            )
            original = HistGradientBoostingClassifier(**params).fit(x, y)
            verify_nodes(original, parents[action]["tree"])
            z, probes = verify_binning(bins[action], action, x, original._bin_mapper)
            result["boundary_probes"] += probes
            control = HistGradientBoostingClassifier(**params).fit(z, y)
            verify_nodes(control, controls[action])
            verify_projection(controls[action], parents[action]["tree"], bins[action])
            assert levels[action]["control"] == [
                q.tolist() for q in control._bin_mapper.bin_thresholds_
            ]
            np.testing.assert_array_equal(control._bin_mapper.transform(z), z)
            result["original_trees"] += 256
            result["control_trees"] += 256
            for arm in ["v142matched", "v142forward"]:
                weights = ledgers[arm].loc[quiet, action + "_weight"].to_numpy(float)
                learner = HistGradientBoostingClassifier(**params).fit(z, y, sample_weight=weights)
                verify_nodes(learner, encoded[arm][action])
                verify_projection(encoded[arm][action], models[arm][action]["tree"], bins[action])
                assert levels[action][arm] == [
                    q.tolist() for q in learner._bin_mapper.bin_thresholds_
                ]
                assert levels[action][arm] == levels[action]["control"]
                np.testing.assert_array_equal(learner._bin_mapper.transform(z), z)
                scaler = StandardScaler().fit(x)
                linear = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                    scaler.transform(x), y, sample_weight=weights
                )
                assert linear.n_iter_.max() < 2000
                saved = models[arm][action]["linear"]
                for key, expected in [
                    ("mean", scaler.mean_),
                    ("scale", scaler.scale_),
                    ("coef", linear.coef_[0]),
                ]:
                    np.testing.assert_array_equal(saved[key], expected)
                assert saved["intercept"] == float(linear.intercept_[0])
                for frame in rows.values():
                    values = frame.loc[
                        independent_groups(frame, action)["quiet"], FEATURES
                    ].to_numpy(float)
                    binned = reference_encode(bins[action], values)
                    predictions = []
                    for raw, source in [
                        (encoded[arm][action], binned),
                        (models[arm][action]["tree"], values),
                    ]:
                        logits = partition_forest(raw["models"][0], source)
                        numeric = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
                        np.testing.assert_array_equal(
                            numeric, CLASSES[action].from_dict(raw).probabilities(source)[:, 0]
                        )
                        predictions.append(numeric)
                    np.testing.assert_array_equal(*predictions)
                    np.testing.assert_allclose(
                        predictions[0], learner.predict_proba(binned)[:, 1], atol=1e-12, rtol=0
                    )
                    np.testing.assert_allclose(
                        linear_probabilities(saved, values),
                        linear.predict_proba(scaler.transform(values))[:, 1],
                        atol=1e-12,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        CLASSES[action]
                        .from_dict(parents[action]["tree"])
                        .probabilities(values)[:, 0],
                        control.predict_proba(binned)[:, 1],
                        atol=1e-12,
                        rtol=0,
                    )
                result["new_final_trees"] += 256
                result["new_linear_models"] += 1
    return result


def verify_forward_final_admission(
    candidate, matched, previous, unchanged, pre_limit, budget, matched_budget, decision
):
    names = {
        action + suffix
        for action in ["reduce", "increase"]
        for suffix in ["_matched_log_loss_gain", "_matched_ap_preserved"]
    }
    names |= {
        "matched_budget_pre_limit_restart_increase_tp_improved",
        "matched_budget_pre_limit_restart_increase_fp_preserved",
    }
    base = {k: v for k, v in decision["checks"].items() if k not in names}
    verify_fixed_admission(
        candidate,
        previous,
        unchanged,
        pre_limit,
        budget,
        {
            "checks": base,
            "fixed_bins_admitted": all(base.values()),
            "profitability_accepted": False,
            "trading_returns_evaluated": False,
        },
        model_name="v142forward",
    )
    for action in ["reduce", "increase"]:
        a, b = candidate[action]["quiet"]["tree"], matched[action]["quiet"]["tree"]
        assert decision["checks"][action + "_matched_log_loss_gain"] == bool(
            (1 - a["log_loss"] / b["log_loss"] if b["log_loss"] else 0) > 0.01 + 1e-12
        )
        assert decision["checks"][action + "_matched_ap_preserved"] == bool(
            a["average_precision"] >= b["average_precision"] - 1e-12
        )
    a = budget["increase__restart__below"]["models"]["v142forward"]["budget_matched"]
    b = matched_budget["increase__restart__below"]["models"]["v142matched"]["budget_matched"]
    expected = {
        "matched_budget_pre_limit_restart_increase_tp_improved": bool(
            a is not None and b is not None and a["true_positive"] > b["true_positive"]
        ),
        "matched_budget_pre_limit_restart_increase_fp_preserved": bool(
            a is not None and b is not None and a["false_positive"] <= b["false_positive"]
        ),
    }
    for key, value in expected.items():
        assert decision["checks"][key] == value
    assert len(decision["checks"]) == 48 and set(decision["checks"]) == set(base) | names
    assert decision["forward_weight_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is decision["trading_returns_evaluated"] is False
