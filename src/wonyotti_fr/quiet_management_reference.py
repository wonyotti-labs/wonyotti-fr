from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES
from .management_capacity_reference import partition_forest, verify_tree_model
from .paired_management_reference import manual_metrics, manual_threshold
from .position_boost_verification import compare_statistics
from .quiet_management import CLASSES, NAMES, groups


def independent_groups(frame, action):
    counts = np.rint(np.expm1(frame[f"past_{action}_log_count_15m"].to_numpy(float))).astype(np.int64)
    seen = frame[f"past_{action}_exists"].to_numpy() == 1
    return {"all": np.ones(len(frame), dtype=bool), "quiet": counts == 0, "recent": counts > 0,
            "first": ~seen, "restart": seen & (counts == 0)}


def verify_quiet_support(rows, support):
    expected = {}
    for phase, frame in rows.items():
        expected[phase] = {}
        for action in ["reduce", "increase"]:
            masks = independent_groups(frame, action)
            actual = groups(frame, action)
            expected[phase][action] = {}
            for name, mask in masks.items():
                np.testing.assert_array_equal(mask, actual[name])
                values = frame.loc[mask, "y_" + action].to_numpy(int)
                expected[phase][action][name] = {"rows": len(values), "positive": int(sum(values)),
                                                "negative": int(sum(values == 0))}
            assert not (masks["first"] & masks["recent"]).any()
            assert np.array_equal(masks["first"] | masks["restart"], masks["quiet"])
    assert support["states"] == expected


def verify_quiet_refits(rows, models):
    tree_count = 0
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            phases = {p: f.loc[independent_groups(f, action)["quiet"], FEATURES].to_numpy(float)
                      for p, f in rows.items()}
            train = rows["training"]
            y = train.loc[independent_groups(train, action)["quiet"], "y_" + action].to_numpy(int)
            model = CLASSES[action].from_dict(models[action]["tree"])
            predictions = {p: model.probabilities(x) for p, x in phases.items()}
            # 공통 검산기의 3행동용 반환 합계 대신 실제 저장된 트리 수를 센다.
            verify_tree_model(model, "depth2_iter256", phases["training"], y[:, None], phases, predictions)
            tree_count += sum(len(v["trees"]) for v in models[action]["tree"]["models"])
            assert models[action]["scope"] == "past_action_count_in_15m_equals_zero"
            x = phases["training"]
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
            assert learner.n_iter_.max() < 2000
            saved = models[action]["linear"]
            for key, values in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", learner.coef_[0])]:
                np.testing.assert_array_equal(saved[key], values)
            assert saved["intercept"] == float(learner.intercept_[0])
    assert tree_count == 512
    return {"independent_histogram_trees": tree_count, "two_linear_refits_exact": True}


def verify_quiet_predictions(source, original, saved, models, offsets, thresholds, constants):
    context = [k for k in saved if k in source]
    pd.testing.assert_frame_equal(saved[context], source[context], check_exact=True)
    np.testing.assert_array_equal(saved.exit_score, original.exit_native_calibrated)
    np.testing.assert_array_equal(saved.exit_request, original.exit_native_calibrated_request)
    for action in ["reduce", "increase"]:
        quiet = independent_groups(source, action)["quiet"]
        x = source.loc[quiet, FEATURES].to_numpy(float)
        for name, model_name in [("baseline", "native"), ("capacity", "capacity")]:
            for suffix, original_suffix in [("score", ""), ("request", "_request")]:
                expected = original[action + "_native_calibrated" + original_suffix].to_numpy().copy()
                expected[quiet] = original.loc[quiet, action + "_" + model_name + "_calibrated" + original_suffix]
                np.testing.assert_array_equal(saved[action + "_" + name + "_" + suffix], expected)
        raw = models[action]["tree"]["models"][0]
        z = partition_forest(raw, x)
        expected = {"tree_raw": 1 / (1 + np.exp(-np.clip(z, -700, 700)))}
        linear = models[action]["linear"]
        z = np.sum((x - linear["mean"]) / linear["scale"] * linear["coef"], axis=1) + linear["intercept"]
        expected["linear_raw"] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        for name, values in expected.items():
            np.testing.assert_allclose(saved.loc[quiet, action + "_" + name + "_score"], values, atol=1e-12, rtol=0)
        for name in ["tree", "linear"]:
            q = np.clip(saved.loc[quiet, action + "_" + name + "_raw_score"].to_numpy(),
                        np.finfo(float).eps, 1 - np.finfo(float).eps)
            z = np.log(q) - np.log1p(-q) + offsets[action][name]
            expected[name] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            np.testing.assert_array_equal(saved.loc[quiet, action + "_" + name + "_score"], expected[name])
        np.testing.assert_array_equal(saved.loc[quiet, action + "_constant_score"], np.full(quiet.sum(), constants[action]))
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            p = saved[action + "_" + name + "_score"].to_numpy()
            r = saved[action + "_" + name + "_request"].to_numpy()
            np.testing.assert_array_equal(p[~quiet], original.loc[~quiet, action + "_native_calibrated"])
            np.testing.assert_array_equal(r[~quiet], original.loc[~quiet, action + "_native_calibrated_request"])
            np.testing.assert_array_equal(r[quiet], p[quiet] >= thresholds[action][name])


def verify_quiet_calibration(saved, offsets, thresholds, details, constants):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        part = saved.loc[independent_groups(saved, action)["quiet"]]
        y = part["y_" + action].to_numpy(int)
        assert constants[action] == y.mean()
        for name in ["tree", "linear"]:
            p = np.clip(part[action + "_" + name + "_raw_score"].to_numpy(), np.finfo(float).eps, 1 - np.finfo(float).eps)
            logit = np.log(p) - np.log1p(-p)
            low, high = -64., 64.
            for _ in range(100):
                center = (low + high) / 2
                mean = np.mean(1 / (1 + np.exp(-np.clip(logit + center, -700, 700))))
                if mean < y.mean():
                    low = center
                else:
                    high = center
            assert abs(offsets[action][name] - (low + high) / 2) < 1e-12
            assert abs(part[action + "_" + name + "_score"].mean() - y.mean()) < 1e-12
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            boundary, result = manual_threshold(y, part[action + "_" + name + "_score"].to_numpy(), beta)
            assert boundary == thresholds[action][name]
            compare_statistics(details[action][name], result)


def verify_quiet_evaluation(saved, evaluation):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        for group, mask in independent_groups(saved, action).items():
            for name in NAMES:
                if not mask.any():
                    assert evaluation[action][group][name] is None
                    continue
                y = saved.loc[mask, "y_" + action].to_numpy(int)
                p = saved.loc[mask, action + "_" + name + "_score"].to_numpy()
                requested = saved.loc[mask, action + "_" + name + "_request"].to_numpy(bool)
                result = manual_metrics(y, p, 0., beta)
                tp, fp = int(sum(requested & (y == 1))), int(sum(requested & (y == 0)))
                fn = int(sum(y)) - tp
                denominator = (1 + beta**2) * tp + beta**2 * fn + fp
                result.update(predicted_positive=tp + fp, true_positive=tp, false_positive=fp,
                    false_negative=fn, precision=tp / (tp + fp) if tp + fp else 0.,
                    recall=tp / (tp + fn) if tp + fn else 0.,
                    f_beta=(1 + beta**2) * tp / denominator if denominator else 0.)
                compare_statistics(evaluation[action][group][name], result)


def verify_quiet_admission(evaluation, unchanged, decision):
    flags = {}
    for action in ["reduce", "increase"]:
        data = evaluation[action]["quiet"]
        candidate = data["tree"]
        for other in ["baseline", "capacity", "linear"]:
            old = data[other]
            flags[action + "_log_loss_" + other] = bool(1 - candidate["log_loss"] / old["log_loss"] > .01 + 1e-12) if old["log_loss"] else False
            flags[action + "_ap_" + other] = bool(candidate["average_precision"] >= old["average_precision"] - 1e-12)
        flags[action + "_raw_preserved"] = bool(candidate["log_loss"] <= data["tree_raw"]["log_loss"] + 1e-12)
        flags[action + "_constant_beaten"] = bool(candidate["log_loss"] < data["constant"]["log_loss"] - 1e-12)
        for key in ["recall", "f_beta"]:
            flags[action + "_quiet_" + key] = bool(candidate[key] > data["baseline"][key] + 1e-12)
        for group in ["all", "first"]:
            m = evaluation[action][group]
            flags[action + "_" + group + "_f_preserved"] = bool(m["tree"]["f_beta"] >= m["baseline"]["f_beta"] - 1e-12)
    flags.update(unchanged)
    assert len(flags) == 27 and decision["checks"] == flags
    assert decision["quiet_models_admitted"] == all(flags.values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
