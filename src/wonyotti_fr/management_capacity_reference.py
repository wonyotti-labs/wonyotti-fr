from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .joint_history_reference import verify_joint_admission
from .management_capacity import FEATURES, LABELS, joint_named_metrics
from .management_market_model_reference import offset_scores
from .minute_management import ACTIONS
from .paired_management_reference import manual_metrics, numeric_linear_scores
from .position_boost_verification import compare_statistics

REFERENCE_GRID = {"depth2_iter64": (2, 4, 64), "depth2_iter256": (2, 4, 256),
                  "depth4_iter64": (4, 16, 64), "depth4_iter256": (4, 16, 256)}


def verify_inner_split(frame, inner, ledger, support):
    cut = pd.Timestamp("2020-07-01", tz="UTC")
    crossing = set()
    for eid, group in frame.groupby("episode_id", sort=False):
        if group.entry_time.min() < cut <= group.label_end.max():
            crossing.add(eid)
    expected, positions = [], {"fit": [], "selection": []}
    counts = dict.fromkeys(["crossing_episode", "left_embargo", "right_embargo", "entry_before_start"], 0)
    for i, row in enumerate(frame[["end", "entry_time", "label_end", "episode_id"]].itertuples(index=False)):
        phase = "fit" if row.end < cut else "selection"
        start = pd.Timestamp("2020-01-01", tz="UTC") if phase == "fit" else cut
        stop = cut if phase == "fit" else pd.Timestamp("2021-01-01", tz="UTC")
        flags = [row.episode_id in crossing, row.end-start < pd.Timedelta(days=1),
                 stop-row.label_end <= pd.Timedelta(days=1), row.entry_time < start]
        reason = next((name for name, flag in zip(counts, flags, strict=True) if flag), "included")
        expected.append([i, row.end, row.entry_time, row.label_end, row.episode_id, phase, *flags, reason])
        if reason == "included":
            positions[phase].append(i)
        else:
            counts[reason] += 1
    independent = pd.DataFrame(expected, columns=ledger.columns)
    pd.testing.assert_frame_equal(independent, ledger, check_exact=True, check_dtype=False)
    result = {}
    for phase, selected in positions.items():
        part = frame.iloc[selected]
        pd.testing.assert_frame_equal(part, inner[phase], check_exact=True)
        positives = {a: int(part["y_"+a].sum()) for a in ACTIONS}
        result[phase] = {"rows": len(part), "episodes": len(set(part.episode_id)), "positive": positives,
                         "negative": {a: len(part)-count for a, count in positives.items()}}
    result.update(crossing_episodes=len(crossing), source_rows=len(frame), exclusions=counts,
                  excluded_rows=sum(counts.values()))
    compare_statistics(support, result)
    return {"source_rows": len(frame), "crossing_episodes": len(crossing), "all_memberships_and_exclusions_verified": True}


def partition_forest(binary, x):
    # 행별 현재 노드를 갱신하는 본 구현과 달리 노드별 행 집합을 재귀 분할한다.
    result = np.full(len(x), binary["baseline"], dtype=float)
    for tree in binary["trees"]:
        pending = [(0, np.arange(len(x)))]
        while pending:
            node, indices = pending.pop()
            if tree["feature"][node] < 0:
                result[indices] += tree["value"][node]
            else:
                left = x[indices, tree["feature"][node]] <= tree["threshold"][node]
                pending.extend([(tree["right"][node], indices[~left]), (tree["left"][node], indices[left])])
    return result


def verify_tree_model(model, setting, x, y, phases, predictions):
    depth, leaves, iterations = REFERENCE_GRID[setting]
    raw = model.to_dict()
    with threadpool_limits(limits=1):
        for i, binary in enumerate(raw["models"]):
            learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=iterations, learning_rate=.05,
                max_leaf_nodes=leaves, max_depth=depth, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                class_weight=None).fit(x, y[:, i])
            assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
            assert learner.n_iter_ == iterations and len(binary["trees"]) == len(learner._predictors) == iterations
            assert binary["baseline"] == float(learner._baseline_prediction[0, 0])
            for stored, trees in zip(binary["trees"], learner._predictors, strict=True):
                assert len(trees) == 1 and len(stored["left"]) == len(trees[0].nodes)
                for j, node in enumerate(trees[0].nodes):
                    assert node["depth"] <= depth and not node["is_categorical"]
                    leaf = bool(node["is_leaf"])
                    expected = {"left": -1 if leaf else int(node["left"]), "right": -1 if leaf else int(node["right"]),
                        "feature": -2 if leaf else int(node["feature_idx"]), "threshold": 0. if leaf else float(node["num_threshold"]),
                        "value": float(node["value"]) if leaf else 0.}
                    assert all(stored[k][j] == value for k, value in expected.items())
            for phase, values in phases.items():
                expected = learner.predict_proba(values)[:, 1]
                logits = partition_forest(binary, values)
                numeric = 1/(1+np.exp(-np.clip(logits, -700, 700)))
                np.testing.assert_allclose(predictions[phase][:, i], expected, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(predictions[phase][:, i], numeric)
    return iterations * 3


def verify_selection(inner, predictions, actual):
    assert list(predictions) == list(REFERENCE_GRID)
    constant = [sum(inner["fit"]["y_"+a])/len(inner["fit"]) for a in ACTIONS]
    np.testing.assert_array_equal(actual["constant"], constant)
    constant_metrics, candidate_metrics, scores = {}, {}, {}

    def metrics(frame, p):
        result = {}
        for i, action in enumerate(ACTIONS):
            v = manual_metrics(frame["y_"+action].to_numpy(int), p[:, i], .5, 1.)
            result[action] = {k: v[k] for k in ["rows", "positive", "log_loss", "average_precision"]}
        return result

    for phase, frame in inner.items():
        constant_metrics[phase] = metrics(frame, np.tile(constant, (len(frame), 1)))
    for name in REFERENCE_GRID:
        candidate_metrics[name] = {phase: metrics(frame, predictions[name][phase]) for phase, frame in inner.items()}
        scores[name] = sum(candidate_metrics[name]["selection"][a]["log_loss"] / constant_metrics["selection"][a]["log_loss"] for a in ACTIONS)/3
    compare_statistics(actual["metrics"], candidate_metrics)
    compare_statistics(actual["constant_metrics"], constant_metrics)
    compare_statistics(actual["scores"], scores)
    assert actual["selection_period"] == ["2020-07-01", "2021-01-01"]
    assert actual["tie_rule"] == "declared_order" and actual["outer_labels_used"] is False
    # 수치 대조 뒤 저장 점수의 정확한 동점 순서를 따로 확인한다.
    assert actual["selected"] == sorted(REFERENCE_GRID, key=lambda name: actual["scores"][name])[0]
    winner = min(REFERENCE_GRID, key=scores.__getitem__)
    assert actual["selected"] == winner


def verify_inner_models(inner, models, predictions):
    x, y = inner["fit"][FEATURES].to_numpy(float), inner["fit"][LABELS].to_numpy(int)
    phases = {k: v[FEATURES].to_numpy(float) for k, v in inner.items()}
    assert list(models) == list(REFERENCE_GRID)
    return sum(verify_tree_model(models[name], name, x, y, phases, predictions[name]) for name in REFERENCE_GRID)


def verify_outer_models(rows, models, selected):
    x, y = rows["training"][FEATURES].to_numpy(float), rows["training"][LABELS].to_numpy(int)
    phases = {k: v[FEATURES].to_numpy(float) for k, v in rows.items()}
    count = 0
    for name, setting in [("native", "depth2_iter64"), ("capacity", selected)]:
        predictions = {k: models[name].probabilities(v) for k, v in phases.items()}
        count += verify_tree_model(models[name], setting, x, y, phases, predictions)
    raw = models["linear"].to_dict()
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(raw["mean"], scaler.mean_)
        np.testing.assert_array_equal(raw["scale"], scaler.scale_)
        predictions = {k: models["linear"].probabilities(v) for k, v in phases.items()}
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(raw["coef"][i], learner.coef_[0])
            assert raw["intercept"][i] == float(learner.intercept_[0])
            for phase, values in phases.items():
                np.testing.assert_allclose(predictions[phase][:, i], learner.predict_proba(scaler.transform(values))[:, 1], atol=1e-12, rtol=0)
    return {"independent_outer_trees": count, "linear_standardization_and_three_fits_exact": True}


def verify_capacity_predictions(frame, models, offsets, constant, scores):
    x = frame[FEATURES].to_numpy(float)
    for name, model in models.items():
        raw = model.to_dict()
        if name == "linear":
            expected = numeric_linear_scores(raw, x)
        else:
            logits = np.column_stack([partition_forest(binary, x) for binary in raw["models"]])
            expected = 1/(1+np.exp(-np.clip(logits, -700, 700)))
        np.testing.assert_allclose(scores[name+"_raw"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(scores[name+"_calibrated"], offset_scores(scores[name+"_raw"], offsets[name].to_dict()))
    np.testing.assert_array_equal(scores["constant"], np.tile(constant, (len(frame), 1)))


def verify_capacity_admission(metrics, decision):
    original = dict(decision)
    original["joint_history_admitted"] = original.pop("capacity_admitted")
    verify_joint_admission(joint_named_metrics(metrics), original)
