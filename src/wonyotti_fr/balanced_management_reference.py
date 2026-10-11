from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .balanced_management import joint_balanced_metrics
from .joint_history_reference import verify_joint_admission
from .management_capacity import FEATURES
from .management_capacity_reference import partition_forest
from .management_market_model_reference import offset_scores
from .minute_management import ACTIONS
from .position_boost_verification import compare_statistics


def verify_weights(labels, actual, support):
    counts = Counter(int(v) for v in labels)
    expected = np.array([len(labels)/(2*counts[int(v)]) for v in labels])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, compute_sample_weight("balanced", labels))
    total = math.fsum(float(v) for v in expected)
    squares = math.fsum(float(v)**2 for v in expected)
    classes = {}
    for key in [0, 1]:
        values = expected[np.asarray(labels) == key]
        weight_sum = math.fsum(float(v) for v in values)
        classes[str(key)] = {"rows": len(values), "weight": len(labels)/(2*counts[key]),
            "sum": weight_sum, "effective_rows": weight_sum**2/math.fsum(float(v)**2 for v in values)}
    reference = {"rows": len(labels), "mean": total/len(labels), "sum": total,
        "effective_rows": total**2/squares, "maximum_share": max(expected)/total, "classes": classes}
    # 총량·유효 표본의 큰 분모는 원래 행 수로 정규화해 같은 오차 경계를 적용한다.
    def normalized(value):
        result = {k: (v/len(labels) if k in {"sum", "effective_rows"} else v) for k, v in value.items() if k != "classes"}
        result["classes"] = {k: {name: (v/len(labels) if name in {"sum", "effective_rows"} else v)
            for name, v in part.items()} for k, part in value["classes"].items()}
        return result
    compare_statistics(normalized(support), normalized(reference))
    assert abs(total/len(labels)-1.) <= 1e-12 and all(abs(v["sum"]/len(labels)-.5) <= 1e-12 for v in classes.values())
    return expected


def verify_balanced_refit(rows, model, weights, support):
    train = rows["training"]
    x = train[FEATURES].to_numpy(float)
    matrices = {phase: part[FEATURES].to_numpy(float) for phase, part in rows.items()}
    predictions = {phase: model.probabilities(values) for phase, values in matrices.items()}
    with threadpool_limits(limits=1):
        for i, action in enumerate(ACTIONS):
            labels = train["y_"+action].to_numpy(int)
            expected_weights = verify_weights(labels, weights[action], support["actions"][action]["weights"])
            learner = HistGradientBoostingClassifier(loss="log_loss", max_iter=256, learning_rate=.05,
                max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                class_weight=None).fit(x, labels, sample_weight=expected_weights)
            assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
            raw = model.to_dict()["models"][action]["models"][0]
            assert raw["action"] == action and raw["baseline"] == float(learner._baseline_prediction[0, 0])
            assert len(raw["trees"]) == learner.n_iter_ == len(learner._predictors) == 256
            for tree, predictors in zip(raw["trees"], learner._predictors, strict=True):
                assert len(predictors) == 1
                nodes = predictors[0].nodes
                assert len(tree["left"]) == len(nodes)
                for j, node in enumerate(nodes):
                    assert node["depth"] <= 2 and not node["is_categorical"]
                    leaf = bool(node["is_leaf"])
                    expected = {"left": -1 if leaf else int(node["left"]), "right": -1 if leaf else int(node["right"]),
                        "feature": -2 if leaf else int(node["feature_idx"]), "threshold": 0. if leaf else float(node["num_threshold"]),
                        "value": float(node["value"]) if leaf else 0.}
                    assert all(tree[k][j] == v for k, v in expected.items())
            for phase, values in matrices.items():
                p = learner.predict_proba(values)[:, 1]
                np.testing.assert_allclose(predictions[phase][:, i], p, atol=1e-12, rtol=0)
                logits = partition_forest(raw, values)
                np.testing.assert_array_equal(predictions[phase][:, i], 1/(1+np.exp(-np.clip(logits, -700, 700))))
    return {"independent_weighted_trees": 768, "all_class_weights_and_full_predictions_verified": True}


def verify_balanced_predictions(frame, model, offset, scores):
    x = frame[FEATURES].to_numpy(float)
    data = model.to_dict()["models"]
    logits = np.column_stack([partition_forest(data[a]["models"][0], x) for a in ACTIONS])
    np.testing.assert_array_equal(scores["balanced_raw"], 1/(1+np.exp(-np.clip(logits, -700, 700))))
    np.testing.assert_array_equal(scores["balanced_calibrated"], offset_scores(scores["balanced_raw"], offset.to_dict()))


def verify_balanced_admission(metrics, decision):
    original = dict(decision)
    original["joint_history_admitted"] = original.pop("balanced_management_admitted")
    verify_joint_admission(joint_balanced_metrics(metrics), original)
