from __future__ import annotations

import copy

import numpy as np

from .first_state_models import validate_first_rows
from .joint_history_models import joint_admission
from .management_calibration import ManagementOffset
from .management_capacity import FEATURES, LABELS, Depth2Iter256, validate_training
from .minute_management import ACTIONS


class BalancedExit(Depth2Iter256):
    actions = ["exit"]
    format = "balanced_management_exit_v1"


class BalancedReduce(Depth2Iter256):
    actions = ["reduce"]
    format = "balanced_management_reduce_v1"


class BalancedIncrease(Depth2Iter256):
    actions = ["increase"]
    format = "balanced_management_increase_v1"


CLASSES = {"exit": BalancedExit, "reduce": BalancedReduce, "increase": BalancedIncrease}


def balanced_weights(labels):
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) < 1000 or not np.isin(y, [0, 1]).all():
        raise ValueError("이진 균형 학습의 정답·차원·표본 오류")
    counts = np.bincount(y.astype(int), minlength=2)
    if (counts < 20).any():
        raise ValueError("이진 균형 학습의 양성·음성 지원 부족")
    weights = len(y)/(2*counts[y.astype(int)])
    effective = float(weights.sum()**2/np.square(weights).sum())
    if (not np.isfinite(weights).all() or (weights <= 0).any()
        or not np.isclose(weights.mean(), 1., rtol=0, atol=1e-12) or effective < 1000
        or any(not np.isclose(weights[y == k].sum()/len(y), .5, rtol=0, atol=1e-12) for k in [0, 1])):
        raise ValueError("이진 균형 학습의 가중치·유효 표본 오류")
    return weights, {"rows": len(y), "effective_rows": effective, "mean": float(weights.mean()),
        "sum": float(weights.sum()), "maximum_share": float(weights.max()/weights.sum()),
        "classes": {str(k): {"rows": int(counts[k]), "weight": float(len(y)/(2*counts[k])),
            "sum": float(weights[y == k].sum()), "effective_rows": float(weights[y == k].sum()**2/np.square(weights[y == k]).sum())}
            for k in [0, 1]}}


class BalancedManagementModels:
    features = FEATURES
    actions = ACTIONS
    format = "balanced_management_bundle_v1"

    def __init__(self, data, components):
        self.data = copy.deepcopy(data)
        self.components = components

    @classmethod
    def from_dict(cls, data):
        if (set(data) != {"format", "features", "weight_rule", "models"}
            or data["format"] != cls.format or data["features"] != cls.features
            or data["weight_rule"] != "binary_training_inverse_frequency"
            or not isinstance(data["models"], dict) or set(data["models"]) != set(ACTIONS)):
            raise ValueError("이진 균형 모형의 형식·행동·가중 규칙 오류")
        components = {action: CLASSES[action].from_dict(data["models"][action]) for action in ACTIONS}
        return cls(data, components)

    @classmethod
    def fit(cls, training, validation_values):
        validate_training(training)
        x = training[FEATURES].to_numpy(float)
        models, weights, support = {}, {}, {}
        for action in ACTIONS:
            y = training["y_"+action].to_numpy(int)
            weights[action], detail = balanced_weights(y)
            models[action], fit = CLASSES[action].fit(x, y[:, None], validation_values, sample_weight=weights[action])
            support[action] = {"weights": detail, "training": fit}
        result = cls.from_dict({"format": cls.format, "features": cls.features,
            "weight_rule": "binary_training_inverse_frequency", "models": {k: v.to_dict() for k, v in models.items()}})
        return result, weights, support

    def probabilities(self, values):
        return np.column_stack([self.components[a].probabilities(values)[:, 0] for a in ACTIONS])

    def to_dict(self):
        return copy.deepcopy(self.data)


def fit_balanced_management(rows):
    validate_first_rows(rows)
    model, weights, support = BalancedManagementModels.fit(rows["training"], rows["calibration"][FEATURES].to_numpy(float))
    raw = model.probabilities(rows["calibration"][FEATURES].to_numpy(float))
    offset, detail = ManagementOffset.fit(raw, rows["calibration"][LABELS].to_numpy(int))
    return model, offset, weights, {"actions": support, "offset": detail}


def joint_balanced_metrics(metrics):
    return {a: {g: {name.replace("balanced_", "joint_"): v for name, v in group.items()}
                for g, group in groups.items()} for a, groups in metrics.items()}


def balanced_admission(metrics):
    result = joint_admission(joint_balanced_metrics(metrics))
    result["balanced_management_admitted"] = result.pop("joint_history_admitted")
    return result
