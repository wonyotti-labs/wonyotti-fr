from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from .extrema_age import EXTRA_FEATURES
from .joint_history_models import joint_admission
from .management_calibration import ManagementOffset
from .management_capacity import FEATURES as NATIVE_FEATURES
from .management_capacity import LABELS, Depth2Iter256
from .order_history import OrderHistoryModels

FEATURES = NATIVE_FEATURES + EXTRA_FEATURES


class ExtremaAgeBoost(Depth2Iter256):
    features = FEATURES
    format = "extrema_age_histogram_v1"


class ExtremaAgeLinear(OrderHistoryModels):
    features = FEATURES
    format = "extrema_age_logistic_v1"


def fit_age_models(rows):
    with threadpool_limits(limits=1):
        x, cx = (rows[phase][FEATURES].to_numpy(float) for phase in ["training", "calibration"])
        y, cy = (rows[phase][LABELS].to_numpy(int) for phase in ["training", "calibration"])
        models, offsets, support = {}, {}, {}
        models["aged"], detail = ExtremaAgeBoost.fit(x, y, cx)
        support["aged"] = {"training": detail}
        models["aged_linear"], _, detail = ExtremaAgeLinear.fit(
            rows["training"], rows["calibration"], "logistic"
        )
        support["aged_linear"] = {"training": detail}
        for name, model in models.items():
            offsets[name], detail = ManagementOffset.fit(model.probabilities(cx), cy)
            support[name]["offset"] = detail
        return models, offsets, support


def age_predictions(native, expanded, models, offsets, constant):
    result = {}
    for name, model in models.items():
        frame = native if model.features == NATIVE_FEATURES else expanded
        raw = model.probabilities(frame[model.features].to_numpy(float))
        result[name + "_raw"], result[name + "_calibrated"] = raw, offsets[name].predict(raw)
    result["constant"] = np.tile(constant, (len(native), 1))
    return result


def age_admission(metrics):
    def renamed(name):
        if name.startswith("aged_linear_"):
            return name.replace("aged_linear_", "linear_", 1)
        if name.startswith("aged_"):
            return name.replace("aged_", "joint_", 1)
        if name.startswith("linear_"):
            return name.replace("linear_", "old_linear_", 1)
        return name

    adapted = {
        a: {g: {renamed(k): v for k, v in group.items()} for g, group in groups.items()}
        for a, groups in metrics.items()
    }
    decision = joint_admission(adapted)
    decision.pop("joint_history_admitted")
    for action, groups in metrics.items():
        new, previous = (groups["all"][name] for name in ["aged_calibrated", "capacity_calibrated"])
        decision["checks"][action + "_capacity_log_loss_preserved"] = bool(
            new["log_loss"] <= previous["log_loss"] + 1e-12
        )
        decision["checks"][action + "_capacity_ap_preserved"] = bool(
            new["average_precision"] >= previous["average_precision"] - 1e-12
        )
    decision["extrema_age_admitted"] = all(decision["checks"].values())
    return decision


def age_feature_usage(model):
    return {
        binary["action"]: {
            name: sum(tree["feature"].count(FEATURES.index(name)) for tree in binary["trees"])
            for name in EXTRA_FEATURES
        }
        for binary in model.to_dict()["models"]
    }
