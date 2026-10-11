from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .management_calibration import ManagementOffset
from .management_flow import EXPANDED_FEATURES, EXTRA_FEATURES
from .management_market_models import market_model_admission, validate_paired_rows
from .minute_management import ACTIONS
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels


class FlowManagementBoostModels(OrderHistoryBoostModels):
    features = EXPANDED_FEATURES
    format = "original_management_flow_histogram_v1"


class FlowManagementLinearModels(OrderHistoryModels):
    features = EXPANDED_FEATURES
    format = "original_management_flow_logistic_v1"


def validate_flow_rows(native, expanded):
    validate_paired_rows(native, native)
    if set(native) != set(expanded):
        raise ValueError("관리 체결 학습의 분할 누락")
    for phase, frame in native.items():
        added = expanded[phase]
        pd.testing.assert_frame_equal(frame, added.drop(columns=EXTRA_FEATURES), check_exact=True)
        if (not np.isfinite(added[EXPANDED_FEATURES]).all().all()
            or added[EXTRA_FEATURES].abs().gt(1).any().any() or not frame.direction.isin([-1, 1]).all()):
            raise ValueError("관리 체결 학습의 비율·입력 지원 오류")
        for name in EXTRA_FEATURES:
            if name.startswith("directional_"):
                np.testing.assert_array_equal(added[name], added[name.removeprefix("directional_")]*frame.direction)


def fit_flow_models(native, expanded):
    validate_flow_rows(native, expanded)
    y = native["training"][["y_"+a for a in ACTIONS]].to_numpy(int)
    cy = native["calibration"][["y_"+a for a in ACTIONS]].to_numpy(int)
    models, offsets, support = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, rows, cls in [("native", native, OrderHistoryBoostModels),
                                ("flow", expanded, FlowManagementBoostModels)]:
            model, detail = cls.fit(rows["training"][cls.features].to_numpy(float), y,
                                   rows["calibration"][cls.features].to_numpy(float))
            models[name], support[name] = model, {"training": detail}
        model, _, detail = FlowManagementLinearModels.fit(expanded["training"], expanded["calibration"], "logistic")
        models["linear"], support["linear"] = model, {"training": detail}
        for name, model in models.items():
            frame = native["calibration"] if name == "native" else expanded["calibration"]
            raw = model.probabilities(frame[model.features].to_numpy(float))
            offsets[name], detail = ManagementOffset.fit(raw, cy)
            support[name]["offset"] = detail
    return models, offsets, support


def flow_predictions(native, expanded, models, offsets, constant):
    result = {}
    for name, model in models.items():
        frame = native if name == "native" else expanded
        raw = model.probabilities(frame[model.features].to_numpy(float))
        result[name+"_raw"], result[name+"_calibrated"] = raw, offsets[name].predict(raw)
    result["constant"] = np.tile(constant, (len(native), 1))
    return result


def flow_model_admission(metrics):
    # v116의 18개 판정식을 그대로 적용하고 후보 이름만 대응한다.
    aligned = {a: {name.replace("flow_", "aligned_"): item for name, item in groups.items()}
               for a, groups in metrics.items()}
    decision = market_model_admission(aligned)
    decision["flow_model_admitted"] = decision.pop("market_model_admitted")
    return decision


def flow_feature_usage(model):
    return {binary["action"]: {
        name: sum(tree["feature"].count(EXPANDED_FEATURES.index(name)) for tree in binary["trees"])
        for name in EXTRA_FEATURES} for binary in model.to_dict()["models"]}
