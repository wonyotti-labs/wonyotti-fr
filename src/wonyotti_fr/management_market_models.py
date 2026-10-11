from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .management_calibration import ManagementOffset
from .management_market import FEATURES, REPLACED
from .minute_management import ACTIONS
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels

MODEL_NAMES = ("native", "aligned", "linear")


def validate_paired_rows(native, aligned):
    if set(native) != {"training", "calibration", "diagnosis"} or set(aligned) != set(native):
        raise ValueError("관리 시장 모형의 세 분할 누락")
    episodes = {}
    for phase in native:
        a, b = native[phase], aligned[phase]
        pd.testing.assert_frame_equal(a.drop(columns=REPLACED), b.drop(columns=REPLACED), check_exact=True)
        y = a[["y_" + action for action in ACTIONS]].to_numpy()
        if (len(a) < (1000 if phase == "training" else 100)
            or not np.isfinite(a[FEATURES]).all().all() or not np.isfinite(b[FEATURES]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any()
            or ((1-y).sum(axis=0) < 20).any() or a.end.duplicated().any()
            or not a.end.is_monotonic_increasing or not a.label_end.gt(a.end).all()):
            raise ValueError("관리 시장 모형의 입력·양성·음성·시간 지원 오류")
        episodes[phase] = set(a.episode_id)
    for first, second in [("training", "calibration"), ("training", "diagnosis"), ("calibration", "diagnosis")]:
        if episodes[first] & episodes[second] or native[first].label_end.max() >= native[second].end.min():
            raise ValueError("관리 시장 모형의 시간·포지션 중첩")


def fit_market_models(native, aligned):
    validate_paired_rows(native, aligned)
    y = native["training"][["y_" + a for a in ACTIONS]].to_numpy(int)
    cy = native["calibration"][["y_" + a for a in ACTIONS]].to_numpy(int)
    models, offsets, support = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, rows in [("native", native), ("aligned", aligned)]:
            model, training = OrderHistoryBoostModels.fit(rows["training"][FEATURES].to_numpy(float), y,
                                                         rows["calibration"][FEATURES].to_numpy(float))
            models[name], support[name] = model, {"training": training}
        linear, _, training = OrderHistoryModels.fit(aligned["training"], aligned["calibration"], "logistic")
        models["linear"], support["linear"] = linear, {"training": training}
        for name in MODEL_NAMES:
            rows = native if name == "native" else aligned
            raw = models[name].probabilities(rows["calibration"][FEATURES].to_numpy(float))
            offsets[name], details = ManagementOffset.fit(raw, cy)
            support[name]["offset"] = details
    return models, offsets, support


def market_predictions(native, aligned, models, offsets, stored_model, stored_offset, constant):
    result = {}
    for name in MODEL_NAMES:
        frame = native if name == "native" else aligned
        raw = models[name].probabilities(frame[FEATURES].to_numpy(float))
        result[name + "_raw"] = raw
        result[name + "_calibrated"] = offsets[name].predict(raw)
    for name, frame in [("stored", native), ("transported", aligned)]:
        raw = stored_model.probabilities(frame[FEATURES].to_numpy(float))
        result[name + "_raw"] = raw
        result[name + "_calibrated"] = stored_offset.predict(raw)
    result["constant"] = np.tile(constant, (len(native), 1))
    return result


def market_model_admission(metrics):
    checks = {}
    for action in ACTIONS:
        groups = metrics[action]
        candidate = groups["aligned_calibrated"]
        for name in ["native_calibrated", "linear_calibrated", "aligned_calibrated", "aligned_raw", "constant"]:
            item = groups[name]
            if (item["average_precision"] is None
                or not np.isfinite([item["log_loss"], item["average_precision"]]).all()
                or item["log_loss"] < 0 or not 0 <= item["average_precision"] <= 1
                or min(item["positive"], item["rows"]-item["positive"]) < 20):
                raise ValueError("관리 시장 모형의 진단 지표·지원 오류")
        for baseline in ["native", "linear"]:
            old = groups[baseline + "_calibrated"]
            gain = 1-candidate["log_loss"]/old["log_loss"] if old["log_loss"] else 0.
            checks[f"{action}_log_loss_{baseline}"] = bool(gain > .01 + 1e-12)
            checks[f"{action}_ap_{baseline}"] = bool(candidate["average_precision"] >= old["average_precision"]-1e-12)
        checks[action + "_raw_preserved"] = bool(candidate["log_loss"] <= groups["aligned_raw"]["log_loss"]+1e-12)
        checks[action + "_constant_beaten"] = bool(candidate["log_loss"] < groups["constant"]["log_loss"]-1e-12)
    return {"market_model_admitted": all(checks.values()), "checks": checks,
            "profitability_accepted": False, "trading_returns_evaluated": False,
            "all_source_periods_already_observed": True}
