from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from .first_state_models import validate_first_rows
from .joint_history_models import joint_admission
from .management_calibration import ManagementOffset
from .management_capacity import FEATURES as NATIVE_FEATURES
from .management_capacity import LABELS, Depth2Iter256
from .order_history import OrderHistoryModels

EXTRA_FEATURES = ["vol_path_current", "vol_path_best", "vol_path_worst", "vol_path_giveback"]
FEATURES = NATIVE_FEATURES + EXTRA_FEATURES
VOL_FLOOR = 1e-8


class VolatilityPathBoost(Depth2Iter256):
    features = FEATURES
    format = "volatility_path_histogram_v1"


class VolatilityPathLinear(OrderHistoryModels):
    features = FEATURES
    format = "volatility_path_logistic_v1"


def volatility_path_values(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(NATIVE_FEATURES) or not np.isfinite(x).all():
        raise ValueError("변동성 경로의 원본 차원·유한성 오류")
    d, vol, current, best, worst, giveback = (x[:, NATIVE_FEATURES.index(k)] for k in
        ["direction", "vol_1d", "favorable_move", "best_close_move", "worst_close_move", "giveback_from_best"])
    moves = np.column_stack([current, best, worst])
    ratios = 1+d[:, None]*moves
    if (not np.isin(d, [-1., 1.]).all() or (vol < 0).any() or not np.isfinite(ratios).all()
        or (ratios <= 0).any() or (best < current).any() or (current < worst).any() or (giveback < 0).any()
        or not np.allclose(giveback, best-current, atol=1e-12, rtol=1e-12)):
        raise ValueError("변동성 경로의 방향·가격 비율·최선/최악 관계 오류")
    denominator = np.maximum(vol, VOL_FLOOR)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logarithms = d[:, None]*np.log1p(d[:, None]*moves)
        # 최선과 현재 로그 값의 차 대신 가격 비율로 작은 되돌림을 보존한다.
        retracement = d*np.log1p(d*giveback/ratios[:, 0])
        result = np.column_stack([logarithms, retracement])/denominator[:, None]
    if not np.isfinite(result).all():
        raise ValueError("변동성 경로의 숫자 범위 초과")
    return result


def attach_volatility_path(rows):
    validate_first_rows(rows)
    result = {}
    for phase, frame in rows.items():
        if set(EXTRA_FEATURES) & set(frame):
            raise ValueError("변동성 경로 입력의 중복 추가")
        added = frame.copy()
        added[EXTRA_FEATURES] = volatility_path_values(frame[NATIVE_FEATURES].to_numpy(float))
        result[phase] = added
    return result


def volatility_boundaries(training):
    values = training.vol_1d.to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("변동성 경계의 학습 입력 오류")
    return np.quantile(values, [.25, .5, .75], method="linear").tolist()


def volatility_groups(frame, boundaries):
    bounds = np.asarray(boundaries, dtype=float)
    values = frame.vol_1d.to_numpy(float)
    if (bounds.shape != (3,) or not np.isfinite(bounds).all() or (bounds < 0).any() or (np.diff(bounds) < 0).any()
        or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("변동성 구간의 경계·입력 오류")
    return np.searchsorted(bounds, values, side="left")


def fit_volatility_models(rows):
    with threadpool_limits(limits=1):
        x, cx = (rows[phase][FEATURES].to_numpy(float) for phase in ["training", "calibration"])
        y, cy = (rows[phase][LABELS].to_numpy(int) for phase in ["training", "calibration"])
        models, offsets, support = {}, {}, {}
        models["scaled"], detail = VolatilityPathBoost.fit(x, y, cx)
        support["scaled"] = {"training": detail}
        models["scaled_linear"], _, detail = VolatilityPathLinear.fit(rows["training"], rows["calibration"], "logistic")
        support["scaled_linear"] = {"training": detail}
        for name, model in models.items():
            offsets[name], detail = ManagementOffset.fit(model.probabilities(cx), cy)
            support[name]["offset"] = detail
        return models, offsets, support


def volatility_predictions(native, expanded, models, offsets, constant):
    result = {}
    for name, model in models.items():
        frame = native if model.features == NATIVE_FEATURES else expanded
        raw = model.probabilities(frame[model.features].to_numpy(float))
        result[name+"_raw"], result[name+"_calibrated"] = raw, offsets[name].predict(raw)
    result["constant"] = np.tile(constant, (len(native), 1))
    return result


def volatility_admission(metrics):
    def renamed(name):
        if name.startswith("scaled_linear_"):
            return name.replace("scaled_linear_", "linear_", 1)
        if name.startswith("scaled_"):
            return name.replace("scaled_", "joint_", 1)
        if name.startswith("linear_"):
            return name.replace("linear_", "old_linear_", 1)
        return name
    adapted = {a: {g: {renamed(k): v for k, v in group.items()} for g, group in groups.items()}
               for a, groups in metrics.items()}
    decision = joint_admission(adapted)
    decision.pop("joint_history_admitted")
    for action, groups in metrics.items():
        new, previous = (groups["all"][name] for name in ["scaled_calibrated", "capacity_calibrated"])
        decision["checks"][action+"_capacity_log_loss_preserved"] = bool(new["log_loss"] <= previous["log_loss"]+1e-12)
        decision["checks"][action+"_capacity_ap_preserved"] = bool(new["average_precision"] >= previous["average_precision"]-1e-12)
    decision["volatility_path_admitted"] = all(decision["checks"].values())
    return decision


def path_feature_usage(model):
    return {binary["action"]: {name: sum(tree["feature"].count(FEATURES.index(name)) for tree in binary["trees"])
            for name in EXTRA_FEATURES} for binary in model.to_dict()["models"]}
