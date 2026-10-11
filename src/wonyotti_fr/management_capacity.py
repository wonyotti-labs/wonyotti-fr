from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from threadpoolctl import threadpool_limits

from .histogram_management import HISTOGRAM_SETTINGS
from .joint_history_models import joint_admission
from .management_calibration import ManagementOffset
from .minute_management import ACTIONS
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels

FEATURES = OrderHistoryModels.features
LABELS = ["y_" + a for a in ACTIONS]
GRID = {"depth2_iter64": (2, 4, 64), "depth2_iter256": (2, 4, 256),
        "depth4_iter64": (4, 16, 64), "depth4_iter256": (4, 16, 256)}
PERIODS = {"fit": ("2020-01-01", "2020-07-01"), "selection": ("2020-07-01", "2021-01-01")}


class Depth2Iter64(OrderHistoryBoostModels):
    format = "management_capacity_depth2_iter64_v1"


class Depth2Iter256(OrderHistoryBoostModels):
    format = "management_capacity_depth2_iter256_v1"
    fixed_settings = {**HISTOGRAM_SETTINGS, "max_iter": 256}


class Depth4Iter64(OrderHistoryBoostModels):
    format = "management_capacity_depth4_iter64_v1"
    fixed_settings = {**HISTOGRAM_SETTINGS, "max_depth": 4, "max_leaf_nodes": 16}


class Depth4Iter256(OrderHistoryBoostModels):
    format = "management_capacity_depth4_iter256_v1"
    fixed_settings = {**HISTOGRAM_SETTINGS, "max_depth": 4, "max_leaf_nodes": 16, "max_iter": 256}


CLASSES = dict(zip(GRID, [Depth2Iter64, Depth2Iter256, Depth4Iter64, Depth4Iter256], strict=True))


def load_capacity_model(raw):
    cls = next((v for v in CLASSES.values() if raw.get("format") == v.format), None)
    if cls is None:
        raise ValueError("관리 복잡도의 허용하지 않는 모형 형식")
    return cls.from_dict(raw)


def validate_training(frame):
    if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("관리 복잡도의 원래 행 순서 오류")
    for column in ["end", "entry_time", "label_end"]:
        values = frame[column]
        if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC" or values.isna().any():
            raise ValueError("관리 복잡도의 UTC 시각 오류")
    y = frame[LABELS].to_numpy()
    if (len(frame) < 1000 or not np.isfinite(frame[FEATURES]).all().all()
        or not np.isin(y, [0, 1]).all() or not np.isin(frame[["past_reduce_exists", "past_increase_exists"]], [0, 1]).all()
        or not frame.end.is_monotonic_increasing or frame.end.duplicated().any()
        or not frame.entry_time.lt(frame.end).all() or not frame.label_end.gt(frame.end).all()
        or not frame.end.ge(pd.Timestamp("2020-01-02", tz="UTC")).all()
        or not frame.entry_time.ge(pd.Timestamp("2020-01-01", tz="UTC")).all()
        or not frame.label_end.lt(pd.Timestamp("2020-12-31", tz="UTC")).all()
        or not np.isfinite(frame.episode_id).all() or frame.episode_id.le(0).any()
        or frame.episode_id.mod(1).ne(0).any() or frame.groupby("episode_id").entry_time.nunique().gt(1).any()):
        raise ValueError("관리 복잡도의 원본 학습 지원·시간·에피소드 오류")


def split_inner(frame):
    validate_training(frame)
    cut = pd.Timestamp("2020-07-01", tz="UTC")
    crossing = set(frame.loc[frame.entry_time.lt(cut) & frame.label_end.ge(cut), "episode_id"])
    ledger = frame[["end", "entry_time", "label_end", "episode_id"]].copy()
    ledger.insert(0, "source_row", np.arange(len(frame)))
    ledger["phase"] = np.where(frame.end.lt(cut), "fit", "selection")
    ledger["crossing_episode"] = frame.episode_id.isin(crossing)
    ledger["left_embargo"], ledger["right_embargo"], ledger["entry_before_start"] = False, False, False
    result, support = {}, {}
    for phase, bounds in PERIODS.items():
        start, stop = (pd.Timestamp(v, tz="UTC") for v in bounds)
        mask = ledger.phase.eq(phase)
        ledger.loc[mask, "left_embargo"] = frame.loc[mask, "end"].lt(start + pd.Timedelta(days=1))
        ledger.loc[mask, "right_embargo"] = frame.loc[mask, "label_end"].ge(stop - pd.Timedelta(days=1))
        ledger.loc[mask, "entry_before_start"] = frame.loc[mask, "entry_time"].lt(start)
    reasons = ["crossing_episode", "left_embargo", "right_embargo", "entry_before_start"]
    ledger["reason"] = np.select([ledger[k] for k in reasons], reasons, default="included")
    for phase in PERIODS:
        indices = ledger.loc[ledger.phase.eq(phase) & ledger.reason.eq("included"), "source_row"]
        part = frame.iloc[indices].copy()
        y = part[LABELS].to_numpy(int)
        if len(part) < 1000 or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any():
            raise ValueError("관리 복잡도의 내부 기간별 양성·음성 지원 부족")
        result[phase] = part
        support[phase] = {"rows": len(part), "episodes": int(part.episode_id.nunique()),
            "positive": dict(zip(ACTIONS, y.sum(axis=0).tolist(), strict=True)),
            "negative": dict(zip(ACTIONS, (1-y).sum(axis=0).tolist(), strict=True))}
    if (set(result["fit"].episode_id) & set(result["selection"].episode_id)
        or result["fit"].label_end.max() >= result["selection"].end.min()):
        raise ValueError("관리 복잡도의 내부 시간·에피소드 중첩")
    support.update(crossing_episodes=len(crossing), source_rows=len(frame),
        exclusions={name: int(ledger.reason.eq(name).sum()) for name in reasons},
        excluded_rows=int(ledger.reason.ne("included").sum()))
    return result, ledger, support


def probability_metrics(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if (y.shape != p.shape or y.ndim != 2 or y.shape[1] != 3 or not len(y)
        or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError("관리 복잡도의 확률·정답 오류")
    return {action: {"rows": len(y), "positive": int(y[:, i].sum()),
        "log_loss": float(log_loss(y[:, i], p[:, i], labels=[0, 1])),
        "average_precision": float(average_precision_score(y[:, i], p[:, i])) if y[:, i].sum() else None}
        for i, action in enumerate(ACTIONS)}


def select_capacity(inner, predictions):
    if list(predictions) != list(GRID) or any(set(v) != set(PERIODS) for v in predictions.values()):
        raise ValueError("관리 복잡도의 후보·기간 누락")
    constant = inner["fit"][LABELS].to_numpy(int).mean(axis=0)
    constants = {phase: probability_metrics(part[LABELS], np.tile(constant, (len(part), 1)))
                 for phase, part in inner.items()}
    metrics, scores = {}, {}
    for name, phases in predictions.items():
        metrics[name] = {phase: probability_metrics(inner[phase][LABELS], p) for phase, p in phases.items()}
        ratios = [metrics[name]["selection"][a]["log_loss"] / constants["selection"][a]["log_loss"] for a in ACTIONS]
        scores[name] = float(np.mean(ratios))
    chosen = min(GRID, key=scores.__getitem__)
    return {"selected": chosen, "scores": scores, "metrics": metrics, "constant": constant.tolist(),
            "constant_metrics": constants, "selection_period": list(PERIODS["selection"]),
            "tie_rule": "declared_order", "outer_labels_used": False}


def fit_inner_models(inner):
    x, vx = (inner[phase][FEATURES].to_numpy(float) for phase in PERIODS)
    y = inner["fit"][LABELS].to_numpy(int)
    models, predictions, support = {}, {}, {}
    for name, cls in CLASSES.items():
        models[name], support[name] = cls.fit(x, y, vx)
        predictions[name] = {phase: models[name].probabilities(part[FEATURES].to_numpy(float)) for phase, part in inner.items()}
    return models, predictions, support


def fit_outer_models(rows, selected):
    if selected not in CLASSES:
        raise ValueError("관리 복잡도의 고정 목록 외 선택")
    x, cx = (rows[p][FEATURES].to_numpy(float) for p in ["training", "calibration"])
    y, cy = (rows[p][LABELS].to_numpy(int) for p in ["training", "calibration"])
    models, offsets, support = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, cls in [("native", OrderHistoryBoostModels), ("capacity", CLASSES[selected])]:
            models[name], detail = cls.fit(x, y, cx)
            support[name] = {"training": detail}
        models["linear"], _, detail = OrderHistoryModels.fit(rows["training"], rows["calibration"], "logistic")
        support["linear"] = {"training": detail}
        for name, model in models.items():
            offsets[name], detail = ManagementOffset.fit(model.probabilities(cx), cy)
            support[name]["offset"] = detail
    return models, offsets, support


def capacity_predictions(frame, models, offsets, constant):
    x = frame[FEATURES].to_numpy(float)
    result = {}
    for name, model in models.items():
        raw = model.probabilities(x)
        result[name + "_raw"], result[name + "_calibrated"] = raw, offsets[name].predict(raw)
    result["constant"] = np.tile(constant, (len(frame), 1))
    return result


def joint_named_metrics(metrics):
    return {a: {g: {name.replace("capacity_", "joint_"): value for name, value in group.items()}
                for g, group in groups.items()} for a, groups in metrics.items()}


def capacity_admission(metrics):
    result = joint_admission(joint_named_metrics(metrics))
    result["capacity_admitted"] = result.pop("joint_history_admitted")
    return result
