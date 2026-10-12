from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .causal_management import causal_admission
from .count_region_management import validate_count_region_inputs
from .first_management import BETAS
from .first_state_models import FEATURES, calibrated_binary, fit_binary_offset, linear_probabilities
from .management_capacity import validate_training
from .management_rank_audit import validate_scores
from .quiet_management import ACTIONS, CLASSES, groups, quiet_mask

WEIGHT_RULE = "training_negative_top_one_percent_ties_high1_low01_mean1"
SCOPE = "past_action_count_in_15m_equals_zero"
MODELS = ("v127", "v140")


def weight_statistics(labels, hard, weights):
    y, h, w = np.asarray(labels), np.asarray(hard), np.asarray(weights)
    if (
        y.ndim != 1
        or h.shape != y.shape
        or w.shape != y.shape
        or h.dtype.kind != "b"
        or not np.isin(y, [0, 1]).all()
        or (h & (y == 1)).any()
        or not np.isfinite(w).all()
        or (w <= 0).any()
    ):
        raise ValueError("꼬리 가중치 집계의 정답·분기·양수 입력 오류")
    masks = {
        "all": np.ones(len(y), bool),
        "positive": y == 1,
        "hard_negative": h,
        "easy_negative": (y == 0) & ~h,
    }
    total = math.fsum(w.tolist())
    result = {}
    for name, mask in masks.items():
        values = w[mask].tolist()
        mass = math.fsum(values)
        squares = math.fsum(v * v for v in values)
        result[name] = {
            "rows": len(values),
            "sum": mass,
            "mean": mass / len(values) if values else None,
            "effective_rows": mass * mass / squares if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "maximum_share": max(values) / total if values else None,
        }
    return result


def tail_weights(labels, scores):
    y, p = np.asarray(labels), validate_scores(scores)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or len(y) < 1000
        or y.dtype.kind not in "bifu"
        or not np.isin(y, [0, 1]).all()
        or min(int((y == 1).sum()), int((y == 0).sum())) < 20
    ):
        raise ValueError("꼬리 가중치의 이진 정답·학습 지원 오류")
    negative = p[y == 0]
    k = (len(negative) + 99) // 100
    q = float(np.sort(negative)[-k])
    hard = (y == 0) & (p >= q)
    # 1 대 0.1을 정수 단위로 정규화해 누적 소수 오차를 피한다.
    units = np.where((y == 1) | hard, 10, 1).astype(np.int64)
    total_units = int(units.sum())
    weights = units.astype(float) * len(y) / total_units
    stats = weight_statistics(y, hard, weights)
    if (
        not math.isclose(stats["all"]["mean"], 1.0, rel_tol=0, abs_tol=1e-12)
        or stats["all"]["effective_rows"] < 1000
    ):
        raise ValueError("꼬리 가중치의 평균·유효 표본 오류")
    return (
        weights,
        hard,
        {
            "rule": WEIGHT_RULE,
            "negative_rows": len(negative),
            "rank": k,
            "boundary_score": q,
            "boundary_ties": int((negative == q).sum()),
            "hard_negative_fraction": float(hard.sum() / len(negative)),
            "normalization_units": total_units,
            "statistics": stats,
        },
    )


def validate_tail_models(models):
    if set(models) != set(ACTIONS):
        raise ValueError("꼬리 가중 관리 모델의 행동 누락")
    for action, value in models.items():
        if (
            set(value) != {"tree", "linear", "scope", "weight_rule"}
            or value["scope"] != SCOPE
            or value["weight_rule"] != WEIGHT_RULE
        ):
            raise ValueError("꼬리 가중 관리 모델의 학습 규칙 오류")
        CLASSES[action].from_dict(value["tree"])
        if value["linear"].get("action") != action:
            raise ValueError("꼬리 가중 관리 선형의 행동 불일치")
        linear_probabilities(value["linear"], np.zeros((1, len(FEATURES))))


def fit_tail_models(training, parent_models):
    validate_tail_rows(training, "training")
    validate_training(training)
    if (
        not training.end.ge(pd.Timestamp("2020-01-01", tz="UTC")).all()
        or not training.end.lt(pd.Timestamp("2021-01-01", tz="UTC")).all()
        or set(parent_models) != set(ACTIONS)
    ):
        raise ValueError("꼬리 가중 학습의 기간·부모 연결 오류")
    models, details = {}, {}
    ledger = training[["end", "episode_id"]].copy()
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            parent = parent_models[action]
            if set(parent) != {"tree", "linear", "scope"} or parent["scope"] != SCOPE:
                raise ValueError("꼬리 가중 학습의 원래 무가중 부모 오류")
            mask = quiet_mask(training, action)
            part = training.loc[mask]
            x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
            scores = CLASSES[action].from_dict(parent["tree"]).probabilities(x)[:, 0]
            weights, hard, details[action] = tail_weights(y, scores)
            tree, _ = CLASSES[action].fit(x, y[:, None], x[:1], sample_weight=weights)
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                scaler.transform(x), y, sample_weight=weights
            )
            if learner.n_iter_.max() >= 2000:
                raise ValueError("꼬리 가중 선형의 수렴 실패")
            linear = {
                "format": "first_state_linear_v1",
                "features": FEATURES,
                "action": action,
                "settings": {"C": 0.1, "max_iter": 2000, "random_state": 0},
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": learner.coef_[0].tolist(),
                "intercept": float(learner.intercept_[0]),
            }
            np.testing.assert_allclose(
                linear_probabilities(linear, x),
                learner.predict_proba(scaler.transform(x))[:, 1],
                atol=1e-12,
                rtol=0,
            )
            models[action] = {
                "tree": tree.to_dict(),
                "linear": linear,
                "scope": SCOPE,
                "weight_rule": WEIGHT_RULE,
            }
            for suffix, values, dtype in [
                ("score", scores, "Float64"),
                ("weight", weights, "Float64"),
                ("hard_negative", hard, "boolean"),
            ]:
                column = action + "_" + suffix
                ledger[column] = pd.array([pd.NA] * len(training), dtype=dtype)
                ledger.loc[mask, column] = values
    validate_tail_models(models)
    return models, ledger, details


def calibrate_tail_models(calibration, models):
    validate_tail_rows(calibration, "calibration")
    validate_tail_models(models)
    if (
        not calibration.end.ge(pd.Timestamp("2021-01-01", tz="UTC")).all()
        or not calibration.end.lt(pd.Timestamp("2021-07-01", tz="UTC")).all()
    ):
        raise ValueError("꼬리 가중 관리의 상반기 보정 기간 오류")
    offsets, thresholds, details, constants = {}, {}, {}, {}
    for action in ACTIONS:
        part = calibration.loc[quiet_mask(calibration, action)]
        x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
        scores = {
            "tree_raw": CLASSES[action].from_dict(models[action]["tree"]).probabilities(x)[:, 0],
            "linear_raw": linear_probabilities(models[action]["linear"], x),
        }
        offsets[action], thresholds[action], details[action] = {}, {}, {}
        for name in ["tree", "linear"]:
            offsets[action][name] = fit_binary_offset(scores[name + "_raw"], y)
            scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
        constants[action] = float(y.mean())
        scores["constant"] = np.full(len(y), constants[action])
        for name, p in scores.items():
            thresholds[action][name], details[action][name] = select_threshold(y, p, BETAS[action])
    return offsets, thresholds, details, constants


def validate_tail_rows(frame, phase):
    bounds = {"training": ("2020-01-01", "2021-01-01"), "calibration": ("2021-01-01", "2021-07-01")}
    if phase not in bounds or not len(frame):
        raise ValueError("꼬리 가중 학습·보정 기간 오류")
    required = {"end", "entry_time", "label_end", "episode_id", "y_exit", "y_reduce", "y_increase"}
    if not required.issubset(frame):
        raise ValueError("꼬리 가중 학습·보정의 시각·정답 누락")
    validate_count_region_inputs(frame)
    for column in ["end", "entry_time", "label_end"]:
        values = frame[column]
        if (
            not isinstance(values.dtype, pd.DatetimeTZDtype)
            or str(values.dt.tz) != "UTC"
            or values.isna().any()
        ):
            raise ValueError("꼬리 가중 학습·보정의 UTC 시각 오류")
    start, stop = (pd.Timestamp(v, tz="UTC") for v in bounds[phase])
    # 정수 변환 전에 분수 정답과 기간 경계 침범을 차단한다.
    if (
        not np.isin(frame[["y_exit", "y_reduce", "y_increase"]], [0, 1]).all()
        or not frame.end.is_monotonic_increasing
        or frame.end.duplicated().any()
        or not frame.end.eq(frame.end.dt.floor("min")).all()
        or not frame.entry_time.lt(frame.end).all()
        or not frame.label_end.gt(frame.end).all()
        or not frame.end.ge(start + pd.Timedelta(days=1)).all()
        or not frame.entry_time.ge(start).all()
        or not frame.label_end.lt(stop - pd.Timedelta(days=1)).all()
        or not np.isfinite(frame.episode_id).all()
        or frame.episode_id.le(0).any()
        or frame.episode_id.ge(2**53).any()
        or frame.episode_id.mod(1).ne(0).any()
        or frame.groupby("episode_id").entry_time.nunique().gt(1).any()
    ):
        raise ValueError("꼬리 가중 학습·보정의 정답·시간·포지션 오류")


def training_weight_groups(frame, ledger):
    pd.testing.assert_frame_equal(
        frame[["end", "episode_id"]], ledger[["end", "episode_id"]], check_exact=True
    )
    result = {}
    for action in ACTIONS:
        quiet = quiet_mask(frame, action)
        for suffix in ["score", "weight", "hard_negative"]:
            if not np.array_equal(ledger[action + "_" + suffix].notna(), quiet):
                raise ValueError("꼬리 가중치 원장의 quiet 지원 불일치")
        result[action] = {}
        partitions = {"all_training": quiet}
        for state, mask in groups(frame, action).items():
            partitions[state] = mask & quiet
            for region in ["below", "capped"]:
                partitions[state + "__" + region] = (
                    mask
                    & quiet
                    & (
                        frame.adds.lt(5).to_numpy()
                        if region == "below"
                        else frame.adds.ge(5).to_numpy()
                    )
                )
        for month, indices in frame.groupby(
            frame.end.dt.strftime("%Y-%m"), sort=True
        ).indices.items():
            mask = np.zeros(len(frame), bool)
            mask[indices] = True
            partitions[month] = mask & quiet
        for name, mask in partitions.items():
            result[action][name] = weight_statistics(
                frame.loc[mask, "y_" + action].to_numpy(),
                ledger.loc[mask, action + "_hard_negative"].to_numpy(bool),
                ledger.loc[mask, action + "_weight"].to_numpy(float),
            )
    return result


def tail_admission(candidate, previous, unchanged, pre_limit, budget):
    result = causal_admission(candidate, previous, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        old, new = previous[action]["quiet"]["tree"], candidate[action]["quiet"]["tree"]
        checks[action + "_tail_weight_log_loss_gain"] = (
            bool(1 - new["log_loss"] / old["log_loss"] > 0.01 + 1e-12) if old["log_loss"] else False
        )
    if set(pre_limit) != set(MODELS):
        raise ValueError("꼬리 가중 관리의 한도 전 대조 누락")
    old, new = pre_limit["v127"], pre_limit["v140"]
    for data in [old, new]:
        if data is None or min(data["positive"], data["rows"] - data["positive"]) < 20:
            raise ValueError("꼬리 가중 관리의 한도 전 지원 부족")
    if (old["rows"], old["positive"]) != (new["rows"], new["positive"]):
        raise ValueError("꼬리 가중 관리의 한도 전 표본 불일치")
    for key in ["recall", "f_beta"]:
        checks["pre_limit_restart_increase_" + key + "_improved"] = bool(
            new[key] > old[key] + 1e-12
        )
    checks["pre_limit_restart_increase_precision_preserved"] = bool(
        new["precision"] >= old["precision"] - 1e-12
    )
    focal = budget["increase__restart__below"]["models"]
    old, new = (focal[model]["budget_matched"] for model in MODELS)
    supported = old is not None and new is not None
    checks["budget_pre_limit_restart_increase_tp_improved"] = bool(
        supported and new["true_positive"] > old["true_positive"]
    )
    checks["budget_pre_limit_restart_increase_fp_preserved"] = bool(
        supported and new["false_positive"] <= old["false_positive"]
    )
    result.pop("causal_distribution_admitted")
    result["tail_weighted_admitted"] = all(checks.values())
    assert len(checks) == 42
    return result
