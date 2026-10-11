from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .first_management import BETAS, FIRST_ACTIONS, decision_metrics
from .management_market_models import validate_paired_rows
from .order_history_boost import OrderHistoryBoostModels

FEATURES = OrderHistoryBoostModels.features


class FirstReduceModel(OrderHistoryBoostModels):
    actions = ["reduce"]
    format = "first_reduce_histogram_v1"


class FirstIncreaseModel(OrderHistoryBoostModels):
    actions = ["increase"]
    format = "first_increase_histogram_v1"


CLASSES = {"reduce": FirstReduceModel, "increase": FirstIncreaseModel}


def first_mask(frame, action):
    if action not in FIRST_ACTIONS:
        raise ValueError("첫 상태의 지원하지 않는 행동")
    flags = frame[f"past_{action}_exists"].to_numpy()
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("첫 상태의 과거 체결 여부 오류")
    return flags == 0


def validate_first_rows(rows):
    validate_paired_rows(rows, rows)
    support = {}
    for phase, frame in rows.items():
        support[phase] = {}
        for action in FIRST_ACTIONS:
            first = first_mask(frame, action)
            support[phase][action] = {}
            for name, mask in [("first", first), ("repeat", ~first)]:
                y = frame.loc[mask, "y_" + action].to_numpy()
                if (len(y) < (1000 if phase == "training" else 100)
                    or min(y.sum(), len(y) - y.sum()) < 20):
                    raise ValueError("첫 상태의 기간·행동별 지원 부족")
                support[phase][action][name] = {"rows": len(y), "positive": int(y.sum()),
                    "negative": int(len(y) - y.sum())}
    return support


def calibrated_binary(scores, offset):
    p = np.asarray(scores, dtype=float)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any() or type(offset) not in (int, float)
        or not np.isfinite(offset) or abs(offset) > 64):
        raise ValueError("첫 상태 보정의 점수·절편 오류")
    q = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return 1 / (1 + np.exp(-np.clip(np.log(q) - np.log1p(-q) + offset, -700, 700)))


def fit_binary_offset(scores, labels):
    p, y = np.asarray(scores), np.asarray(labels)
    calibrated_binary(p, 0.)
    if (y.shape != p.shape or len(y) < 100 or not np.isin(y, [0, 1]).all()
        or min(y.sum(), len(y) - y.sum()) < 20):
        raise ValueError("첫 상태 절편의 정답·지원 부족")
    target, low, high = float(y.mean()), -64., 64.
    if not calibrated_binary(p, low).mean() < target < calibrated_binary(p, high).mean():
        raise ValueError("첫 상태 절편의 탐색 범위 초과")
    for _ in range(100):
        middle = (low + high) / 2
        if calibrated_binary(p, middle).mean() < target:
            low = middle
        else:
            high = middle
    value = (low + high) / 2
    if abs(calibrated_binary(p, value).mean() - target) >= 1e-12:
        raise ValueError("첫 상태 절편의 수렴 실패")
    return value


def linear_probabilities(raw, values):
    if (raw.get("format") != "first_state_linear_v1" or raw.get("features") != FEATURES
        or raw.get("action") not in FIRST_ACTIONS
        or raw.get("settings") != {"C": .1, "max_iter": 2000, "random_state": 0}):
        raise ValueError("첫 상태 선형 모형의 형식·고정 설정 오류")
    parameters = {k: np.asarray(raw[k], dtype=float) for k in ["mean", "scale", "coef"]}
    if (any(v.shape != (len(FEATURES),) or not np.isfinite(v).all() for v in parameters.values())
        or (parameters["scale"] <= 0).any() or type(raw.get("intercept")) not in (int, float)
        or not np.isfinite(raw["intercept"])):
        raise ValueError("첫 상태 선형 모형의 숫자·척도 오류")
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("첫 상태 선형 모형의 입력 오류")
    z = ((x - parameters["mean"]) / parameters["scale"]) @ parameters["coef"] + raw["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("첫 상태 선형 모형의 연산 범위 초과")
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit_first_models(rows):
    support = validate_first_rows(rows)
    models, offsets, thresholds, details = {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in FIRST_ACTIONS:
            train, calibration = (rows[k].loc[first_mask(rows[k], action)]
                                  for k in ["training", "calibration"])
            x, cx = (v[FEATURES].to_numpy(float) for v in [train, calibration])
            y, cy = (v["y_" + action].to_numpy(int) for v in [train, calibration])
            tree, _ = CLASSES[action].fit(x, y[:, None], cx)
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
            if learner.n_iter_.max() >= 2000:
                raise ValueError("첫 상태 선형 적합의 수렴 실패")
            linear = {"format": "first_state_linear_v1", "features": FEATURES, "action": action,
                "settings": {"C": .1, "max_iter": 2000, "random_state": 0},
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
            for xx in [x, cx]:
                np.testing.assert_allclose(linear_probabilities(linear, xx),
                    learner.predict_proba(scaler.transform(xx))[:, 1], rtol=0, atol=1e-12)
            models[action] = {"tree": tree.to_dict(), "linear": linear}
            scores = {"tree_raw": tree.probabilities(cx)[:, 0], "linear_raw": linear_probabilities(linear, cx)}
            offsets[action] = {}
            for name in ["tree", "linear"]:
                offsets[action][name] = fit_binary_offset(scores[name + "_raw"], cy)
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
            scores["constant"] = np.full(len(cx), cy.mean())
            thresholds[action], details[action] = {}, {}
            for name, p in scores.items():
                boundary, selected = select_threshold(cy, p, BETAS[action])
                thresholds[action][name], details[action][name] = boundary, selected
    return models, offsets, thresholds, details, support


def first_scores(frame, action, models, offsets):
    x = frame.loc[first_mask(frame, action), FEATURES].to_numpy(float)
    model = CLASSES[action].from_dict(models[action]["tree"])
    result = {"tree_raw": model.probabilities(x)[:, 0],
              "linear_raw": linear_probabilities(models[action]["linear"], x)}
    for name in ["tree", "linear"]:
        result[name] = calibrated_binary(result[name + "_raw"], offsets[action][name])
    return result


def metrics(labels, scores, requests, beta):
    y, p = np.asarray(labels), np.asarray(scores)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("첫 상태 평가의 점수 오류")
    if not len(y):
        return None
    result = decision_metrics(y, requests, beta)
    result.update(log_loss=float(log_loss(y, p, labels=[0, 1])),
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(brier_score_loss(y, p)))
    return result


def prediction_frame(frame, policy, models, offsets, thresholds, constants):
    columns = ["end", "episode_id", *["y_" + a for a in ["exit", *FIRST_ACTIONS]],
               "past_reduce_exists", "past_increase_exists"]
    result = frame[columns].copy()
    native = policy.manager.probabilities(frame[FEATURES].to_numpy(float))
    result["exit_score"] = native[:, 0]
    result["exit_request"] = native[:, 0] >= policy.thresholds["exit"]
    for j, action in enumerate(FIRST_ACTIONS, start=1):
        first = first_mask(frame, action)
        baseline = native[:, j]
        limits = np.where(first, policy.first_thresholds[action], policy.thresholds[action])
        requests = baseline >= limits
        result[action + "_baseline_score"] = baseline
        result[action + "_baseline_request"] = requests
        scores = first_scores(frame, action, models, offsets)
        scores["constant"] = np.full(int(first.sum()), constants[action])
        for name, p in scores.items():
            combined, predicted = baseline.copy(), requests.copy()
            combined[first], predicted[first] = p, p >= thresholds[action][name]
            result[action + "_" + name + "_score"] = combined
            result[action + "_" + name + "_request"] = predicted
    return result


def evaluate_frame(frame):
    result = {}
    for action in FIRST_ACTIONS:
        first = first_mask(frame, action)
        result[action] = {}
        for group, mask in [("all", np.ones(len(frame), dtype=bool)), ("first", first), ("repeat", ~first)]:
            result[action][group] = {}
            for name in ["baseline", "tree_raw", "linear_raw", "tree", "linear", "constant"]:
                result[action][group][name] = metrics(frame.loc[mask, "y_" + action].to_numpy(),
                    frame.loc[mask, action + "_" + name + "_score"].to_numpy(),
                    frame.loc[mask, action + "_" + name + "_request"].to_numpy(), BETAS[action])
    return result


def first_state_admission(evaluation, unchanged):
    checks = {}
    for action in FIRST_ACTIONS:
        part, full = evaluation[action]["first"], evaluation[action]["all"]
        new = part["tree"]
        for name in ["baseline", "linear", "tree", "tree_raw", "constant"]:
            m = part[name]
            if (min(m["positive"], m["rows"] - m["positive"]) < 20
                or not np.isfinite([m["log_loss"], m["average_precision"], m["recall"], m["f_beta"]]).all()
                or m["log_loss"] < 0 or not 0 <= m["average_precision"] <= 1):
                raise ValueError("첫 상태 판정의 지표·지원 오류")
        for name in ["baseline", "linear"]:
            gain = 1 - new["log_loss"] / part[name]["log_loss"] if part[name]["log_loss"] else 0.
            checks[action + "_log_loss_" + name] = bool(gain > .01 + 1e-12)
            checks[action + "_ap_" + name] = bool(new["average_precision"] >= part[name]["average_precision"] - 1e-12)
        checks[action + "_raw_preserved"] = bool(new["log_loss"] <= part["tree_raw"]["log_loss"] + 1e-12)
        checks[action + "_constant_beaten"] = bool(new["log_loss"] < part["constant"]["log_loss"] - 1e-12)
        for key in ["recall", "f_beta"]:
            checks[action + "_first_" + key] = bool(new[key] > part["baseline"][key] + 1e-12)
        checks[action + "_all_f_preserved"] = bool(full["tree"]["f_beta"] >= full["baseline"]["f_beta"] - 1e-12)
    if set(unchanged) != {"exit", "repeat_reduce", "repeat_increase"} or any(type(v) is not bool for v in unchanged.values()):
        raise ValueError("첫 상태의 보존 조건 오류")
    checks.update(unchanged)
    return {"first_state_models_admitted": all(checks.values()), "checks": checks,
        "profitability_accepted": False, "trading_returns_evaluated": False,
        "all_source_periods_already_observed": True}
