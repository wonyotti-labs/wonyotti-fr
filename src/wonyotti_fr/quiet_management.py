from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .first_management import BETAS
from .first_state_models import (
    FEATURES,
    calibrated_binary,
    fit_binary_offset,
    linear_probabilities,
    metrics,
    validate_first_rows,
)
from .management_capacity import Depth2Iter256

ACTIONS = ("reduce", "increase")
NAMES = ("baseline", "capacity", "tree_raw", "linear_raw", "tree", "linear", "constant")


class QuietReduceModel(Depth2Iter256):
    actions = ["reduce"]
    format = "quiet_reduce_histogram_v1"


class QuietIncreaseModel(Depth2Iter256):
    actions = ["increase"]
    format = "quiet_increase_histogram_v1"


CLASSES = {"reduce": QuietReduceModel, "increase": QuietIncreaseModel}


def quiet_mask(frame, action):
    if action not in ACTIONS:
        raise ValueError("최근 행동 분리의 지원하지 않는 행동")
    logged = frame[f"past_{action}_log_count_15m"].to_numpy(float)
    exists = frame[f"past_{action}_exists"].to_numpy()
    if (not np.isfinite(logged).all() or (logged < 0).any() or (logged > np.log1p(2**31)).any()
        or not np.isin(exists, [0, 1]).all() or (logged[exists == 0] != 0).any()):
        raise ValueError("최근 행동 분리의 횟수·존재 입력 오류")
    counts = np.expm1(logged)
    if (not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-7)
        or ((counts < .5) & (logged != 0)).any()):
        raise ValueError("최근 행동 분리의 비정수 횟수")
    return logged == 0


def groups(frame, action):
    quiet = quiet_mask(frame, action)
    first = frame[f"past_{action}_exists"].eq(0).to_numpy()
    return {"all": np.ones(len(frame), dtype=bool), "quiet": quiet, "recent": ~quiet,
            "first": first, "restart": quiet & ~first}


def validate_quiet_rows(rows):
    original = validate_first_rows(rows)
    support = {}
    for phase, frame in rows.items():
        support[phase] = {}
        for action in ACTIONS:
            support[phase][action] = {}
            for group, mask in groups(frame, action).items():
                y = frame.loc[mask, "y_" + action].to_numpy(int)
                support[phase][action][group] = {
                    "rows": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum())}
                if group in {"quiet", "recent"} and (
                    len(y) < (1000 if phase == "training" else 100)
                    or min(y.sum(), len(y) - y.sum()) < 20
                ):
                    raise ValueError("최근 행동 분리의 기간·상태별 지원 부족")
    return {"original_first_support": original, "states": support}


def fit_quiet_models(train, calibration):
    models, offsets, thresholds, details, constants = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            parts = [f.loc[quiet_mask(f, action)] for f in [train, calibration]]
            x, cx = [f[FEATURES].to_numpy(float) for f in parts]
            y, cy = [f["y_" + action].to_numpy(int) for f in parts]
            if any(min(v.sum(), len(v) - v.sum()) < 20 for v in [y, cy]):
                raise ValueError("최근 행동 분리 학습의 양성·음성 부족")
            tree, _ = CLASSES[action].fit(x, y[:, None], cx)
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
            if learner.n_iter_.max() >= 2000:
                raise ValueError("최근 행동 분리 선형의 수렴 실패")
            linear = {"format": "first_state_linear_v1", "features": FEATURES, "action": action,
                "settings": {"C": .1, "max_iter": 2000, "random_state": 0},
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
            for values in [x, cx]:
                np.testing.assert_allclose(linear_probabilities(linear, values),
                    learner.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
            models[action] = {"tree": tree.to_dict(), "linear": linear,
                              "scope": "past_action_count_in_15m_equals_zero"}
            scores = {"tree_raw": tree.probabilities(cx)[:, 0],
                      "linear_raw": linear_probabilities(linear, cx)}
            offsets[action], thresholds[action], details[action] = {}, {}, {}
            for name in ["tree", "linear"]:
                offsets[action][name] = fit_binary_offset(scores[name + "_raw"], cy)
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
            constants[action] = float(cy.mean())
            scores["constant"] = np.full(len(cx), constants[action])
            for name, values in scores.items():
                thresholds[action][name], details[action][name] = select_threshold(cy, values, BETAS[action])
    return models, offsets, thresholds, details, constants


def quiet_predictions(frame, original, models, offsets, thresholds, constants):
    context = ["end", "episode_id", "y_exit", "y_reduce", "y_increase",
               "past_reduce_exists", "past_increase_exists",
               "past_reduce_log_count_15m", "past_increase_log_count_15m"]
    result = frame[context].copy()
    result["exit_score"] = original.exit_native_calibrated.to_numpy()
    result["exit_request"] = original.exit_native_calibrated_request.to_numpy()
    for action in ACTIONS:
        quiet = quiet_mask(frame, action)
        x = frame.loc[quiet, FEATURES].to_numpy(float)
        baseline = original[action + "_native_calibrated"].to_numpy()
        requested = original[action + "_native_calibrated_request"].to_numpy()
        result[action + "_baseline_score"] = baseline
        result[action + "_baseline_request"] = requested
        capacity, capacity_request = baseline.copy(), requested.copy()
        capacity[quiet] = original.loc[quiet, action + "_capacity_calibrated"]
        capacity_request[quiet] = original.loc[quiet, action + "_capacity_calibrated_request"]
        result[action + "_capacity_score"] = capacity
        result[action + "_capacity_request"] = capacity_request
        tree = CLASSES[action].from_dict(models[action]["tree"])
        scores = {"tree_raw": tree.probabilities(x)[:, 0],
                  "linear_raw": linear_probabilities(models[action]["linear"], x)}
        for name in ["tree", "linear"]:
            scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
        scores["constant"] = np.full(int(quiet.sum()), constants[action])
        for name, values in scores.items():
            combined, requests = baseline.copy(), requested.copy()
            combined[quiet], requests[quiet] = values, values >= thresholds[action][name]
            result[action + "_" + name + "_score"] = combined
            result[action + "_" + name + "_request"] = requests
    return result


def evaluate_quiet(frame):
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for group, mask in groups(frame, action).items():
            result[action][group] = {name: metrics(frame.loc[mask, "y_" + action].to_numpy(),
                frame.loc[mask, action + "_" + name + "_score"].to_numpy(),
                frame.loc[mask, action + "_" + name + "_request"].to_numpy(), BETAS[action])
                for name in NAMES}
    return result


def quiet_admission(evaluation, unchanged):
    checks = {}
    for action in ACTIONS:
        part = evaluation[action]["quiet"]
        candidate = part["tree"]
        for name in NAMES:
            value = part[name]
            if (min(value["positive"], value["rows"] - value["positive"]) < 20
                or not np.isfinite([value[k] for k in ["log_loss", "average_precision", "recall", "f_beta"]]).all()):
                raise ValueError("최근 행동 분리 판정의 지표·지원 부족")
        for name in ["baseline", "capacity", "linear"]:
            gain = 1 - candidate["log_loss"] / part[name]["log_loss"] if part[name]["log_loss"] else 0.
            checks[action + "_log_loss_" + name] = bool(gain > .01 + 1e-12)
            checks[action + "_ap_" + name] = bool(candidate["average_precision"] >= part[name]["average_precision"] - 1e-12)
        checks[action + "_raw_preserved"] = bool(candidate["log_loss"] <= part["tree_raw"]["log_loss"] + 1e-12)
        checks[action + "_constant_beaten"] = bool(candidate["log_loss"] < part["constant"]["log_loss"] - 1e-12)
        for key in ["recall", "f_beta"]:
            checks[action + "_quiet_" + key] = bool(candidate[key] > part["baseline"][key] + 1e-12)
        for group in ["all", "first"]:
            values = evaluation[action][group]
            checks[action + "_" + group + "_f_preserved"] = bool(values["tree"]["f_beta"] >= values["baseline"]["f_beta"] - 1e-12)
    if set(unchanged) != {"exit", "recent_reduce", "recent_increase"} or any(type(v) is not bool for v in unchanged.values()):
        raise ValueError("최근 행동 분리의 보존 조건 오류")
    checks.update(unchanged)
    return {"quiet_models_admitted": all(checks.values()), "checks": checks,
            "profitability_accepted": False, "trading_returns_evaluated": False,
            "all_source_periods_already_observed": True}
