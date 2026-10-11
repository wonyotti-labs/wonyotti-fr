from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .add_count_audit import count_masks, validate_count_predictions
from .causal_management import causal_admission
from .first_management import BETAS
from .first_state_models import FEATURES as BASE_FEATURES
from .first_state_models import calibrated_binary, fit_binary_offset, metrics
from .management_capacity import Depth2Iter256
from .quiet_management import ACTIONS, groups, quiet_mask

FEATURES = ["adds" if name == "adds_capped" else name for name in BASE_FEATURES]
COMPARISONS = ("v127", "v135")
LINEAR_SETTINGS = {"C": .1, "max_iter": 2000, "random_state": 0}
SCOPE = "past_action_count_in_15m_equals_zero"


class UncappedReduceModel(Depth2Iter256):
    features = FEATURES
    actions = ["reduce"]
    format = "uncapped_quiet_reduce_v1"


class UncappedIncreaseModel(Depth2Iter256):
    features = FEATURES
    actions = ["increase"]
    format = "uncapped_quiet_increase_v1"


CLASSES = {"reduce": UncappedReduceModel, "increase": UncappedIncreaseModel}


def validate_uncapped_inputs(frame):
    if not set([*BASE_FEATURES, "adds"]).issubset(frame) or not np.isfinite(frame[[*BASE_FEATURES, "adds"]]).all().all():
        raise ValueError("누적 횟수 관리의 입력 누락·비유한 값")
    count_masks(frame)
    if not frame.direction.isin([-1, 1]).all() or not np.array_equal(frame.adds_capped, np.minimum(frame.adds, 5)):
        raise ValueError("누적 횟수 관리의 방향·상한 연결 오류")
    for action in ACTIONS:
        quiet_mask(frame, action)


def uncapped_linear_probabilities(raw, values, action):
    if (action not in ACTIONS or raw.get("format") != "uncapped_quiet_linear_v1"
        or raw.get("features") != FEATURES or raw.get("action") != action
        or raw.get("settings") != LINEAR_SETTINGS):
        raise ValueError("누적 횟수 관리 선형의 형식·입력·행동 설정 오류")
    parameters = {k: np.asarray(raw[k], dtype=float) for k in ["mean", "scale", "coef"]}
    if (any(v.shape != (len(FEATURES),) or not np.isfinite(v).all() for v in parameters.values())
        or (parameters["scale"] <= 0).any() or type(raw.get("intercept")) not in (int, float)
        or not np.isfinite(raw["intercept"])):
        raise ValueError("누적 횟수 관리 선형의 숫자·척도 오류")
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("누적 횟수 관리 선형의 입력 오류")
    z = ((x - parameters["mean"]) / parameters["scale"]) @ parameters["coef"] + raw["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("누적 횟수 관리 선형의 연산 범위 초과")
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit_uncapped_models(train, calibration):
    for frame in [train, calibration]:
        validate_uncapped_inputs(frame)
    models, offsets, thresholds, details, constants = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            parts = [f.loc[quiet_mask(f, action)] for f in [train, calibration]]
            x, cx = [f[FEATURES].to_numpy(float) for f in parts]
            y, cy = [f["y_" + action].to_numpy() for f in parts]
            if any(not np.isin(v, [0, 1]).all() or min(v.sum(), len(v)-v.sum()) < 20 for v in [y, cy]):
                raise ValueError("누적 횟수 관리 학습의 양성·음성 지원 부족")
            tree, _ = CLASSES[action].fit(x, y[:, None], cx)
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(**LINEAR_SETTINGS).fit(scaler.transform(x), y)
            if learner.n_iter_.max() >= 2000:
                raise ValueError("누적 횟수 관리 선형의 수렴 실패")
            linear = {"format": "uncapped_quiet_linear_v1", "features": FEATURES.copy(), "action": action,
                "settings": LINEAR_SETTINGS.copy(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
            for values in [x, cx]:
                np.testing.assert_allclose(uncapped_linear_probabilities(linear, values, action),
                    learner.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
            models[action] = {"tree": tree.to_dict(), "linear": linear, "scope": SCOPE}
            scores = {"tree_raw": tree.probabilities(cx)[:, 0], "linear_raw": uncapped_linear_probabilities(linear, cx, action)}
            offsets[action], thresholds[action], details[action] = {}, {}, {}
            for name in ["tree", "linear"]:
                offsets[action][name] = fit_binary_offset(scores[name + "_raw"], cy)
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
            constants[action] = float(cy.mean())
            scores["constant"] = np.full(len(cy), constants[action])
            for name, values in scores.items():
                thresholds[action][name], details[action][name] = select_threshold(cy, values, BETAS[action])
    return models, offsets, thresholds, details, constants


def uncapped_quiet_predictions(frame, previous, models, offsets, thresholds, constants):
    validate_uncapped_inputs(frame)
    context = [k for k in previous if k in frame]
    pd.testing.assert_frame_equal(frame[context], previous[context], check_exact=True)
    result = previous.copy()
    for action in ACTIONS:
        if models[action].get("scope") != SCOPE:
            raise ValueError("누적 횟수 관리 모델의 학습 범위 오류")
        tree = CLASSES[action].from_dict(models[action]["tree"])
        quiet = quiet_mask(frame, action)
        if not quiet.any():
            continue
        x = frame.loc[quiet, FEATURES].to_numpy(float)
        scores = {"tree_raw": tree.probabilities(x)[:, 0],
                  "linear_raw": uncapped_linear_probabilities(models[action]["linear"], x, action)}
        for name in ["tree", "linear"]:
            scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
        if type(constants[action]) not in (int, float) or not np.isfinite(constants[action]) or not 0 < constants[action] < 1:
            raise ValueError("누적 횟수 관리 빈도 상수 오류")
        scores["constant"] = np.full(int(quiet.sum()), constants[action])
        for name, values in scores.items():
            threshold = thresholds[action][name]
            if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("누적 횟수 관리 문턱 오류")
            result.loc[quiet, action + "_" + name + "_score"] = values
            result.loc[quiet, action + "_" + name + "_request"] = values >= threshold
    return result


def pre_limit_restart_metrics(frame, predictions):
    validate_count_predictions(frame, predictions, COMPARISONS)
    mask = groups(frame, "increase")["restart"] & frame.adds.lt(5).to_numpy()
    y = frame.loc[mask, "y_increase"].to_numpy()
    return {name: metrics(y, values.loc[mask, "increase_tree_score"].to_numpy(),
        values.loc[mask, "increase_tree_request"].to_numpy(), .5) for name, values in predictions.items()}


def uncapped_admission(candidate, previous, unchanged, pre_limit):
    result = causal_admission(candidate, previous, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        old, new = previous[action]["quiet"]["tree"], candidate[action]["quiet"]["tree"]
        checks[action + "_uncapped_log_loss_gain"] = bool(1 - new["log_loss"] / old["log_loss"] > .01 + 1e-12) if old["log_loss"] else False
    if set(pre_limit) != set(COMPARISONS):
        raise ValueError("누적 횟수 제한 전 대조 누락")
    for value in pre_limit.values():
        if value is None or min(value["positive"], value["rows"] - value["positive"]) < 20 or not np.isfinite([value[k] for k in ["recall", "precision", "f_beta"]]).all():
            raise ValueError("누적 횟수 제한 전 평가 지원 부족")
    old, new = pre_limit["v127"], pre_limit["v135"]
    if (old["rows"], old["positive"]) != (new["rows"], new["positive"]):
        raise ValueError("누적 횟수 제한 전 대조 지원 불일치")
    for key in ["recall", "f_beta"]:
        checks["pre_limit_restart_increase_" + key + "_improved"] = bool(new[key] > old[key] + 1e-12)
    checks["pre_limit_restart_increase_precision_preserved"] = bool(new["precision"] >= old["precision"] - 1e-12)
    result.pop("causal_distribution_admitted")
    result["uncapped_count_admitted"] = all(checks.values())
    return result
