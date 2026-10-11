from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .causal_management import causal_admission
from .first_management import BETAS
from .first_state_models import FEATURES as BASE_FEATURES
from .first_state_models import calibrated_binary, fit_binary_offset
from .management_capacity import Depth2Iter256
from .order_price_history import PRICE_FEATURES
from .quiet_management import ACTIONS, quiet_mask

FEATURES = BASE_FEATURES + PRICE_FEATURES
LINEAR_SETTINGS = {"C": .1, "max_iter": 2000, "random_state": 0}
SCOPE = "past_action_count_in_15m_equals_zero"


class PriceReduceModel(Depth2Iter256):
    features = FEATURES
    actions = ["reduce"]
    format = "order_price_quiet_reduce_v1"


class PriceIncreaseModel(Depth2Iter256):
    features = FEATURES
    actions = ["increase"]
    format = "order_price_quiet_increase_v1"


CLASSES = {"reduce": PriceReduceModel, "increase": PriceIncreaseModel}


def validate_price_inputs(frame):
    if not set(FEATURES).issubset(frame) or not np.isfinite(frame[FEATURES]).all().all():
        raise ValueError("가격 관리 모델의 입력 누락·비유한 값")
    if not frame.direction.isin([-1, 1]).all():
        raise ValueError("가격 관리 모델의 포지션 방향 오류")
    for action in ACTIONS:
        quiet_mask(frame, action)
        raw = frame[f"past_{action}_log_price_move"].to_numpy()
        directional = frame[f"past_{action}_directional_log_price_move"].to_numpy()
        if (not np.array_equal(directional, raw * frame.direction.to_numpy())
            or (raw[frame[f"past_{action}_exists"].eq(0)] != 0).any()):
            raise ValueError("가격 관리 모델의 가격·방향·부재 연결 오류")


def price_linear_probabilities(raw, values, action):
    if (action not in ACTIONS or raw.get("format") != "order_price_quiet_linear_v1"
        or raw.get("features") != FEATURES or raw.get("action") != action
        or raw.get("settings") != LINEAR_SETTINGS):
        raise ValueError("가격 관리 선형의 형식·입력·행동 설정 오류")
    parameters = {k: np.asarray(raw[k], dtype=float) for k in ["mean", "scale", "coef"]}
    if (any(v.shape != (len(FEATURES),) or not np.isfinite(v).all() for v in parameters.values())
        or (parameters["scale"] <= 0).any() or type(raw.get("intercept")) not in (int, float)
        or not np.isfinite(raw["intercept"])):
        raise ValueError("가격 관리 선형의 숫자·척도 오류")
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("가격 관리 선형의 입력 오류")
    z = ((x - parameters["mean"]) / parameters["scale"]) @ parameters["coef"] + raw["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("가격 관리 선형의 연산 범위 초과")
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit_price_models(train, calibration):
    for frame in [train, calibration]:
        validate_price_inputs(frame)
    models, offsets, thresholds, details, constants = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            parts = [f.loc[quiet_mask(f, action)] for f in [train, calibration]]
            x, cx = [f[FEATURES].to_numpy(float) for f in parts]
            y, cy = [f["y_" + action].to_numpy() for f in parts]
            if any(not np.isin(v, [0, 1]).all() or min(v.sum(), len(v)-v.sum()) < 20 for v in [y, cy]):
                raise ValueError("가격 관리 학습의 양성·음성 지원 부족")
            tree, _ = CLASSES[action].fit(x, y[:, None], cx)
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(**LINEAR_SETTINGS).fit(scaler.transform(x), y)
            if learner.n_iter_.max() >= 2000:
                raise ValueError("가격 관리 선형의 수렴 실패")
            linear = {"format": "order_price_quiet_linear_v1", "features": FEATURES.copy(), "action": action,
                "settings": LINEAR_SETTINGS.copy(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
            for values in [x, cx]:
                np.testing.assert_allclose(price_linear_probabilities(linear, values, action),
                    learner.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
            models[action] = {"tree": tree.to_dict(), "linear": linear, "scope": SCOPE}
            scores = {"tree_raw": tree.probabilities(cx)[:, 0], "linear_raw": price_linear_probabilities(linear, cx, action)}
            offsets[action], thresholds[action], details[action] = {}, {}, {}
            for name in ["tree", "linear"]:
                offsets[action][name] = fit_binary_offset(scores[name + "_raw"], cy)
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
            constants[action] = float(cy.mean())
            scores["constant"] = np.full(len(cy), constants[action])
            for name, values in scores.items():
                thresholds[action][name], details[action][name] = select_threshold(cy, values, BETAS[action])
    return models, offsets, thresholds, details, constants


def price_quiet_predictions(frame, previous, models, offsets, thresholds, constants):
    validate_price_inputs(frame)
    context = [k for k in previous if k in frame]
    pd.testing.assert_frame_equal(frame[context], previous[context], check_exact=True)
    result = previous.copy()
    for action in ACTIONS:
        if models[action].get("scope") != SCOPE:
            raise ValueError("가격 관리 모델의 학습 범위 오류")
        tree = CLASSES[action].from_dict(models[action]["tree"])
        quiet = quiet_mask(frame, action)
        if not quiet.any():
            continue
        x = frame.loc[quiet, FEATURES].to_numpy(float)
        scores = {"tree_raw": tree.probabilities(x)[:, 0],
                  "linear_raw": price_linear_probabilities(models[action]["linear"], x, action)}
        for name in ["tree", "linear"]:
            scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
        if type(constants[action]) not in (int, float) or not np.isfinite(constants[action]) or not 0 < constants[action] < 1:
            raise ValueError("가격 관리 빈도 상수 오류")
        scores["constant"] = np.full(int(quiet.sum()), constants[action])
        for name, values in scores.items():
            threshold = thresholds[action][name]
            if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("가격 관리 문턱 오류")
            result.loc[quiet, action + "_" + name + "_score"] = values
            result.loc[quiet, action + "_" + name + "_request"] = values >= threshold
    return result


def price_admission(candidate, previous, unchanged):
    result = causal_admission(candidate, previous, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        old, new = previous[action]["quiet"]["tree"], candidate[action]["quiet"]["tree"]
        checks[action + "_price_log_loss_gain"] = bool(1 - new["log_loss"] / old["log_loss"] > .01 + 1e-12) if old["log_loss"] else False
    result.pop("causal_distribution_admitted")
    result["order_price_admitted"] = all(checks.values())
    return result
