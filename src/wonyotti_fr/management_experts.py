from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .first_management import BETAS
from .first_state_models import FEATURES, calibrated_binary, fit_binary_offset, linear_probabilities
from .management_capacity import Depth2Iter256
from .quiet_management import ACTIONS, groups, validate_quiet_rows
from .quiet_state_calibration import STATES, state_admission


class ExpertReduceModel(Depth2Iter256):
    actions = ["reduce"]
    format = "state_expert_reduce_histogram_v1"


class ExpertIncreaseModel(Depth2Iter256):
    actions = ["increase"]
    format = "state_expert_increase_histogram_v1"


CLASSES = {"reduce": ExpertReduceModel, "increase": ExpertIncreaseModel}


def load_expert(models, action, state):
    if action not in ACTIONS or state not in STATES:
        raise ValueError("관리 분리 모델의 행동·상태 오류")
    raw = models[action][state]
    if (set(raw) != {"action", "state", "tree", "linear"}
        or raw["action"] != action or raw["state"] != state
        or raw["linear"].get("action") != action):
        raise ValueError("관리 분리 모델의 상태 연결 오류")
    return CLASSES[action].from_dict(raw["tree"])


def expert_support(rows):
    support = validate_quiet_rows(rows)
    for phase, actions in support["states"].items():
        for states in actions.values():
            for state in STATES:
                part = states[state]
                if (part["rows"] < (1000 if phase == "training" else 100)
                    or min(part["positive"], part["negative"]) < 20):
                    raise ValueError("관리 분리 모델의 최초·재시작 지원 부족")
    return support


def fit_experts(training, calibration):
    models, offsets, thresholds, details, constants = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            for result in [models, offsets, thresholds, details, constants]:
                result[action] = {}
            for state in STATES:
                train = training.loc[groups(training, action)[state]]
                calibrate = calibration.loc[groups(calibration, action)[state]]
                x, cx = [f[FEATURES].to_numpy(float) for f in [train, calibrate]]
                y, cy = [f["y_" + action].to_numpy(int) for f in [train, calibrate]]
                if (len(y) < 1000 or len(cy) < 100
                    or any(not np.isin(v, [0, 1]).all() or min(v.sum(), len(v) - v.sum()) < 20 for v in [y, cy])):
                    raise ValueError("관리 분리 모델의 학습·보정 표본 부족")
                tree, _ = CLASSES[action].fit(x, y[:, None], cx)
                scaler = StandardScaler().fit(x)
                learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y)
                if learner.n_iter_.max() >= 2000:
                    raise ValueError("관리 분리 선형 모델의 수렴 실패")
                linear = {"format": "first_state_linear_v1", "features": FEATURES, "action": action,
                    "settings": {"C": .1, "max_iter": 2000, "random_state": 0},
                    "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
                for values in [x, cx]:
                    np.testing.assert_allclose(linear_probabilities(linear, values),
                        learner.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
                models[action][state] = {"action": action, "state": state,
                                         "tree": tree.to_dict(), "linear": linear}
                scores = {"tree_raw": tree.probabilities(cx)[:, 0], "linear_raw": linear_probabilities(linear, cx)}
                offsets[action][state], thresholds[action][state], details[action][state] = {}, {}, {}
                for model in ["tree", "linear"]:
                    offset = fit_binary_offset(scores[model + "_raw"], cy)
                    offsets[action][state][model] = offset
                    scores[model] = calibrated_binary(scores[model + "_raw"], offset)
                constants[action][state] = float(cy.mean())
                scores["constant"] = np.full(len(cy), constants[action][state])
                for model, probabilities in scores.items():
                    threshold, selected = select_threshold(cy, probabilities, BETAS[action])
                    thresholds[action][state][model], details[action][state][model] = threshold, selected
    return models, offsets, thresholds, details, constants


def expert_predictions(source, previous, models, offsets, thresholds, constants):
    result = previous.copy()
    for action in ACTIONS:
        for state in STATES:
            mask = groups(source, action)[state]
            if not mask.any():
                continue
            x = source.loc[mask, FEATURES].to_numpy(float)
            model = load_expert(models, action, state)
            linear = models[action][state]["linear"]
            scores = {"tree_raw": model.probabilities(x)[:, 0], "linear_raw": linear_probabilities(linear, x)}
            for name in ["tree", "linear"]:
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][state][name])
            scores["constant"] = np.full(int(mask.sum()), constants[action][state])
            for name, values in scores.items():
                result.loc[mask, action + "_" + name + "_score"] = values
                result.loc[mask, action + "_" + name + "_request"] = values >= thresholds[action][state][name]
    return result


def expert_admission(candidate, previous, unchanged):
    result = state_admission(candidate, previous, unchanged)
    # 이전 단계 이름을 결과의 모델 변경 범위에 맞게 구분한다.
    result["state_experts_admitted"] = result.pop("state_calibration_admitted")
    return result
