from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .causal_management import causal_admission, supported
from .causal_score_distribution import scores_array
from .first_management import BETAS
from .first_state_models import calibrated_binary, fit_binary_offset

KINDS = ("absolute", "joint")
FEATURES = {"absolute": ["absolute_logit"], "joint": ["absolute_logit", "past_quantile_logit"]}
SETTINGS = {"C": .1, "solver": "lbfgs", "fit_intercept": True,
            "max_iter": 2000, "tol": 1e-8, "random_state": 0, "class_weight": None}


def fusion_matrix(absolute, relative, kind):
    if kind not in KINDS:
        raise ValueError("관리 결합 보정의 입력 종류 오류")
    a, b = scores_array(absolute), scores_array(relative)
    if len(a) != len(b):
        raise ValueError("관리 결합 보정의 입력 행 수 오류")
    columns = [a] if kind == "absolute" else [a, b]
    clipped = np.clip(np.column_stack(columns), np.finfo(float).eps, 1 - np.finfo(float).eps)
    return np.log(clipped) - np.log1p(-clipped)


def fusion_probabilities(raw, absolute, relative, action, state, source, kind):
    keys = {"format", "action", "state", "source", "kind", "features", "settings", "mean", "scale", "coef", "intercept"}
    if (action not in {"reduce", "increase"} or state not in {"first", "restart"}
        or source not in {"tree", "linear"} or kind not in KINDS or set(raw) != keys
        or raw["format"] != "management_score_fusion_v1" or raw["settings"] != SETTINGS
        or any(raw[k] != v for k, v in [("action", action), ("state", state), ("source", source), ("kind", kind)])
        or raw["features"] != FEATURES[kind]):
        raise ValueError("관리 결합 보정의 모델·상태·입력 연결 오류")
    arrays = {key: np.asarray(raw[key], dtype=float) for key in ["mean", "scale", "coef"]}
    if (any(v.shape != (len(FEATURES[kind]),) or not np.isfinite(v).all() for v in arrays.values())
        or (arrays["scale"] <= 0).any() or type(raw["intercept"]) not in (float, int)
        or not np.isfinite(raw["intercept"])):
        raise ValueError("관리 결합 보정의 숫자·차원·척도 오류")
    values = fusion_matrix(absolute, relative, kind)
    z = ((values - arrays["mean"]) / arrays["scale"]) @ arrays["coef"] + raw["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("관리 결합 보정의 연산 범위 오류")
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit_score_fusion(calibration, ledger):
    result = {kind: {key: {} for key in ["models", "offsets", "thresholds", "details", "constants"]} for kind in KINDS}
    with threadpool_limits(limits=1):
        for kind, fitted in result.items():
            for action in ["reduce", "increase"]:
                for target in fitted.values():
                    target[action] = {}
                for state in ["first", "restart"]:
                    mask = supported(calibration, ledger, action, state)
                    y = calibration.loc[mask, "y_" + action].to_numpy(int)
                    if len(y) < 100 or not np.isin(calibration.loc[mask, "y_" + action], [0, 1]).all() or min(y.sum(), len(y) - y.sum()) < 20:
                        raise ValueError("관리 결합 보정의 상반기 지원 부족")
                    for key in ["models", "offsets", "thresholds", "details"]:
                        fitted[key][action][state] = {}
                    probabilities = {}
                    for source in ["tree", "linear"]:
                        a = calibration.loc[mask, action + "_" + source + "_raw_score"].to_numpy()
                        b = ledger.loc[mask, action + "_" + source + "_mapped"].to_numpy()
                        x = fusion_matrix(a, b, kind)
                        scaler = StandardScaler().fit(x)
                        learner = LogisticRegression(**SETTINGS).fit(scaler.transform(x), y)
                        if learner.n_iter_.max() >= SETTINGS["max_iter"]:
                            raise ValueError("관리 결합 보정의 수렴 실패")
                        raw = {"format": "management_score_fusion_v1", "action": action, "state": state,
                            "source": source, "kind": kind, "features": list(FEATURES[kind]), "settings": dict(SETTINGS),
                            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                            "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
                        scores = fusion_probabilities(raw, a, b, action, state, source, kind)
                        np.testing.assert_allclose(scores, learner.predict_proba(scaler.transform(x))[:, 1], rtol=0, atol=1e-12)
                        offset = fit_binary_offset(scores, y)
                        fitted["models"][action][state][source] = raw
                        fitted["offsets"][action][state][source] = offset
                        probabilities[source + "_raw"] = scores
                        probabilities[source] = calibrated_binary(scores, offset)
                    fitted["constants"][action][state] = float(y.mean())
                    probabilities["constant"] = np.full(len(y), y.mean())
                    for name, scores in probabilities.items():
                        threshold, selected = select_threshold(y, scores, BETAS[action])
                        fitted["thresholds"][action][state][name] = threshold
                        fitted["details"][action][state][name] = selected
    return result


def fusion_predictions(original, ledger, fitted, kind):
    if kind not in KINDS:
        raise ValueError("관리 결합 보정의 예측 종류 오류")
    parameters, result = fitted[kind], original.copy()
    for action in ["reduce", "increase"]:
        for state in ["first", "restart"]:
            mask = supported(original, ledger, action, state)
            if not mask.any():
                continue
            probabilities = {}
            for source in ["tree", "linear"]:
                raw = parameters["models"][action][state][source]
                a = original.loc[mask, action + "_" + source + "_raw_score"].to_numpy()
                b = ledger.loc[mask, action + "_" + source + "_mapped"].to_numpy()
                p = fusion_probabilities(raw, a, b, action, state, source, kind)
                probabilities[source + "_raw"] = p
                probabilities[source] = calibrated_binary(p, parameters["offsets"][action][state][source])
            probabilities["constant"] = np.full(mask.sum(), parameters["constants"][action][state])
            for name, values in probabilities.items():
                result.loc[mask, action + "_" + name + "_score"] = values
                result.loc[mask, action + "_" + name + "_request"] = values >= parameters["thresholds"][action][state][name]
    return result


def fusion_admission(candidate, absolute, original, unchanged):
    result = causal_admission(candidate, original, unchanged)
    checks = result["checks"]
    for action in ["reduce", "increase"]:
        new, control = candidate[action]["quiet"]["tree"], absolute[action]["quiet"]["tree"]
        gain = 1 - new["log_loss"] / control["log_loss"] if control["log_loss"] else 0.
        checks[action + "_absolute_log_loss_gain"] = bool(gain > .01 + 1e-12)
        checks[action + "_absolute_ap_preserved"] = bool(new["average_precision"] >= control["average_precision"] - 1e-12)
    result.pop("causal_distribution_admitted")
    result["score_fusion_admitted"] = all(checks.values())
    return result
