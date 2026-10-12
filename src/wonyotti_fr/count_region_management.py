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
from .histogram_management import HISTOGRAM_SETTINGS
from .order_history_boost import OrderHistoryBoostModels
from .quiet_management import ACTIONS, NAMES, groups, quiet_mask

FEATURES = BASE_FEATURES.copy()
REGIONS = ("below", "capped")
COMPARISONS = ("v127", "v137")
LINEAR_SETTINGS = {"C": .1, "max_iter": 2000, "random_state": 0}
SCOPE = "past_action_count_in_15m_equals_zero"


class RegionBaseModel(OrderHistoryBoostModels):
    features = FEATURES
    fixed_settings = {**HISTOGRAM_SETTINGS, "max_iter": 128}


class BelowReduceModel(RegionBaseModel):
    actions = ["reduce"]
    format = "count_region_below_reduce_v1"


class CappedReduceModel(RegionBaseModel):
    actions = ["reduce"]
    format = "count_region_capped_reduce_v1"


class BelowIncreaseModel(RegionBaseModel):
    actions = ["increase"]
    format = "count_region_below_increase_v1"


class CappedIncreaseModel(RegionBaseModel):
    actions = ["increase"]
    format = "count_region_capped_increase_v1"


CLASSES = {"reduce": {"below": BelowReduceModel, "capped": CappedReduceModel},
           "increase": {"below": BelowIncreaseModel, "capped": CappedIncreaseModel}}


def validate_count_region_inputs(frame):
    if not frame.columns.is_unique or not frame.index.is_unique or not set([*BASE_FEATURES, "adds"]).issubset(frame) or not np.isfinite(frame[[*BASE_FEATURES, "adds"]]).all().all():
        raise ValueError("횟수 분리 관리의 입력 누락·비유한 값")
    count_masks(frame)
    if not frame.direction.isin([-1, 1]).all() or not np.array_equal(frame.adds_capped, np.minimum(frame.adds, 5)):
        raise ValueError("횟수 분리 관리의 방향·상한 연결 오류")
    for action in ACTIONS:
        quiet_mask(frame, action)


def count_region_linear_probabilities(raw, values, action, region):
    if (action not in ACTIONS or region not in REGIONS or raw.get("region") != region or raw.get("format") != "count_region_quiet_linear_v1"
        or raw.get("features") != FEATURES or raw.get("action") != action
        or raw.get("settings") != LINEAR_SETTINGS):
        raise ValueError("횟수 분리 관리 선형의 형식·입력·행동 설정 오류")
    parameters = {k: np.asarray(raw[k], dtype=float) for k in ["mean", "scale", "coef"]}
    if (any(v.shape != (len(FEATURES),) or not np.isfinite(v).all() for v in parameters.values())
        or (parameters["scale"] <= 0).any() or type(raw.get("intercept")) not in (int, float)
        or not np.isfinite(raw["intercept"])):
        raise ValueError("횟수 분리 관리 선형의 숫자·척도 오류")
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("횟수 분리 관리 선형의 입력 오류")
    z = ((x - parameters["mean"]) / parameters["scale"]) @ parameters["coef"] + raw["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("횟수 분리 관리 선형의 연산 범위 초과")
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def region_masks(frame, action):
    count_masks(frame)
    if not np.array_equal(frame.adds_capped, np.minimum(frame.adds, 5)):
        raise ValueError("횟수 분리의 기존 상한 연결 오류")
    quiet = quiet_mask(frame, action)
    below = frame.adds_capped.lt(5).to_numpy()
    return {"below": quiet & below, "capped": quiet & ~below}


def region_support(frame, action):
    result = {}
    for region, mask in region_masks(frame, action).items():
        part = frame.loc[mask]
        y = part["y_" + action].to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError("횟수 분리의 이진 정답 지원 오류")
        result[region] = {"rows": len(part), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
            "episodes": int(part.episode_id.nunique()), "positive_episodes": int(part.loc[y == 1, "episode_id"].nunique())}
    return result


def validate_region_support(rows):
    result = {}
    for phase, frame in rows.items():
        validate_count_region_inputs(frame)
        if phase not in {"training", "calibration", "diagnosis"}:
            raise ValueError("횟수 분리의 지원 기간 오류")
        result[phase] = {a: region_support(frame, a) for a in ACTIONS}
        for regions in result[phase].values():
            for value in regions.values():
                if value["rows"] < (1000 if phase == "training" else 100) or min(value["positive"], value["negative"]) < 20:
                    raise ValueError("횟수 분리 학습의 양성·음성 지원 부족")
    return result


def region_raw_scores(frame, action, models):
    if set(models) != {*REGIONS, "scope"} or models.get("scope") != SCOPE:
        raise ValueError("횟수 분리 모델의 학습 범위 오류")
    quiet = quiet_mask(frame, action)
    part = frame.loc[quiet]
    scores = {name: np.full(len(part), np.nan) for name in ["tree_raw", "linear_raw"]}
    for region, mask in region_masks(part, action).items():
        model = models[region]
        if model.get("region") != region:
            raise ValueError("횟수 분리 모델의 분기 연결 오류")
        tree = CLASSES[action][region].from_dict(model["tree"])
        x = part.loc[mask, FEATURES].to_numpy(float)
        scores["linear_raw"][mask] = count_region_linear_probabilities(model["linear"], x, action, region)
        if mask.any():
            scores["tree_raw"][mask] = tree.probabilities(x)[:, 0]
    if any(not np.isfinite(p).all() for p in scores.values()):
        raise ValueError("횟수 분리 원시 점수의 연결 누락")
    return scores


def fit_count_region_models(train, calibration):
    validate_region_support({"training": train, "calibration": calibration})
    models, offsets, thresholds, details, constants = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            models[action] = {"scope": SCOPE}
            for region in REGIONS:
                parts = [f.loc[region_masks(f, action)[region]] for f in [train, calibration]]
                x, cx = [f[FEATURES].to_numpy(float) for f in parts]
                y = parts[0]["y_" + action].to_numpy()
                tree, _ = CLASSES[action][region].fit(x, y[:, None], cx)
                scaler = StandardScaler().fit(x)
                learner = LogisticRegression(**LINEAR_SETTINGS).fit(scaler.transform(x), y)
                if learner.n_iter_.max() >= 2000:
                    raise ValueError("횟수 분리 관리 선형의 수렴 실패")
                linear = {"format": "count_region_quiet_linear_v1", "features": FEATURES.copy(), "action": action,
                    "region": region, "settings": LINEAR_SETTINGS.copy(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "coef": learner.coef_[0].tolist(), "intercept": float(learner.intercept_[0])}
                for values in [x, cx]:
                    np.testing.assert_allclose(count_region_linear_probabilities(linear, values, action, region),
                        learner.predict_proba(scaler.transform(values))[:, 1], rtol=0, atol=1e-12)
                models[action][region] = {"tree": tree.to_dict(), "linear": linear, "region": region}
            scores = region_raw_scores(calibration, action, models[action])
            cy = calibration.loc[quiet_mask(calibration, action), "y_" + action].to_numpy()
            offsets[action], thresholds[action], details[action] = {}, {}, {}
            for name in ["tree", "linear"]:
                offsets[action][name] = fit_binary_offset(scores[name + "_raw"], cy)
                scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
            constants[action] = float(cy.mean())
            scores["constant"] = np.full(len(cy), constants[action])
            for name, values in scores.items():
                thresholds[action][name], details[action][name] = select_threshold(cy, values, BETAS[action])
    return models, offsets, thresholds, details, constants


def count_region_quiet_predictions(frame, previous, models, offsets, thresholds, constants):
    validate_count_region_inputs(frame)
    context = [k for k in previous if k in frame]
    pd.testing.assert_frame_equal(frame[context], previous[context], check_exact=True)
    result = previous.copy()
    for action in ACTIONS:
        quiet = quiet_mask(frame, action)
        scores = region_raw_scores(frame, action, models[action])
        if not quiet.any():
            continue
        for name in ["tree", "linear"]:
            scores[name] = calibrated_binary(scores[name + "_raw"], offsets[action][name])
        if type(constants[action]) not in (int, float) or not np.isfinite(constants[action]) or not 0 < constants[action] < 1:
            raise ValueError("횟수 분리 관리 빈도 상수 오류")
        scores["constant"] = np.full(int(quiet.sum()), constants[action])
        for name, values in scores.items():
            threshold = thresholds[action][name]
            if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("횟수 분리 관리 문턱 오류")
            result.loc[quiet, action + "_" + name + "_score"] = values
            result.loc[quiet, action + "_" + name + "_request"] = values >= threshold
    return result


def pre_limit_restart_metrics(frame, predictions):
    validate_count_predictions(frame, predictions, COMPARISONS)
    mask = groups(frame, "increase")["restart"] & frame.adds.lt(5).to_numpy()
    y = frame.loc[mask, "y_increase"].to_numpy()
    return {name: metrics(y, values.loc[mask, "increase_tree_score"].to_numpy(),
        values.loc[mask, "increase_tree_request"].to_numpy(), .5) for name, values in predictions.items()}


def count_region_admission(candidate, previous, unchanged, pre_limit):
    result = causal_admission(candidate, previous, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        old, new = previous[action]["quiet"]["tree"], candidate[action]["quiet"]["tree"]
        checks[action + "_count_region_log_loss_gain"] = bool(1 - new["log_loss"] / old["log_loss"] > .01 + 1e-12) if old["log_loss"] else False
    if set(pre_limit) != set(COMPARISONS):
        raise ValueError("횟수 분리 제한 전 대조 누락")
    for value in pre_limit.values():
        if value is None or min(value["positive"], value["rows"] - value["positive"]) < 20 or not np.isfinite([value[k] for k in ["recall", "precision", "f_beta"]]).all():
            raise ValueError("횟수 분리 제한 전 평가 지원 부족")
    old, new = pre_limit["v127"], pre_limit["v137"]
    if (old["rows"], old["positive"]) != (new["rows"], new["positive"]):
        raise ValueError("횟수 분리 제한 전 대조 지원 불일치")
    for key in ["recall", "f_beta"]:
        checks["pre_limit_restart_increase_" + key + "_improved"] = bool(new[key] > old[key] + 1e-12)
    checks["pre_limit_restart_increase_precision_preserved"] = bool(new["precision"] >= old["precision"] - 1e-12)
    result.pop("causal_distribution_admitted")
    result["count_region_admitted"] = all(checks.values())
    return result


def evaluate_count_regions(frame, predictions):
    validate_count_predictions(frame, predictions, COMPARISONS)
    result = {}
    for action in ACTIONS:
        supports = region_support(frame, action)
        result[action] = {}
        for region, mask in region_masks(frame, action).items():
            y = frame.loc[mask, "y_" + action].to_numpy()
            models = {}
            for model, values in predictions.items():
                models[model] = {}
                for name in NAMES:
                    p = values.loc[mask, action + "_" + name + "_score"].to_numpy()
                    r = values.loc[mask, action + "_" + name + "_request"].to_numpy()
                    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or r.dtype.kind != "b":
                        raise ValueError("횟수 분리 대조의 확률·요청 오류")
                    models[model][name] = metrics(y, p, r, BETAS[action])
            result[action][region] = {"support": supports[region], "models": models}
    return result
