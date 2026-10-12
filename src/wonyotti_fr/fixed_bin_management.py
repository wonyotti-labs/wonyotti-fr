from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .causal_management import causal_admission
from .first_state_models import FEATURES
from .management_capacity import validate_training
from .quiet_management import ACTIONS, CLASSES, quiet_mask
from .tail_weighted_management import SCOPE, validate_tail_models, validate_tail_rows
from .tail_weighted_reference import verify_tail_weights

MODELS = ("v127", "v141")


def model_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def training_digest(frame, action):
    part = frame.loc[quiet_mask(frame, action)]
    digest = hashlib.sha256()
    for values, dtype in [
        (part[FEATURES], "<f8"),
        (part["y_" + action], "<i8"),
        (part.episode_id, "<i8"),
        (part.end.astype("datetime64[ns, UTC]").array.asi8, "<i8"),
    ]:
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_bins(raw, action):
    if (
        action not in ACTIONS
        or not isinstance(raw, dict)
        or set(raw)
        != {
            "format",
            "features",
            "action",
            "rows",
            "parent_sha256",
            "training_sha256",
            "thresholds",
        }
        or raw["format"] != "fixed_unweighted_quiet_bins_v1"
        or raw["features"] != FEATURES
        or raw["action"] != action
        or type(raw["rows"]) is not int
        or raw["rows"] < 1000
        or any(
            not isinstance(raw[k], str) or not re.fullmatch("[0-9a-f]{64}", raw[k])
            for k in ["parent_sha256", "training_sha256"]
        )
        or not isinstance(raw["thresholds"], list)
        or len(raw["thresholds"]) != len(FEATURES)
    ):
        raise ValueError("고정 구간의 형식·입력·행동·학습 연결 오류")
    for values in raw["thresholds"]:
        if (
            not isinstance(values, list)
            or len(values) > 254
            or any(type(v) not in (int, float) for v in values)
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError("고정 구간의 경계 수·유한성·순서 오류")


def encode_bins(raw, action, values):
    validate_bins(raw, action)
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("고정 구간의 입력 차원·유한성 오류")
    return np.column_stack(
        [np.searchsorted(q, x[:, j], side="left") for j, q in enumerate(raw["thresholds"])]
    ).astype(np.uint8)


def export_histogram(learner, action):
    cls = CLASSES[action]
    if (
        learner.n_iter_ != 256
        or learner.do_early_stopping_
        or not np.array_equal(learner.classes_, [0, 1])
        or learner._baseline_prediction.shape != (1, 1)
    ):
        raise ValueError("고정 구간 학습기의 반복·클래스·초기 점수 오류")
    trees = []
    for values in learner._predictors:
        if len(values) != 1:
            raise ValueError("고정 구간 학습기의 이진 트리 구조 오류")
        nodes = values[0].nodes
        if nodes["is_categorical"].any() or (nodes["depth"] > 2).any():
            raise ValueError("고정 구간 학습기의 범주·깊이 오류")
        leaf = nodes["is_leaf"].astype(bool)
        trees.append(
            {
                "left": np.where(leaf, -1, nodes["left"].astype(int)).tolist(),
                "right": np.where(leaf, -1, nodes["right"].astype(int)).tolist(),
                "feature": np.where(leaf, -2, nodes["feature_idx"].astype(int)).tolist(),
                "threshold": np.where(leaf, 0.0, nodes["num_threshold"]).tolist(),
                "value": np.where(leaf, nodes["value"], 0.0).tolist(),
            }
        )
    raw = {
        "format": cls.format,
        "features": FEATURES,
        "settings": cls.fixed_settings,
        "models": [
            {
                "action": action,
                "baseline": float(learner._baseline_prediction[0, 0]),
                "trees": trees,
            }
        ],
    }
    return cls.from_dict(raw).to_dict()


def lift_thresholds(encoded, bins, action):
    validate_bins(bins, action)
    result = CLASSES[action].from_dict(encoded).to_dict()
    for tree in result["models"][0]["trees"]:
        for i, feature in enumerate(tree["feature"]):
            if feature < 0:
                continue
            threshold = tree["threshold"][i]
            boundaries = bins["thresholds"][feature]
            if not 0 <= threshold < len(boundaries):
                raise ValueError("고정 구간 트리의 대응 범위 밖 분기")
            # 정수 구간 <= t는 원래 입력 <= q[floor(t)]와 동치다.
            tree["threshold"][i] = boundaries[int(np.floor(threshold))]
    return CLASSES[action].from_dict(result).to_dict()


def check_level_mapper(learner, encoded):
    if learner._bin_mapper.is_categorical_.any() or learner._bin_mapper.n_bins != 256:
        raise ValueError("고정 구간 학습기의 구간 설정 오류")
    np.testing.assert_array_equal(learner._bin_mapper.transform(encoded), encoded)
    return [q.tolist() for q in learner._bin_mapper.bin_thresholds_]


def fit_fixed_bins(training, parents, weighted, ledger, details):
    validate_tail_rows(training, "training")
    validate_training(training)
    validate_tail_models(weighted)
    if set(parents) != set(ACTIONS) or set(details) != set(ACTIONS):
        raise ValueError("고정 구간 학습의 원래 모형·가중치 연결 누락")
    pd.testing.assert_frame_equal(
        training[["end", "episode_id"]], ledger[["end", "episode_id"]], check_exact=True
    )
    models, bins, controls, encoded_models, levels = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            parent = parents[action]
            if set(parent) != {"tree", "linear", "scope"} or parent["scope"] != SCOPE:
                raise ValueError("고정 구간 학습의 무가중 부모 오류")
            quiet = quiet_mask(training, action)
            part = training.loc[quiet]
            x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
            for suffix in ["score", "weight", "hard_negative"]:
                if not np.array_equal(ledger[action + "_" + suffix].notna(), quiet):
                    raise ValueError("고정 구간의 가중치 원장 지원 불일치")
            weights = ledger.loc[quiet, action + "_weight"].to_numpy(float)
            scores = CLASSES[action].from_dict(parent["tree"]).probabilities(x)[:, 0]
            np.testing.assert_array_equal(
                scores, ledger.loc[quiet, action + "_score"].to_numpy(float)
            )
            verify_tail_weights(
                y,
                scores,
                weights,
                ledger.loc[quiet, action + "_hard_negative"].to_numpy(bool),
                details[action],
            )
            original = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(x, y)
            if export_histogram(original, action) != parent["tree"]:
                raise ValueError("고정 구간의 원래 무가중 모델 재현 실패")
            bins[action] = {
                "format": "fixed_unweighted_quiet_bins_v1",
                "features": FEATURES.copy(),
                "action": action,
                "rows": len(x),
                "parent_sha256": model_digest(parent["tree"]),
                "training_sha256": training_digest(training, action),
                "thresholds": [q.tolist() for q in original._bin_mapper.bin_thresholds_],
            }
            z = encode_bins(bins[action], action, x)
            np.testing.assert_array_equal(z, original._bin_mapper.transform(x))
            # 정수 수준을 연속형으로 적합하되 구간 수와 행 순서는 그대로 유지한다.
            control = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(z, y)
            levels[action] = {"control": check_level_mapper(control, z)}
            controls[action] = export_histogram(control, action)
            if lift_thresholds(controls[action], bins[action], action) != parent["tree"]:
                raise ValueError("고정 구간의 무가중 동등성 관문 실패")
            learner = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(
                z, y, sample_weight=weights
            )
            levels[action]["weighted"] = check_level_mapper(learner, z)
            if levels[action]["control"] != levels[action]["weighted"]:
                raise ValueError("고정 구간의 가중·무가중 수준 경계 불일치")
            encoded_models[action] = export_histogram(learner, action)
            models[action] = copy.deepcopy(weighted[action])
            models[action]["tree"] = lift_thresholds(encoded_models[action], bins[action], action)
            for learner_, raw in [
                (original, parent["tree"]),
                (control, controls[action]),
                (learner, encoded_models[action]),
            ]:
                values = x if learner_ is original else z
                np.testing.assert_allclose(
                    CLASSES[action].from_dict(raw).probabilities(values)[:, 0],
                    learner_.predict_proba(values)[:, 1],
                    atol=1e-12,
                    rtol=0,
                )
    validate_tail_models(models)
    return models, bins, controls, encoded_models, levels


def fixed_bin_admission(candidate, previous, unchanged, pre_limit, budget, *, model_name="v141"):
    if (
        not isinstance(model_name, str)
        or not re.fullmatch("v[0-9]+[a-z]*", model_name)
        or model_name == "v127"
    ):
        raise ValueError("고정 구간 판정의 후보 식별자 오류")
    models = ("v127", model_name)
    result = causal_admission(candidate, previous, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        old, new = previous[action]["quiet"]["tree"], candidate[action]["quiet"]["tree"]
        checks[action + "_tail_weight_log_loss_gain"] = (
            bool(1 - new["log_loss"] / old["log_loss"] > 0.01 + 1e-12) if old["log_loss"] else False
        )
    if set(pre_limit) != set(models):
        raise ValueError("꼬리 가중 관리의 한도 전 대조 누락")
    old, new = pre_limit["v127"], pre_limit[model_name]
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
    if set(focal) != set(models):
        raise ValueError("고정 구간 판정의 예산 후보 식별자 불일치")
    old, new = (focal[model]["budget_matched"] for model in models)
    supported = old is not None and new is not None
    checks["budget_pre_limit_restart_increase_tp_improved"] = bool(
        supported and new["true_positive"] > old["true_positive"]
    )
    checks["budget_pre_limit_restart_increase_fp_preserved"] = bool(
        supported and new["false_positive"] <= old["false_positive"]
    )
    result.pop("causal_distribution_admitted")
    result["fixed_bins_admitted"] = all(checks.values())
    assert len(checks) == 42
    return result
