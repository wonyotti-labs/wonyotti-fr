from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .first_management import BETAS
from .first_state_models import FEATURES, calibrated_binary, fit_binary_offset, linear_probabilities
from .fixed_bin_management import (
    check_level_mapper,
    encode_bins,
    export_histogram,
    fixed_bin_admission,
    lift_thresholds,
    model_digest,
    training_digest,
    validate_bins,
)
from .forward_management import ARMS, WEIGHT_RULE, build_forward_weights
from .forward_management_reference import reference_forward_partition, verify_forward_weights
from .management_capacity import validate_training
from .quiet_management import ACTIONS, CLASSES, quiet_mask
from .tail_weighted_management import SCOPE, validate_tail_rows


def validate_forward_models(models, arm):
    if arm not in ARMS or set(models) != set(ACTIONS):
        raise ValueError("과거 예측 최종 모델의 방식·행동 오류")
    for action, value in models.items():
        if (
            set(value) != {"tree", "linear", "scope", "weight_rule", "arm"}
            or value["scope"] != SCOPE
            or value["weight_rule"] != WEIGHT_RULE
            or value["arm"] != arm
            or value["linear"].get("action") != action
        ):
            raise ValueError("과거 예측 최종 모델의 가중치·방식 연결 오류")
        CLASSES[action].from_dict(value["tree"])
        linear_probabilities(value["linear"], np.zeros((1, len(FEATURES))))


def fit_forward_models(training, parents, bins, auxiliary, weighted):
    validate_tail_rows(training, "training")
    validate_training(training)
    auxiliaries, scores, support = auxiliary
    ledgers, details, overlap = weighted
    if set(parents) != set(ACTIONS) or set(bins) != set(ACTIONS):
        raise ValueError("과거 예측 최종 학습의 원래 모델·구간 누락")
    reference_forward_partition(training, scores, support)
    regenerated = build_forward_weights(training, parents, auxiliaries, scores)
    for arm in ARMS:
        pd.testing.assert_frame_equal(regenerated[0][arm], ledgers[arm], check_exact=True)
    if regenerated[1:] != (details, overlap):
        raise ValueError("과거 예측 최종 학습의 가중치 경계·교집합 변경")
    verify_forward_weights(training, parents, scores, ledgers, details, overlap)
    models = {arm: {} for arm in ARMS}
    encoded = {arm: {} for arm in ARMS}
    controls, levels = {}, {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            validate_bins(bins[action], action)
            quiet = quiet_mask(training, action)
            part = training.loc[quiet]
            x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
            if (
                bins[action]["parent_sha256"] != model_digest(parents[action]["tree"])
                or bins[action]["training_sha256"] != training_digest(training, action)
                or bins[action]["rows"] != len(x)
            ):
                raise ValueError("과거 예측 최종 학습의 고정 구간 출처 오류")
            original = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(x, y)
            if (
                export_histogram(original, action) != parents[action]["tree"]
                or [q.tolist() for q in original._bin_mapper.bin_thresholds_]
                != bins[action]["thresholds"]
            ):
                raise ValueError("과거 예측 최종 학습의 원래 모델·구간 재현 실패")
            z = encode_bins(bins[action], action, x)
            np.testing.assert_array_equal(z, original._bin_mapper.transform(x))
            control = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(z, y)
            controls[action] = export_histogram(control, action)
            levels[action] = {"control": check_level_mapper(control, z)}
            if lift_thresholds(controls[action], bins[action], action) != parents[action]["tree"]:
                raise ValueError("과거 예측 최종 학습의 무가중 동등성 실패")
            scaler = StandardScaler().fit(x)
            for arm in ARMS:
                weights = ledgers[arm].loc[quiet, action + "_weight"].to_numpy(float)
                learner = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(
                    z, y, sample_weight=weights
                )
                levels[action][arm] = check_level_mapper(learner, z)
                if levels[action][arm] != levels[action]["control"]:
                    raise ValueError("과거 예측 최종 학습의 가중 구간 변경")
                encoded[arm][action] = export_histogram(learner, action)
                lifted = lift_thresholds(encoded[arm][action], bins[action], action)
                linear = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(
                    scaler.transform(x), y, sample_weight=weights
                )
                if linear.n_iter_.max() >= 2000:
                    raise ValueError("과거 예측 최종 선형 학습의 수렴 실패")
                raw_linear = {
                    "format": "first_state_linear_v1",
                    "features": FEATURES.copy(),
                    "action": action,
                    "settings": {"C": 0.1, "max_iter": 2000, "random_state": 0},
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": linear.coef_[0].tolist(),
                    "intercept": float(linear.intercept_[0]),
                }
                models[arm][action] = {
                    "tree": lifted,
                    "linear": raw_linear,
                    "scope": SCOPE,
                    "weight_rule": WEIGHT_RULE,
                    "arm": arm,
                }
                np.testing.assert_allclose(
                    CLASSES[action].from_dict(lifted).probabilities(x)[:, 0],
                    learner.predict_proba(z)[:, 1],
                    atol=1e-12,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    linear_probabilities(raw_linear, x),
                    linear.predict_proba(scaler.transform(x))[:, 1],
                    atol=1e-12,
                    rtol=0,
                )
    for arm in ARMS:
        validate_forward_models(models[arm], arm)
    return models, controls, encoded, levels


def calibrate_forward_models(calibration, models, arm):
    validate_tail_rows(calibration, "calibration")
    validate_forward_models(models, arm)
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
        for name, probabilities in scores.items():
            thresholds[action][name], details[action][name] = select_threshold(
                y, probabilities, BETAS[action]
            )
    return offsets, thresholds, details, constants


def forward_final_admission(
    candidate, matched, previous, unchanged, pre_limit, budget, matched_budget
):
    result = fixed_bin_admission(
        candidate, previous, unchanged, pre_limit, budget, model_name="v142forward"
    )
    checks = result["checks"]
    for action in ACTIONS:
        a, b = candidate[action]["quiet"]["tree"], matched[action]["quiet"]["tree"]
        if (a["rows"], a["positive"]) != (b["rows"], b["positive"]) or min(
            b["positive"], b["rows"] - b["positive"]
        ) < 20:
            raise ValueError("과거 예측 판정의 같은 지원 대조 불일치")
        for item in [a, b]:
            if (
                not np.isfinite([item["log_loss"], item["average_precision"]]).all()
                or item["log_loss"] < 0
            ):
                raise ValueError("과거 예측 판정의 대조 지표 오류")
        checks[action + "_matched_log_loss_gain"] = bool(
            (1 - a["log_loss"] / b["log_loss"] if b["log_loss"] else 0) > 0.01 + 1e-12
        )
        checks[action + "_matched_ap_preserved"] = bool(
            a["average_precision"] >= b["average_precision"] - 1e-12
        )
    old_group, new_group = [v["increase__restart__below"] for v in [matched_budget, budget]]
    if (
        old_group["support"] != new_group["support"]
        or old_group["models"]["v127"] != new_group["models"]["v127"]
    ):
        raise ValueError("과거 예측 판정의 공통 예산 지원 불일치")
    if set(old_group["models"]) != {"v127", "v142matched"}:
        raise ValueError("과거 예측 판정의 같은 지원 대조 식별자 오류")
    old, new = (
        old_group["models"]["v142matched"]["budget_matched"],
        new_group["models"]["v142forward"]["budget_matched"],
    )
    supported = old is not None and new is not None
    checks["matched_budget_pre_limit_restart_increase_tp_improved"] = bool(
        supported and new["true_positive"] > old["true_positive"]
    )
    checks["matched_budget_pre_limit_restart_increase_fp_preserved"] = bool(
        supported and new["false_positive"] <= old["false_positive"]
    )
    result.pop("fixed_bins_admitted")
    result["forward_weight_admitted"] = all(checks.values())
    assert len(checks) == 48
    return result
