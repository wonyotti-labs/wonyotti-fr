from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .action_model import select_threshold
from .first_management import BETAS
from .first_state_models import metrics, validate_first_rows
from .management_calibration import ManagementOffset
from .management_market_models import market_model_admission
from .minute_management import ACTIONS
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels
from .paired_management_evaluation import choose_thresholds

STATES = ("00", "01", "10", "11")
EXTRA_FEATURES = ["history_joint_" + state for state in STATES]
FEATURES = OrderHistoryModels.features + EXTRA_FEATURES


class JointHistoryBoostModels(OrderHistoryBoostModels):
    features = FEATURES
    format = "joint_history_histogram_v1"


class JointHistoryLinearModels(OrderHistoryModels):
    features = FEATURES
    format = "joint_history_logistic_v1"


def joint_codes(frame):
    flags = frame[["past_increase_exists", "past_reduce_exists"]].to_numpy()
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("이력 조합의 과거 체결 여부 오류")
    return (2 * flags[:, 0] + flags[:, 1]).astype(np.int8)


def attach_joint_history(rows):
    validate_first_rows(rows)
    expanded = {}
    for phase, frame in rows.items():
        if set(EXTRA_FEATURES) & set(frame):
            raise ValueError("이력 조합 입력의 중복 추가")
        codes = joint_codes(frame)
        added = frame.copy()
        for index, name in enumerate(EXTRA_FEATURES):
            added[name] = (codes == index).astype(float)
        expanded[phase] = added
    return expanded


def verify_joint_inputs(native, expanded):
    validate_first_rows(native)
    if set(native) != set(expanded):
        raise ValueError("이력 조합의 분할 누락")
    count = 0
    for phase, frame in native.items():
        part = expanded[phase]
        pd.testing.assert_frame_equal(frame, part.drop(columns=EXTRA_FEATURES), check_exact=True)
        # 비트 수식과 별도로 두 불리언의 네 가지 조건을 전수 대조한다.
        a, b = frame.past_increase_exists.eq(1), frame.past_reduce_exists.eq(1)
        for name, mask in zip(EXTRA_FEATURES, [~a & ~b, ~a & b, a & ~b, a & b], strict=True):
            np.testing.assert_array_equal(part[name], mask.astype(float))
        np.testing.assert_array_equal(part[EXTRA_FEATURES].sum(axis=1), np.ones(len(frame)))
        count += len(frame) * 4
    return count


def fit_joint_models(native, expanded):
    verify_joint_inputs(native, expanded)
    y, cy = (native[p][["y_" + a for a in ACTIONS]].to_numpy(int) for p in ["training", "calibration"])
    models, offsets, support = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, rows, cls in [("native", native, OrderHistoryBoostModels),
                                ("joint", expanded, JointHistoryBoostModels)]:
            model, detail = cls.fit(rows["training"][cls.features].to_numpy(float), y,
                                   rows["calibration"][cls.features].to_numpy(float))
            models[name], support[name] = model, {"training": detail}
        model, _, detail = JointHistoryLinearModels.fit(expanded["training"], expanded["calibration"], "logistic")
        models["linear"], support["linear"] = model, {"training": detail}
        for name, model in models.items():
            frame = native["calibration"] if name == "native" else expanded["calibration"]
            raw = model.probabilities(frame[model.features].to_numpy(float))
            offsets[name], detail = ManagementOffset.fit(raw, cy)
            support[name]["offset"] = detail
    return models, offsets, support


def joint_predictions(native, expanded, models, offsets, constant):
    result = {}
    for name, model in models.items():
        frame = native if name == "native" else expanded
        raw = model.probabilities(frame[model.features].to_numpy(float))
        result[name + "_raw"], result[name + "_calibrated"] = raw, offsets[name].predict(raw)
    result["constant"] = np.tile(constant, (len(native), 1))
    return result


def joint_thresholds(calibration, predictions):
    full, full_details = choose_thresholds(calibration[["y_" + a for a in ACTIONS]].to_numpy(int), predictions)
    first, details = {}, {}
    for name, scores in predictions.items():
        first[name], details[name] = {}, {}
        for index, action in [(1, "reduce"), (2, "increase")]:
            mask = calibration[f"past_{action}_exists"].eq(0).to_numpy()
            boundary, selection = select_threshold(calibration.loc[mask, "y_" + action], scores[mask, index], BETAS[action])
            first[name][action], details[name][action] = boundary, selection
    return {"full": full, "first": first}, {"full": full_details, "first": details}


def joint_requests(frame, scores, full, first):
    p = np.asarray(scores)
    if (p.shape != (len(frame), 3) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
        or set(full) != set(ACTIONS) or set(first) != {"reduce", "increase"}
        or not np.isfinite([*full.values(), *first.values()]).all()
        or any(not 0 <= value <= 1 for value in [*full.values(), *first.values()])):
        raise ValueError("이력 조합의 확률·문턱 오류")
    codes = joint_codes(frame)
    limits = np.tile([full[a] for a in ACTIONS], (len(frame), 1))
    limits[np.isin(codes, [0, 2]), 1] = first["reduce"]
    limits[np.isin(codes, [0, 1]), 2] = first["increase"]
    return p >= limits


def evaluate_joint(frame, predictions, thresholds):
    codes = joint_codes(frame)
    result = {a: {} for a in ACTIONS}
    if set(predictions) != set(thresholds["full"]) or set(predictions) != set(thresholds["first"]):
        raise ValueError("이력 조합 평가의 모형·문턱 누락")
    requests = {name: joint_requests(frame, p, thresholds["full"][name], thresholds["first"][name])
                for name, p in predictions.items()}
    for index, action in enumerate(ACTIONS):
        groups = {"all": np.ones(len(frame), dtype=bool), **{"joint_" + state: codes == j for j, state in enumerate(STATES)}}
        if action != "exit":
            first = frame[f"past_{action}_exists"].eq(0).to_numpy()
            groups.update(first=first, repeat=~first)
        for group, mask in groups.items():
            result[action][group] = {name: metrics(frame.loc[mask, "y_" + action].to_numpy(),
                p[mask, index], requests[name][mask, index], BETAS[action]) for name, p in predictions.items()}
    return result, requests


def joint_admission(evaluation):
    adapted = {a: {name.replace("joint_", "aligned_"): value for name, value in groups["all"].items()}
               for a, groups in evaluation.items()}
    decision = market_model_admission(adapted)
    checks = decision["checks"]
    for action in ACTIONS:
        groups = evaluation[action]
        checks[action + "_all_f_preserved"] = bool(groups["all"]["joint_calibrated"]["f_beta"] >= groups["all"]["native_calibrated"]["f_beta"] - 1e-12)
        if action != "exit":
            first = groups["first"]
            for key in ["recall", "f_beta"]:
                checks[action + "_first_" + key + "_preserved"] = bool(first["joint_calibrated"][key] >= first["native_calibrated"][key] - 1e-12)
    decision.pop("market_model_admitted")
    decision["joint_history_admitted"] = all(checks.values())
    return decision


def joint_feature_usage(model):
    return {binary["action"]: {name: sum(tree["feature"].count(FEATURES.index(name)) for tree in binary["trees"])
            for name in EXTRA_FEATURES} for binary in model.to_dict()["models"]}
