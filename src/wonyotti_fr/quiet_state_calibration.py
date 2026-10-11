from __future__ import annotations

import numpy as np

from .action_model import select_threshold
from .first_management import BETAS
from .first_state_models import calibrated_binary, fit_binary_offset
from .quiet_management import ACTIONS, groups, quiet_admission

STATES = ("first", "restart")


def validate_state_support(frame):
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for state in STATES:
            part = frame.loc[groups(frame, action)[state]]
            labels = part["y_" + action].to_numpy()
            if (len(labels) < 100 or not np.isin(labels, [0, 1]).all()
                or min(labels.sum(), len(labels) - labels.sum()) < 20):
                raise ValueError("상태별 보정의 최초·재시작 지원 부족")
            for model in ["tree", "linear"]:
                for suffix in ["_raw_score", "_score"]:
                    p = part[action + "_" + model + suffix].to_numpy(float)
                    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise ValueError("상태별 보정의 확률 오류")
            result[action][state] = {"rows": len(labels), "positive": int(labels.sum()),
                                     "negative": int(len(labels) - labels.sum())}
    return result


def fit_state_calibration(calibration):
    support = validate_state_support(calibration)
    offsets, thresholds, controls, details = {}, {}, {}, {}
    for action in ACTIONS:
        offsets[action], thresholds[action], controls[action], details[action] = {}, {}, {}, {}
        masks = groups(calibration, action)
        for state in STATES:
            part = calibration.loc[masks[state]]
            y = part["y_" + action].to_numpy(int)
            offsets[action][state], thresholds[action][state] = {}, {}
            controls[action][state], details[action][state] = {}, {}
            for model in ["tree", "linear"]:
                raw = part[action + "_" + model + "_raw_score"].to_numpy()
                offset = fit_binary_offset(raw, y)
                calibrated = calibrated_binary(raw, offset)
                threshold, selected = select_threshold(y, calibrated, BETAS[action])
                control, control_detail = select_threshold(y, part[action + "_" + model + "_score"].to_numpy(), BETAS[action])
                offsets[action][state][model], thresholds[action][state][model] = offset, threshold
                controls[action][state][model] = control
                details[action][state][model] = {"split": selected, "threshold_only": control_detail}
    return offsets, thresholds, controls, details, support


def state_predictions(original, offsets, thresholds, controls):
    candidate, control = original.copy(), original.copy()
    for action in ACTIONS:
        masks = groups(original, action)
        for state in STATES:
            mask = masks[state]
            if not mask.any():
                continue
            for model in ["tree", "linear"]:
                stem = action + "_" + model
                scores = calibrated_binary(original.loc[mask, stem + "_raw_score"].to_numpy(), offsets[action][state][model])
                candidate.loc[mask, stem + "_score"] = scores
                candidate.loc[mask, stem + "_request"] = scores >= thresholds[action][state][model]
                control.loc[mask, stem + "_request"] = original.loc[mask, stem + "_score"].to_numpy() >= controls[action][state][model]
    return candidate, control


def state_means(original, candidate):
    result = {}
    for action in ACTIONS:
        result[action] = {}
        masks = groups(original, action)
        for state in STATES:
            mask = masks[state]
            rows = int(mask.sum())
            result[action][state] = {"rows": rows,
                "positive": int(original.loc[mask, "y_" + action].sum()),
                "frequency": float(original.loc[mask, "y_" + action].mean()) if rows else None,
                "models": {model: {
                    "raw": float(original.loc[mask, action + "_" + model + "_raw_score"].mean()) if rows else None,
                    "shared": float(original.loc[mask, action + "_" + model + "_score"].mean()) if rows else None,
                    "split": float(candidate.loc[mask, action + "_" + model + "_score"].mean()) if rows else None,
                } for model in ["tree", "linear"]}}
    return result


def state_admission(candidate, original, unchanged):
    result = quiet_admission(candidate, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        current, previous = candidate[action]["quiet"]["tree"], original[action]["quiet"]["tree"]
        checks[action + "_shared_log_loss_preserved"] = bool(current["log_loss"] <= previous["log_loss"] + 1e-12)
        checks[action + "_shared_ap_preserved"] = bool(current["average_precision"] >= previous["average_precision"] - 1e-12)
    return {"state_calibration_admitted": all(checks.values()), "checks": checks,
        "profitability_accepted": False, "trading_returns_evaluated": False,
        "all_source_periods_already_observed": True}
