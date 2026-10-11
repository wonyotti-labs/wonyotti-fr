from __future__ import annotations

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .causal_score_distribution import MIN_HISTORY, map_past_scores, scores_array, time_array
from .first_management import BETAS
from .first_state_models import calibrated_binary, fit_binary_offset
from .quiet_management import ACTIONS, groups
from .quiet_state_calibration import STATES, state_admission

PHASES = ("training", "calibration", "diagnosis")
MODELS = ("tree", "linear")


def reference_distributions(training):
    time_array(training.end)
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for state in STATES:
            part = training.loc[groups(training, action)[state]]
            if len(part) < MIN_HISTORY:
                raise ValueError("과거 관리 보정의 학습 기준 분포 부족")
            result[action][state] = {model: np.sort(scores_array(part[action + "_" + model + "_raw_score"]))
                                     for model in MODELS}
    return result


def map_management_history(originals, references):
    if tuple(originals) != PHASES:
        raise ValueError("과거 관리 보정의 기간 순서 오류")
    combined = pd.concat(originals.values(), ignore_index=True)
    time_array(combined.end)
    ledger = combined[["end", "episode_id"]].copy()
    for action in ACTIONS:
        for model in MODELS:
            stem = action + "_" + model + "_"
            for field in ["count", "less", "equal"]:
                ledger[stem + field] = np.full(len(combined), -1, dtype=np.int64)
            ledger[stem + "mapped"] = np.full(len(combined), np.nan)
            for state in STATES:
                mask = groups(combined, action)[state]
                part = combined.loc[mask]
                mapped = map_past_scores(part.end, part[stem + "raw_score"], references[action][state][model])
                for field in ["count", "less", "equal", "mapped"]:
                    ledger.loc[mask, stem + field] = mapped[field].to_numpy()
        np.testing.assert_array_equal(ledger[action + "_tree_count"], ledger[action + "_linear_count"])
    result, start = {}, 0
    for phase, frame in originals.items():
        part = ledger.iloc[start:start + len(frame)].copy()
        part.index = frame.index
        result[phase] = part
        start += len(frame)
    return result


def supported(frame, ledger, action, state):
    pd.testing.assert_frame_equal(frame[["end", "episode_id"]], ledger[["end", "episode_id"]], check_exact=True)
    return groups(frame, action)[state] & ledger[action + "_tree_count"].ge(MIN_HISTORY).to_numpy()


def mapping_support(frame, ledger):
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for state in STATES:
            mask = groups(frame, action)[state]
            usable = supported(frame, ledger, action, state)
            result[action][state] = {}
            for name, selected in [("supported", usable), ("fallback", mask & ~usable)]:
                y = frame.loc[selected, "y_" + action].to_numpy()
                result[action][state][name] = {"rows": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum())}
    return result


def fit_causal_calibration(calibration, ledger):
    offsets, thresholds, details, constants = {}, {}, {}, {}
    for action in ACTIONS:
        for item in [offsets, thresholds, details, constants]:
            item[action] = {}
        for state in STATES:
            mask = supported(calibration, ledger, action, state)
            y = calibration.loc[mask, "y_" + action].to_numpy()
            if len(y) < 100 or not np.isin(y, [0, 1]).all() or min(y.sum(), len(y) - y.sum()) < 20:
                raise ValueError("과거 관리 보정의 상반기 지원 정답 부족")
            offsets[action][state], thresholds[action][state], details[action][state] = {}, {}, {}
            probabilities = {}
            for model in MODELS:
                raw = ledger.loc[mask, action + "_" + model + "_mapped"].to_numpy()
                offset = fit_binary_offset(raw, y)
                offsets[action][state][model] = offset
                probabilities[model + "_raw"] = raw
                probabilities[model] = calibrated_binary(raw, offset)
            constants[action][state] = float(y.mean())
            probabilities["constant"] = np.full(len(y), constants[action][state])
            for name, scores in probabilities.items():
                threshold, selected = select_threshold(y, scores, BETAS[action])
                thresholds[action][state][name], details[action][state][name] = threshold, selected
    return offsets, thresholds, details, constants


def causal_predictions(original, ledger, offsets, thresholds, constants):
    candidate = original.copy()
    for action in ACTIONS:
        for state in STATES:
            mask = supported(original, ledger, action, state)
            if not mask.any():
                continue
            probabilities = {}
            for model in MODELS:
                raw = scores_array(ledger.loc[mask, action + "_" + model + "_mapped"])
                probabilities[model + "_raw"] = raw
                probabilities[model] = calibrated_binary(raw, offsets[action][state][model])
            probabilities["constant"] = np.full(mask.sum(), constants[action][state])
            for name, values in probabilities.items():
                candidate.loc[mask, action + "_" + name + "_score"] = values
                candidate.loc[mask, action + "_" + name + "_request"] = values >= thresholds[action][state][name]
    return candidate


def causal_admission(candidate, original, unchanged):
    result = state_admission(candidate, original, unchanged)
    checks = result["checks"]
    for action in ACTIONS:
        checks[action + "_quiet_precision_preserved"] = bool(candidate[action]["quiet"]["tree"]["precision"] >=
                                                               candidate[action]["quiet"]["baseline"]["precision"] - 1e-12)
        checks[action + "_restart_recall_improved"] = bool(candidate[action]["restart"]["tree"]["recall"] >
                                                            original[action]["restart"]["tree"]["recall"] + 1e-12)
    result.pop("state_calibration_admitted")
    result["causal_distribution_admitted"] = all(checks.values())
    return result
