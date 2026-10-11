from __future__ import annotations

import numpy as np
import pandas as pd

from .paired_management_reference import manual_threshold
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups, verify_quiet_admission


def verify_state_calibration(calibration, offsets, thresholds, controls, details, support):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        masks = independent_groups(calibration, action)
        for state in ["first", "restart"]:
            part = calibration.loc[masks[state]]
            y = part["y_" + action].to_numpy(int)
            assert support[action][state] == {"rows": len(y), "positive": int(sum(y)), "negative": int(sum(y == 0))}
            for model in ["tree", "linear"]:
                raw = np.clip(part[action + "_" + model + "_raw_score"].to_numpy(), np.finfo(float).eps, 1 - np.finfo(float).eps)
                logit = np.log(raw) - np.log1p(-raw)
                low, high = -64., 64.
                for _ in range(100):
                    middle = (low + high) / 2
                    p = 1 / (1 + np.exp(-np.clip(logit + middle, -700, 700)))
                    if p.mean() < y.mean():
                        low = middle
                    else:
                        high = middle
                assert abs(offsets[action][state][model] - (low + high) / 2) < 1e-12
                p = 1 / (1 + np.exp(-np.clip(logit + offsets[action][state][model], -700, 700)))
                assert abs(p.mean() - y.mean()) < 1e-12
                for name, scores, threshold in [
                    ("split", p, thresholds[action][state][model]),
                    ("threshold_only", part[action + "_" + model + "_score"].to_numpy(), controls[action][state][model]),
                ]:
                    selected, evidence = manual_threshold(y, scores, beta)
                    assert selected == threshold
                    compare_statistics(details[action][state][model][name], evidence)


def verify_state_predictions(original, candidate, control, offsets, thresholds, controls):
    changed = {a + "_" + m + "_" + suffix for a in ["reduce", "increase"]
               for m in ["tree", "linear"] for suffix in ["score", "request"]}
    preserved = [k for k in original if k not in changed]
    pd.testing.assert_frame_equal(original[preserved], candidate[preserved], check_exact=True)
    pd.testing.assert_frame_equal(original[preserved], control[preserved], check_exact=True)
    for action in ["reduce", "increase"]:
        masks = independent_groups(original, action)
        for model in ["tree", "linear"]:
            stem = action + "_" + model
            np.testing.assert_array_equal(control[stem + "_score"], original[stem + "_score"])
            for suffix in ["score", "request"]:
                for frame in [candidate, control]:
                    np.testing.assert_array_equal(frame.loc[masks["recent"], stem + "_" + suffix], original.loc[masks["recent"], stem + "_" + suffix])
            for state in ["first", "restart"]:
                mask = masks[state]
                q = np.clip(original.loc[mask, stem + "_raw_score"].to_numpy(), np.finfo(float).eps, 1 - np.finfo(float).eps)
                expected = 1 / (1 + np.exp(-np.clip(np.log(q) - np.log1p(-q) + offsets[action][state][model], -700, 700)))
                np.testing.assert_array_equal(candidate.loc[mask, stem + "_score"], expected)
                np.testing.assert_array_equal(candidate.loc[mask, stem + "_request"], expected >= thresholds[action][state][model])
                np.testing.assert_array_equal(control.loc[mask, stem + "_request"], original.loc[mask, stem + "_score"].to_numpy() >= controls[action][state][model])
            # 상태 내 단조 변환으로 요청이 달라지면 수치 경계를 조사하기 전 중단한다.
            np.testing.assert_array_equal(candidate[stem + "_request"], control[stem + "_request"])
    return {"exit": True, "recent_reduce": True, "recent_increase": True}


def verify_state_means(original, candidate, means):
    for action in ["reduce", "increase"]:
        for state in ["first", "restart"]:
            mask = independent_groups(original, action)[state]
            part = means[action][state]
            y = original.loc[mask, "y_" + action].to_numpy()
            assert part["rows"] == len(y) and part["positive"] == sum(y)
            expected = {"rows": len(y), "positive": int(sum(y)),
                        "frequency": float(sum(y) / len(y)) if len(y) else None, "models": {}}
            for model in ["tree", "linear"]:
                expected["models"][model] = {}
                for label, source, suffix in [("raw", original, "_raw_score"), ("shared", original, "_score"), ("split", candidate, "_score")]:
                    p = source.loc[mask, action + "_" + model + suffix].to_numpy()
                    expected["models"][model][label] = float(np.sum(p) / len(p)) if len(p) else None
            compare_statistics(part, expected)


def verify_state_admission(candidate, original, unchanged, decision):
    previous_keys = {a + "_shared_" + k for a in ["reduce", "increase"] for k in ["log_loss_preserved", "ap_preserved"]}
    base = {k: v for k, v in decision["checks"].items() if k not in previous_keys}
    verify_quiet_admission(candidate, unchanged, {"checks": base, "quiet_models_admitted": all(base.values()),
                            "profitability_accepted": False, "trading_returns_evaluated": False})
    for action in ["reduce", "increase"]:
        a, b = candidate[action]["quiet"]["tree"], original[action]["quiet"]["tree"]
        assert decision["checks"][action + "_shared_log_loss_preserved"] == bool(a["log_loss"] <= b["log_loss"] + 1e-12)
        assert decision["checks"][action + "_shared_ap_preserved"] == bool(a["average_precision"] >= b["average_precision"] - 1e-12)
    assert len(decision["checks"]) == 31
    assert decision["state_calibration_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
