from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .causal_management_reference import (
    verify_causal_admission,
    verify_causal_calibration,
    verify_causal_predictions,
)
from .quiet_management_reference import independent_groups


def reference_inputs(frame, ledger, action, state, source, kind):
    mask = independent_groups(frame, action)[state] & ledger[action + "_tree_count"].ge(1000).to_numpy()
    columns = [frame.loc[mask, action + "_" + source + "_raw_score"].to_numpy()]
    if kind == "joint":
        columns.append(ledger.loc[mask, action + "_" + source + "_mapped"].to_numpy())
    p = np.clip(np.stack(columns, axis=1), np.finfo(float).eps, 1 - np.finfo(float).eps)
    return mask, np.log(p) - np.log1p(-p)


def verify_fusion_refits(calibration, ledger, fitted):
    count = 0
    with threadpool_limits(limits=1):
        for kind in ["absolute", "joint"]:
            for action in ["reduce", "increase"]:
                for state in ["first", "restart"]:
                    for source in ["tree", "linear"]:
                        mask, x = reference_inputs(calibration, ledger, action, state, source, kind)
                        y = calibration.loc[mask, "y_" + action].to_numpy(int)
                        scaler = StandardScaler().fit(x)
                        learner = LogisticRegression(C=.1, solver="lbfgs", fit_intercept=True,
                            max_iter=2000, tol=1e-8, random_state=0, class_weight=None).fit(scaler.transform(x), y)
                        assert learner.n_iter_.max() < 2000
                        raw = fitted[kind]["models"][action][state][source]
                        assert raw["action"] == action and raw["state"] == state
                        assert raw["source"] == source and raw["kind"] == kind
                        assert raw["features"] == (["absolute_logit", "past_quantile_logit"] if kind == "joint" else ["absolute_logit"])
                        for name, values in [("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", learner.coef_[0])]:
                            np.testing.assert_array_equal(raw[name], values)
                        assert raw["intercept"] == float(learner.intercept_[0])
                        count += 1
    assert count == 16
    return {"independent_calibrator_refits": count, "same_calibration_support": True}


def verify_fusion_predictions(original, ledger, saved, fitted, kind, *, calibration=False):
    parameters, transformed = fitted[kind], ledger.copy()
    for action in ["reduce", "increase"]:
        for state in ["first", "restart"]:
            for source in ["tree", "linear"]:
                mask, x = reference_inputs(original, ledger, action, state, source, kind)
                raw = parameters["models"][action][state][source]
                z = np.sum((x - raw["mean"]) / raw["scale"] * raw["coef"], axis=1) + raw["intercept"]
                expected = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
                scores = saved.loc[mask, action + "_" + source + "_raw_score"].to_numpy()
                np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
                transformed.loc[mask, action + "_" + source + "_mapped"] = scores
    preserved = verify_causal_predictions(original, transformed, saved, parameters["offsets"],
                                            parameters["thresholds"], parameters["constants"])
    if calibration:
        verify_causal_calibration(original, transformed, parameters["offsets"], parameters["thresholds"],
                                  parameters["details"], parameters["constants"])
    return preserved


def verify_fusion_admission(candidate, absolute, original, unchanged, decision):
    extras = {a + suffix for a in ["reduce", "increase"] for suffix in ["_absolute_log_loss_gain", "_absolute_ap_preserved"]}
    base = {k: v for k, v in decision["checks"].items() if k not in extras}
    verify_causal_admission(candidate, original, unchanged, {"checks": base, "causal_distribution_admitted": all(base.values()),
        "profitability_accepted": False, "trading_returns_evaluated": False})
    for action in ["reduce", "increase"]:
        a, b = candidate[action]["quiet"]["tree"], absolute[action]["quiet"]["tree"]
        gain = 1 - a["log_loss"] / b["log_loss"] if b["log_loss"] else 0.
        assert decision["checks"][action + "_absolute_log_loss_gain"] == bool(gain > .01 + 1e-12)
        assert decision["checks"][action + "_absolute_ap_preserved"] == bool(a["average_precision"] >= b["average_precision"] - 1e-12)
    assert len(decision["checks"]) == 39
    assert decision["score_fusion_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
