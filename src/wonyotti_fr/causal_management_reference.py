from __future__ import annotations

import numpy as np
import pandas as pd

from .causal_score_distribution import verify_past_mapping
from .paired_management_reference import manual_threshold
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups
from .quiet_state_reference import verify_state_admission


def verify_management_mapping(originals, references, ledgers):
    whole = pd.concat(originals.values(), ignore_index=True)
    ledger = pd.concat(ledgers.values(), ignore_index=True)
    expected_columns = {"end", "episode_id", *[a + "_" + m + "_" + field
        for a in ["reduce", "increase"] for m in ["tree", "linear"] for field in ["count", "less", "equal", "mapped"]]}
    assert set(ledger) == expected_columns
    pd.testing.assert_frame_equal(whole[["end", "episode_id"]], ledger[["end", "episode_id"]], check_exact=True)
    result = {}
    for action in ["reduce", "increase"]:
        masks = independent_groups(whole, action)
        for model in ["tree", "linear"]:
            stem = action + "_" + model + "_"
            assert ledger.loc[masks["recent"], stem + "mapped"].isna().all()
            for name in ["count", "less", "equal"]:
                assert ledger[stem + name].dtype == np.dtype(np.int64)
                assert ledger.loc[masks["recent"], stem + name].eq(-1).all()
            for state in ["first", "restart"]:
                train = originals["training"]
                pool = np.sort(train.loc[independent_groups(train, action)[state], stem + "raw_score"].to_numpy(float))
                np.testing.assert_array_equal(references[action][state][model], pool)
                subset = ledger.loc[masks[state], [stem + key for key in ["count", "less", "equal", "mapped"]]].copy()
                subset.columns = ["count", "less", "equal", "mapped"]
                part = whole.loc[masks[state]]
                result[action + "/" + state + "/" + model] = verify_past_mapping(part.end, part[stem + "raw_score"], pool, subset)
        np.testing.assert_array_equal(ledger[action + "_tree_count"], ledger[action + "_linear_count"])
    return result


def verify_mapping_support(frame, ledger, support):
    for action in ["reduce", "increase"]:
        for state in ["first", "restart"]:
            mask = independent_groups(frame, action)[state]
            active = ledger[action + "_tree_count"].to_numpy() >= 1000
            for name, selected in [("supported", mask & active), ("fallback", mask & ~active)]:
                y = frame.loc[selected, "y_" + action].to_numpy()
                assert support[action][state][name] == {"rows": len(y), "positive": int(sum(y)), "negative": int(sum(y == 0))}


def verify_causal_calibration(frame, ledger, offsets, thresholds, details, constants):
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        for state in ["first", "restart"]:
            mask = independent_groups(frame, action)[state] & ledger[action + "_tree_count"].ge(1000).to_numpy()
            y = frame.loc[mask, "y_" + action].to_numpy(int)
            assert len(y) >= 100 and min(sum(y), sum(y == 0)) >= 20
            assert constants[action][state] == y.mean()
            scores = {"constant": np.full(len(y), y.mean())}
            for model in ["tree", "linear"]:
                raw = ledger.loc[mask, action + "_" + model + "_mapped"].to_numpy(float)
                scores[model + "_raw"] = raw
                q = np.clip(raw, np.finfo(float).eps, 1 - np.finfo(float).eps)
                logit = np.log(q) - np.log1p(-q)
                low, high = -64., 64.
                for _ in range(100):
                    midpoint = (low + high) / 2
                    p = 1 / (1 + np.exp(-np.clip(logit + midpoint, -700, 700)))
                    if p.mean() < y.mean():
                        low = midpoint
                    else:
                        high = midpoint
                assert abs(offsets[action][state][model] - (low + high) / 2) < 1e-12
                scores[model] = 1 / (1 + np.exp(-np.clip(logit + offsets[action][state][model], -700, 700)))
                assert abs(scores[model].mean() - y.mean()) < 1e-12
            for name, values in scores.items():
                threshold, detail = manual_threshold(y, values, beta)
                assert thresholds[action][state][name] == threshold
                compare_statistics(details[action][state][name], detail)


def verify_causal_predictions(original, ledger, saved, offsets, thresholds, constants):
    changed = {a + "_" + m + "_" + suffix for a in ["reduce", "increase"]
               for m in ["tree_raw", "linear_raw", "tree", "linear", "constant"] for suffix in ["score", "request"]}
    preserved = [key for key in original if key not in changed]
    pd.testing.assert_frame_equal(saved[preserved], original[preserved], check_exact=True)
    for action in ["reduce", "increase"]:
        masks = independent_groups(original, action)
        eligible = ledger[action + "_tree_count"].ge(1000).to_numpy()
        for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
            for suffix in ["score", "request"]:
                column = action + "_" + name + "_" + suffix
                pd.testing.assert_series_equal(saved.loc[~eligible, column], original.loc[~eligible, column], check_exact=True)
        for state in ["first", "restart"]:
            mask = masks[state] & eligible
            for model in ["tree", "linear"]:
                raw = ledger.loc[mask, action + "_" + model + "_mapped"].to_numpy()
                np.testing.assert_array_equal(saved.loc[mask, action + "_" + model + "_raw_score"], raw)
                q = np.clip(raw, np.finfo(float).eps, 1 - np.finfo(float).eps)
                expected = 1 / (1 + np.exp(-np.clip(np.log(q) - np.log1p(-q) + offsets[action][state][model], -700, 700)))
                np.testing.assert_array_equal(saved.loc[mask, action + "_" + model + "_score"], expected)
            np.testing.assert_array_equal(saved.loc[mask, action + "_constant_score"], np.full(mask.sum(), constants[action][state]))
            for name in ["tree_raw", "linear_raw", "tree", "linear", "constant"]:
                stem = action + "_" + name
                np.testing.assert_array_equal(saved.loc[mask, stem + "_request"], saved.loc[mask, stem + "_score"].to_numpy() >= thresholds[action][state][name])
    return {"exit": True, "recent_reduce": True, "recent_increase": True}


def verify_causal_admission(candidate, original, unchanged, decision):
    extra = {a + suffix for a in ["reduce", "increase"] for suffix in ["_quiet_precision_preserved", "_restart_recall_improved"]}
    base = {k: v for k, v in decision["checks"].items() if k not in extra}
    verify_state_admission(candidate, original, unchanged, {"checks": base, "state_calibration_admitted": all(base.values()),
        "profitability_accepted": False, "trading_returns_evaluated": False})
    for action in ["reduce", "increase"]:
        part = candidate[action]["quiet"]
        assert decision["checks"][action + "_quiet_precision_preserved"] == bool(part["tree"]["precision"] >= part["baseline"]["precision"] - 1e-12)
        assert decision["checks"][action + "_restart_recall_improved"] == bool(candidate[action]["restart"]["tree"]["recall"] > original[action]["restart"]["tree"]["recall"] + 1e-12)
    assert len(decision["checks"]) == 35
    assert decision["causal_distribution_admitted"] == all(decision["checks"].values())
    assert decision["profitability_accepted"] is False and decision["trading_returns_evaluated"] is False
