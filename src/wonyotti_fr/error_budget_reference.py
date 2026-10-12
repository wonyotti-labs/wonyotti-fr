from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .add_count_reference import requested_metrics
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups


def reference_masks(frame):
    result = {}
    for action in ["reduce", "increase"]:
        states = independent_groups(frame, action)
        counts = np.asarray(frame.adds)
        for state in ["quiet", "first", "restart"]:
            for region in ["all", "below", "capped"]:
                count_mask = (
                    np.ones(len(frame), dtype=bool)
                    if region == "all"
                    else (counts < 5 if region == "below" else counts >= 5)
                )
                result["__".join([action, state, region])] = states[state] & count_mask
    return result


def reference_support(frame, action):
    labels = frame["y_" + action].to_numpy()
    positives = labels == 1
    return {
        "rows": len(frame),
        "positive": int(np.count_nonzero(positives)),
        "negative": int(np.count_nonzero(~positives)),
        "episodes": len(set(frame.episode_id)),
        "positive_episodes": len(set(frame.loc[positives, "episode_id"])),
    }


def reference_cutoff(y, p, budget):
    negatives = [float(score) for label, score in zip(y, p, strict=True) if label == 0]
    result = {
        "status": "supported" if len(y) else "no_calibration_rows",
        "budget": budget,
        "source_rows": len(y),
        "source_negative": len(negatives),
        "threshold": None,
        "boundary_score": None,
        "boundary_negative_ties": 0,
        "false_positive": None,
        "unused_budget": None,
    }
    if not len(y):
        return result
    threshold = 0.0
    frequencies = Counter(negatives)
    if len(negatives) > budget:
        running = 0
        for q in sorted(frequencies, reverse=True):
            running += frequencies[q]
            if running > budget:
                threshold = math.nextafter(q, math.inf)
                result.update(boundary_score=q, boundary_negative_ties=frequencies[q])
                break
    fp = sum(score >= threshold for score in negatives)
    result.update(threshold=threshold, false_positive=fp, unused_budget=budget - fp)
    assert fp <= budget
    if threshold > 0:
        assert sum(score >= math.nextafter(threshold, -math.inf) for score in negatives) > budget
    return result


def verify_budget_cutoffs(frame, predictions, actual):
    assert frame.end.ge(pd.Timestamp("2021-01-01", tz="UTC")).all()
    assert frame.end.lt(pd.Timestamp("2021-07-01", tz="UTC")).all()
    expected = {}
    for key, mask in reference_masks(frame).items():
        action = key.split("__")[0]
        part = frame.iloc[np.flatnonzero(mask)]
        y = part["y_" + action].to_numpy(int)
        r = predictions["v127"].loc[mask, action + "_tree_request"].to_numpy()
        budget = sum(int(label == 0 and requested) for label, requested in zip(y, r, strict=True))
        expected[key] = {"support": reference_support(part, action), "budget": budget, "models": {}}
        for model in ["v127", "v137"]:
            p = predictions[model].loc[mask, action + "_tree_score"].to_numpy()
            expected[key]["models"][model] = reference_cutoff(y, p, budget)
            # 인접한 부동소수점 문턱은 통계 허용 오차로 같다고 처리하지 않는다.
            assert (
                actual[key]["models"][model]["threshold"]
                == expected[key]["models"][model]["threshold"]
            )
            assert (
                actual[key]["models"][model]["boundary_score"]
                == expected[key]["models"][model]["boundary_score"]
            )
    compare_statistics(actual, expected)


def reference_metrics(y, p, request, beta):
    result = requested_metrics(y, p, request, beta)
    if result is not None:
        negative = int(np.count_nonzero(y == 0))
        result["false_positive_rate"] = result["false_positive"] / negative if negative else None
    return result


def verify_budget_evaluation(frame, predictions, cutoffs, actual, ledger):
    pd.testing.assert_frame_equal(
        ledger[["end", "episode_id"]], frame[["end", "episode_id"]], check_exact=True
    )
    expected = {}
    expected_columns = ["end", "episode_id"]
    for key, mask in reference_masks(frame).items():
        action = key.split("__")[0]
        part = frame.iloc[np.flatnonzero(mask)]
        y = part["y_" + action].to_numpy(int)
        beta = 1.0 if action == "reduce" else 0.5
        expected[key] = {"support": reference_support(part, action), "models": {}}
        for model in ["v127", "v137"]:
            column = model + "__" + key
            expected_columns.append(column)
            stored = ledger[column]
            assert str(stored.dtype) == "boolean"
            p = predictions[model].loc[mask, action + "_tree_score"].to_numpy()
            old = predictions[model].loc[mask, action + "_tree_request"].to_numpy(bool)
            cutoff = cutoffs[key]["models"][model]
            original = reference_metrics(y, p, old, beta)
            supported = cutoff["threshold"] is not None
            selected, overlap = None, None
            missing = ~mask if supported else np.ones(len(frame), dtype=bool)
            np.testing.assert_array_equal(stored.isna().to_numpy(), missing)
            if supported:
                request = np.fromiter(
                    (score >= cutoff["threshold"] for score in p), dtype=bool, count=len(p)
                )
                np.testing.assert_array_equal(stored.loc[mask].to_numpy(bool), request)
                selected = reference_metrics(y, p, request, beta)
                overlap = {
                    k: {"true_positive": 0, "false_positive": 0}
                    for k in ["intersection", "added", "removed"]
                }
                for label, before, after in zip(y, old, request, strict=True):
                    name = (
                        "intersection"
                        if before and after
                        else "added"
                        if after
                        else "removed"
                        if before
                        else None
                    )
                    if name:
                        overlap[name]["true_positive" if label else "false_positive"] += 1
                if original is not None:
                    for metric in ["log_loss", "average_precision", "brier"]:
                        assert (
                            actual[key]["models"][model]["original"][metric]
                            == actual[key]["models"][model]["budget_matched"][metric]
                        )
            expected[key]["models"][model] = {
                "status": cutoff["status"],
                "unsupported_rows": 0 if supported else len(part),
                "original": original,
                "budget_matched": selected,
                "overlap": overlap,
            }
    assert list(ledger.columns) == expected_columns
    compare_statistics(actual, expected)


def verify_focal_retention(frame, predictions, cutoffs, ledger, actual):
    key = "increase__restart__below"
    mask = reference_masks(frame)[key]
    y = frame.y_increase.to_numpy(int)
    before = predictions["v127"].increase_tree_request.to_numpy(bool)
    candidate = predictions["v137"].increase_tree_request.to_numpy(bool)
    new_tp = [i for i in np.flatnonzero(mask) if y[i] == 1 and candidate[i] and not before[i]]
    supported = cutoffs[key]["models"]["v137"]["threshold"] is not None
    retained = sum(bool(ledger.iloc[i]["v137__" + key]) for i in new_tp) if supported else None
    expected = {
        "rows": int(mask.sum()),
        "positive": int(sum(y[mask])),
        "original_candidate_new_true_positive": len(new_tp),
        "budget_supported": supported,
        "retained_new_true_positive": retained,
        "removed_new_true_positive": len(new_tp) - retained if supported else None,
    }
    compare_statistics(actual, expected)
