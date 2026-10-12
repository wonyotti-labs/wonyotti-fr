from __future__ import annotations

import numpy as np
import pandas as pd

from .add_count_audit import validate_count_predictions
from .count_region_management import validate_count_region_inputs
from .error_budget import (
    GROUPS,
    apply_budget_cutoff,
    budget_masks,
    choose_budget_cutoff,
    support,
    validate_budget_cutoff,
    with_fpr,
)
from .error_budget_reference import (
    reference_cutoff,
    reference_masks,
    reference_metrics,
    reference_support,
)
from .first_management import BETAS
from .first_state_models import metrics
from .position_boost_verification import compare_statistics


def validate_paired_budget_inputs(frame, predictions):
    names = tuple(predictions)
    if len(names) != 2 or any(not isinstance(n, str) or not n.isalnum() for n in names):
        raise ValueError("공통 예산의 두 모델 식별자 오류")
    validate_count_region_inputs(frame)
    validate_count_predictions(frame, predictions, names)
    if (
        not isinstance(frame.end.dtype, pd.DatetimeTZDtype)
        or str(frame.end.dt.tz) != "UTC"
        or frame.end.isna().any()
        or frame.end.duplicated().any()
        or not frame.end.is_monotonic_increasing
        or not frame.end.eq(frame.end.dt.floor("min")).all()
        or not np.isfinite(frame.episode_id).all()
        or frame.episode_id.le(0).any()
        or frame.episode_id.ge(2**53).any()
        or frame.episode_id.mod(1).ne(0).any()
    ):
        raise ValueError("공통 예산의 시각·포지션 오류")
    return names


def validate_paired_cutoffs(cutoffs, names):
    if set(cutoffs) != {"__".join(g) for g in GROUPS}:
        raise ValueError("공통 예산의 집단 누락")
    for value in cutoffs.values():
        if set(value) != {"support", "budget", "models"} or set(value["models"]) != set(names):
            raise ValueError("공통 예산의 대조 연결 오류")
        s = value["support"]
        if (
            set(s) != {"rows", "positive", "negative", "episodes", "positive_episodes"}
            or any(type(v) is not int or v < 0 for v in s.values())
            or s["positive"] + s["negative"] != s["rows"]
            or not s["positive_episodes"] <= s["episodes"] <= s["rows"]
            or s["positive_episodes"] > s["positive"]
            or type(value["budget"]) is not int
            or not 0 <= value["budget"] <= s["negative"]
        ):
            raise ValueError("공통 예산의 지원 값 오류")
        for cutoff in value["models"].values():
            validate_budget_cutoff(cutoff)
            if (
                cutoff["budget"] != value["budget"]
                or cutoff["source_rows"] != s["rows"]
                or cutoff["source_negative"] != s["negative"]
            ):
                raise ValueError("공통 예산의 문턱·지원 연결 오류")


def fit_paired_budget(frame, predictions, baseline):
    names = validate_paired_budget_inputs(frame, predictions)
    if (
        baseline not in names
        or not frame.end.ge(pd.Timestamp("2021-01-01", tz="UTC")).all()
        or not frame.end.lt(pd.Timestamp("2021-07-01", tz="UTC")).all()
    ):
        raise ValueError("공통 예산의 기준 모델·보정 기간 오류")
    result = {}
    for key, mask in budget_masks(frame).items():
        action = key.split("__")[0]
        y = frame.loc[mask, "y_" + action].to_numpy()
        original = predictions[baseline].loc[mask, action + "_tree_request"].to_numpy()
        b = int(((y == 0) & original).sum())
        result[key] = {
            "support": support(frame.loc[mask], action),
            "budget": b,
            "models": {
                name: choose_budget_cutoff(
                    y, values.loc[mask, action + "_tree_score"].to_numpy(), b
                )
                for name, values in predictions.items()
            },
        }
    validate_paired_cutoffs(result, names)
    return result


def evaluate_paired_budget(frame, predictions, cutoffs):
    names = validate_paired_budget_inputs(frame, predictions)
    validate_paired_cutoffs(cutoffs, names)
    result, ledger = {}, frame[["end", "episode_id"]].copy()
    for key, mask in budget_masks(frame).items():
        action = key.split("__")[0]
        y = frame.loc[mask, "y_" + action].to_numpy()
        entries = {}
        for name, values in predictions.items():
            p = values.loc[mask, action + "_tree_score"].to_numpy()
            old = values.loc[mask, action + "_tree_request"].to_numpy()
            cutoff = cutoffs[key]["models"][name]
            new = apply_budget_cutoff(p, cutoff)
            original = with_fpr(metrics(y, p, old, BETAS[action]))
            selected = None if new is None else with_fpr(metrics(y, p, new, BETAS[action]))
            overlap = None
            data, missing = np.zeros(len(frame), bool), np.ones(len(frame), bool)
            if new is not None:
                data[mask], missing[mask] = new, False
                overlap = {
                    part: {
                        "true_positive": int((r & (y == 1)).sum()),
                        "false_positive": int((r & (y == 0)).sum()),
                    }
                    for part, r in [
                        ("intersection", old & new),
                        ("added", ~old & new),
                        ("removed", old & ~new),
                    ]
                }
                if original is not None:
                    assert all(
                        original[k] == selected[k]
                        for k in ["log_loss", "average_precision", "brier"]
                    )
            entries[name] = {
                "status": cutoff["status"],
                "unsupported_rows": len(y) if new is None else 0,
                "original": original,
                "budget_matched": selected,
                "overlap": overlap,
            }
            ledger[name + "__" + key] = pd.arrays.BooleanArray(data, missing)
        result[key] = {"support": support(frame.loc[mask], action), "models": entries}
    return result, ledger


def verify_paired_cutoffs(frame, predictions, baseline, actual):
    assert frame.end.ge(pd.Timestamp("2021-01-01", tz="UTC")).all()
    assert frame.end.lt(pd.Timestamp("2021-07-01", tz="UTC")).all()
    expected = {}
    for key, mask in reference_masks(frame).items():
        action = key.split("__")[0]
        part = frame.loc[mask]
        y = part["y_" + action].to_numpy(int)
        old = predictions[baseline].loc[mask, action + "_tree_request"].to_numpy()
        b = sum(int(label == 0 and requested) for label, requested in zip(y, old, strict=True))
        expected[key] = {"support": reference_support(part, action), "budget": b, "models": {}}
        for name, values in predictions.items():
            ref = reference_cutoff(y, values.loc[mask, action + "_tree_score"].to_numpy(), b)
            expected[key]["models"][name] = ref
            for field in ["threshold", "boundary_score"]:
                assert actual[key]["models"][name][field] == ref[field]
    compare_statistics(actual, expected)


def verify_paired_budget(frame, predictions, cutoffs, actual, ledger):
    pd.testing.assert_frame_equal(
        ledger[["end", "episode_id"]], frame[["end", "episode_id"]], check_exact=True
    )
    expected, columns = {}, ["end", "episode_id"]
    for key, mask in reference_masks(frame).items():
        action = key.split("__")[0]
        part = frame.loc[mask]
        y = part["y_" + action].to_numpy(int)
        beta = 1.0 if action == "reduce" else 0.5
        expected[key] = {"support": reference_support(part, action), "models": {}}
        for name, values in predictions.items():
            column = name + "__" + key
            columns.append(column)
            stored = ledger[column]
            assert str(stored.dtype) == "boolean"
            p = values.loc[mask, action + "_tree_score"].to_numpy()
            old = values.loc[mask, action + "_tree_request"].to_numpy(bool)
            cut = cutoffs[key]["models"][name]
            supported = cut["threshold"] is not None
            original = reference_metrics(y, p, old, beta)
            selected, overlap = None, None
            assert stored.loc[~mask].isna().all()
            if supported:
                new = np.asarray([float(score) >= cut["threshold"] for score in p], dtype=bool)
                assert stored.loc[mask].notna().all()
                np.testing.assert_array_equal(stored.loc[mask].to_numpy(bool), new)
                selected = reference_metrics(y, p, new, beta)
                overlap = {
                    part: {"true_positive": 0, "false_positive": 0}
                    for part in ["intersection", "added", "removed"]
                }
                for label, a, b in zip(y, old, new, strict=True):
                    part = "intersection" if a and b else "added" if b else "removed" if a else None
                    if part:
                        overlap[part]["true_positive" if label else "false_positive"] += 1
            else:
                assert stored.isna().all()
            expected[key]["models"][name] = {
                "status": cut["status"],
                "unsupported_rows": 0 if supported else len(y),
                "original": original,
                "budget_matched": selected,
                "overlap": overlap,
            }
    assert list(ledger) == columns
    compare_statistics(actual, expected)
