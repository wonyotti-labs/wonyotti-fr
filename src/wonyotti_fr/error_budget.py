from __future__ import annotations

import numpy as np
import pandas as pd

from .add_count_audit import validate_count_predictions
from .count_region_management import validate_count_region_inputs
from .first_management import BETAS
from .first_state_models import metrics
from .management_rank_audit import validate_scores
from .quiet_management import groups

MODELS = ("v127", "v137")
GROUPS = [
    (a, s, r)
    for a in ["reduce", "increase"]
    for s in ["quiet", "first", "restart"]
    for r in ["all", "below", "capped"]
]
CUTOFF_FIELDS = {
    "status",
    "budget",
    "source_rows",
    "source_negative",
    "threshold",
    "boundary_score",
    "boundary_negative_ties",
    "false_positive",
    "unused_budget",
}


def validate_budget_inputs(frame, predictions):
    validate_count_region_inputs(frame)
    validate_count_predictions(frame, predictions, MODELS)
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
        raise ValueError("오신호 한도 진단의 시각·포지션 오류")


def budget_masks(frame):
    result = {}
    for action in ["reduce", "increase"]:
        states = groups(frame, action)
        for state in ["quiet", "first", "restart"]:
            for region in ["all", "below", "capped"]:
                mask = states[state].copy()
                if region != "all":
                    mask &= (
                        frame.adds.lt(5).to_numpy()
                        if region == "below"
                        else frame.adds.ge(5).to_numpy()
                    )
                result["__".join([action, state, region])] = mask
    return result


def support(frame, action):
    y = frame["y_" + action].to_numpy()
    return {
        "rows": len(y),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "episodes": int(frame.episode_id.nunique()),
        "positive_episodes": int(frame.loc[y == 1, "episode_id"].nunique()),
    }


def choose_budget_cutoff(labels, scores, budget):
    p, y = validate_scores(scores), np.asarray(labels)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or y.dtype.kind not in "bifu"
        or not np.isin(y, [0, 1]).all()
        or type(budget) is not int
        or not 0 <= budget <= int((y == 0).sum())
    ):
        raise ValueError("오신호 한도 진단의 정답·정수 예산 오류")
    negative = p[y == 0]
    result = {
        "status": "supported" if len(y) else "no_calibration_rows",
        "budget": budget,
        "source_rows": len(y),
        "source_negative": len(negative),
        "threshold": None,
        "boundary_score": None,
        "boundary_negative_ties": 0,
        "false_positive": None,
        "unused_budget": None,
    }
    if len(y):
        threshold = 0.0
        if len(negative) > budget:
            q = float(np.sort(negative)[::-1][budget])
            threshold = float(np.nextafter(q, np.inf))
            result.update(boundary_score=q, boundary_negative_ties=int((negative == q).sum()))
        fp = int((negative >= threshold).sum())
        result.update(threshold=threshold, false_positive=fp, unused_budget=budget - fp)
    validate_budget_cutoff(result)
    return result


def validate_budget_cutoff(value):
    if set(value) != CUTOFF_FIELDS:
        raise ValueError("오신호 한도 진단의 문턱 형식 오류")
    integers = ["budget", "source_rows", "source_negative", "boundary_negative_ties"]
    if (
        any(type(value[k]) is not int or value[k] < 0 for k in integers)
        or not value["budget"] <= value["source_negative"] <= value["source_rows"]
    ):
        raise ValueError("오신호 한도 진단의 문턱 지원 오류")
    if value["status"] == "no_calibration_rows":
        if any(value[k] != 0 for k in integers) or any(
            value[k] is not None
            for k in ["threshold", "boundary_score", "false_positive", "unused_budget"]
        ):
            raise ValueError("오신호 한도 진단의 미지원 문턱 오류")
        return
    t, q, fp = value["threshold"], value["boundary_score"], value["false_positive"]
    if (
        value["status"] != "supported"
        or not value["source_rows"]
        or type(t) not in (int, float)
        or not np.isfinite(t)
        or not 0 <= t <= np.nextafter(1.0, np.inf)
        or type(fp) is not int
        or not 0 <= fp <= value["budget"]
        or type(value["unused_budget"]) is not int
        or value["unused_budget"] != value["budget"] - fp
    ):
        raise ValueError("오신호 한도 진단의 문턱 값 오류")
    if value["source_negative"] <= value["budget"]:
        if (
            t != 0
            or q is not None
            or value["boundary_negative_ties"] != 0
            or fp != value["source_negative"]
        ):
            raise ValueError("오신호 한도 진단의 전체 음성 경계 오류")
    elif (
        type(q) not in (int, float)
        or not np.isfinite(q)
        or not 0 <= q <= 1
        or t != np.nextafter(q, np.inf)
        or value["boundary_negative_ties"] < 1
        or not value["budget"] < fp + value["boundary_negative_ties"] <= value["source_negative"]
    ):
        raise ValueError("오신호 한도 진단의 동점·인접 문턱 오류")


def apply_budget_cutoff(scores, cutoff):
    p = validate_scores(scores)
    validate_budget_cutoff(cutoff)
    return None if cutoff["status"] == "no_calibration_rows" else p >= cutoff["threshold"]


def fit_budget_cutoffs(frame, predictions):
    validate_budget_inputs(frame, predictions)
    if not (
        frame.end.ge(pd.Timestamp("2021-01-01", tz="UTC"))
        & frame.end.lt(pd.Timestamp("2021-07-01", tz="UTC"))
    ).all():
        raise ValueError("오신호 한도 문턱의 상반기 보정 범위 오류")
    result = {}
    for key, mask in budget_masks(frame).items():
        action = key.split("__")[0]
        part = frame.loc[mask]
        y = part["y_" + action].to_numpy()
        original = predictions["v127"].loc[mask, action + "_tree_request"].to_numpy()
        budget = int(((y == 0) & original).sum())
        result[key] = {
            "support": support(part, action),
            "budget": budget,
            "models": {
                name: choose_budget_cutoff(
                    y, values.loc[mask, action + "_tree_score"].to_numpy(), budget
                )
                for name in MODELS
                for values in [predictions[name]]
            },
        }
    return result


def validate_cutoff_collection(cutoffs):
    if set(cutoffs) != {"__".join(g) for g in GROUPS}:
        raise ValueError("오신호 한도 진단의 집단 누락")
    for value in cutoffs.values():
        if set(value) != {"support", "budget", "models"} or set(value["models"]) != set(MODELS):
            raise ValueError("오신호 한도 진단의 대조 누락")
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
            raise ValueError("오신호 한도 진단의 저장 지원 오류")
        for record in value["models"].values():
            validate_budget_cutoff(record)
            if (
                record["budget"] != value["budget"]
                or record["source_rows"] != s["rows"]
                or record["source_negative"] != s["negative"]
            ):
                raise ValueError("오신호 한도 진단의 공통 예산 연결 오류")


def with_fpr(value):
    if value is not None:
        negative = value["rows"] - value["positive"]
        value["false_positive_rate"] = value["false_positive"] / negative if negative else None
    return value


def evaluate_budget(frame, predictions, cutoffs):
    validate_budget_inputs(frame, predictions)
    validate_cutoff_collection(cutoffs)
    result = {}
    ledger = frame[["end", "episode_id"]].copy()
    for key, mask in budget_masks(frame).items():
        action = key.split("__")[0]
        part = frame.loc[mask]
        y = part["y_" + action].to_numpy()
        models = {}
        for model in MODELS:
            values = predictions[model]
            p = values.loc[mask, action + "_tree_score"].to_numpy()
            old = values.loc[mask, action + "_tree_request"].to_numpy()
            cutoff = cutoffs[key]["models"][model]
            matched = apply_budget_cutoff(p, cutoff)
            original = with_fpr(metrics(y, p, old, BETAS[action]))
            selected = None if matched is None else with_fpr(metrics(y, p, matched, BETAS[action]))
            overlap = None
            data, missing = np.zeros(len(frame), dtype=bool), np.ones(len(frame), dtype=bool)
            if matched is not None:
                data[mask], missing[mask] = matched, False
                overlap = {
                    name: {
                        "true_positive": int((choice & (y == 1)).sum()),
                        "false_positive": int((choice & (y == 0)).sum()),
                    }
                    for name, choice in [
                        ("intersection", old & matched),
                        ("added", ~old & matched),
                        ("removed", old & ~matched),
                    ]
                }
                if original is not None:
                    assert all(
                        original[k] == selected[k]
                        for k in ["log_loss", "average_precision", "brier"]
                    )
            ledger[model + "__" + key] = pd.arrays.BooleanArray(data, missing)
            models[model] = {
                "status": cutoff["status"],
                "unsupported_rows": len(part) if matched is None else 0,
                "original": original,
                "budget_matched": selected,
                "overlap": overlap,
            }
        result[key] = {"support": support(part, action), "models": models}
    return result, ledger


def focal_retention(frame, predictions, cutoffs, ledger):
    key = "increase__restart__below"
    mask = budget_masks(frame)[key]
    original = (
        mask
        & frame.y_increase.eq(1).to_numpy()
        & predictions["v137"].increase_tree_request.to_numpy()
        & ~predictions["v127"].increase_tree_request.to_numpy()
    )
    supported = cutoffs[key]["models"]["v137"]["status"] == "supported"
    matched = ledger["v137__" + key].fillna(False).to_numpy(bool)
    return {
        "rows": int(mask.sum()),
        "positive": int((mask & frame.y_increase.eq(1).to_numpy()).sum()),
        "original_candidate_new_true_positive": int(original.sum()),
        "budget_supported": supported,
        "retained_new_true_positive": int((original & matched).sum()) if supported else None,
        "removed_new_true_positive": int((original & ~matched).sum()) if supported else None,
    }
