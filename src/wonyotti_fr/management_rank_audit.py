from __future__ import annotations

import math

import numpy as np
import pandas as pd

TAILS = {"0.1%": .001, "0.5%": .005, "1%": .01, "5%": .05}
QUANTILES = {"min": 0., "q10": .1, "q25": .25, "median": .5,
             "q75": .75, "q90": .9, "q99": .99, "max": 1.}
COLUMNS = ["end", "episode_id", "label", "score", "request"]


def validate_scores(values):
    x = np.asarray(values)
    if (x.ndim != 1 or x.dtype.kind not in "fiu" or not np.isfinite(x).all()
        or ((x < 0) | (x > 1)).any()):
        raise ValueError("점수 진단의 확률 차원·범위 오류")
    return x.astype(float)


def validate_frame(frame, threshold):
    if (set(frame) != set(COLUMNS) or isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("점수 진단의 열·문턱 오류")
    scores = validate_scores(frame.score)
    if (not isinstance(frame.end.dtype, pd.DatetimeTZDtype) or str(frame.end.dt.tz) != "UTC"
        or frame.end.isna().any() or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or not frame.end.eq(frame.end.dt.floor("min")).all()):
        raise ValueError("점수 진단의 시각 오류")
    ids = frame.episode_id.to_numpy()
    if (ids.dtype.kind not in "fiu" or not np.isfinite(ids).all()
        or (ids < 0).any() or (ids >= 2**53).any() or (ids != np.floor(ids)).any()
        or not np.isin(frame.label, [0, 1]).all() or frame.label.dtype.kind not in "bifu"
        or frame.request.dtype != np.dtype(bool)):
        raise ValueError("점수 진단의 포지션·정답·요청 오류")
    if not np.array_equal(frame.request.to_numpy(), scores >= threshold):
        raise ValueError("점수 진단의 기존 문턱·요청 불일치")


def tail_cutoffs(scores):
    x = validate_scores(scores)
    if not len(x):
        raise ValueError("점수 진단의 절단점 표본 부족")
    ordered = np.sort(x)[::-1]
    result = {}
    for name, fraction in TAILS.items():
        budget = math.ceil(len(x) * fraction)
        threshold = float(ordered[budget - 1])
        selected = int((x >= threshold).sum())
        result[name] = {"cutoff": threshold, "source_rows": len(x), "nominal_rows": budget,
                        "selected_rows": selected, "tie_excess": selected - budget}
    return result


def validate_cutoffs(cutoffs):
    if set(cutoffs) != set(TAILS):
        raise ValueError("점수 진단의 절단점 종류 오류")
    last, source_rows = 1., None
    for key, fraction in TAILS.items():
        item = cutoffs[key]
        if (set(item) != {"cutoff", "source_rows", "nominal_rows", "selected_rows", "tie_excess"}
            or type(item["cutoff"]) not in (int, float) or not np.isfinite(item["cutoff"])
            or not 0 <= item["cutoff"] <= last
            or any(type(item[k]) is not int for k in item if k != "cutoff")
            or item["source_rows"] < 1
            or item["nominal_rows"] != math.ceil(item["source_rows"] * fraction)
            or not item["nominal_rows"] <= item["selected_rows"] <= item["source_rows"]
            or item["tie_excess"] != item["selected_rows"] - item["nominal_rows"]
            or source_rows is not None and item["source_rows"] != source_rows):
            raise ValueError("점수 진단의 절단점 값 오류")
        last, source_rows = item["cutoff"], item["source_rows"]


def distribution(values):
    return {name: float(np.quantile(values, q, method="linear")) if len(values) else None
            for name, q in QUANTILES.items()}


def confusion(labels, requests, original):
    y, r, old = np.asarray(labels, bool), np.asarray(requests, bool), np.asarray(original, bool)
    tp, fp, fn = int((y & r).sum()), int((~y & r).sum()), int((y & ~r).sum())
    return {"rows": len(y), "positive": int(y.sum()), "requests": int(r.sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "added_tp": int((y & r & ~old).sum()), "added_fp": int((~y & r & ~old).sum()),
            "removed_tp": int((y & ~r & old).sum()), "removed_fp": int((~y & ~r & old).sum())}


def concentration(episodes):
    result = {"episodes": len(episodes), "positive_episodes": sum(v["positive"] > 0 for v in episodes),
              "requested_episodes": sum(v["requests"] > 0 for v in episodes),
              "tp_episodes": sum(v["tp"] > 0 for v in episodes)}
    for key in ["positive", "fp"]:
        values = sorted([v[key] for v in episodes], reverse=True)
        total = sum(values)
        result[key + "_top1_fraction"] = sum(values[:1]) / total if total else None
        result[key + "_top3_fraction"] = sum(values[:3]) / total if total else None
    return result


def audit_scores(frame, threshold, calibration_cutoffs):
    validate_frame(frame, threshold)
    validate_cutoffs(calibration_cutoffs)
    x, y, old = frame.score.to_numpy(float), frame.label.to_numpy(bool), frame.request.to_numpy(bool)
    positive = np.flatnonzero(y)
    sorted_all, sorted_neg = np.sort(x), np.sort(x[~y])
    rank_start = len(x) - np.searchsorted(sorted_all, x[positive], side="right") + 1
    rank_end = len(x) - np.searchsorted(sorted_all, x[positive], side="left")
    neg_above = len(sorted_neg) - np.searchsorted(sorted_neg, x[positive], side="right")
    neg_equal = (np.searchsorted(sorted_neg, x[positive], side="right")
                 - np.searchsorted(sorted_neg, x[positive], side="left"))
    positives = [{"end": frame.end.iloc[i].isoformat(), "episode_id": int(frame.episode_id.iloc[i]),
                  "score": float(x[i]), "requested": bool(old[i]), "rank_start": int(rank_start[j]),
                  "rank_end": int(rank_end[j]), "negative_above": int(neg_above[j]),
                  "negative_equal": int(neg_equal[j])} for j, i in enumerate(positive)]
    counted = frame[["episode_id"]].copy()
    counted = counted.assign(rows=1, positive=y.astype(int), requests=old.astype(int),
                             tp=(y & old).astype(int), fp=(~y & old).astype(int), fn=(y & ~old).astype(int))
    episodes = [{k: int(v) for k, v in item.items()} for item in
                counted.groupby("episode_id", sort=True).sum().reset_index().to_dict("records")]
    current = tail_cutoffs(x) if len(x) else None
    tails = {}
    for name, cutoffs in [("calibration_fixed", calibration_cutoffs), ("contemporaneous_hindsight", current)]:
        tails[name] = {}
        for key in TAILS:
            spec = cutoffs[key] if cutoffs is not None else None
            selected = x >= spec["cutoff"] if spec else np.zeros(len(x), bool)
            tails[name][key] = {"source": spec, "metrics": confusion(y, selected, old)}
    return {"threshold": float(threshold), "original": confusion(y, old, old),
            "scores": {"all": distribution(x), "positive": distribution(x[y]), "negative": distribution(x[~y])},
            "positive_max_below_threshold": bool(x[y].max() < threshold) if y.any() else None,
            "positive_rank_start_fraction": distribution(rank_start / len(x) if len(x) else rank_start),
            "positive_rank_end_fraction": distribution(rank_end / len(x) if len(x) else rank_end),
            "concentration": concentration(episodes), "tails": tails}, {"positives": positives, "episodes": episodes}
