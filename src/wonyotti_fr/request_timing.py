from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"early": (1, 14), "late": (-14, -1), "symmetric": (-14, 14)}
CATEGORIES = ("future_only", "past_only", "both", "far", "no_positive")


def validate_frame(frame, request, label):
    if (
        not len(frame)
        or frame.columns.duplicated().any()
        or not {"end", "episode_id", request, label}.issubset(frame)
    ):
        raise ValueError("시간 진단의 행·열 누락 또는 중복")
    end = frame.end
    if (
        not isinstance(end.dtype, pd.DatetimeTZDtype)
        or str(end.dt.tz) != "UTC"
        or end.isna().any()
        or end.duplicated().any()
        or not end.is_monotonic_increasing
        or not end.eq(end.dt.floor("min")).all()
        or frame.episode_id.isna().any()
        or not pd.api.types.is_numeric_dtype(frame.episode_id)
        or pd.api.types.is_bool_dtype(frame.episode_id)
        or not np.isfinite(frame.episode_id.to_numpy(dtype=float)).all()
        or not frame.episode_id.mod(1).eq(0).all()
        or (frame.episode_id <= 0).any()
        or (frame.episode_id > 2**53 - 1).any()
        or frame[request].dtype != bool
        or not frame[label].isin([0, 1]).all()
    ):
        raise ValueError("시간 진단의 시각·포지션·요청·정답 오류")
    episodes = frame.episode_id.to_numpy()
    starts = np.r_[True, episodes[1:] != episodes[:-1]]
    if len(np.unique(episodes[starts])) != int(starts.sum()):
        raise ValueError("종료한 포지션의 재등장")
    return end.dt.as_unit("ns").astype("int64").to_numpy() // 60_000_000_000


def match_minutes(requests, targets, lower, upper):
    # 동일한 폭의 시간 구간은 가장 이른 대응부터 확정해 최대 건수를 보존한다.
    i = j = 0
    pairs = []
    while i < len(requests) and j < len(targets):
        lag = int(targets[j]) - int(requests[i])
        if lag < lower:
            j += 1
        elif lag > upper:
            i += 1
        else:
            pairs.append((int(requests[i]), int(targets[j])))
            i += 1
            j += 1
    return pairs


def audit_action(frame, request, label):
    times = validate_frame(frame, request, label)
    predicted, positive = frame[request].to_numpy(), frame[label].eq(1).to_numpy()
    tp = int((predicted & positive).sum())
    summary = {
        "rows": len(frame), "positives": int(positive.sum()), "requests": int(predicted.sum()),
        "tp": tp, "fp": int((predicted & ~positive).sum()),
        "fn": int((~predicted & positive).sum()),
        "false_positive_timing": dict.fromkeys(CATEGORIES, 0),
        "matching": {w: {"additional_pairs": 0, "loose_remaining_targets": 0}
                     for w in WINDOWS},
        "clusters": {"count": 0, "singletons": 0, "max_requests": 0,
                     "max_duration_minutes": 0, "all_fp_count": 0, "all_fp_requests": 0},
    }
    requests_ledger, pair_ledger, cluster_ledger = [], [], []
    for episode, indices in frame.groupby("episode_id", sort=False).indices.items():
        r = times[indices[predicted[indices]]]
        y = times[indices[positive[indices]]]
        exact = np.intersect1d(r, y, assume_unique=True)
        remaining_r = np.setdiff1d(r, exact, assume_unique=True)
        remaining_y = np.setdiff1d(y, exact, assume_unique=True)
        for minute in r:
            if minute in exact:
                category = "exact"
            elif not len(y):
                category = "no_positive"
            else:
                index = np.searchsorted(y, minute)
                future = index < len(y) and y[index] - minute <= 14
                past = index > 0 and minute - y[index - 1] <= 14
                category = "both" if future and past else (
                    "future_only" if future else "past_only" if past else "far"
                )
            if category != "exact":
                summary["false_positive_timing"][category] += 1
            requests_ledger.append({"episode_id": int(episode), "minute": int(minute),
                                    "category": category})
        for window, (lower, upper) in WINDOWS.items():
            pairs = match_minutes(remaining_r, remaining_y, lower, upper)
            summary["matching"][window]["additional_pairs"] += len(pairs)
            # 느슨한 포착은 하나의 요청을 여러 양성 분에 재사용할 수 있다.
            left = np.searchsorted(remaining_r, remaining_y - upper, side="left")
            right = np.searchsorted(remaining_r, remaining_y - lower, side="right")
            summary["matching"][window]["loose_remaining_targets"] += int((right > left).sum())
            for before, after in pairs:
                pair_ledger.append({"episode_id": int(episode), "window": window,
                                    "request_minute": before, "target_minute": after})
        starts = np.flatnonzero(np.r_[True, np.diff(r) > 3]) if len(r) else []
        stops = [*starts[1:], len(r)] if len(r) else []
        for start, stop in zip(starts, stops, strict=True):
            group = r[start:stop]
            exact_count = int(np.isin(group, exact).sum())
            count, duration = len(group), int(group[-1] - group[0])
            cluster = summary["clusters"]
            cluster["count"] += 1
            cluster["singletons"] += int(count == 1)
            cluster["max_requests"] = max(cluster["max_requests"], count)
            cluster["max_duration_minutes"] = max(cluster["max_duration_minutes"], duration)
            cluster["all_fp_count"] += int(exact_count == 0)
            cluster["all_fp_requests"] += count if exact_count == 0 else 0
            cluster_ledger.append({"episode_id": int(episode), "start_minute": int(group[0]),
                                   "end_minute": int(group[-1]), "requests": count,
                                   "exact_count": exact_count})
    for result in summary["matching"].values():
        result["total_pairs_with_exact"] = tp + result["additional_pairs"]
    if sum(summary["false_positive_timing"].values()) != summary["fp"]:
        raise ValueError("오탐 시간 분해 합계 불일치")
    return summary, {"requests": requests_ledger, "pairs": pair_ledger, "clusters": cluster_ledger}
