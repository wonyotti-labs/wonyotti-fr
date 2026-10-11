from __future__ import annotations

from collections import deque

import numpy as np


def graph_match_count(requests, targets, lower, upper):
    edges = {int(r): [int(y) for y in targets if lower <= int(y) - int(r) <= upper]
             for r in requests}
    owners = {}
    for root in edges:
        queue, seen, path = deque([root]), {root}, {}
        terminal = None
        while queue and terminal is None:
            request = queue.popleft()
            for target in edges[request]:
                if target in path:
                    continue
                path[target] = request
                if target not in owners:
                    terminal = target
                    break
                previous = owners[target]
                if previous not in seen:
                    seen.add(previous)
                    queue.append(previous)
        if terminal is not None:
            reverse = {request: target for target, request in owners.items()}
            while terminal is not None:
                request = path[terminal]
                previous_target = reverse.get(request)
                owners[terminal] = request
                terminal = previous_target
    return len(owners)


def verify_action(frame, request, label, summary, ledgers):
    result = {"rows": len(frame), "positives": 0, "requests": 0, "tp": 0, "fp": 0, "fn": 0,
              "false_positive_timing": dict.fromkeys(
                  ["future_only", "past_only", "both", "far", "no_positive"], 0),
              "matching": {}, "clusters": {"count": 0, "singletons": 0, "max_requests": 0,
                  "max_duration_minutes": 0, "all_fp_count": 0, "all_fp_requests": 0}}
    expected_requests, expected_clusters = [], []
    pairs = ledgers["pairs"]
    checked_pairs = 0
    for episode, part in frame.groupby("episode_id", sort=False):
        # Timestamp 연산으로 구현의 정수 배열 변환과 별도로 분을 재구성한다.
        r = [int(t.timestamp() // 60) for t in part.loc[part[request], "end"]]
        y = [int(t.timestamp() // 60) for t in part.loc[part[label].eq(1), "end"]]
        exact = set(r) & set(y)
        result["positives"] += len(y)
        result["requests"] += len(r)
        result["tp"] += len(exact)
        for minute in r:
            differences = [t - minute for t in y]
            if 0 in differences:
                category = "exact"
            elif not differences:
                category = "no_positive"
            else:
                early = any(0 < d < 15 for d in differences)
                late = any(-15 < d < 0 for d in differences)
                category = "both" if early and late else (
                    "future_only" if early else "past_only" if late else "far"
                )
            if category != "exact":
                result["false_positive_timing"][category] += 1
            expected_requests.append({"episode_id": int(episode), "minute": minute,
                                      "category": category})
        unmatched_r, unmatched_y = sorted(set(r) - exact), sorted(set(y) - exact)
        for window, lower, upper in [("early", 1, 14), ("late", -14, -1), ("symmetric", -14, 14)]:
            count = graph_match_count(unmatched_r, unmatched_y, lower, upper)
            loose = sum(any(lower <= t - r <= upper for r in unmatched_r) for t in unmatched_y)
            row = result["matching"].setdefault(window, {"additional_pairs": 0,
                                                        "loose_remaining_targets": 0})
            row["additional_pairs"] += count
            row["loose_remaining_targets"] += loose
            found = [p for p in pairs if p["episode_id"] == episode and p["window"] == window]
            assert len(found) == count
            assert len({p["request_minute"] for p in found}) == count
            assert len({p["target_minute"] for p in found}) == count
            for pair in found:
                assert pair["request_minute"] in unmatched_r
                assert pair["target_minute"] in unmatched_y
                assert lower <= pair["target_minute"] - pair["request_minute"] <= upper
            checked_pairs += count
        clusters = []
        for minute in r:
            if not clusters or minute - clusters[-1][-1] > 3:
                clusters.append([])
            clusters[-1].append(minute)
        for group in clusters:
            size, duration = len(group), group[-1] - group[0]
            hits = len(set(group) & set(y))
            row = result["clusters"]
            row["count"] += 1
            row["singletons"] += int(size == 1)
            row["max_requests"] = max(row["max_requests"], size)
            row["max_duration_minutes"] = max(row["max_duration_minutes"], duration)
            row["all_fp_count"] += int(hits == 0)
            row["all_fp_requests"] += size if hits == 0 else 0
            expected_clusters.append({"episode_id": int(episode), "start_minute": group[0],
                                      "end_minute": group[-1], "requests": size, "exact_count": hits})
    result["fp"] = result["requests"] - result["tp"]
    result["fn"] = result["positives"] - result["tp"]
    for row in result["matching"].values():
        row["total_pairs_with_exact"] = result["tp"] + row["additional_pairs"]
    assert result == summary
    assert checked_pairs == len(pairs)
    assert expected_requests == ledgers["requests"]
    assert expected_clusters == ledgers["clusters"]
    assert np.isfinite(list(result["clusters"].values())).all()
    return {"all_rows": len(frame), "request_rows": len(expected_requests),
            "unique_pair_rows": checked_pairs, "cluster_rows": len(expected_clusters)}
