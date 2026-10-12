from __future__ import annotations

from collections import deque

import pandas as pd


def graph_cardinality(adjacency):
    owners = {}
    for start in range(len(adjacency)):
        queue, visited, via = deque([start]), {start}, {}
        end = None
        while queue and end is None:
            request = queue.popleft()
            for event in adjacency[request]:
                if event in via:
                    continue
                via[event] = request
                if event not in owners:
                    end = event
                    break
                owner = owners[event]
                if owner not in visited:
                    visited.add(owner)
                    queue.append(owner)
        if end is not None:
            assigned = {request: event for event, request in owners.items()}
            while True:
                request = via[end]
                previous = assigned.get(request)
                owners[end] = request
                if previous is None:
                    break
                end = previous
    return len(owners)


def verify_temporal(frame, request_mask, target_mask, horizon, requests, targets):
    original = list(frame[["end", "episode_id"]].itertuples(index=False, name=None))
    requested = [row for row, flag in zip(original, request_mask, strict=True) if flag]
    events = [row for row, flag in zip(original, target_mask, strict=True) if flag]
    assert (
        list(requests[["request_end", "episode_id"]].itertuples(index=False, name=None))
        == requested
    )
    assert list(targets[["event_end", "episode_id"]].itertuples(index=False, name=None)) == events
    event_indices = {row: i for i, row in enumerate(events)}
    observed = {t for t, _ in original}
    adjacency = [
        [
            event_indices[key]
            for step in range(horizon)
            if (key := (time + pd.Timedelta(minutes=step), episode)) in event_indices
        ]
        for time, episode in requested
    ]
    max_count = graph_cardinality(adjacency)
    linked = requests.loc[requests.event_end.notna()]
    assert len(linked) == max_count == int(targets.request_end.notna().sum())
    pairs, used = {}, set()
    for i, row in enumerate(requests.itertuples(index=False)):
        key = (row.request_end, row.episode_id)
        assert row.original_positive == (key in event_indices)
        assert row.has_target == bool(adjacency[i])
        assert row.window_grid_complete == all(
            row.request_end + pd.Timedelta(minutes=j) in observed for j in range(horizon)
        )
        assert row.window_beyond_phase == (
            row.request_end + pd.Timedelta(minutes=horizon)
            > original[-1][0] + pd.Timedelta(minutes=1)
        )
        if pd.isna(row.event_end):
            assert pd.isna(row.lead_minutes)
            continue
        target = (row.event_end, row.episode_id)
        assert target in event_indices and target not in used
        assert event_indices[target] in adjacency[i]
        assert row.lead_minutes == int((row.event_end - row.request_end).total_seconds() / 60)
        used.add(target)
        pairs[target] = row.request_end
    reachable = {event for neighbors in adjacency for event in neighbors}
    for i, row in enumerate(targets.itertuples(index=False)):
        key = (row.event_end, row.episode_id)
        assert row.has_request == (i in reachable)
        if key in pairs:
            assert row.request_end == pairs[key]
        else:
            assert pd.isna(row.request_end)
    return max_count


def reference_summary(requests, targets, horizon):
    rr, ee = list(requests.itertuples(index=False)), list(targets.itertuples(index=False))
    matched = sum(pd.notna(r.event_end) for r in rr)
    captured = sum(pd.notna(e.request_end) for e in ee)
    return {
        "requests": len(rr),
        "original_true_positive": sum(bool(r.original_positive) for r in rr),
        "matched_requests": matched,
        "unmatched_requests": len(rr) - matched,
        "request_match_rate": matched / len(rr) if rr else None,
        "requests_with_any_target": sum(bool(r.has_target) for r in rr),
        "complete_observation_windows": sum(bool(r.window_grid_complete) for r in rr),
        "windows_beyond_phase": sum(bool(r.window_beyond_phase) for r in rr),
        "lead_minutes_counts": {
            str(i): sum(pd.notna(r.lead_minutes) and r.lead_minutes == i for r in rr)
            for i in range(horizon)
        },
        "targets": len(ee),
        "matched_targets": captured,
        "unmatched_targets": len(ee) - captured,
        "target_capture_rate": captured / len(ee) if ee else None,
        "targets_with_any_request": sum(bool(r.has_request) for r in ee),
    }


def verify_temporal_summaries(requests, targets, horizon, total, monthly):
    assert total == reference_summary(requests, targets, horizon)
    for month, result in monthly.items():
        rr = requests.loc[[t.strftime("%Y-%m") == month for t in requests.request_end]]
        ee = targets.loc[[t.strftime("%Y-%m") == month for t in targets.event_end]]
        assert result == reference_summary(rr, ee, horizon)
    for field in [
        "requests",
        "original_true_positive",
        "matched_requests",
        "unmatched_requests",
        "requests_with_any_target",
        "complete_observation_windows",
        "windows_beyond_phase",
        "targets",
        "matched_targets",
        "unmatched_targets",
        "targets_with_any_request",
    ]:
        assert sum(v[field] for v in monthly.values()) == total[field]
    for lead in total["lead_minutes_counts"]:
        assert (
            sum(v["lead_minutes_counts"][lead] for v in monthly.values())
            == total["lead_minutes_counts"][lead]
        )
