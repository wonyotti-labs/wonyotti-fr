from __future__ import annotations

import numpy as np
import pandas as pd

from .paired_management_reference import manual_metrics
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups


def verify_add_count_links(frame, actions, events, links):
    trade = events.exectype.eq("Trade").to_numpy()
    signed = np.where(trade, np.where(events.side.eq("Buy"), events.quantity, -events.quantity), 0).astype(np.int64)
    if sum(abs(int(v)) for v in signed) >= 2**62:
        raise ValueError("추가 횟수 독립 복원의 수량 누적 범위 초과")
    quantities = np.cumsum(signed)
    before = np.r_[0, quantities[:-1]]
    np.testing.assert_array_equal(actions.before_qty, before)
    np.testing.assert_array_equal(actions.after_qty, quantities)
    direction = np.sign(quantities)
    born = (quantities != 0) & ((before == 0) | (np.sign(before) != direction))
    birth_ids = np.cumsum(born)
    episodes = np.where(quantities != 0, birth_ids, 0)
    np.testing.assert_array_equal(actions.episode_id, birth_ids)
    increased = trade & (before != 0) & (np.sign(before) == direction) & (np.abs(quantities) > np.abs(before))
    labels = np.where(~trade, "funding", np.where(before == 0, "open",
        np.where(quantities == 0, "close", np.where(born, "reverse", np.where(increased, "increase", "reduce")))))
    np.testing.assert_array_equal(actions.action, labels)
    first = np.zeros(len(events), dtype=bool)
    seen = pd.DataFrame({"birth": birth_ids[trade], "key": events.loc[trade, "order_key"].to_numpy()})
    first[trade] = ~seen.duplicated(["birth", "key"]).to_numpy()
    increments = (increased & first).astype(np.int64)
    counts = pd.Series(increments).groupby(birth_ids, sort=False).cumsum().to_numpy(copy=True)
    counts[quantities == 0] = 0
    entry_indices = np.maximum.accumulate(np.where(born, np.arange(len(events)), -1))
    event_times = events.time.astype("datetime64[ns, UTC]").array.asi8
    observed = np.empty(len(frame), dtype=np.int64)
    cursor = -1
    for i, end in enumerate(frame.end.astype("datetime64[ns, UTC]").array.asi8):
        while cursor + 1 < len(events) and event_times[cursor + 1] < end:
            cursor += 1
        if cursor < 0:
            raise ValueError("추가 횟수 독립 복원의 이전 사건 누락")
        observed[i] = cursor
    np.testing.assert_array_equal(links.event_row, observed)
    np.testing.assert_array_equal(frame.adds, counts[observed])
    np.testing.assert_array_equal(frame.adds_capped, np.minimum(counts[observed], 5))
    np.testing.assert_array_equal(frame.episode_id, episodes[observed])
    np.testing.assert_array_equal(frame.direction, direction[observed])
    np.testing.assert_array_equal(frame.state_time.astype("datetime64[ns, UTC]").array.asi8, event_times[observed])
    held = quantities[observed] != 0
    assert (entry_indices[observed][held] >= 0).all()
    entry_times = np.full(len(frame), pd.NaT.value, dtype=np.int64)
    entry_times[held] = event_times[entry_indices[observed][held]]
    np.testing.assert_array_equal(frame.entry_time.astype("datetime64[ns, UTC]").array.asi8, entry_times)
    expected_buckets = np.array([str(v) if v <= 5 else ">5" for v in counts[observed]])
    np.testing.assert_array_equal(links.bucket, expected_buckets)
    pd.testing.assert_frame_equal(links[frame.columns.intersection(links.columns)], frame[frame.columns.intersection(links.columns)], check_exact=True)
    return {"rows": len(frame), "events": len(events), "distinct_increase_events": int(increments.sum()),
            "all_counts_state_keys_and_strict_times_verified": True}


def requested_metrics(y, p, requested, beta):
    if not len(y):
        return None
    value = manual_metrics(y, p, 0., beta)
    tp = int(np.count_nonzero(requested & (y == 1)))
    fp = int(np.count_nonzero(requested & (y == 0)))
    fn = int(np.count_nonzero(y == 1)) - tp
    denominator = (1 + beta**2)*tp + beta**2*fn + fp
    value.update(predicted_positive=tp+fp, true_positive=tp, false_positive=fp, false_negative=fn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f_beta=(1+beta**2)*tp/denominator if denominator else 0.)
    return value


def verify_count_strata(frame, predictions, saved):
    bucket_names = ["all", "0", "1", "2", "3", "4", "5", ">5"]
    expected = {}
    for action, beta in [("reduce", 1.), ("increase", .5)]:
        expected[action] = {}
        for state, state_mask in independent_groups(frame, action).items():
            expected[action][state] = {}
            for bucket in bucket_names:
                count_mask = np.ones(len(frame), dtype=bool) if bucket == "all" else (
                    frame.adds.to_numpy() > 5 if bucket == ">5" else frame.adds.to_numpy() == int(bucket))
                positions = np.flatnonzero(state_mask & count_mask)
                part = frame.iloc[positions]
                y = part["y_" + action].to_numpy(int)
                actual_counts = set(int(v) for v in part.adds)
                value = {"support": {"rows": len(y), "positive": int(np.count_nonzero(y)),
                    "negative": int(np.count_nonzero(y == 0)), "episodes": len(set(part.episode_id)),
                    "distinct_counts": len(actual_counts), "minimum_adds": min(actual_counts) if len(y) else None,
                    "maximum_adds": max(actual_counts) if len(y) else None}, "models": {}}
                for name in ["v127", "v133"]:
                    selected = predictions[name].iloc[positions]
                    value["models"][name] = requested_metrics(y, selected[action + "_tree_score"].to_numpy(),
                        selected[action + "_tree_request"].to_numpy(bool), beta)
                expected[action][state][bucket] = value
            buckets = saved[action][state]
            if set(buckets) != set(bucket_names):
                raise ValueError("추가 횟수 진단의 고정 구간 변경")
            for key in ["rows", "positive", "negative"]:
                assert sum(buckets[b]["support"][key] for b in bucket_names[1:]) == buckets["all"]["support"][key]
            for name in ["v127", "v133"]:
                total = buckets["all"]["models"][name]
                parts = [buckets[b]["models"][name] for b in bucket_names[1:] if buckets[b]["models"][name] is not None]
                if total is None:
                    assert not parts
                    continue
                for key in ["rows", "positive", "predicted_positive", "true_positive", "false_positive", "false_negative"]:
                    assert sum(v[key] for v in parts) == total[key]
                for key in ["log_loss", "brier"]:
                    compare_statistics(sum(v[key]*v["rows"] for v in parts)/total["rows"], total[key])
    compare_statistics(saved, expected)
