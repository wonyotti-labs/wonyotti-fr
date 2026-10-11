from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .extrema_age import EXTRA_FEATURES

STATE_KEYS = {"entry", "direction", "end", "first", "low", "high", "low_end", "high_end"}


def validate_clock_state(state):
    if state is None:
        return
    if (
        set(state) != STATE_KEYS
        or type(state["direction"]) is not int
        or state["direction"] not in (-1, 0, 1)
    ):
        raise ValueError("극값 시계 복원의 필드·방향 오류")
    if state["direction"] == 0:
        end = pd.Timestamp(state["end"])
        if (
            pd.isna(end)
            or end.tzinfo is None
            or end.utcoffset().total_seconds()
            or end.value % 60_000_000_000
            or any(state[k] is not None for k in STATE_KEYS - {"direction", "end"})
        ):
            raise ValueError("극값 시계 복원의 무포지션 상태 오류")
        return
    times = {k: pd.Timestamp(state[k]) for k in ["entry", "end", "first", "low_end", "high_end"]}
    if any(pd.isna(t) or t.tzinfo is None or t.utcoffset().total_seconds() for t in times.values()):
        raise ValueError("극값 시계 복원의 UTC 오류")
    if (
        not times["entry"] < times["first"] <= times["end"]
        or any(not times["first"] <= times[k] <= times["end"] for k in ["low_end", "high_end"])
        or any(times[k].value % 60_000_000_000 for k in ["first", "end", "low_end", "high_end"])
        or any(
            type(state[k]) not in (float, int) or not math.isfinite(state[k])
            for k in ["low", "high"]
        )
        or not 0 < state["low"] <= state["high"]
    ):
        raise ValueError("극값 시계 복원의 시간·가격 범위 오류")


def streaming_extrema(frame, initial=None):
    validate_clock_state(initial)
    state = None if initial is None else dict(initial)
    result = []
    previous = pd.Timestamp(state["end"]) if state else None
    for row in frame[["end", "close", "entry_time", "direction"]].itertuples(index=False):
        end, close, entry, direction = row
        if (
            pd.isna(end)
            or end.tzinfo is None
            or end.utcoffset().total_seconds()
            or end.value % 60_000_000_000
            or not math.isfinite(close)
            or close <= 0
            or direction not in (-1, 0, 1)
            or (previous is not None and end - previous != pd.Timedelta(minutes=1))
        ):
            raise ValueError("극값 시계 스트림의 시간·가격·방향 오류")
        previous = end
        if not direction:
            state = {k: None for k in STATE_KEYS}
            state.update(direction=0, end=end.isoformat())
            continue
        if (
            pd.isna(entry)
            or entry.tzinfo is None
            or entry.utcoffset().total_seconds()
            or entry >= end
        ):
            raise ValueError("극값 시계 스트림의 진입 시각 오류")
        if (
            state is None
            or state["direction"] != direction
            or pd.Timestamp(state["entry"]) != entry
        ):
            state = {
                "entry": entry.isoformat(),
                "direction": int(direction),
                "end": end.isoformat(),
                "first": end.isoformat(),
                "low": float(close),
                "high": float(close),
                "low_end": end.isoformat(),
                "high_end": end.isoformat(),
            }
        else:
            if close < state["low"]:
                state["low"], state["low_end"] = float(close), end.isoformat()
            if close > state["high"]:
                state["high"], state["high_end"] = float(close), end.isoformat()
            state["end"] = end.isoformat()
        best = pd.Timestamp(state["high_end"] if direction == 1 else state["low_end"])
        worst = pd.Timestamp(state["low_end"] if direction == 1 else state["high_end"])
        best_minutes, worst_minutes = (
            int((end.value - t.value) // 60_000_000_000) for t in [best, worst]
        )
        result.append(
            [
                end,
                pd.Timestamp(state["first"]),
                state["low"],
                state["high"],
                best,
                worst,
                best_minutes,
                worst_minutes,
                math.log1p(best_minutes),
                math.log1p(worst_minutes),
            ]
        )
    validate_clock_state(state)
    names = [
        "end",
        "path_first_end",
        "path_low",
        "path_high",
        "best_record_end",
        "worst_record_end",
        "minutes_since_best",
        "minutes_since_worst",
        *EXTRA_FEATURES,
    ]
    return pd.DataFrame(result, columns=names), state


def verify_extrema_inputs(source, table, rows, expanded, links):
    direct, _ = streaming_extrema(source)
    pd.testing.assert_frame_equal(
        table[direct.columns], direct, check_exact=False, atol=1e-12, rtol=0
    )
    result = {"source_rows": len(source), "source_held_rows": len(table), "periods": {}}
    indexed = direct.set_index("end")
    for phase, frame in rows.items():
        pd.testing.assert_frame_equal(
            frame, expanded[phase].drop(columns=EXTRA_FEATURES), check_exact=True
        )
        expected = indexed.loc[frame.end]
        np.testing.assert_allclose(
            expanded[phase][EXTRA_FEATURES], expected[EXTRA_FEATURES], atol=1e-12, rtol=0
        )
        link = links[phase]
        for name in ["best_record_end", "worst_record_end", "path_first_end"]:
            np.testing.assert_array_equal(
                link[name].astype("datetime64[ns, UTC]").array.asi8,
                expected[name].astype("datetime64[ns, UTC]").array.asi8,
            )
        result["periods"][phase] = {
            "rows": len(frame),
            "new_values": 2 * len(frame),
            "ages": {
                label: {
                    "minimum": int(link["minutes_since_" + label].min()),
                    "maximum": int(link["minutes_since_" + label].max()),
                    "zero_rows": int(link["minutes_since_" + label].eq(0).sum()),
                }
                for label in ["best", "worst"]
            },
        }
    return result


def verify_age_groups(link, label, groups):
    expected = []
    for value in link["minutes_since_" + label]:
        expected.append(
            0 if value == 0 else 1 if value < 15 else 2 if value < 60 else 3 if value < 1440 else 4
        )
    np.testing.assert_array_equal(groups, expected)
