from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .entry_boundary_support import utc_times


def occupancy_summary(curve, episodes):
    times = pd.to_datetime(curve.time, utc=True)
    utc_times(times, unique=True)
    if (
        curve.empty
        or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        or not np.isfinite(curve.quantity).all()
        or curve.policy_event.isna().any()
    ):
        raise ValueError("진입 경로의 분별 계좌 시간·수량 오류")
    months = (times - pd.Timedelta(minutes=1)).dt.strftime("%Y-%m")
    signals = (
        (episodes.signal_time - pd.Timedelta(minutes=1)).dt.strftime("%Y-%m")
        if len(episodes)
        else pd.Series([], dtype=str)
    )

    def describe(part, waits):
        held = part.quantity.ne(0)
        return {
            "minutes": len(part),
            "held_close_minutes": int(held.sum()),
            "flat_close_minutes": int((~held).sum()),
            "held_close_fraction": float(held.mean()) if len(part) else None,
            "daily_halt_minutes": int(part.daily_halted.sum()),
            "halted_minutes": int(part.halted.sum()),
            "policy_events": {
                str(k): int(v) for k, v in part.policy_event.value_counts().sort_index().items()
            },
            "flat_policy_events": {
                str(k): int(v)
                for k, v in part.loc[~held, "policy_event"].value_counts().sort_index().items()
            },
            "waits_started": len(waits),
            "wait_status": {
                str(k): int(v) for k, v in waits.status.value_counts().sort_index().items()
            }
            if len(waits)
            else {},
            "waits_executed": int(waits.executed.sum()) if len(waits) else 0,
            "triggers_without_fill": int((waits.status.eq("triggered") & ~waits.executed).sum())
            if len(waits)
            else 0,
        }

    result = {"total": describe(curve, episodes), "monthly": {}}
    for month in sorted(set(months)):
        result["monthly"][month] = describe(
            curve.loc[months.eq(month)], episodes.loc[signals.eq(month)]
        )
    return result


def verify_occupancy_counts(curve, episodes, summary):
    groups = {}
    for stamp, quantity, event, daily, halted in zip(
        curve.time,
        curve.quantity,
        curve.policy_event,
        curve.daily_halted,
        curve.halted,
        strict=True,
    ):
        month = str(pd.Timestamp(stamp) - pd.Timedelta(minutes=1))[:7]
        for key in ["total", month]:
            if key not in groups:
                groups[key] = {
                    "minutes": 0,
                    "held_close_minutes": 0,
                    "flat_close_minutes": 0,
                    "daily_halt_minutes": 0,
                    "halted_minutes": 0,
                    "policy_events": Counter(),
                    "flat_policy_events": Counter(),
                    "waits_started": 0,
                    "wait_status": Counter(),
                    "waits_executed": 0,
                    "triggers_without_fill": 0,
                }
            group = groups[key]
            group["minutes"] += 1
            group["held_close_minutes"] += int(quantity != 0)
            group["flat_close_minutes"] += int(quantity == 0)
            group["daily_halt_minutes"] += int(daily)
            group["halted_minutes"] += int(halted)
            group["policy_events"][event] += 1
            if quantity == 0:
                group["flat_policy_events"][event] += 1
    for row in episodes.itertuples(index=False):
        month = str(row.signal_time - pd.Timedelta(minutes=1))[:7]
        for key in ["total", month]:
            group = groups[key]
            group["waits_started"] += 1
            group["wait_status"][row.status] += 1
            group["waits_executed"] += int(row.executed)
            group["triggers_without_fill"] += int(row.status == "triggered" and not row.executed)
    assert set(groups) - {"total"} == set(summary["monthly"])
    for key, expected in groups.items():
        actual = summary["total"] if key == "total" else summary["monthly"][key]
        for name, value in expected.items():
            assert actual[name] == value
        assert actual["held_close_fraction"] == expected["held_close_minutes"] / expected["minutes"]


def verify_holding_intervals(curve, trades, fills):
    ends = pd.to_datetime(curve.time, utc=True).astype("datetime64[ns, UTC]").array.asi8
    direction = np.zeros(len(curve), dtype=np.int8)
    entries = fills.loc[fills.reason.eq("entry")] if len(fills) else pd.DataFrame()
    entry_lookup = {pd.Timestamp(row.time): row for row in entries.itertuples(index=False)}
    if len(entry_lookup) != len(entries):
        raise ValueError("진입 경로의 중복 진입 체결")
    total, closed_at_end, matched, previous_exit = 0, 0, set(), None
    for row in trades.itertuples(index=False):
        entry, exit_ = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        if (
            entry.tzinfo is None
            or exit_.tzinfo is None
            or entry.utcoffset().total_seconds()
            or exit_.utcoffset().total_seconds()
            or entry.value % 60_000_000_000
            or exit_.value % 60_000_000_000
            or exit_ < entry
            or row.direction not in (-1, 1)
        ):
            raise ValueError("진입 경로의 거래 시간·방향 오류")
        hold = int((exit_ - entry).total_seconds() / 60)
        if hold != row.hold_bars:
            raise ValueError("진입 경로의 거래 보유 분 불일치")
        fill = entry_lookup.get(entry)
        if (
            fill is None
            or entry in matched
            or row.direction != int(np.sign(fill.delta_quantity))
            or row.entry_price != fill.price
            or previous_exit is not None
            and entry < previous_exit
        ):
            raise ValueError("진입 경로의 체결·거래 연결 또는 보유 시간 중첩")
        matched.add(entry)
        previous_exit = exit_
        at_end = row.exit_reason in {"intrabar_stop", "end_of_test"}
        first = int(np.searchsorted(ends, entry.value, side="right"))
        last = int(np.searchsorted(ends, exit_.value, side="left" if at_end else "right"))
        if first > last or (direction[first:last] != 0).any() or last - first != hold - int(at_end):
            raise ValueError("진입 경로의 보유 구간 중복·시가 및 분말 경계 오류")
        direction[first:last] = row.direction
        total += hold
        closed_at_end += int(at_end)
    np.testing.assert_array_equal(direction, np.sign(curve.quantity.to_numpy()).astype(np.int8))
    if len(matched) != len(entries):
        raise ValueError("진입 경로의 거래에 연결되지 않은 실제 진입")
    assert int(np.count_nonzero(direction)) == total - closed_at_end
    return {
        "closed_trades": len(trades),
        "sum_trade_hold_minutes": total,
        "closes_at_bar_end": closed_at_end,
        "held_close_minutes": int(np.count_nonzero(direction)),
        "all_close_directions_exact": True,
        "exit_reasons": {
            str(reason): {
                "trades": len(part),
                "hold_minutes": int(part.hold_bars.sum()),
                "net_pnl": float(part.net_pnl.sum()),
            }
            for reason, part in trades.groupby("exit_reason", sort=True)
        }
        if len(trades)
        else {},
    }
