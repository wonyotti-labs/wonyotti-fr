from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def verify_wait_episodes(curve, fills, bars, episodes, *, offset=16, ttl=5):
    if offset != 16 or ttl != 5:
        raise ValueError("진입 경로의 고정 대기 설정 오류")
    times = pd.to_datetime(curve.time, utc=True).astype("datetime64[ns, UTC]")
    pd.testing.assert_series_equal(
        times.reset_index(drop=True).rename("end"),
        bars.end.reset_index(drop=True),
        check_exact=True,
    )
    starts = np.flatnonzero(curve.policy_event.eq("armed").to_numpy())
    if len(starts) != len(episodes):
        raise ValueError("진입 경로의 시작과 대기 원장 수 불일치")
    available = (
        bars.loc[bars["count"].gt(0) & bars.volume.gt(0), "time"]
        .astype("datetime64[ns, UTC]")
        .array.asi8
    )
    entries = (
        fills.loc[fills.reason.eq("entry")].copy() if len(fills) else pd.DataFrame(columns=["time"])
    )
    entry_times = pd.to_datetime(entries.time, utc=True).astype("datetime64[ns, UTC]").array.asi8
    if len(set(entry_times)) != len(entries):
        raise ValueError("진입 경로의 동일 시각 중복 진입")
    entry_lookup = {
        int(stamp): row
        for stamp, row in zip(entry_times, entries.itertuples(index=False), strict=True)
    }
    matched, covered = set(), np.zeros(len(curve), bool)
    event_values, close = curve.policy_event.to_numpy(), bars.close.to_numpy(float)
    results = []
    for start, saved in zip(starts, episodes.itertuples(index=False), strict=True):
        state = json.loads(curve.policy_state.iloc[start])
        stamp = times.iloc[start]
        if (
            set(state) != {"signal_time", "expires_at", "direction", "reference_price"}
            or pd.Timestamp(state["signal_time"]) != stamp
            or stamp.value % (300 * 10**9)
            or pd.Timestamp(state["expires_at"]) != stamp + pd.Timedelta(minutes=ttl)
            or state["direction"] not in (-1, 1)
            or state["reference_price"] != close[start]
            or curve.quantity.iloc[start] != 0
        ):
            raise ValueError("진입 경로의 최초 신호 상태·가격 불일치")
        stop, kind = None, None
        for i in range(start + 1, min(start + ttl + 2, len(curve))):
            age = (times.iloc[i] - stamp).total_seconds() / 60
            favorable = state["direction"] * math.log(state["reference_price"] / close[i]) * 10000
            event = event_values[i]
            if event == "cleared":
                if not (
                    curve.quantity.iloc[i] != 0
                    or curve.halted.iloc[i]
                    or curve.daily_halted.iloc[i]
                    or curve.completed.iloc[i]
                ):
                    raise ValueError("진입 경로의 근거 없는 대기 취소")
                stop, kind = i, event
                break
            expected = (
                "triggered"
                if age <= ttl and favorable >= offset
                else "expired"
                if age >= ttl
                else "waiting"
            )
            if event != expected:
                raise ValueError("진입 경로의 가격 조건·만료 상태 불일치")
            if event == "waiting":
                if json.loads(curve.policy_state.iloc[i]) != state:
                    raise ValueError("진입 경로의 대기 중 원래 신호 변경")
            else:
                stop, kind = i, event
                break
        if stop is None or covered[start : stop + 1].any():
            raise ValueError("진입 경로의 대기 미종료·중첩")
        covered[start : stop + 1] = True
        if json.loads(curve.policy_state.iloc[stop]):
            raise ValueError("진입 경로의 종료 후 대기 상태 잔류")
        decision = times.iloc[stop]
        if (
            saved.signal_time != stamp
            or saved.decision_time != decision
            or saved.status != kind
            or saved.direction != state["direction"]
            or saved.reference_price != close[start]
            or saved.decision_close != close[stop]
            or saved.wait_minutes != (decision - stamp).total_seconds() / 60
        ):
            raise ValueError("진입 경로의 저장 대기 원장 불일치")
        intent = "enter_long" if state["direction"] > 0 else "enter_short"
        if curve.next_intent.iloc[stop] != (intent if kind == "triggered" else "hold"):
            raise ValueError("진입 경로의 종료 의도 불일치")
        j = int(np.searchsorted(available, decision.value))
        executable = int(available[j]) if j < len(available) else None
        entry = entry_lookup.get(executable) if kind == "triggered" else None
        if bool(saved.executed) != (entry is not None) or saved.expected_entry_time != decision:
            raise ValueError("진입 경로의 실제 진입 연결 불일치")
        if executable is not None:
            if pd.Timestamp(saved.executable_entry_time).value != executable:
                raise ValueError("진입 경로의 다음 거래 가능 시각 불일치")
            assert saved.liquidity_wait_minutes == (executable - decision.value) / 60_000_000_000
        elif not pd.isna(saved.executable_entry_time):
            raise ValueError("진입 경로의 없는 거래 가능 시각")
        decision_bps = state["direction"] * math.log(close[start] / close[stop]) * 10000
        np.testing.assert_allclose(
            saved.favorable_at_decision_bps, decision_bps, rtol=0, atol=1e-12
        )
        if entry is not None:
            if (
                executable in matched
                or int(np.sign(entry.delta_quantity)) != state["direction"]
                or saved.entry_price != entry.price
            ):
                raise ValueError("진입 경로의 중복·방향·체결 가격 불일치")
            matched.add(executable)
            expected = [
                state["direction"] * math.log(close[start] / entry.price) * 10000,
                state["direction"] * math.log(close[stop] / entry.price) * 10000,
            ]
            np.testing.assert_allclose(
                [saved.favorable_at_fill_bps, saved.fill_vs_decision_bps],
                expected,
                rtol=0,
                atol=1e-12,
            )
        results.append(kind)
    active = curve.policy_event.isin(["armed", "waiting", "triggered", "expired"]).to_numpy()
    if (active & ~covered).any() or len(matched) != len(entries):
        raise ValueError("진입 경로의 연결 없는 대기·실제 진입")
    for i in np.flatnonzero(curve.policy_event.eq("cleared").to_numpy() & ~covered):
        if (
            i == 0
            or not json.loads(curve.policy_state.iloc[i - 1])
            or "signal_time" in json.loads(curve.policy_state.iloc[i - 1])
        ):
            raise ValueError("진입 경로의 관리 초기화 구분 오류")
    return {
        "waits": len(starts),
        "matched_entry_fills": len(matched),
        "all_wait_states_prices_expiries_and_entries_verified": True,
        "management_resets": int((curve.policy_event.eq("cleared").to_numpy() & ~covered).sum()),
    }
