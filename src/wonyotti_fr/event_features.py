from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = ["ret_5m", "ret_15m", "ret_1h", "ret_4h", "ret_1d", "trend_15m_1h", "trend_1h_4h",
                   "trend_4h_16h", "vol_1h", "vol_1d", "volume_ratio_1h", "volume_ratio_1d",
                   "range_fraction", "close_location"]
STATE_FEATURES = ["direction", "favorable_move", "log_hold_minutes", "adds_capped"]


def event_features(bars: pd.DataFrame) -> pd.DataFrame:
    if len(bars) < 300 or bars.time.duplicated().any() or not bars.time.is_monotonic_increasing:
        raise ValueError("특징 계산에는 시간순의 중복 없는 시세가 300개 이상 필요합니다.")
    if not ((bars.end - bars.time) == pd.Timedelta(minutes=5)).all():
        raise ValueError("사건별 후보는 5분봉만 사용합니다.")
    frame = bars.reset_index(drop=True)
    close = frame.close.astype(float)
    returns = np.log(close).diff()
    result = frame[["time", "end"]].copy().astype({"time": "datetime64[ns, UTC]", "end": "datetime64[ns, UTC]"})
    for name, size in [("ret_5m", 1), ("ret_15m", 3), ("ret_1h", 12), ("ret_4h", 48), ("ret_1d", 288)]:
        result[name] = close.pct_change(size, fill_method=None)
    segments = frame.time.diff().ne(pd.Timedelta(minutes=5)).cumsum()
    averages = {span: close.groupby(segments).transform(lambda s, window=span: s.ewm(span=window, adjust=False, min_periods=window).mean())
                for span in [3, 12, 48, 192]}
    for name, first, second in [("trend_15m_1h", 3, 12), ("trend_1h_4h", 12, 48), ("trend_4h_16h", 48, 192)]:
        result[name] = averages[first] / averages[second] - 1
    result["vol_1h"] = returns.rolling(12, min_periods=12).std() * np.sqrt(12)
    result["vol_1d"] = returns.rolling(288, min_periods=288).std() * np.sqrt(288)
    for name, window in [("volume_ratio_1h", 12), ("volume_ratio_1d", 288)]:
        result[name] = frame.volume / frame.volume.rolling(window, min_periods=window).mean().replace(0, np.nan)
    result["range_fraction"] = (frame.high - frame.low) / close
    result["close_location"] = (close - frame.low) / (frame.high - frame.low).replace(0, np.nan)
    result[MARKET_FEATURES] = result[MARKET_FEATURES].replace([np.inf, -np.inf], np.nan)
    broken = frame.time.diff().ne(pd.Timedelta(minutes=5)).rolling(288, min_periods=1).max().astype(bool)
    result.loc[broken, MARKET_FEATURES] = np.nan
    return result


def teacher_states(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if len(actions) != len(events) or not actions.time.equals(events.time) or not actions.order_key.equals(events.order_key):
        raise ValueError("원본 행동과 사건의 순서가 다릅니다.")
    basis, adds, entry, seen = 0.0, 0, pd.NaT, set()
    rows = []
    for action, event in zip(actions.itertuples(index=False), events.itertuples(index=False), strict=True):
        if event.exectype != "Funding":
            unit = abs(event.cost_satoshi) / event.quantity
            if action.action in {"open", "reverse"}:
                basis, adds, entry, seen = unit, 0, action.time, set()
            elif action.action == "increase":
                basis = (basis * abs(action.before_qty) + unit * abs(action.after_qty - action.before_qty)) / abs(action.after_qty)
                adds += int(action.order_key not in seen)
            elif action.action == "close":
                basis, adds, entry = 0.0, 0, pd.NaT
            seen.add(action.order_key)
        rows.append((action.time, int(np.sign(action.after_qty)), 1e8 / basis if basis else 0.0,
                     entry, adds, action.episode_id if action.after_qty else 0))
    result = pd.DataFrame(rows, columns=["state_time", "direction", "average_entry", "entry_time", "adds", "episode_id"])
    for key in ["state_time", "entry_time"]:
        result[key] = pd.to_datetime(result[key], utc=True).astype("datetime64[ns, UTC]")
    return result.drop_duplicates("state_time", keep="last")


def independent_orders(executions: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    if not executions.time.is_monotonic_increasing or not actions.time.is_monotonic_increasing:
        raise ValueError("원본 체결과 행동은 시간순이어야 합니다.")
    source = executions[(executions.symbol == "XBTUSD") & executions.exectype.eq("Trade")
                        & ~executions.order_key.str.startswith("special:")]
    first = source.drop_duplicates("order_key", keep="first")
    linked = first[["order_key", "orderqty", "side", "time"]].merge(
        actions[actions.action.ne("funding")].drop_duplicates("order_key", keep="first")[["order_key", "before_qty", "episode_id"]],
        on="order_key", how="left", validate="one_to_one")
    if linked.before_qty.isna().any() or linked.orderqty.le(0).any():
        raise ValueError("최초 주문 수량 또는 직전 포지션이 불완전합니다.")
    direction = np.where(linked.side.eq("Buy"), 1, -1)
    existing = np.sign(linked.before_qty)
    entry = np.where(direction > 0, "enter_long", "enter_short")
    linked["target"] = np.where(existing == 0, entry,
                                 np.where(existing == direction, "increase",
                                          np.where(linked.orderqty < linked.before_qty.abs(), "reduce",
                                                   np.where(linked.orderqty == linked.before_qty.abs(), "exit", entry))))
    linked = linked.rename(columns={"time": "target_time", "episode_id": "target_episode_id"})
    linked["target_time"] = linked.target_time.astype("datetime64[ns, UTC]")
    return linked.sort_values("target_time", kind="stable")


def training_events(bars: pd.DataFrame, actions: pd.DataFrame, events: pd.DataFrame,
                    executions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = event_features(bars)
    states = teacher_states(actions, events)
    orders = independent_orders(executions, actions)
    joined = pd.merge_asof(features, states, left_on="end", right_on="state_time", direction="backward", allow_exact_matches=False)
    joined[["direction", "average_entry", "adds", "episode_id"]] = joined[["direction", "average_entry", "adds", "episode_id"]].fillna(0)
    joined["favorable_move"] = (bars.close.to_numpy() / joined.average_entry.replace(0, np.nan) - 1) * joined.direction
    joined["favorable_move"] = joined.favorable_move.fillna(0)
    hold = (joined.end - joined.entry_time).dt.total_seconds().div(60).fillna(0)
    joined["log_hold_minutes"] = np.log1p(hold.clip(lower=0))
    joined["adds_capped"] = joined["adds"].clip(upper=5)
    joined = pd.merge_asof(joined, orders[["target_time", "target", "target_episode_id"]], left_on="end", right_on="target_time",
                           direction="forward", tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))
    joined["target"] = joined.target.fillna("hold")
    joined["label_end"] = joined.end + pd.Timedelta(minutes=5)
    times = orders.target_time.array.asi8
    starts = joined.end.array.asi8
    stops = joined.label_end.array.asi8
    joined["orders_in_window"] = np.searchsorted(times, stops, side="left") - np.searchsorted(times, starts, side="left")
    flat = joined.direction.eq(0)
    compatible = (~flat) | joined.target.isin(["hold", "enter_long", "enter_short"])
    same_entry = ((joined.direction > 0) & joined.target.eq("enter_long")) | ((joined.direction < 0) & joined.target.eq("enter_short"))
    compatible &= ~same_entry
    covered = (joined.end >= actions.time.min().floor("5min")) & (joined.label_end <= actions.time.max())
    valid_features = joined[MARKET_FEATURES + STATE_FEATURES].notna().all(axis=1)
    joined["usable"] = covered & compatible & valid_features
    diagnostics = {"bars": len(joined), "independent_orders": len(orders),
                   "covered_bars": int(covered.sum()), "usable_bars": int(joined.usable.sum()),
                   "incompatible_state_windows": int((covered & ~compatible).sum()),
                   "multiple_order_windows": int((covered & joined.orders_in_window.gt(1)).sum()),
                   "label_rule": "다음 5분의 첫 독립 주문. 창 안의 다른 주문은 건수만 별도 기록.",
                   "source_limit": "최초 체결 시각이며 실제 주문 제출·취소 시각은 알 수 없음."}
    return joined, diagnostics


def purged_train(data: pd.DataFrame, episodes: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    boundary = pd.Timestamp(cutoff, tz="UTC")
    cross = episodes[(episodes.entry_time < boundary) & (episodes.exit_time.isna() | (episodes.exit_time >= boundary))].episode_id
    mask = data.usable & (data.label_end < boundary - pd.Timedelta(days=1))
    mask &= ~data.episode_id.isin(cross) & ~data.target_episode_id.isin(cross)
    return data[mask].copy()
