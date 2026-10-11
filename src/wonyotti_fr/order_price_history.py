from __future__ import annotations

import numpy as np
import pandas as pd

from .order_history import HISTORY_ACTIONS, HISTORY_FEATURES, attach_order_history
from .quiet_management import groups

PRICE_FEATURES = [f"past_{action}_{name}" for action in HISTORY_ACTIONS
                  for name in ["log_price_move", "directional_log_price_move"]]


def price_ledger(frame, actions):
    required = {"end", "episode_id", "close", "direction", *HISTORY_FEATURES}
    if not required.issubset(frame) or not {"price", "time", "order_key", "action", "episode_id", "before_qty", "after_qty"}.issubset(actions):
        raise ValueError("과거 주문 가격의 필수 열 누락")
    if (frame.empty or actions.empty or not frame.index.is_unique
        or any(not isinstance(f[k].dtype, pd.DatetimeTZDtype) or str(f[k].dt.tz) != "UTC"
               for f, k in [(frame, "end"), (actions, "time")])
        or not np.isfinite(frame[["close", "episode_id", "direction"]]).all().all()
        or frame.close.le(0).any() or frame.episode_id.le(0).any() or frame.episode_id.mod(1).ne(0).any()
        or not frame.direction.isin([-1, 1]).all()):
        raise ValueError("과거 주문 가격의 시각·포지션·종가 오류")
    orders = actions.action.ne("funding")
    if not np.isfinite(actions.loc[orders, "price"]).all() or actions.loc[orders, "price"].le(0).any():
        raise ValueError("과거 주문 가격의 비유한·비양수 체결")
    restored, old_ledger = attach_order_history(frame.drop(columns=HISTORY_FEATURES), actions)
    pd.testing.assert_frame_equal(restored[HISTORY_FEATURES], frame[HISTORY_FEATURES], check_exact=True)
    first = actions.assign(action_row=np.arange(len(actions))).loc[orders].drop_duplicates("order_key", keep="first")
    ledger = first.loc[first.action.isin(HISTORY_ACTIONS), [*old_ledger.columns, "price", "action_row"]].copy()
    pd.testing.assert_frame_equal(ledger[old_ledger.columns], old_ledger, check_exact=True)
    return ledger.reset_index(drop=True)


def attach_order_prices(frame, actions):
    if set(PRICE_FEATURES) & set(frame):
        raise ValueError("과거 주문 가격 입력의 중복 추가")
    ledger = price_ledger(frame, actions)
    result, links = frame.copy(), frame[["end", "episode_id"]].copy()
    ends = frame.end.astype("datetime64[ns, UTC]").array.asi8
    events = {(int(eid), action): part for (eid, action), part in ledger.groupby(["episode_id", "action"], sort=False)}
    feature_values = np.zeros((len(frame), 4), dtype=float)
    for i, action in enumerate(HISTORY_ACTIONS):
        references = np.full(len(frame), -1, dtype=np.int64)
        times = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
        prices = np.zeros(len(frame))
        for eid, locations in frame.groupby("episode_id", sort=False).indices.items():
            part = events.get((int(eid), action))
            if part is None:
                continue
            event_times = part.time.astype("datetime64[ns, UTC]").array.asi8
            # 현재 시각의 체결은 다음 판단부터 관측한다.
            before = np.searchsorted(event_times, ends[locations], side="left") - 1
            available = before >= 0
            chosen = part.iloc[before[available]]
            where = locations[available]
            if not np.array_equal(np.sign(chosen.before_qty.to_numpy()), frame.direction.to_numpy()[where]):
                raise ValueError("과거 주문 가격의 포지션 방향 불일치")
            references[where] = chosen.action_row.to_numpy()
            times[where] = event_times[before[available]]
            prices[where] = chosen.price.to_numpy()
        valid = references >= 0
        np.testing.assert_array_equal(valid, frame[f"past_{action}_exists"].to_numpy() == 1)
        move = np.zeros(len(frame))
        move[valid] = np.log(frame.close.to_numpy()[valid]) - np.log(prices[valid])
        feature_values[:, 2*i] = move
        feature_values[:, 2*i+1] = move * frame.direction.to_numpy()
        links[action + "_action_row"] = references
        links[action + "_time"] = pd.to_datetime(times, utc=True)
        links[action + "_price"] = prices
    if not np.isfinite(feature_values).all():
        raise ValueError("과거 주문 가격 입력의 연산 범위 오류")
    result[PRICE_FEATURES] = feature_values
    pd.testing.assert_frame_equal(result[frame.columns], frame, check_exact=True)
    return result, links, ledger


def price_support(frame):
    result = {}
    for action in HISTORY_ACTIONS:
        result[action] = {}
        for state, mask in groups(frame, action).items():
            part = frame.loc[mask]
            value = {"rows": len(part), "positive": int(part["y_" + action].sum()), "references": {}, "features": {}}
            for reference in HISTORY_ACTIONS:
                observed = part[f"past_{reference}_exists"].eq(1)
                value["references"][reference] = {"present": int(observed.sum()), "absent": int((~observed).sum()),
                    "positive_present": int(part.loc[observed, "y_" + action].sum()),
                    "positive_absent": int(part.loc[~observed, "y_" + action].sum())}
            for name in PRICE_FEATURES:
                p = part[name].to_numpy()
                value["features"][name] = None if not len(p) else {
                    "minimum": float(p.min()), "q25": float(np.quantile(p, .25)), "median": float(np.median(p)),
                    "q75": float(np.quantile(p, .75)), "maximum": float(p.max()), "zero": int((p == 0).sum())}
            result[action][state] = value
    return result
