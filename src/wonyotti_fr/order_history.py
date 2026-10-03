from __future__ import annotations

import numpy as np
import pandas as pd

from .minute_inventory import MinuteInventoryModels

HISTORY_ACTIONS = ['increase', 'reduce']
HISTORY_FEATURES = [f'past_{a}_{name}' for a in HISTORY_ACTIONS
                    for name in ['exists', 'log_minutes_since', 'log_count_15m']]


class OrderHistoryModels(MinuteInventoryModels):
    features = MinuteInventoryModels.features + HISTORY_FEATURES
    format = 'minute_order_history_action_v1'


def attach_order_history(frame, actions):
    if (frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or frame.end.isna().any() or actions.time.isna().any() or not actions.time.is_monotonic_increasing
        or not np.isfinite(actions[['before_qty', 'after_qty', 'episode_id']]).all().all()
        or (actions.episode_id < 0).any() or actions.episode_id.mod(1).ne(0).any()
        or not actions.before_qty.iloc[1:].reset_index(drop=True).equals(actions.after_qty.iloc[:-1].reset_index(drop=True))):
        raise ValueError('과거 관리 주문의 시각·포지션·수량 연속성 오류')
    known = {'open', 'close', 'reverse', 'increase', 'reduce', 'funding'}
    if not actions.action.isin(known).all():
        raise ValueError('과거 관리 주문의 지원하지 않는 행동')
    for action in HISTORY_ACTIONS:
        selected = actions.loc[actions.action.eq(action)]
        same = selected.before_qty.ne(0) & (np.sign(selected.before_qty) == np.sign(selected.after_qty))
        changed = (selected.after_qty.abs() > selected.before_qty.abs()) if action == 'increase' else (selected.after_qty.abs() < selected.before_qty.abs())
        if not (same & changed & selected.episode_id.gt(0)).all():
            raise ValueError('과거 관리 주문의 관측 행동·수량 불일치')
    orders = actions.loc[actions.action.ne('funding')]
    if orders.order_key.isna().any() or orders.order_key.astype(str).str.len().eq(0).any():
        raise ValueError('과거 관리 주문의 식별자 누락')
    # 전체 주문의 최종 의도 대신 첫 체결에서 이미 관측한 행동만 사용한다.
    first = orders.drop_duplicates('order_key', keep='first')
    ledger = first[first.action.isin(HISTORY_ACTIONS)][['time', 'order_key', 'action', 'episode_id', 'before_qty', 'after_qty']].copy()
    result = frame.copy()
    ends = frame.end.astype('datetime64[ns, UTC]').array.asi8
    features = np.zeros((len(frame), len(HISTORY_FEATURES)), dtype=float)
    events = {(int(eid), action): part.time.astype('datetime64[ns, UTC]').array.asi8
              for (eid, action), part in ledger.groupby(['episode_id', 'action'], sort=False)}
    for eid, locations in frame.groupby('episode_id', sort=False).indices.items():
        if eid <= 0:
            continue
        current = ends[locations]
        for i, action in enumerate(HISTORY_ACTIONS):
            times = events.get((int(eid), action))
            if times is None:
                continue
            before = np.searchsorted(times, current, side='left')
            available = before > 0
            count = before - np.searchsorted(times, current - pd.Timedelta(minutes=15).value, side='left')
            features[locations, 3*i] = available
            features[locations[available], 3*i+1] = np.log1p((current[available] - times[before[available]-1]) / pd.Timedelta(minutes=1).value)
            features[locations, 3*i+2] = np.log1p(count)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError('과거 관리 주문 입력의 유한 범위 오류')
    result[HISTORY_FEATURES] = features
    pd.testing.assert_frame_equal(result[frame.columns], frame, check_exact=True)
    return result, ledger
