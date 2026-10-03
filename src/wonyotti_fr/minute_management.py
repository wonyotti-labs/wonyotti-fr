from __future__ import annotations

import numpy as np
import pandas as pd

from .event_features import MARKET_FEATURES, STATE_FEATURES, event_features, independent_orders

ACTIONS = ['exit', 'reduce', 'increase']
FEATURES = MARKET_FEATURES + [f'directional_{name}' for name in MARKET_FEATURES] + STATE_FEATURES


def purged_window(data: pd.DataFrame, start: str, stop: str) -> pd.DataFrame:
    first, last = pd.Timestamp(start, tz='UTC'), pd.Timestamp(stop, tz='UTC')
    if first >= last or last - first <= pd.Timedelta(days=2):
        raise ValueError('관리 학습 창의 시작·종료 오류')
    # 경계를 넘는 포지션의 앞부분도 제거해 같은 보유 이력이 양쪽에 섞이지 않게 한다.
    crossing = data.loc[data.entry_time.lt(last) & data.end.ge(last), 'episode_id'].unique()
    return data[data.usable & data.end.ge(first + pd.Timedelta(days=1)) & data.entry_time.ge(first)
                & data.label_end.lt(last - pd.Timedelta(days=1)) & ~data.episode_id.isin(crossing)].copy()


def management_orders(executions: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    orders = independent_orders(executions, actions)
    # 반전 행의 episode_id는 새 포지션이므로 청산 대상은 직전 행의 포지션이다.
    before = actions.assign(before_episode_id=np.where(actions.before_qty.ne(0), actions.episode_id.shift(fill_value=0), 0))
    first = before[before.action.ne('funding')].drop_duplicates('order_key', keep='first')
    result = orders.merge(first[['order_key', 'before_episode_id']], on='order_key', how='left', validate='one_to_one')
    if result.before_episode_id.isna().any() or (result.before_qty.ne(0) & result.before_episode_id.le(0)).any():
        raise ValueError('관리 주문의 직전 포지션 연결 오류')
    return result


def management_values(market, direction, favorable_move, hold_minutes, adds):
    market = np.asarray(market, dtype=float)
    return np.r_[market, market * direction, direction, favorable_move,
                 np.log1p(max(hold_minutes, 0)), min(adds, 5)]


def management_events(minute: pd.DataFrame, five: pd.DataFrame, states: pd.DataFrame,
                      orders: pd.DataFrame, last_source_time: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if (minute.end.duplicated().any() or not minute.end.is_monotonic_increasing
        or not minute.end.sub(minute.time).eq(pd.Timedelta(minutes=1)).all()
        or (minute.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=1).value).any()
        or not minute.end.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()):
        raise ValueError('관리 정답의 분봉 경계·연속성 오류')
    if (states.state_time.duplicated().any() or not states.state_time.is_monotonic_increasing
        or orders.order_key.duplicated().any() or not orders.target_time.is_monotonic_increasing):
        raise ValueError('관리 상태·독립 주문의 중복·순서 오류')
    features = event_features(five).rename(columns={'end': 'feature_end'})
    base = minute[['time', 'end', 'close']].copy().astype({'time': 'datetime64[ns, UTC]', 'end': 'datetime64[ns, UTC]'})
    frame = pd.merge_asof(base, features[['feature_end', *MARKET_FEATURES]], left_on='end', right_on='feature_end',
                          direction='backward', tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))
    frame = pd.merge_asof(frame, states, left_on='end', right_on='state_time', direction='backward', allow_exact_matches=False)
    frame[['direction', 'average_entry', 'adds', 'episode_id']] = frame[['direction', 'average_entry', 'adds', 'episode_id']].fillna(0)
    frame['favorable_move'] = (frame.close / frame.average_entry.replace(0, np.nan) - 1) * frame.direction
    frame['log_hold_minutes'] = np.log1p((frame.end - frame.entry_time).dt.total_seconds().div(60).clip(lower=0))
    frame['adds_capped'] = frame['adds'].clip(upper=5)
    for name in MARKET_FEATURES:
        frame[f'directional_{name}'] = frame[name] * frame.direction
    frame['label_end'] = frame.end + pd.Timedelta(minutes=1)
    frame['target_episode_id'] = frame.episode_id
    frame['usable'] = (frame.direction.isin([-1, 1]) & frame.label_end.le(last_source_time)
                       & np.isfinite(frame[FEATURES]).all(axis=1))
    ledger = orders.copy()
    ledger['window_end'] = ledger.target_time.dt.floor('min')
    ledger = ledger.merge(frame[['end', 'usable', 'direction', 'episode_id']], how='left',
                          left_on='window_end', right_on='end', validate='many_to_one')
    ledger['action'] = np.where(ledger.before_qty.ne(0) & ledger.target.isin(['enter_long', 'enter_short']), 'exit', ledger.target)
    # 분 안의 새 진입 이후 관리는 경계에서 이미 보유한 포지션의 정답과 구분한다.
    ledger['reason'] = np.select([
        ledger.end.isna(), ledger.before_qty.eq(0), ledger.direction.ne(np.sign(ledger.before_qty)),
        ledger.episode_id.ne(ledger.before_episode_id), ledger.usable.ne(True),
    ], ['outside_minutes', 'new_entry', 'different_direction', 'different_episode', 'unusable_features_or_range'], default='linked')
    selected = ledger[ledger.reason.eq('linked')]
    if not selected.action.isin(ACTIONS).all():
        raise ValueError('관리 정답에 지원하지 않는 행동')
    for action in ACTIONS:
        counts = selected[selected.action.eq(action)].groupby('window_end').size()
        frame[f'{action}_count'] = frame.end.map(counts).fillna(0).astype(int)
        frame[f'y_{action}'] = frame[f'{action}_count'].gt(0).astype(int)
        if int(frame[f'{action}_count'].sum()) != int(selected.action.eq(action).sum()):
            raise ValueError('분별 정답과 독립 주문 원장의 건수 불일치')
    summary = {'minute_windows': len(frame), 'usable_windows': int(frame.usable.sum()),
               'independent_orders': len(orders), 'linked_orders': len(selected),
               'reasons': {str(k): int(v) for k, v in ledger.reason.value_counts().items()},
               'action_order_counts': {a: int(frame[f'{a}_count'].sum()) for a in ACTIONS},
               'action_window_counts': {a: int(frame[f'y_{a}'].sum()) for a in ACTIONS},
               'multiple_action_windows': int(frame[[f'y_{a}' for a in ACTIONS]].sum(axis=1).gt(1).sum()),
               'multiple_order_windows': int(frame[[f'{a}_count' for a in ACTIONS]].sum(axis=1).gt(1).sum()),
               'rule': '1분 안의 모든 독립 관리 주문, 경계 직전 방향·에피소드 일치, 체결 시각 기준'}
    if sum(summary['reasons'].values()) != len(orders):
        raise ValueError('독립 주문 연결 원장의 합계 오류')
    return frame, ledger, summary
