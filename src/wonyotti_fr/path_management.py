from __future__ import annotations

import numpy as np
import pandas as pd

from .action_model import ActionModels, MinuteActionPolicy
from .engine import PolicyDecision
from .minute_management import FEATURES

PATH_FEATURES = ['best_close_move', 'worst_close_move', 'giveback_from_best']
PATH_STATE = {'path_entry_time', 'path_direction', 'path_low', 'path_high', 'path_end'}


def price_path_values(direction, average_entry, close, low, high):
    best = np.where(direction > 0, high, low)
    worst = np.where(direction > 0, low, high)
    favorable = direction * (close / average_entry - 1)
    best_move = direction * (best / average_entry - 1)
    worst_move = direction * (worst / average_entry - 1)
    return best_move, worst_move, best_move - favorable


def add_price_path(frame: pd.DataFrame) -> pd.DataFrame:
    if (not frame.end.is_monotonic_increasing or frame.end.duplicated().any()
        or not frame.end.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()):
        raise ValueError('보유 가격 경로의 시간 경계 오류')
    result = frame.copy()
    held = frame.direction.isin([-1, 1])
    if (frame.loc[held, 'average_entry'].le(0).any() or frame.close.le(0).any()
        or not np.isfinite(frame.loc[held, 'average_entry']).all() or not np.isfinite(frame.close).all()
        or frame.loc[held, 'entry_time'].isna().any() or frame.loc[held, 'episode_id'].le(0).any()
        or frame.loc[held, 'entry_time'].ge(frame.loc[held, 'end']).any()):
        raise ValueError('보유 가격 경로의 상태·가격 오류')
    # 전체 시계열에서 누적한 뒤 구간을 나눠야 구간 시작 전의 알려진 경로가 보존된다.
    prices = frame.close.where(held)
    low = prices.groupby(frame.episode_id).cummin()
    high = prices.groupby(frame.episode_id).cummax()
    values = price_path_values(frame.direction, frame.average_entry.replace(0, np.nan), frame.close, low, high)
    for name, value in zip(PATH_FEATURES, values, strict=True):
        result[name] = value
    result['usable'] &= np.isfinite(result[PATH_FEATURES]).all(axis=1)
    return result


class PathActionModels(ActionModels):
    features = FEATURES + PATH_FEATURES
    format = 'minute_path_action_v1'


def validate_path_state(stored, previous_end=None):
    if not PATH_STATE <= set(stored):
        raise ValueError('보유 가격 경로의 저장 필드 누락')
    entry, end = pd.Timestamp(stored['path_entry_time']), pd.Timestamp(stored['path_end'])
    if (pd.isna(entry) or pd.isna(end) or entry.tzinfo is None or end.tzinfo is None
        or entry.utcoffset().total_seconds() or end.utcoffset().total_seconds()
        or entry >= end or end.value % pd.Timedelta(minutes=1).value
        or (previous_end is not None and end != pd.Timestamp(previous_end))):
        raise ValueError('보유 가격 경로의 저장 시각 오류')
    low, high, direction = (stored[k] for k in ['path_low', 'path_high', 'path_direction'])
    if (type(direction) is not int or direction not in (-1, 1) or type(low) not in (float, int)
        or type(high) not in (float, int) or not np.isfinite([low, high]).all() or not 0 < low <= high):
        raise ValueError('보유 가격 경로의 저장 숫자 오류')


class PathActionPolicy(MinuteActionPolicy):
    def feature_values(self, bar, state):
        low, high = state['_path_bounds']
        return np.r_[super().feature_values(bar, state), price_path_values(
            state['direction'], state['average_entry'], bar['close'], low, high)]

    def path_context(self, bar, state):
        stored = state['policy_state']
        has_path = bool(PATH_STATE & set(stored))
        if has_path:
            validate_path_state(stored, pd.Timestamp(bar['end']) - pd.Timedelta(minutes=1))
        clean = {k: v for k, v in stored.items() if k not in PATH_STATE}
        base_state = {**state, 'policy_state': clean}
        if not state['direction'] or state['halted'] or self.baseline == 'cash':
            return base_state, {}
        entry, end = pd.Timestamp(state['position_entry_time']), pd.Timestamp(bar['end'])
        if (pd.isna(entry) or entry.tzinfo is None or entry.utcoffset().total_seconds() or entry >= end
            or not np.isfinite(state['average_entry']) or state['average_entry'] <= 0):
            raise ValueError('보유 가격 경로의 현재 진입 상태 오류')
        same = has_path and pd.Timestamp(stored['path_entry_time']) == entry and stored['path_direction'] == state['direction']
        if not same and state['hold_bars'] > 1:
            raise ValueError('이미 보유한 포지션의 가격 경로 누락')
        low = min(stored['path_low'], bar['close']) if same else bar['close']
        high = max(stored['path_high'], bar['close']) if same else bar['close']
        path = {'path_entry_time': entry.isoformat(), 'path_direction': state['direction'],
                'path_low': float(low), 'path_high': float(high), 'path_end': end.isoformat()}
        base_state['_path_bounds'] = low, high
        return base_state, path

    def __call__(self, bar, state):
        base_state, path = self.path_context(bar, state)
        decision = super().__call__(bar, base_state)
        return PolicyDecision(decision.intent, {**decision.state, **path}, decision.event)


class ReversalPathPolicy(PathActionPolicy):
    def __call__(self, bar, state):
        decision = super().__call__(bar, state)
        if decision.intent != 'exit':
            return decision
        direction = -state['direction']
        intent = 'enter_long' if direction > 0 else 'enter_short'
        # 반전 체결 이후의 새 방향에도 중복 관리 대기를 유지한다.
        stored = {**decision.state, 'management_direction': direction}
        return PolicyDecision(intent, stored, f'action_reverse_{"long" if direction > 0 else "short"}')
