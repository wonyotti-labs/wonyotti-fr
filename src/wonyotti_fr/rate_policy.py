from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import PolicyDecision
from .minute_management import ACTIONS
from .path_management import PATH_STATE, PathActionPolicy
from .pullback_policy import PullbackPolicy

RATE_STATE = {'rate_entry_time', *[f'rate_{a}' for a in ACTIONS]}


def validate_rate_state(stored):
    if not RATE_STATE <= set(stored) or not PATH_STATE <= set(stored):
        raise ValueError('행동 누적량의 저장 필드 누락')
    if stored['rate_entry_time'] != stored['path_entry_time']:
        raise ValueError('행동 누적량과 보유 가격 경로의 포지션 불일치')
    values = [stored[f'rate_{a}'] for a in ACTIONS]
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('행동 누적량의 저장 숫자 오류')


def calibrate_rates(model, data, period=('2020-07-01', '2021-01-01')):
    if period not in [('2020-07-01', '2021-01-01'), ('2021-07-01', '2022-01-01')]:
        raise ValueError('행동 빈도 보정의 사전 고정 기간 오류')
    first, last = (pd.Timestamp(value, tz='UTC') for value in period)
    if (data.end.min() < first + pd.Timedelta(days=1)
        or data.label_end.max() >= last - pd.Timedelta(days=1)):
        raise ValueError('행동 빈도 보정의 시간 격리 오류')
    scores = model.probabilities(data[model.features].to_numpy(dtype=float))
    labels = data[[f'y_{a}' for a in ACTIONS]].to_numpy()
    if (not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()
        or (labels.sum(axis=0) < 20).any() or (scores.sum(axis=0) <= 0).any()):
        raise ValueError('행동 빈도 보정의 지원 표본·점수 오류')
    scales = dict(zip(ACTIONS, (labels.sum(axis=0) / scores.sum(axis=0)).tolist(), strict=True))
    return {'scales': scales, 'rows': len(data), 'first_end': data.end.min(), 'last_label_end': data.label_end.max(),
            'positive_minutes': dict(zip(ACTIONS, labels.sum(axis=0).tolist(), strict=True)),
            'predicted_mass': dict(zip(ACTIONS, scores.sum(axis=0).tolist(), strict=True))}


class RateActionPolicy(PathActionPolicy):
    def __init__(self, entry, manager, thresholds, multiplier, scales):
        super().__init__(entry, manager, thresholds, multiplier)
        if (set(scales) != set(ACTIONS) or any(type(v) not in (int, float)
            or not np.isfinite(v) or not 0 < v <= 100 for v in scales.values())):
            raise ValueError('행동 빈도 보정 배율 오류')
        self.scales = dict(scales)

    def __call__(self, bar, state):
        if state['bar_seconds'] != 60:
            raise ValueError('분별 누적 정책의 실행 간격 오류')
        stored = state['policy_state']
        has_rate = bool(RATE_STATE & set(stored))
        if has_rate:
            validate_rate_state(stored)
        elif PATH_STATE & set(stored):
            raise ValueError('기존 보유 상태의 행동 누적량 누락')
        clean = {k: v for k, v in stored.items() if k not in RATE_STATE}
        clean_state = {**state, 'policy_state': clean}
        if not state['direction'] or state['halted'] or self.baseline == 'cash':
            return super().__call__(bar, clean_state)
        context, path = self.path_context(bar, clean_state)
        same = has_rate and stored['rate_entry_time'] == path['path_entry_time']
        counters = {a: stored[f'rate_{a}'] if same else 0. for a in ACTIONS}
        scores = self.manager.probabilities(self.feature_values(bar, context).reshape(1, -1))[0]
        available = np.isfinite(scores).all()
        if available:
            if ((scores < 0) | (scores > 1)).any():
                raise ValueError('다음 분 사건 점수의 범위 오류')
            for a, score in zip(ACTIONS, scores, strict=True):
                counters[a] = min(1., counters[a] + min(1., float(score) * self.scales[a]))
        memory = {'rate_entry_time': path['path_entry_time'], **{f'rate_{a}': counters[a] for a in ACTIONS}}
        existing = context['policy_state']
        current = pd.Timestamp(bar['end'])

        def decide(intent, control, event):
            return PolicyDecision(intent, {**control, **path, **memory}, event)

        if 'management_after' in existing:
            if set(existing) != {'management_after', 'management_direction'} or existing['management_direction'] not in (-1, 1):
                raise ValueError('누적 정책의 관리 대기 상태 오류')
            until = pd.Timestamp(existing['management_after'])
            if (pd.isna(until) or until.tzinfo is None or until.utcoffset().total_seconds()
                or until.value % pd.Timedelta(minutes=1).value or until > current + pd.Timedelta(minutes=3)):
                raise ValueError('누적 정책의 관리 대기 시각 오류')
            if existing['management_direction'] == state['direction'] and current < until:
                return decide('hold', existing, 'action_rate_cooldown')
        elif existing:
            # 지연 진입 전에 생긴 신규 대기는 기존 진입 연결 기록을 남기고 해제한다.
            if set(existing) != {'signal_time', 'expires_at', 'direction', 'reference_price'}:
                raise ValueError('누적 정책의 알 수 없는 관리 상태')
            decision = PullbackPolicy.__call__(self, bar, context)
            return decide(decision.intent, decision.state, decision.event)
        if state['pending'] != 'hold':
            return decide('hold', existing, 'action_rate_pending')
        if not available:
            return decide('hold', {}, 'action_rate_unavailable')
        for a in ACTIONS:
            if counters[a] >= 1. - 1e-12:
                memory[f'rate_{a}'] = 0.
                control = {'management_after': (current + pd.Timedelta(minutes=3)).isoformat(),
                           'management_direction': state['direction']}
                return decide(a, control, f'action_{a}')
        return decide('hold', {}, 'action_rate_hold')


class ReversalRatePolicy(RateActionPolicy):
    def __call__(self, bar, state):
        decision = super().__call__(bar, state)
        if decision.intent != 'exit':
            return decision
        direction = -state['direction']
        intent = 'enter_long' if direction > 0 else 'enter_short'
        # 누적량과 경로는 실제 반전 체결 후 새 포지션 식별 시각으로 초기화한다.
        stored = {**decision.state, 'management_direction': direction}
        return PolicyDecision(intent, stored, f'action_reverse_{"long" if direction > 0 else "short"}')
