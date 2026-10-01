from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import PolicyDecision


class PullbackPolicy:
    def __init__(self, base, offset_bps: int, ttl_minutes: int, baseline: str | None = None):
        if (type(offset_bps) is not int or offset_bps not in (8, 16, 32)
            or type(ttl_minutes) is not int or ttl_minutes not in (5, 15)
            or baseline not in (None, 'cash', 'immediate')):
            raise ValueError('진입 대기 후보의 기준 오류')
        self.base, self.offset_bps, self.ttl_minutes, self.baseline = base, offset_bps, ttl_minutes, baseline

    def prepare(self, frame):
        boundary = frame.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=5).value == 0
        self.base.prepare(frame.loc[boundary])

    def __call__(self, bar: dict, state: dict) -> PolicyDecision:
        if state['bar_seconds'] != 60:
            raise ValueError('진입 대기 정책은 1분 실행만 지원합니다.')
        current = pd.Timestamp(bar['end'])
        waiting = state['policy_state']
        if state['direction'] or state['halted'] or self.baseline == 'cash':
            return PolicyDecision('hold', {}, 'cleared' if waiting else 'idle')
        if waiting:
            if set(waiting) != {'signal_time', 'expires_at', 'direction', 'reference_price'}:
                raise ValueError('진입 대기 상태의 필드 오류')
            first, expiry = (pd.Timestamp(waiting[key]) for key in ['signal_time', 'expires_at'])
            if (type(waiting['direction']) is not int or waiting['direction'] not in (-1, 1)
                or type(waiting['reference_price']) not in (int, float) or not np.isfinite(waiting['reference_price'])
                or waiting['reference_price'] <= 0 or first.tzinfo is None or expiry.tzinfo is None
                or first.utcoffset().total_seconds() != 0 or expiry.utcoffset().total_seconds() != 0
                or first.value % pd.Timedelta(minutes=5).value
                or expiry - first != pd.Timedelta(minutes=self.ttl_minutes) or current <= first):
                raise ValueError('진입 대기 상태의 시간·방향·가격 오류')
            favorable = waiting['direction'] * np.log(waiting['reference_price'] / bar['close']) * 10000
            if current <= expiry and favorable >= self.offset_bps:
                intent = 'enter_long' if waiting['direction'] > 0 else 'enter_short'
                return PolicyDecision(intent, {}, 'triggered')
            if current >= expiry:
                return PolicyDecision('hold', {}, 'expired')
            return PolicyDecision('hold', waiting, 'waiting')
        if state['pending'] != 'hold' or current.value % pd.Timedelta(minutes=5).value:
            return PolicyDecision('hold', {}, 'idle')
        intent = self.base(bar, state)
        if intent not in ('enter_long', 'enter_short'):
            return PolicyDecision('hold', {}, 'no_signal')
        if self.baseline == 'immediate':
            return PolicyDecision(intent, {}, 'immediate')
        waiting = {'signal_time': current.isoformat(),
                   'expires_at': (current + pd.Timedelta(minutes=self.ttl_minutes)).isoformat(),
                   'direction': 1 if intent == 'enter_long' else -1, 'reference_price': float(bar['close'])}
        return PolicyDecision('hold', waiting, 'armed')
