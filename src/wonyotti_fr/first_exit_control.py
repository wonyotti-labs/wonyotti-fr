from __future__ import annotations

import numpy as np

from .close_effect import CLOSE_FEATURES
from .engine import PolicyDecision


class FirstExitControl:
    def __init__(self, parent, *, enabled=True, record=None):
        if type(enabled) is not bool or parent.manager.features != CLOSE_FEATURES or isinstance(parent, FirstExitControl):
            raise ValueError('첫 청산 대조의 부모·활성 설정 오류')
        self.parent, self.enabled, self.record = parent, enabled, record

    def prepare(self, data):
        self.parent.prepare(data)

    def __call__(self, bar, state):
        if not self.enabled:
            return self.parent(bar, state)
        captured = []
        original = self.parent.feature_values

        def capture(current, view):
            values = original(current, view)
            captured.append(np.asarray(values, dtype=float).copy())
            return values

        # 부모가 실제 계산한 입력만 관찰하고 호출이 실패해도 원래 메서드를 복원한다.
        self.parent.feature_values = capture
        try:
            decision = self.parent(bar, state)
        finally:
            self.parent.feature_values = original
        if not isinstance(decision, PolicyDecision) or len(captured) > 1:
            raise ValueError('첫 청산 대조의 부모 판단·특징 호출 오류')
        if captured and captured[0].shape != (len(CLOSE_FEATURES),):
            raise ValueError('첫 청산 대조의 현재 특징 차원 오류')
        supported = bool(captured and np.isfinite(captured[0]).all())
        reason = ('flat' if not state['direction'] else 'halted' if state['halted'] else
            'no_management_call' if not captured else 'nonfinite_features' if not supported else
            'original_exit' if decision.intent == 'exit' else 'first_eligible_exit')
        changed = reason == 'first_eligible_exit'
        if changed and decision.intent not in {'hold', 'increase', 'reduce'}:
            raise ValueError('첫 청산 대조의 지원하지 않는 보유 의도')
        result = PolicyDecision('exit', decision.state, 'first_exit_control') if changed else decision
        if self.record is not None:
            self.record({'decision_time': bar['end'], 'direction': state['direction'],
                'feature_calls': len(captured), 'features_supported': supported,
                'original_intent': decision.intent, 'final_intent': result.intent,
                'original_event': decision.event, 'changed': changed, 'reason': reason})
        return result
