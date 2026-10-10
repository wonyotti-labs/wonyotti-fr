from __future__ import annotations

import numpy as np

from wonyotti_fr.exit_state import ExitStatePolicy
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


class DisabledManagementScores:
    features = OrderHistoryBoostModels.features

    def probabilities(self, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError('보유 관리 중지 대조의 특징 차원 오류')
        result = np.zeros((len(matrix), 3))
        result[~np.isfinite(matrix).all(axis=1)] = np.nan
        return result


class HoldOnlyControl:
    def __init__(self, parent, enabled=True):
        if not isinstance(parent, ExitStatePolicy) or type(enabled) is not bool:
            raise ValueError('보유 관리 중지는 기존 직접 청산 정책의 대조만 지원')
        thresholds = [*parent.thresholds.values(), *parent.first_thresholds.values()]
        if not thresholds or not np.isfinite(thresholds).all() or min(thresholds) <= 0:
            raise ValueError('보유 관리 중지의 원래 문턱은 양수여야 함')
        self.parent, self.enabled = parent, enabled
        self.disabled = DisabledManagementScores()

    def prepare(self, bars):
        self.parent.prepare(bars)

    def __call__(self, bar, state):
        if not self.enabled:
            return self.parent(bar, state)
        original = self.parent.manager
        self.parent.manager = self.disabled
        try:
            # 부모의 실제 체결 이력·가격 경로·진입 대기는 그대로 갱신한다.
            decision = self.parent(bar, state)
            if decision.intent in {'increase', 'reduce', 'exit'}:
                raise ValueError('보유 관리 중지 후 관리 요청이 남음')
            return decision
        finally:
            self.parent.manager = original
