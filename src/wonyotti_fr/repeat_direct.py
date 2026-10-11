from __future__ import annotations

from .exit_state import ExitStatePolicy


class RepeatDirectPolicy(ExitStatePolicy):
    def __init__(self, parent):
        if type(parent) is not ExitStatePolicy or parent.multiplier != 1.5:
            raise ValueError("반복 직접 문턱에는 원래 v48 정책이 필요함")
        super().__init__(parent)

    def action_threshold(self, action, state):
        if action in {"reduce", "increase"}:
            exists = state["_fill_features"][3 if action == "reduce" else 0]
            if exists not in (0.0, 1.0):
                raise ValueError("반복 직접 문턱의 과거 체결 상태 오류")
            if exists == 1:
                return self.thresholds[action]
        return super().action_threshold(action, state)


class ReferenceRepeatPolicy:
    def __init__(self, parent):
        if type(parent) is not ExitStatePolicy or parent.multiplier != 1.5:
            raise ValueError("반복 문턱 검산에는 원래 v48 정책이 필요함")
        self.parent = parent

    def prepare(self, bars):
        self.parent.prepare(bars)

    def __call__(self, bar, state):
        original = self.parent.action_threshold
        had_instance = "action_threshold" in vars(self.parent)
        previous = vars(self.parent).get("action_threshold")

        def threshold(action, current):
            old = original(action, current)
            if action == "reduce" and current["_fill_features"][3] == 1:
                return self.parent.thresholds["reduce"]
            if action == "increase" and current["_fill_features"][0] == 1:
                return self.parent.thresholds["increase"]
            return old

        # 검산은 원래 정책의 문턱 호출만 감싸며 예외 후에도 부모를 복원한다.
        self.parent.action_threshold = threshold
        try:
            return self.parent(bar, state)
        finally:
            if had_instance:
                self.parent.action_threshold = previous
            else:
                del self.parent.action_threshold
