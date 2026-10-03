from __future__ import annotations

from .expansion_model import ExpansionPolicy
from .pullback_policy import PullbackPolicy
from .rate_policy import RateActionPolicy


def direction_only_manager(policy: RateActionPolicy) -> RateActionPolicy:
    base = policy.base
    if (not isinstance(policy, RateActionPolicy) or not isinstance(base, ExpansionPolicy)
        or base.activity_threshold <= 0 or (policy.offset_bps, policy.ttl_minutes, policy.baseline) != (16, 5, None)):
        raise ValueError('방향 기반 기회의 고정 관리·진입 기반 오류')
    # 기존 객체를 수정하지 않아 원래 활동 관문을 쓰는 대조를 보존한다.
    direction = ExpansionPolicy(base.activity, base.direction, 0., base.direction_threshold, base.min_hold_bars)
    entry = PullbackPolicy(direction, policy.offset_bps, policy.ttl_minutes)
    return RateActionPolicy(entry, policy.manager, policy.thresholds, policy.multiplier, policy.scales)
