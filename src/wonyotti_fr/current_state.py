from __future__ import annotations

import json

from .common import sha256
from .minute_inventory import MinuteInventoryPolicy
from .path_management import PathActionPolicy
from .probe_entry import copy_boosted_parent, load_probe_entry_selection
from .rate_policy import RATE_STATE


class CurrentStatePolicy(MinuteInventoryPolicy):
    def __init__(self, parent):
        super().__init__(parent, parent.manager, parent.thresholds, parent.multiplier, parent.scales, parent.size_model)

    def __call__(self, bar, state):
        if RATE_STATE & set(state['policy_state']):
            raise ValueError('현재 상태 정책에 이전 누적량 상태 연결 불가')
        # 확정된 현재 점수로 판단하고 기존 경로·대기·축소 크기를 보존한다.
        decision = PathActionPolicy.__call__(self, bar, state)
        return self.size_reduction(bar, state, decision)


def copy_probe_parent(reference, out):
    copy_boosted_parent(reference, out)
    (out / 'boost_selection.json').write_bytes((reference / 'boost_selection.json').read_bytes())
    (out / 'probe_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_current_parent(selection, frozen):
    path = selection / 'probe_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['probe_selection_sha256']:
        raise ValueError('현재 상태 정책의 이전 선택 지문 오류')
    parent = json.loads(path.read_text())
    if parent.get('protocol') != 'probe_entry_v29':
        raise ValueError('현재 상태 정책의 기반 모형 오류')
    return load_probe_entry_selection(selection, parent)


def load_current_selection(selection, frozen):
    parent, original = load_current_parent(selection, frozen)
    if (frozen.get('protocol') != 'current_state_v31'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})):
        raise ValueError('현재 상태 정책의 고정 설정 변경')
    return frozen, CurrentStatePolicy(original)
