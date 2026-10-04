from __future__ import annotations

import json
from pathlib import Path

from .common import sha256
from .expansion_model import ExpansionPolicy
from .first_state import FirstStatePolicy
from .holding_support import copy_first_parent, load_holding_selection


def without_activity_gate(original):
    base = original.base
    if not 0 < base.activity_threshold <= 1 or base.direction_threshold != .65 or base.min_hold_bars != 12:
        raise ValueError('활동 관문 제거의 기존 진입 설정 오류')
    candidate = FirstStatePolicy(original, original.first_thresholds)
    candidate.base = ExpansionPolicy(base.activity, base.direction, 0., base.direction_threshold, base.min_hold_bars)
    return candidate


def copy_holding_parent(reference, out):
    copy_first_parent(reference, out)
    for name in ['first_selection.json', 'holding_support.json']:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'holding_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_activity_parent(selection, frozen):
    path = selection / 'holding_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['holding_selection_sha256']:
        raise ValueError('활동 관문 제거의 이전 선택 지문 오류')
    return load_holding_selection(selection, json.loads(path.read_text()))


def load_activity_ablation_selection(selection, frozen):
    parent, original = load_activity_parent(selection, frozen)
    if (frozen.get('protocol') != 'activity_ablation_v46' or parent.get('protocol') != 'holding_support_v40'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or frozen.get('activity_gate_disabled') is not True
        or type(frozen.get('effective_activity_threshold')) is not float or frozen['effective_activity_threshold'] != 0.
        or frozen.get('ablation_protocol_sha256') != sha256(Path('docs/EXPERIMENT_V46.md'))):
        raise ValueError('활동 관문 제거의 고정 모델·문턱·위험 오류')
    return frozen, without_activity_gate(original)

