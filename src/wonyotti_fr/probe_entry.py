from __future__ import annotations

import json
from dataclasses import replace

from .boosted_direction import BOOST_FILES, copy_direction_parent, load_boosted_direction_selection
from .common import sha256
from .engine import EngineConfig


def probe_risks(parent):
    risk = EngineConfig(**parent['risk'])
    if risk.allocation != .5 or risk.entry_fraction != .5 or risk.addition_fraction != .25:
        raise ValueError('최초 노출 분리의 고정 기반 위험 설정 오류')
    return replace(risk, entry_fraction=.125), risk, replace(risk, allocation=.125)


def copy_boosted_parent(reference, out):
    copy_direction_parent(reference, out)
    for name in ['direction_selection.json', *BOOST_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'boost_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_probe_parent(selection, frozen):
    path = selection / 'boost_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['boost_selection_sha256']:
        raise ValueError('최초 노출 분리의 이전 선택 지문 오류')
    parent = json.loads(path.read_text())
    if parent.get('protocol') != 'boosted_direction_v28':
        raise ValueError('최초 노출 분리의 기반 모형 오류')
    return load_boosted_direction_selection(selection, parent)


def load_probe_entry_selection(selection, frozen):
    parent, original = load_probe_parent(selection, frozen)
    risk, _, _ = probe_risks(parent)
    if (frozen.get('protocol') != 'probe_entry_v29' or frozen.get('risk') != risk.__dict__
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'risk', 'development_metrics'})):
        raise ValueError('최초 노출 분리의 고정 설정 변경')
    return frozen, original
