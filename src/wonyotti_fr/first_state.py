from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import sha256
from .first_management import FIRST_ACTIONS, first_management_admission
from .history_state import (
    HISTORY_POLICY_FILES,
    HistoryStatePolicy,
    copy_current_parent,
    load_history_state_selection,
)

FIRST_POLICY_FILES = ['first_policy_thresholds.json', 'first_admission.json']


class FirstStatePolicy(HistoryStatePolicy):
    def __init__(self, parent, first_thresholds):
        super().__init__(parent, parent.manager, parent.thresholds)
        if (set(first_thresholds) != set(FIRST_ACTIONS)
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in first_thresholds.values())):
            raise ValueError('첫 자체 체결 관리의 문턱 오류')
        self.first_thresholds = dict(first_thresholds)

    def action_threshold(self, action, state):
        if action in FIRST_ACTIONS:
            exists = state['_fill_features'][0 if action == 'increase' else 3]
            if exists not in (0., 1.):
                raise ValueError('첫 자체 체결 관리의 과거 존재 여부 오류')
            if not exists:
                return self.first_thresholds[action]
        return super().action_threshold(action, state)


def validate_first_admission(evidence):
    decision = evidence['decision']
    expected = first_management_admission(evidence['metrics'], decision['unchanged_masks'])
    settings, summary = evidence['settings'], evidence['summary']
    if (not expected['first_thresholds_admitted'] or decision != expected
        or any(summary.get(k) != v for k, v in expected.items()) or summary.get('complete') is not True
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V38.md'))
        or settings.get('first_multiplier') != 1. or settings.get('new_models_fitted') is not False):
        raise ValueError('첫 자체 체결 관리의 사전 진단 통과 근거 오류')
    for action in FIRST_ACTIONS:
        for phase in ['first', 'repeat']:
            counts = set()
            for metric in evidence['metrics'][action][phase].values():
                n, positive = metric['rows'], metric['positive']
                if type(n) is not int or type(positive) is not int or min(positive, n-positive) < 20:
                    raise ValueError('첫 자체 체결 관리 진단의 지원 부족')
                counts.add((n, positive))
            if len(counts) != 1:
                raise ValueError('첫 자체 체결 관리 진단의 비교 표본 불일치')
    return expected


def copy_history_parent(reference, out):
    copy_current_parent(reference, out)
    for name in ['current_selection.json', *HISTORY_POLICY_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'history_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_first_parent(selection, frozen):
    path = selection / 'history_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['history_selection_sha256']:
        raise ValueError('첫 자체 체결 관리의 이전 선택 지문 오류')
    return load_history_state_selection(selection, json.loads(path.read_text()))


def load_first_state_selection(selection, frozen):
    parent, original = load_first_parent(selection, frozen)
    if (frozen.get('protocol') != 'first_state_v39'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['first_files_sha256']) != set(FIRST_POLICY_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['first_files_sha256'][n]
               for n in FIRST_POLICY_FILES)):
        raise ValueError('첫 자체 체결 관리의 고정 설정·모델 지문 오류')
    evidence = json.loads((selection / 'first_admission.json').read_text())
    validate_first_admission(evidence)
    value = json.loads((selection / 'first_policy_thresholds.json').read_text())
    if (value['first_multiplier'] != 1. or original.multiplier != 1.5 or value['multiplier'] != original.multiplier
        or value['calibration_period'] != ['2021-01-01', '2021-07-01']
        or value['minimum_predicted_positive'] != 20 or value['betas'] != {'reduce': 1., 'increase': .5}
        or evidence['settings']['selection_sha256'] != frozen['history_selection_sha256']):
        raise ValueError('첫 자체 체결 관리의 문턱·부모 연결 오류')
    return frozen, FirstStatePolicy(original, value['thresholds'])
