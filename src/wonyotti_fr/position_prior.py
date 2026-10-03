from __future__ import annotations

import json

import numpy as np

from .common import sha256
from .expansion_model import BinaryModel
from .new_position import POSITION_FILES, copy_recent_parent, load_new_position_selection
from .recent_entry import ENTRY_PERIOD, RecentEntryPolicy

PRIOR_FILES = ['position_direction_model.json', 'direction_prior.json']


def prior_offset(old, new):
    for counts in [old, new]:
        if (set(counts) != {'buy', 'sell'} or any(type(v) is not int or v < 50 for v in counts.values())):
            raise ValueError('방향 빈도 보정의 양쪽 지원 표본 부족')
    if sum(old.values()) < 1000 or sum(new.values()) < 200:
        raise ValueError('방향 빈도 보정의 전체 지원 표본 부족')
    return float(np.log(new['buy'] / new['sell']) - np.log(old['buy'] / old['sell']))


def adjust_direction(model, old, new):
    if model.data['kind'] != 'logistic':
        raise ValueError('방향 빈도 보정은 고정 로지스틱 모형만 지원합니다.')
    data = model.to_dict()
    data['intercept'] += prior_offset(old, new)
    return BinaryModel.from_dict(data)


def copy_position_parent(reference, out):
    copy_recent_parent(reference, out)
    for name in ['recent_selection.json', *POSITION_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'position_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_position_prior_selection(selection, frozen):
    path = selection / 'position_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['position_selection_sha256']:
        raise ValueError('방향 빈도 보정의 기존 활동 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'position_prior_v26' or parent.get('protocol') != 'new_position_v25'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['prior_files_sha256']) != set(PRIOR_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['prior_files_sha256'][n]
               for n in PRIOR_FILES)):
        raise ValueError('방향 빈도 보정의 고정 설정·모델 지문 오류')
    _, original = load_new_position_selection(selection, parent)
    metadata = json.loads((selection / 'direction_prior.json').read_text())
    if metadata['training_period'] != ENTRY_PERIOD:
        raise ValueError('방향 빈도 보정의 학습 기간 오류')
    offset = prior_offset(metadata['old_counts'], metadata['new_counts'])
    if offset != metadata['offset'] or offset != frozen['direction_prior_offset']:
        raise ValueError('방향 빈도 보정의 절편 이동 오류')
    expected = adjust_direction(original.base.direction, metadata['old_counts'], metadata['new_counts'])
    direction = BinaryModel.from_dict(json.loads((selection / 'position_direction_model.json').read_text()))
    if direction.to_dict() != expected.to_dict():
        raise ValueError('방향 빈도 보정 외 모형 변경')
    return frozen, RecentEntryPolicy(original, original.base.activity, direction, original.base.activity_threshold)
