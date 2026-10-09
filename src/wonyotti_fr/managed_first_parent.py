from __future__ import annotations

import json
from pathlib import Path

from .close_effect import CLOSE_FEATURES
from .common import sha256
from .event_research import load_selection
from .history_state import CalibratedHistoryModels
from .management_calibration import ManagementOffset
from .order_history_boost import OrderHistoryBoostModels


def pinned_json(path, expected):
    if path.is_symlink() or path.parent.is_symlink() or sha256(path) != expected:
        raise ValueError('고정 관리 모델의 메타데이터 연결 지문 오류')
    return json.loads(path.read_text())


def checked_manager_parent(reference):
    prior = json.loads((reference/'reference_evidence.json').read_text())
    source = Path(prior['reference'])
    metadata = {}
    for name in ['manifest.json', 'expansion_evidence.json']:
        path = source/name
        metadata['linear_'+name] = {'path': str(path), 'sha256': prior['files'][name]}
    manifest = pinned_json(source/'manifest.json', prior['files']['manifest.json'])
    expansion = pinned_json(source/'expansion_evidence.json', prior['files']['expansion_evidence.json'])
    if manifest['settings']['expansion'] != expansion['expansion']:
        raise ValueError('고정 관리 모델의 확장 참조 불일치')
    path = Path(expansion['expansion'])/'manifest.json'
    generation = pinned_json(path, expansion['files']['manifest.json'])['settings']
    metadata['generation_manifest.json'] = {'path': str(path), 'sha256': expansion['files']['manifest.json']}
    parent = Path(generation['reference'])
    frozen = pinned_json(parent/'frozen_selection.json', generation['reference_sha256'])
    if frozen['protocol'] != 'exit_move_v54':
        raise ValueError('고정 관리 모델의 원래 부모 규약 오류')
    loaded, policy = load_selection(parent)
    if loaded != frozen or policy.manager.features != CLOSE_FEATURES:
        raise ValueError('고정 관리 모델의 실제 부모·특징 연결 오류')
    model, offset = policy.manager.model.to_dict(), policy.manager.offset.to_dict()
    manager = CalibratedHistoryModels(OrderHistoryBoostModels.from_dict(model), ManagementOffset.from_dict(offset))
    return manager, {'parent': str(parent), 'parent_sha256': generation['reference_sha256'], 'metadata': metadata,
        'model': model, 'offset': offset, 'management_refitted': False, 'source_current_features': CLOSE_FEATURES}
