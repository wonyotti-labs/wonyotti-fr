from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import save_json, sha256
from .horizon_exit_state import HORIZON_EXIT_FILES, HorizonExitPolicy, load_horizon_exit_selection
from .recent_history import copy_exit_parent

NET_EXIT_RULE = {'format': 'current_net_exit_guard_v1', 'minimum_estimated_exit_net': 0.,
    'valuation_price': 'current_closed_bar', 'includes': ['realized_partial_pnl', 'remaining_quantity_pnl',
        'paid_fees', 'paid_funding', 'estimated_exit_slippage', 'estimated_exit_fee'],
    'negative_net_threshold': 'original_v48', 'nonnegative_net_threshold': 'horizon_v52',
    'original_exit_and_risk_exits_preserved': True, 'future_fill_price_used': False, 'new_models_fitted': False}


class NetExitPolicy(HorizonExitPolicy):
    def __init__(self, parent):
        super().__init__(parent, parent.horizon_threshold)
        if self.horizon_threshold > self.thresholds['exit']:
            raise ValueError('현재 비용 청산의 범위 문턱은 기존 문턱 이하여야 합니다.')

    def action_threshold(self, action, state):
        if action == 'exit':
            value = state.get('estimated_exit_net')
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError('현재 비용 청산의 추정 순손익 누락·숫자 오류')
            return self.horizon_threshold if value >= 0 else self.thresholds['exit']
        return super().action_threshold(action, state)


def copy_horizon_parent(reference, out):
    copy_exit_parent(reference, out)
    for name in ['exit_selection.json', *HORIZON_EXIT_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'horizon_exit_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
    save_json(out / 'net_exit_rule.json', {**NET_EXIT_RULE, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V53.md'))})


def load_net_exit_parent(selection, frozen):
    path = selection / 'horizon_exit_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['horizon_exit_selection_sha256']:
        raise ValueError('현재 비용 청산의 이전 선택 지문 오류')
    return load_horizon_exit_selection(selection, json.loads(path.read_text()))


def load_net_exit_selection(selection, frozen):
    parent, original = load_net_exit_parent(selection, frozen)
    path = selection / 'net_exit_rule.json'
    if (frozen.get('protocol') != 'net_exit_v53' or parent['protocol'] != 'horizon_exit_v52'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or path.stat().st_size > 1024**2 or sha256(path) != frozen['net_exit_rule_sha256']
        or json.loads(path.read_text()) != {**NET_EXIT_RULE, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V53.md'))}):
        raise ValueError('현재 비용 청산의 고정 모델·위험·규칙 오류')
    return frozen, NetExitPolicy(original)
