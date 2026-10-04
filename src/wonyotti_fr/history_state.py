from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import sha256
from .current_state import CurrentStatePolicy, copy_probe_parent, load_current_selection
from .engine import PolicyDecision
from .history_calibration_diagnostics import history_calibration_admission
from .management_calibration import ManagementOffset
from .minute_inventory import MinuteInventoryPolicy
from .order_history import HISTORY_ACTIONS
from .order_history_boost import OrderHistoryBoostModels
from .path_management import PATH_STATE, validate_path_state

FILL_STATE = {'fill_adds', 'fill_fraction', 'fill_increase_last', 'fill_reduce_last',
              'fill_increase_mask', 'fill_reduce_mask'}
HISTORY_POLICY_FILES = ['history_manager.json', 'history_offset.json', 'history_admission.json',
                        'history_thresholds.json']
MINUTE_NS = pd.Timedelta(minutes=1).value


def validate_fill_history(stored):
    if not FILL_STATE <= set(stored):
        raise ValueError('자체 체결 이력의 저장 필드 누락')
    if (type(stored['fill_adds']) is not int or not 0 <= stored['fill_adds'] <= 5
        or type(stored['fill_fraction']) not in (int, float) or not np.isfinite(stored['fill_fraction'])
        or not 0 < stored['fill_fraction'] <= 1):
        raise ValueError('자체 체결 이력의 저장 수량 오류')
    end = pd.Timestamp(stored['path_end']).value // MINUTE_NS
    entry = pd.Timestamp(stored['path_entry_time']).value // MINUTE_NS
    for action in HISTORY_ACTIONS:
        last, mask = (stored[f'fill_{action}_{key}'] for key in ['last', 'mask'])
        if (type(last) is not int or type(mask) is not int or not 0 <= mask < 2**15
            or (last != 0 and not entry < last < end)):
            raise ValueError('자체 체결 이력의 저장 시각·범위 오류')
        age = end - last
        if (last == 0 and mask != 0) or (last and age > 15 and mask != 0):
            raise ValueError('자체 체결 이력의 최근 범위 불일치')
        if last and age <= 15 and (not mask or (mask & -mask) != 1 << (age - 1)):
            raise ValueError('자체 체결 이력의 마지막 체결 불일치')
        if any(mask & (1 << bit) and end - bit - 1 <= entry for bit in range(15)):
            raise ValueError('자체 체결 이력의 진입 전 사건')
    if stored['fill_increase_mask'] & stored['fill_reduce_mask']:
        raise ValueError('같은 분의 복수 부분 체결 오류')
    if (bool(stored['fill_adds']) != bool(stored['fill_increase_last'])
        or stored['fill_increase_mask'].bit_count() > stored['fill_adds']):
        raise ValueError('자체 체결 이력의 추가 횟수 불일치')


def fill_context(bar, state):
    stored = state['policy_state']
    has_path, has_history = bool(PATH_STATE & set(stored)), bool(FILL_STATE & set(stored))
    if has_path != has_history:
        raise ValueError('자체 체결 이력과 보유 경로의 연결 누락')
    if has_path:
        validate_path_state(stored, pd.Timestamp(bar['end']) - pd.Timedelta(minutes=1))
        validate_fill_history(stored)
    clean = {**state, 'policy_state': {k: v for k, v in stored.items() if k not in FILL_STATE}}
    if not state['direction'] or state['halted']:
        return clean, {}, np.zeros(6)
    adds, fraction = state['adds'], state['remaining_fraction']
    if (type(adds) is not int or not 0 <= adds <= 5 or type(fraction) not in (int, float)
        or not np.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('자체 체결 이력의 현재 수량 오류')
    same = (has_path and pd.Timestamp(stored['path_entry_time']) == pd.Timestamp(state['position_entry_time'])
            and stored['path_direction'] == state['direction'])
    history = {k: stored[k] for k in FILL_STATE} if same else dict.fromkeys(FILL_STATE, 0)
    if not same and (state['hold_bars'] != 1 or adds != 0 or fraction != 1.):
        raise ValueError('새 포지션의 자체 체결 이력 누락')
    changes = dict.fromkeys(HISTORY_ACTIONS, False)
    if same:
        delta = adds - stored['fill_adds']
        if (delta not in (0, 1) or (delta == 1 and fraction < stored['fill_fraction'])
            or (delta == 0 and fraction > stored['fill_fraction'])):
            raise ValueError('한 분의 자체 체결 수량 변화 오류')
        changes['increase'] = delta == 1
        changes['reduce'] = delta == 0 and fraction < stored['fill_fraction']
    end = pd.Timestamp(bar['end']).value // MINUTE_NS
    values = []
    for action in HISTORY_ACTIONS:
        last_key, mask_key = f'fill_{action}_last', f'fill_{action}_mask'
        history[mask_key] = ((history[mask_key] << 1) | int(changes[action])) & (2**15 - 1)
        if changes[action]:
            history[last_key] = int(end - 1)
        last = history[last_key]
        values.extend([float(last != 0), float(np.log1p(end - last)) if last else 0.,
                       float(np.log1p(history[mask_key].bit_count()))])
    history.update(fill_adds=adds, fill_fraction=float(fraction))
    clean['_fill_features'] = np.asarray(values)
    return clean, history, np.asarray(values)


class CalibratedHistoryModels:
    features = OrderHistoryBoostModels.features

    def __init__(self, model, offset):
        self.model, self.offset = model, offset

    def probabilities(self, values):
        raw = self.model.probabilities(values)
        valid = np.isfinite(raw).all(axis=1)
        result = np.full_like(raw, np.nan)
        if valid.any():
            result[valid] = self.offset.predict(raw[valid])
        return result


class HistoryStatePolicy(CurrentStatePolicy):
    def __init__(self, parent, manager, thresholds):
        super().__init__(parent)
        self.manager, self.thresholds = manager, dict(thresholds)
        if (set(thresholds) != {'exit', 'reduce', 'increase'}
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values())):
            raise ValueError('자체 이력 관리 문턱 오류')

    def feature_values(self, bar, state):
        return np.r_[super().feature_values(bar, state), state['_fill_features']]

    def size_feature_values(self, bar, state):
        return MinuteInventoryPolicy.feature_values(self, bar, state)

    def __call__(self, bar, state):
        context, history, _ = fill_context(bar, state)
        decision = super().__call__(bar, context)
        return PolicyDecision(decision.intent, {**decision.state, **history}, decision.event, decision.reduction_fraction)


def validate_history_admission(evidence):
    for group in evidence['metrics'].values():
        counts = set()
        for metric in group.values():
            n, positive, negative = (metric[k] for k in ['rows', 'positive', 'negative'])
            if (any(type(v) is not int for v in [n, positive, negative])
                or n < 100 or min(positive, negative) < 20 or n != positive + negative):
                raise ValueError('자체 이력 관리 진단의 정답 지원 오류')
            counts.add((n, positive, negative))
        if len(counts) != 1:
            raise ValueError('자체 이력 관리 진단의 비교 표본 불일치')
    expected = history_calibration_admission(evidence['metrics'])
    settings, summary = evidence['settings'], evidence['summary']
    if (not expected['calibrated_histogram_admitted'] or evidence['decision'] != expected
        or any(summary.get(k) != v for k, v in expected.items()) or summary.get('complete') is not True
        or summary['episode_intersection'] != 0
        or settings['periods'] != {k: list(v) for k, v in PERIODS.items()}
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V35.md'))
        or 'history_files_sha256' not in settings):
        raise ValueError('자체 이력 관리의 사전 진단 통과 근거 오류')
    return expected


def copy_current_parent(reference, out):
    copy_probe_parent(reference, out)
    (out / 'probe_selection.json').write_bytes((reference / 'probe_selection.json').read_bytes())
    (out / 'current_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_history_parent(selection, frozen):
    path = selection / 'current_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['current_selection_sha256']:
        raise ValueError('자체 이력 관리의 이전 선택 지문 오류')
    return load_current_selection(selection, json.loads(path.read_text()))


def load_history_state_selection(selection, frozen):
    parent, original = load_history_parent(selection, frozen)
    if (frozen.get('protocol') != 'history_state_v36'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['history_files_sha256']) != set(HISTORY_POLICY_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['history_files_sha256'][n]
               for n in HISTORY_POLICY_FILES)):
        raise ValueError('자체 이력 관리의 고정 설정·모델 지문 오류')
    values = {n: json.loads((selection / n).read_text()) for n in HISTORY_POLICY_FILES}
    validate_history_admission(values['history_admission.json'])
    thresholds = values['history_thresholds.json']
    if (thresholds['calibration_period'] != list(PERIODS['calibration'])
        or thresholds['betas'] != [2., 1., .5] or thresholds['model_refitted'] is not False
        or thresholds['minimum_predicted_positive'] != 20 or thresholds['profit_selected'] is not False):
        raise ValueError('자체 이력 관리의 문턱 선택 범위 오류')
    manager = CalibratedHistoryModels(OrderHistoryBoostModels.from_dict(values['history_manager.json']),
                                     ManagementOffset.from_dict(values['history_offset.json']))
    return frozen, HistoryStatePolicy(original, manager, thresholds['thresholds'])
