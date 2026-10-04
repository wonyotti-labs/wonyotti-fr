from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .common import save_json, sha256
from .episode_balance import verified_files
from .exit_horizon import (
    HORIZON_MINUTES,
    exit_offset_predict,
    horizon_requests,
    load_exit_horizon_inputs,
)
from .exit_state import ExitStatePolicy
from .management_calibration import ManagementOffset
from .recent_history import load_recent_history_parent

HORIZON_EXIT_FILES = ['horizon_exit_threshold.json', 'horizon_exit_evidence.json']


class HorizonExitPolicy(ExitStatePolicy):
    def __init__(self, parent, threshold):
        super().__init__(parent)
        if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('청산 범위 문턱의 숫자 오류')
        self.horizon_threshold = threshold

    def action_threshold(self, action, state):
        return self.horizon_threshold if action == 'exit' else super().action_threshold(action, state)


def horizon_exit_comparison(requests):
    if set(requests) != {'candidate', 'previous_v48'}:
        raise ValueError('청산 범위 문턱의 대조 누락')
    for r in requests.values():
        if (not np.isfinite([r['f2'], r['event_recall']]).all()
            or not all(0 <= r[k] <= 1 for k in ['f2', 'event_recall'])
            or type(r['original_events']) is not int or r['original_events'] < 20
            or type(r['detected_events']) is not int or not 0 <= r['detected_events'] <= r['original_events']
            or r['event_recall'] != r['detected_events']/r['original_events']):
            raise ValueError('청산 범위 문턱의 사건 지원·지표 오류')
    if requests['candidate']['original_events'] != requests['previous_v48']['original_events']:
        raise ValueError('청산 범위 문턱의 사건 표본 불일치')
    return {k+'_preserved': requests['candidate'][k] >= requests['previous_v48'][k] for k in ['f2', 'event_recall']}


def select_horizon_exit(rows, original, recalibration, expected_threshold):
    cal, val = rows['calibration'], rows['diagnosis']
    cp, vp = (original.manager.probabilities(p[original.manager.features].to_numpy())[:, 0] for p in [cal, val])
    threshold, support = select_threshold(cal.y_exit.to_numpy(), cp, 2.)
    for name, probabilities in [('calibration', cp), ('diagnosis', vp)]:
        if not np.array_equal(probabilities >= threshold, exit_offset_predict(recalibration, probabilities) >= expected_threshold):
            raise ValueError(f'청산 범위 문턱의 {name} 단조 보정 요청 불일치')
    requests = {name: horizon_requests(val, vp, t)[0] for name, t in [
        ('candidate', threshold), ('previous_v48', original.thresholds['exit'])]}
    comparison = horizon_exit_comparison(requests)
    if not all(comparison.values()):
        raise ValueError('청산 범위 문턱의 기존 F2·사건 포착 기준 미달')
    return {'threshold': threshold, 'original_threshold': original.thresholds['exit'], 'support': support,
        'beta': 2., 'minimum_predicted_positive': 20, 'calibration_period': ['2021-01-01', '2021-07-01'],
        'horizon_minutes': HORIZON_MINUTES, 'selected_by_profit': False}, {
        'requests': requests, 'comparison': comparison, 'all_calibration_and_diagnosis_requests_equal': True}


def prepare_horizon_exit(reference, diagnosis, original, out):
    verified_files(diagnosis)
    settings = json.loads((diagnosis / 'manifest.json').read_text())['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V51.md')) or settings['horizon_minutes'] != 15
        or settings['trading_returns_evaluated'] is not False):
        raise ValueError('청산 범위 문턱의 기존 범위 진단 오류')
    source = Path(settings['source'])
    prior = json.loads((reference / 'history_admission.json').read_text())
    if settings['source_files_sha256'] != sha256(source / 'files.json') or prior['files_sha256'] != settings['source_files_sha256']:
        raise ValueError('청산 범위 문턱의 기반 관리 진단 연결 오류')
    rows, _, _, targets, baseline = load_exit_horizon_inputs(source)
    pd.testing.assert_frame_equal(targets, pd.read_parquet(diagnosis / 'horizon_labels.parquet'), check_exact=True)
    for name, frame in rows.items():
        pd.testing.assert_frame_equal(frame, pd.read_parquet(diagnosis / f'{name}_used.parquet'), check_exact=True)
    if (original.manager.model.to_dict() != baseline.model.to_dict()
        or original.manager.offset.to_dict() != baseline.offset.to_dict()):
        raise ValueError('청산 범위 문턱의 기존 모형 변경')
    offset = ManagementOffset.from_dict(json.loads((diagnosis / 'offsets.json').read_text())['original'])
    threshold = json.loads((diagnosis / 'thresholds.json').read_text())['original']
    choices, checks = select_horizon_exit(rows, original, offset, threshold)
    if checks['requests']['candidate'] != json.loads((diagnosis / 'requests.json').read_text())['original']:
        raise ValueError('청산 범위 문턱의 기존 사건 포착 불일치')
    evidence = {**checks, 'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'source_files_sha256': sha256(source / 'files.json'), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'model_files_sha256': {n: sha256(reference / n) for n in ['history_manager.json', 'history_offset.json']},
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V52.md')), 'new_models_fitted': False,
        'previous_new_model_admitted': json.loads((diagnosis / 'decision.json').read_text())['exit_horizon_admitted'],
        'already_observed_diagnosis': True, 'profitability_accepted': False}
    save_json(out / HORIZON_EXIT_FILES[0], choices)
    save_json(out / HORIZON_EXIT_FILES[1], evidence)


def load_horizon_exit_selection(selection, frozen):
    parent, original = load_recent_history_parent(selection, frozen)
    if (frozen.get('protocol') != 'horizon_exit_v52' or parent['protocol'] != 'exit_state_v48'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['horizon_exit_files_sha256']) != set(HORIZON_EXIT_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['horizon_exit_files_sha256'][n]
               for n in HORIZON_EXIT_FILES)):
        raise ValueError('청산 범위 문턱의 고정 설정·지문 오류')
    choices, evidence = (json.loads((selection / n).read_text()) for n in HORIZON_EXIT_FILES)
    prior = json.loads((selection / 'history_admission.json').read_text())
    if (evidence['reference_sha256'] != frozen['exit_selection_sha256']
        or evidence['source_files_sha256'] != prior['files_sha256']
        or evidence['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V52.md'))
        or evidence['new_models_fitted'] is not False or evidence['already_observed_diagnosis'] is not True
        or evidence['profitability_accepted'] is not False or evidence['all_calibration_and_diagnosis_requests_equal'] is not True
        or set(evidence['model_files_sha256']) != {'history_manager.json', 'history_offset.json'}
        or any(sha256(selection / n) != h for n, h in evidence['model_files_sha256'].items())
        or evidence['comparison'] != horizon_exit_comparison(evidence['requests']) or not all(evidence['comparison'].values())
        or choices['original_threshold'] != original.thresholds['exit'] or choices['beta'] != 2.
        or choices['minimum_predicted_positive'] != 20 or choices['horizon_minutes'] != 15
        or choices['calibration_period'] != ['2021-01-01', '2021-07-01'] or choices['selected_by_profit'] is not False
        or choices['support']['actual_positive'] < 20 or choices['support']['predicted_positive'] < 20
        or choices['support']['beta'] != 2.):
        raise ValueError('청산 범위 문턱의 보정·지원·이전 모형 대조 오류')
    return frozen, HorizonExitPolicy(original, choices['threshold'])
