from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .common import save_json, sha256
from .episode_balance import verified_files
from .exit_state import (
    EXIT_POLICY_FILES,
    ExitStatePolicy,
    copy_activity_parent,
    load_exit_state_selection,
)
from .first_management import BETAS, FIRST_ACTIONS
from .history_state import CalibratedHistoryModels, HistoryStatePolicy
from .management_calibration import ManagementOffset
from .minute_management import ACTIONS, purged_window
from .order_history import HISTORY_FEATURES
from .order_history_boost import OrderHistoryBoostModels

RECENT_PERIODS = {'training': ['2020-01-01', '2021-07-01'],
                  'calibration': ['2021-07-01', '2022-01-01']}
RECENT_HISTORY_FILES = ['recent_history_manager.json', 'recent_history_offset.json',
                        'recent_history_thresholds.json', 'recent_history_training.json']


class RecentHistoryPolicy(ExitStatePolicy):
    def __init__(self, parent, manager, thresholds, first_thresholds):
        updated = HistoryStatePolicy(parent, manager, thresholds)
        updated.first_thresholds = first_thresholds
        super().__init__(updated)


def validate_recent_history_rows(rows):
    if set(rows) != set(RECENT_PERIODS):
        raise ValueError('최신 관리 이력의 고정 분할 오류')
    support = {}
    for name, period in RECENT_PERIODS.items():
        frame = rows[name]
        pd.testing.assert_frame_equal(frame, purged_window(frame, *period).reset_index(drop=True), check_exact=True)
        ids = frame.episode_id.to_numpy()
        y = frame[[f'y_{a}' for a in ACTIONS]].to_numpy()
        if (len(frame) < (1000 if name == 'training' else 100)
            or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
            or frame[['end', 'entry_time', 'label_end', 'episode_id']].isna().any().any()
            or not np.isfinite(frame[OrderHistoryBoostModels.features]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any()
            or not np.isfinite(ids).all() or ((ids <= 0) | (ids >= 2**53) | (ids != np.floor(ids))).any()
            or frame.entry_time.ge(frame.end).any() or frame.end.gt(frame.label_end).any()
            or frame[HISTORY_FEATURES].lt(0).any().any()
            or any(not frame[f'past_{a}_exists'].isin([0, 1]).all() for a in FIRST_ACTIONS)):
            raise ValueError('최신 관리 이력의 시간·지원·입력 오류')
        support[name] = {'period': period, 'rows': len(frame), 'episodes': int(frame.episode_id.nunique()),
            'positive': {a: int(frame[f'y_{a}'].sum()) for a in ACTIONS}}
    train, calibration = rows['training'], rows['calibration']
    if set(train.episode_id) & set(calibration.episode_id) or train.label_end.max() >= calibration.end.min():
        raise ValueError('최신 관리 이력의 학습·보정 중첩')
    for a in FIRST_ACTIONS:
        for exists in [0, 1]:
            y = calibration.loc[calibration[f'past_{a}_exists'].eq(exists), f'y_{a}'].to_numpy()
            if min(y.sum(), len(y)-y.sum()) < 20:
                raise ValueError('최신 관리 이력의 첫·반복 행동 지원 부족')
    return support


def fit_recent_history(rows):
    splits = validate_recent_history_rows(rows)
    train, calibration = rows['training'], rows['calibration']
    x, cx = (p[OrderHistoryBoostModels.features].to_numpy(dtype=float) for p in [train, calibration])
    y, cy = (p[[f'y_{a}' for a in ACTIONS]].to_numpy() for p in [train, calibration])
    model, fitted = OrderHistoryBoostModels.fit(x, y, cx)
    raw = model.probabilities(cx)
    offset, calibrated = ManagementOffset.fit(raw, cy)
    probabilities = offset.predict(raw)
    thresholds, support, first, first_support = {}, {}, {}, {}
    for i, action in enumerate(ACTIONS):
        thresholds[action], support[action] = select_threshold(cy[:, i], probabilities[:, i], BETAS[action])
        if action in FIRST_ACTIONS:
            mask = calibration[f'past_{action}_exists'].eq(0).to_numpy()
            first[action], first_support[action] = select_threshold(cy[mask, i], probabilities[mask, i], BETAS[action])
    choices = {'thresholds': thresholds, 'first_thresholds': first, 'support': support, 'first_support': first_support,
        'betas': BETAS, 'calibration_period': RECENT_PERIODS['calibration'], 'minimum_predicted_positive': 20,
        'application': {'exit': 1., 'first': 1., 'repeat': 1.5}, 'selected_by_profit': False}
    return model, offset, choices, {'splits': splits, 'model': fitted, 'offset': calibrated}


def load_recent_history_rows(reference, diagnosis):
    evidence = json.loads((reference / 'history_admission.json').read_text())
    verified_files(diagnosis)
    if sha256(diagnosis / 'files.json') != evidence['files_sha256']:
        raise ValueError('최신 관리 재학습의 기존 진단 연결 오류')
    settings = json.loads((diagnosis / 'manifest.json').read_text())['settings']
    history = Path(settings['history'])
    files = verified_files(history)
    required = {'manifest.json', 'training_used.parquet', 'diagnosis_used.parquet'}
    if not required <= set(files) or sha256(history / 'files.json') != settings['history_files_sha256']:
        raise ValueError('최신 관리 재학습의 과거 이력 파일 오류')
    old = json.loads((history / 'manifest.json').read_text())['settings']
    if (old['training_period'] != RECENT_PERIODS['training'] or old['diagnosis_period'] != RECENT_PERIODS['calibration']
        or old['selection_sha256'] != settings['selection_sha256']
        or old['labels_files_sha256'] != settings['labels_files_sha256'] or old['new_features'] != HISTORY_FEATURES
        or any(sha256(Path(old['audit']) / n) != h for n, h in old['audit_sha256'].items())):
        raise ValueError('최신 관리 재학습의 원본·기간·특징 연결 오류')
    for filename, family in [('models.json', 'history_manager.json'), ('offsets.json', 'history_offset.json')]:
        if json.loads((diagnosis / filename).read_text())['histogram'] != json.loads((reference / family).read_text()):
            raise ValueError('최신 관리 재학습의 기존 모형 계열 불일치')
    rows = {n: pd.read_parquet(history / f'{stored}_used.parquet')
            for n, stored in [('training', 'training'), ('calibration', 'diagnosis')]}
    validate_recent_history_rows(rows)
    pd.testing.assert_frame_equal(rows['calibration'], pd.read_parquet(diagnosis / 'diagnosis_used.parquet'), check_exact=True)
    for n, period in [('training', ['2020-01-01', '2021-01-01']), ('calibration', ['2021-01-01', '2021-07-01'])]:
        part = purged_window(rows['training'], *period).reset_index(drop=True)
        pd.testing.assert_frame_equal(part, pd.read_parquet(diagnosis / f'{n}_used.parquet'), check_exact=True)
    return rows, {'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'history_files_sha256': sha256(history / 'files.json'), 'history': str(history),
        'training_sha256': files['training_used.parquet'], 'calibration_sha256': files['diagnosis_used.parquet']}


def prepare_recent_history(reference, diagnosis, out):
    rows, sources = load_recent_history_rows(reference, diagnosis)
    model, offset, choices, support = fit_recent_history(rows)
    save_json(out / 'recent_history_manager.json', model.to_dict())
    save_json(out / 'recent_history_offset.json', offset.to_dict())
    save_json(out / 'recent_history_thresholds.json', choices)
    save_json(out / 'recent_history_training.json', {**support, 'sources': sources,
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'periods': RECENT_PERIODS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V50.md')), 'new_model_family': False,
        'late_2021_is_calibration': True, 'profitability_accepted': False})


def copy_exit_parent(reference, out):
    copy_activity_parent(reference, out)
    for name in ['activity_selection.json', *EXIT_POLICY_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'exit_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_recent_history_parent(selection, frozen):
    path = selection / 'exit_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['exit_selection_sha256']:
        raise ValueError('최신 관리 이력의 이전 선택 지문 오류')
    return load_exit_state_selection(selection, json.loads(path.read_text()))


def load_recent_history_selection(selection, frozen):
    parent, original = load_recent_history_parent(selection, frozen)
    if (frozen.get('protocol') != 'recent_history_v50' or parent['protocol'] != 'exit_state_v48'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['recent_history_files_sha256']) != set(RECENT_HISTORY_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['recent_history_files_sha256'][n]
               for n in RECENT_HISTORY_FILES)):
        raise ValueError('최신 관리 이력의 고정 설정·파일 지문 오류')
    choices = json.loads((selection / 'recent_history_thresholds.json').read_text())
    support = json.loads((selection / 'recent_history_training.json').read_text())
    evidence = json.loads((selection / 'history_admission.json').read_text())
    if (support['reference_sha256'] != frozen['exit_selection_sha256'] or support['periods'] != RECENT_PERIODS
        or support['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V50.md'))
        or support['new_model_family'] is not False or support['late_2021_is_calibration'] is not True
        or support['profitability_accepted'] is not False
        or support['sources']['diagnosis_files_sha256'] != evidence['files_sha256']
        or support['sources']['history_files_sha256'] != evidence['settings']['history_files_sha256']
        or choices['application'] != {'exit': 1., 'first': 1., 'repeat': 1.5}
        or choices['calibration_period'] != RECENT_PERIODS['calibration'] or choices['betas'] != BETAS
        or choices['minimum_predicted_positive'] != 20 or choices['selected_by_profit'] is not False):
        raise ValueError('최신 관리 이력의 기간·보정·문턱 적용 오류')
    for mapping, expected in [('support', ACTIONS), ('first_support', FIRST_ACTIONS)]:
        if set(choices[mapping]) != set(expected) or any(
            choices[mapping][a]['actual_positive'] < 20 or choices[mapping][a]['predicted_positive'] < 20
            or choices[mapping][a]['beta'] != BETAS[a] for a in expected):
            raise ValueError('최신 관리 이력의 문턱 지원 오류')
    model = OrderHistoryBoostModels.from_dict(json.loads((selection / 'recent_history_manager.json').read_text()))
    offset = ManagementOffset.from_dict(json.loads((selection / 'recent_history_offset.json').read_text()))
    return frozen, RecentHistoryPolicy(original, CalibratedHistoryModels(model, offset),
        choices['thresholds'], choices['first_thresholds'])
