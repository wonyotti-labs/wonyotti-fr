from __future__ import annotations

import json
import math
from pathlib import Path

from .common import sha256
from .direction_diagnostics import DIAGNOSIS_PERIOD, TRAIN_PERIOD, direction_admission
from .expansion_model import BinaryModel
from .position_direction import (
    DIRECTION_FILES,
    DIRECTION_PERIOD,
    copy_prior_parent,
    load_position_direction_selection,
)
from .recent_entry import RecentEntryPolicy

BOOST_FILES = ['boosted_direction.json', 'boosted_direction_training.json', 'direction_admission.json']


def validate_admission(evidence):
    summary, settings, metrics = (evidence[k] for k in ['summary', 'settings', 'metrics'])
    expected = direction_admission(metrics)
    supports = evidence['training_support']
    counts = [supports[k] for k in ['logistic', 'boosted']]
    if any(r['rows'] != summary['training_rows'] or r['rows'] != r['positive'] + r['negative']
           or min(r['positive'], r['negative']) < 20 for r in counts):
        raise ValueError('방향 진단의 학습 지원 불일치')
    for kind in ['logistic', 'boosted', 'constant']:
        n, fraction = metrics[kind]['rows'], metrics[kind]['actual_buy_fraction']
        if not math.isfinite(fraction) or min(n * fraction, n * (1 - fraction)) < 20:
            raise ValueError('방향 진단의 양쪽 정답 지원 부족')
    if (not expected['boosted_admitted'] or evidence['decision'] != expected
        or any(summary.get(k) != v for k, v in expected.items())
        or summary.get('complete') is not True or summary['episode_intersection'] != 0
        or summary['training_rows'] < 1000 or summary['diagnosis_rows'] < 100
        or any(metrics[k]['rows'] != summary['diagnosis_rows'] for k in ['logistic', 'boosted', 'constant'])
        or settings['training_period'] != TRAIN_PERIOD or settings['diagnosis_period'] != DIAGNOSIS_PERIOD
        or settings['model_count'] != 2 or settings['trading_returns_evaluated'] is not False
        or settings['all_source_periods_already_observed'] is not True):
        raise ValueError('방향 부스팅 후보의 사전 진단 통과 근거 오류')
    return expected


def read_admission(diagnosis, hashes):
    files = json.loads((diagnosis / 'files.json').read_text())
    required = {'summary.json', 'decision.json', 'metrics.json', 'manifest.json', 'models.json',
                'training_support.json', 'training_used.parquet', 'diagnosis_used.parquet', 'predictions.parquet'}
    if not required <= set(files):
        raise ValueError('방향 진단의 필수 출력 누락')
    for name, digest in files.items():
        path = (diagnosis / name).resolve()
        if not path.is_relative_to(diagnosis.resolve()) or sha256(path) != digest:
            raise ValueError('방향 진단의 출력 지문 오류')
    evidence = {k: json.loads((diagnosis / f'{k}.json').read_text()) for k in ['summary', 'decision', 'metrics', 'training_support']}
    evidence['settings'] = json.loads((diagnosis / 'manifest.json').read_text())['settings']
    if (any(evidence['settings'].get(k) != v for k, v in hashes.items())
        or evidence['settings']['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V28.md'))):
        raise ValueError('방향 진단과 후속 학습의 원본·사전 계획 불일치')
    evidence['files_sha256'] = sha256(diagnosis / 'files.json')
    validate_admission(evidence)
    return evidence


def copy_direction_parent(reference, out):
    copy_prior_parent(reference, out)
    for name in ['prior_selection.json', *DIRECTION_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'direction_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_boosted_direction_selection(selection, frozen):
    path = selection / 'direction_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['direction_selection_sha256']:
        raise ValueError('방향 부스팅의 이전 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'boosted_direction_v28' or parent.get('protocol') != 'position_direction_v27'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['boost_files_sha256']) != set(BOOST_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['boost_files_sha256'][n] for n in BOOST_FILES)):
        raise ValueError('방향 부스팅의 고정 설정·모델 지문 오류')
    _, original = load_position_direction_selection(selection, parent)
    validate_admission(json.loads((selection / 'direction_admission.json').read_text()))
    metadata = json.loads((selection / 'boosted_direction_training.json').read_text())
    direction = BinaryModel.from_dict(json.loads((selection / 'boosted_direction.json').read_text()))
    if (metadata['training_period'] != DIRECTION_PERIOD or metadata['prior_offset_applied'] is not False
        or metadata['direction_threshold'] != .65 or direction.data['kind'] != 'boosted'
        or len(direction.data['trees']) != 64 or direction.data['learning_rate'] != .05
        or metadata['model']['rows'] < 1000 or min(metadata['model']['positive'], metadata['model']['negative']) < 20
        or metadata['model']['rows'] != metadata['model']['positive'] + metadata['model']['negative']):
        raise ValueError('방향 부스팅의 학습 범위·지원·고정 모델 설정 오류')
    return frozen, RecentEntryPolicy(original, original.base.activity, direction, original.base.activity_threshold)
