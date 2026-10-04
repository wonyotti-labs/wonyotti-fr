from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import sha256
from .engine import EngineConfig
from .first_state import FIRST_POLICY_FILES, copy_history_parent, load_first_state_selection

MINUTE_NS = pd.Timedelta(minutes=1).value


def holding_support(rows):
    support, episodes = {}, {}
    for name in ['training', 'calibration']:
        frame = rows[name]
        first, last = (pd.Timestamp(v, tz='UTC') for v in PERIODS[name])
        if (frame[['end', 'entry_time', 'label_end', 'episode_id']].isna().any().any()
            or frame.end.gt(frame.label_end).any() or len(frame) < (1000 if name == 'training' else 100)
            or frame.end.min() < first + pd.Timedelta(days=1) or frame.entry_time.min() < first
            or frame.label_end.max() >= last - pd.Timedelta(days=1)
            or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing):
            raise ValueError('보유 시간 지원 범위의 표본·시간 오류')
        duration = frame.end.astype('datetime64[ns, UTC]').array.asi8 - frame.entry_time.astype('datetime64[ns, UTC]').array.asi8
        if (duration <= 0).any() or not np.isfinite(frame.log_hold_minutes).all():
            raise ValueError('보유 시간 지원 범위의 진입·보유 값 오류')
        if not np.allclose(np.log1p(duration / MINUTE_NS), frame.log_hold_minutes, atol=1e-12, rtol=0):
            raise ValueError('보유 시간 지원 범위와 원래 모델 입력 불일치')
        support[name] = {'rows': len(frame), 'period': list(PERIODS[name]),
            'max_duration_ns': int(duration.max()), 'first_end': frame.end.min().isoformat(),
            'last_label_end': frame.label_end.max().isoformat()}
        episodes[name] = set(frame.episode_id)
    if (episodes['training'] & episodes['calibration']
        or rows['training'].label_end.max() >= rows['calibration'].end.min()):
        raise ValueError('보유 시간 지원 범위의 포지션·시각 중첩')
    value = {'format': 'holding_support_v1', 'support': support,
        'max_hold_bars': max(v['max_duration_ns'] for v in support.values()) // MINUTE_NS,
        'bar_seconds': 60, 'rounding': 'floor', 'diagnosis_used': False, 'profit_selected': False}
    validate_holding_support(value)
    return value


def validate_holding_support(value):
    if (value.get('format') != 'holding_support_v1' or value.get('bar_seconds') != 60
        or value.get('rounding') != 'floor' or value.get('diagnosis_used') is not False
        or value.get('profit_selected') is not False or set(value['support']) != {'training', 'calibration'}
        or type(value.get('max_hold_bars')) is not int or value['max_hold_bars'] <= 0):
        raise ValueError('보유 시간 한도의 고정 형식·범위 오류')
    for name, row in value['support'].items():
        first, last = (pd.Timestamp(v, tz='UTC') for v in PERIODS[name])
        start, end = pd.Timestamp(row['first_end']), pd.Timestamp(row['last_label_end'])
        if (row['period'] != list(PERIODS[name]) or type(row['rows']) is not int
            or row['rows'] < (1000 if name == 'training' else 100)
            or type(row['max_duration_ns']) is not int or not MINUTE_NS <= row['max_duration_ns'] < (last-first).value
            or pd.isna(start) or pd.isna(end) or not first+pd.Timedelta(days=1) <= start <= end < last-pd.Timedelta(days=1)):
            raise ValueError('보유 시간 한도의 학습·보정 지원 오류')
    if value['max_hold_bars'] != max(v['max_duration_ns'] for v in value['support'].values()) // MINUTE_NS:
        raise ValueError('보유 시간 한도의 정수 분 내림 불일치')


def copy_first_parent(reference, out):
    copy_history_parent(reference, out)
    for name in ['history_selection.json', *FIRST_POLICY_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'first_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_holding_parent(selection, frozen):
    path = selection / 'first_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['first_selection_sha256']:
        raise ValueError('보유 시간 한도의 이전 선택 지문 오류')
    return load_first_state_selection(selection, json.loads(path.read_text()))


def load_holding_selection(selection, frozen):
    parent, original = load_holding_parent(selection, frozen)
    path = selection / 'holding_support.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['holding_support_sha256']:
        raise ValueError('보유 시간 한도의 산출 기록 지문 오류')
    value = json.loads(path.read_text())
    validate_holding_support(value)
    source = value['source']
    evidence = json.loads((selection / 'first_admission.json').read_text())
    if (source['files_sha256'] != evidence['settings']['diagnosis_files_sha256']
        or source['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V40.md'))
        or set(source['row_files_sha256']) != {'training_used.parquet', 'calibration_used.parquet'}
        or any(type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
               for h in source['row_files_sha256'].values())):
        raise ValueError('보유 시간 한도의 학습 원본 연결 오류')
    risk = EngineConfig(**parent['risk'])
    if (frozen.get('protocol') != 'holding_support_v40' or risk.max_hold_bars != 0
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'risk', 'development_metrics'})
        or frozen.get('risk') != replace(risk, max_hold_bars=value['max_hold_bars']).__dict__):
        raise ValueError('보유 시간 한도의 고정 모델·위험 변경')
    return frozen, original
