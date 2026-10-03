from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .event_features import purged_train, teacher_states
from .expansion_data import source_inputs
from .minute_management import ACTIONS, management_events, management_orders
from .path_management import PATH_FEATURES, add_price_path
from .timing_study import aggregate_comparison, read_minute_history


def write_labels(out, data, ledger, summary, train, calibration, check):
    data.to_parquet(out / 'events.parquet', index=False)
    ledger.to_parquet(out / 'order_ledger.parquet', index=False)
    splits = {}
    for name, frame in [('training', train), ('calibration', calibration), ('check_2021', check)]:
        frame.to_parquet(out / f'{name}.parquet', index=False)
        splits[name] = {'rows': len(frame), 'positive_windows': {a: int(frame[f'y_{a}'].sum()) for a in ACTIONS},
                        'first_end': frame.end.min(), 'last_label_end': frame.label_end.max(),
                        'episodes': int(frame.episode_id.nunique())}
    save_json(out / 'summary.json', {**summary, 'complete': True, 'split': splits})
    save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})


def run_management_labels(audit: Path, study: Path, history: Path, minute_history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    minute, minute_hashes = read_minute_history(minute_history, history)
    digest = sha256(audit / 'xbtusd_events.parquet')
    if digest != json.loads((study / 'manifest.json').read_text())['settings']['audit_sha256']['xbtusd_events.parquet']:
        raise ValueError('관리 정답의 원본 사건 지문 오류')
    out = new_run(output, 'minute-management-labels', {**hashes, **minute_hashes, 'xbtusd_events_sha256': digest,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V10.md'))})
    try:
        _, comparison = aggregate_comparison(minute, source['bars'])
        if comparison['incomplete_buckets'] or comparison['value_mismatch_buckets']:
            raise ValueError('원본 1분·5분 시세 집계 불일치')
        states = teacher_states(source['actions'], pd.read_parquet(audit / 'xbtusd_events.parquet'))
        orders = management_orders(source['executions'], source['actions'])
        data, ledger, summary = management_events(minute, source['bars'], states, orders, source['actions'].time.max())
        train = purged_train(data, source['episodes'], '2020-01-01')
        calibration = purged_train(data, source['episodes'], '2021-01-01')
        calibration = calibration[calibration.end.ge('2020-01-02') & calibration.entry_time.ge('2020-01-01')]
        check = data[data.usable & data.end.ge('2021-01-02') & data.label_end.lt('2022-01-01') & data.entry_time.ge('2021-01-01')]
        write_labels(out, data, ledger, {**summary, 'aggregate_comparison': comparison}, train, calibration, check)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def run_path_labels(labels: Path, output: Path) -> Path:
    hashes = json.loads((labels / 'files.json').read_text())
    names = ['events.parquet', 'order_ledger.parquet', 'summary.json', 'training.parquet', 'calibration.parquet', 'check_2021.parquet']
    if any(sha256(labels / name) != hashes[name] for name in names):
        raise ValueError('가격 경로 정답의 기반 지문 오류')
    summary = json.loads((labels / 'summary.json').read_text())
    if summary.get('complete') is not True or 'path_features' in summary:
        raise ValueError('가격 경로 정답의 기반 형식 오류')
    out = new_run(output, 'path-management-labels', {'labels': str(labels), 'files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V12.md'))})
    try:
        original = pd.read_parquet(labels / 'events.parquet')
        data = add_price_path(original)
        if not data.usable.equals(original.usable):
            raise ValueError('가격 경로 특징 추가 후 정답 지원 표본 변경')
        splits = []
        for name in ['training', 'calibration', 'check_2021']:
            ends = pd.read_parquet(labels / f'{name}.parquet', columns=['end']).end
            part = data[data.end.isin(ends)]
            if len(part) != len(ends):
                raise ValueError('가격 경로 정답의 분할 행 연결 오류')
            splits.append(part)
        write_labels(out, data, pd.read_parquet(labels / 'order_ledger.parquet'),
                     {**summary, 'path_features': PATH_FEATURES}, *splits)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
