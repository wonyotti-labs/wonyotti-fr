from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_effect import run_close_effect_labels
from .common import sha256
from .source_snapshot import copy_snapshot


def preserve_generation_source(output, hashes):
    source = output/'code_snapshot'
    target = output/'generation_source/wonyotti_fr'
    if target.is_symlink() or (output/'generation_source').is_symlink():
        raise ValueError('분별 청산 생성 코드의 링크 경로 오류')
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, expected in hashes.items():
        path = source/name
        if Path(name).name != name or path.is_symlink() or sha256(path) != expected:
            raise ValueError('분별 청산 생성 코드의 스냅샷 불일치')
        if not (target/name).exists():
            copy_snapshot(path, target/name)
        if (target/name).is_symlink() or sha256(target/name) != expected:
            raise ValueError('분별 청산 생성 코드의 보존 지문 불일치')
    if {p.name for p in target.glob('*.py')} != set(hashes):
        raise ValueError('분별 청산 생성 코드의 파일 범위 오류')


def validate_legacy_reference(labels, settings):
    from .close_learning_inputs import load_close_training
    legacy, _ = load_close_training(labels)
    previous = json.loads((labels/'manifest.json').read_text())['settings']
    fields = ['reference', 'reference_sha256', 'reference_outputs_sha256', 'market', 'features',
              'market_manifest_sha256', 'feature_manifest_sha256', 'period']
    if (any(settings[name] != previous[name] for name in fields)
        or str(labels) != settings['legacy_labels'] or sha256(labels/'files.json') != settings['legacy_files_sha256']):
        raise ValueError('분별 청산의 기존 부모·시세·기간 연결 오류')
    return legacy


def verify_legacy_grid(frame, legacy, complete):
    if (type(complete) is not bool or legacy.empty or frame.empty or frame.columns.tolist() != legacy.columns.tolist()
        or frame.decision_time.isna().any() or frame.decision_time.duplicated().any()
        or not frame.decision_time.is_monotonic_increasing
        or not isinstance(frame.decision_time.dtype, pd.DatetimeTZDtype) or str(frame.decision_time.dt.tz) != 'UTC'):
        raise ValueError('분별 청산의 기존 격자·시간·열 오류')
    times = frame.decision_time.astype('datetime64[ns, UTC]').array.asi8
    if (times % (60*10**9)).any():
        raise ValueError('분별 청산의 분 경계 오류')
    selected = frame.loc[times % (300*10**9) == 0].reset_index(drop=True)
    expected = legacy if complete else legacy.loc[legacy.decision_time.le(frame.decision_time.iloc[-1])].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, expected, check_exact=True)
    return {'complete_source': complete, 'legacy_rows': len(legacy), 'matched_rows': len(selected),
        'all_completed_legacy_rows_exact': True, 'minute_rows': len(frame), 'additional_rows': len(frame)-len(selected)}


def run_minute_close_effect_labels(labels: Path, output: Path, *, resume: Path | None = None,
                                   max_opportunities: int | None = None) -> Path:
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    return run_close_effect_labels(Path(settings['reference']), Path(settings['market']), Path(settings['features']), output,
        resume=resume, max_opportunities=max_opportunities, decision_seconds=60, legacy_labels=labels)
