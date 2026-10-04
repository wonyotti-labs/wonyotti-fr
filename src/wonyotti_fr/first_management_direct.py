from __future__ import annotations

import json
from pathlib import Path

from .common import sha256
from .first_management import run_first_management_diagnosis
from .inventory_labels import verify_files


def run_first_management_direct_diagnosis(reference: Path, output: Path) -> Path:
    files = json.loads((reference / 'files.json').read_text())
    required = {'manifest.json', 'summary.json', 'first_thresholds.json', 'history_manager.json',
                'history_offset.json', 'history_thresholds.json', 'metrics.json', 'decision.json', 'predictions.parquet'}
    if (not required <= set(files)
        or any(not (reference / n).resolve().is_relative_to(reference.resolve()) for n in files)):
        raise ValueError('첫 문턱 직접 적용의 기반 파일·경로 오류')
    verify_files(reference, list(files))
    settings = json.loads((reference / 'manifest.json').read_text())['settings']
    selection, diagnosis = Path(settings['selection']), Path(settings['diagnosis'])
    thresholds = json.loads((reference / 'first_thresholds.json').read_text())
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V37.md'))
        or sha256(selection / 'frozen_selection.json') != settings['selection_sha256']
        or sha256(diagnosis / 'files.json') != settings['diagnosis_files_sha256']
        or thresholds['multiplier'] != 1.5
        or any((reference / n).read_bytes() != (selection / n).read_bytes()
               for n in ['history_manager.json', 'history_offset.json', 'history_thresholds.json'])):
        raise ValueError('첫 문턱 직접 적용의 기존 모델·문턱·원본 변경')
    context = {'thresholds': thresholds, 'metadata': {'reference': str(reference),
        'reference_files_sha256': sha256(reference / 'files.json'), 'first_multiplier': 1.}}
    return run_first_management_diagnosis(selection, diagnosis, output, _first_context=context)
