from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import sha256
from .expanded_first_reference import trusted_proof
from .managed_first_evaluation import MANAGED_POLICIES
from .managed_first_reference import (
    MANAGED_CALIBRATION_FILES,
    MANAGED_INPUT_FILES,
    MANAGED_REFERENCE_FILES,
)
from .retained_weight_close_reference import checked_files

PROBABILITY_INPUT_FILES = MANAGED_INPUT_FILES | {'managed_first_training.parquet', 'managed_first_calibration.parquet'}
PROBABILITY_REFERENCE_FILES = MANAGED_REFERENCE_FILES | {'manager_evidence.json', 'linear_manifest.json',
    'linear_expansion_evidence.json', 'generation_manifest.json', 'manager_parent_selection.json',
    'managed_first_training.parquet', 'managed_first_calibration.parquet'}
PROBABILITY_CALIBRATION_FILES = MANAGED_CALIBRATION_FILES | {'positions_managed_first_linear.parquet'}
PROBABILITY_REVIEW_CHECKS = ['complete', 'previous_verified_expanded_inputs_and_fourteen_policies_reused',
    'all_first_rows_all_positions_and_equal_weights_verified', 'all192_manager_trees_offsets_and77_linear_inputs_verified',
    'all_fifteen_policies_twelve_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']


def checked_probability_reference(reference, verification, expected):
    proof = trusted_proof(verification, expected)
    if (any(proof.get(key) is not True for key in PROBABILITY_REVIEW_CHECKS) or proof.get('synthetic_only') is not False
        or proof.get('calibration_passed') is not False or proof.get('profitability_accepted') is not False
        or proof.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('세 확률 첫 선형의 이전 검산 완료·참조 오류')
    files = checked_files(reference)
    if set(files) != PROBABILITY_REFERENCE_FILES:
        raise ValueError('세 확률 첫 선형의 이전 파일 목록 오류')
    manifest, summary, decision = (json.loads((reference/name).read_text()) for name in ['manifest.json', 'summary.json', 'decision.json'])
    settings = manifest['settings']
    if (settings['policies'] != MANAGED_POLICIES or settings['score_rows'] != 'first_eligible_only'
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V87.md'))
        or settings['whole_policy_historically_available_claimed'] is not False
        or summary['complete'] is not True or summary['profitability_accepted'] is not False or summary['diagnosis_evaluated'] is not False
        or decision['calibration_passed'] is not False or decision['threshold_search'] is not False or decision['fallback_used'] is not False
        or decision['threshold'] != .5 or decision['candidate'] != 'managed_first_linear'
        or any((reference/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])):
        raise ValueError('세 확률 첫 선형의 이전 계획·첫 판단·진단 차단 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (snapshot.is_symlink() or set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or (snapshot/name).is_symlink() or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('세 확률 첫 선형의 이전 생성 코드 오류')
    calibration = checked_files(reference/'calibration')
    if set(calibration) != PROBABILITY_CALIBRATION_FILES or sha256(reference/'calibration/files.json') != summary['calibration_files_sha256']:
        raise ValueError('세 확률 첫 선형의 이전 열다섯 정책 봉인 오류')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'), 'verification': str(verification),
        'verification_sha256': expected, 'files': files, 'calibration_files': calibration, 'source_sha256': sources,
        'previous_verified_evidence_reused': True, 'previous_models_refitted': False, 'external_nested_runs_reverified': False,
        'generation_engine_rerun': False}


def load_probability_baseline(reference):
    folder = reference/'calibration'
    result = {name: json.loads((folder/f'{name}.json').read_text()) for name in ['metrics', 'first_metrics', 'probability_metrics',
        'breakdown', 'first_breakdown', 'probability_breakdown', 'first_probability_metrics']}
    result['positions'] = {name: pd.read_parquet(folder/f'positions_{name}.parquet') for name in MANAGED_POLICIES}
    for name in ['predictions', 'first_membership']:
        result[name] = pd.read_parquet(folder/f'{name}.parquet')
    result['calibration_decision'] = json.loads((reference/'decision.json').read_text())
    return result
