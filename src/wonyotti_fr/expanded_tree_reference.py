from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import sha256
from .expanded_first_evaluation import EXPANDED_FIRST_POLICIES
from .expanded_first_reference import EXPANDED_CALIBRATION_FILES, trusted_proof
from .first_linear_reference import LINEAR_INPUT_FILES
from .retained_weight_close_reference import checked_files

EXPANDED_TREE_INPUT_FILES = LINEAR_INPUT_FILES | {'added_training_labels.parquet', 'combined_training_costs.parquet',
    'combined_first_training.parquet', 'added_first_opportunity_ledger.parquet', 'added_all_positions.parquet',
    'added_training_membership.parquet', 'original_training_positions.parquet'}
EXPANDED_TREE_REFERENCE_FILES = EXPANDED_TREE_INPUT_FILES | {'summary.json', 'model.json', 'training_support.json',
    'previous_files.json', 'expansion_evidence.json', 'reference_evidence.json', 'REPORT.md', 'expansion_verification.json',
    'previous_verification.json', 'previous_training_support.json', 'manifest.json', 'decision.json', 'expansion_files.json',
    'previous_calibration_decision.json'}
EXPANDED_TREE_CALIBRATION_FILES = EXPANDED_CALIBRATION_FILES | {'positions_expanded_first_linear.parquet', 'positions_expanded_first_constant.parquet'}
EXPANDED_TREE_REVIEW_CHECKS = ['complete', 'previous_verified_expansion_inputs_and_eleven_policies_reused',
    'all_first_rows_all_positions_and_equal_weights_verified', 'all_new_linear_parameters_and_first_calibration_scores_verified',
    'all_thirteen_policies_ten_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']


def checked_expanded_tree_reference(reference, verification, expected):
    proof = trusted_proof(verification, expected)
    if (any(proof.get(key) is not True for key in EXPANDED_TREE_REVIEW_CHECKS) or proof.get('synthetic_only') is not False
        or proof.get('calibration_passed') is not False or proof.get('profitability_accepted') is not False
        or proof.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('확장 첫 트리의 이전 검산 완료·참조 오류')
    files = checked_files(reference)
    if set(files) != EXPANDED_TREE_REFERENCE_FILES:
        raise ValueError('확장 첫 트리의 이전 파일 목록 오류')
    manifest, summary, decision = (json.loads((reference/name).read_text()) for name in ['manifest.json', 'summary.json', 'decision.json'])
    settings = manifest['settings']
    if (settings['policies'] != EXPANDED_FIRST_POLICIES or settings['score_rows'] != 'first_eligible_only'
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V85.md'))
        or settings['whole_policy_historically_available_claimed'] is not False
        or summary['complete'] is not True or summary['profitability_accepted'] is not False or summary['diagnosis_evaluated'] is not False
        or decision['calibration_passed'] is not False or decision['threshold_search'] is not False or decision['fallback_used'] is not False
        or decision['threshold'] != .5 or decision['candidate'] != 'expanded_first_linear'
        or any((reference/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])):
        raise ValueError('확장 첫 트리의 이전 계획·첫 판단·진단 차단 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (snapshot.is_symlink() or set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or (snapshot/name).is_symlink() or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('확장 첫 트리의 이전 생성 코드 오류')
    calibration = checked_files(reference/'calibration')
    if set(calibration) != EXPANDED_TREE_CALIBRATION_FILES or sha256(reference/'calibration/files.json') != summary['calibration_files_sha256']:
        raise ValueError('확장 첫 트리의 이전 열세 정책 봉인 오류')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'), 'verification': str(verification),
        'verification_sha256': expected, 'files': files, 'calibration_files': calibration, 'source_sha256': sources,
        'previous_verified_evidence_reused': True, 'previous_models_refitted': False, 'external_nested_runs_reverified': False,
        'generation_engine_rerun': False}


def load_expanded_tree_baseline(reference):
    folder = reference/'calibration'
    result = {name: json.loads((folder/f'{name}.json').read_text()) for name in ['metrics', 'first_metrics', 'probability_metrics',
        'breakdown', 'first_breakdown', 'probability_breakdown', 'first_probability_metrics']}
    result['positions'] = {name: pd.read_parquet(folder/f'positions_{name}.parquet') for name in EXPANDED_FIRST_POLICIES}
    for name in ['predictions', 'first_membership']:
        result[name] = pd.read_parquet(folder/f'{name}.parquet')
    result['calibration_decision'] = json.loads((reference/'decision.json').read_text())
    return result
