from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .common import sha256
from .first_linear_evaluation import LINEAR_POLICIES
from .first_linear_reference import FIRST_CALIBRATION_FILES, FIRST_FAILURE_FILES
from .retained_weight_close_reference import checked_files

LINEAR_FAILURE_FILES = FIRST_FAILURE_FILES | {'previous_model.json', 'previous_training_support.json', 'earlier_regression_decision.json'}
LINEAR_CALIBRATION_FILES = FIRST_CALIBRATION_FILES | {'positions_first_linear.parquet'}
WEEKLY_FIRST_REVIEW_CHECKS = ['complete', 'previous_verified_first_inputs_and_calibration_reused',
    'all_first_rows_all_positions_and_equal_weights_verified', 'all_new_linear_parameters_and_first_calibration_scores_verified',
    'all_nine_policies_seven_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']


def checked_weekly_first_reference(reference, verification, expected_sha256):
    if (type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
        or verification.is_symlink() or sha256(verification) != expected_sha256):
        raise ValueError('첫 판단 주간 모델의 신뢰한 검산 지문 불일치')
    proof = json.loads(verification.read_text())
    if (any(proof.get(key) is not True for key in WEEKLY_FIRST_REVIEW_CHECKS) or proof.get('synthetic_only') is not False
        or proof.get('calibration_passed') is not False or proof.get('profitability_accepted') is not False
        or proof.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('첫 판단 주간 모델의 이전 검산 완료·봉인 오류')
    files = checked_files(reference)
    if set(files) != LINEAR_FAILURE_FILES:
        raise ValueError('첫 판단 주간 모델의 이전 파일 목록 오류')
    manifest, summary, decision = (json.loads((reference/name).read_text()) for name in ['manifest.json', 'summary.json', 'decision.json'])
    if (manifest['settings']['policies'] != LINEAR_POLICIES or manifest['settings']['score_rows'] != 'first_eligible_only'
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or summary['diagnosis_evaluated'] is not False or decision['calibration_passed'] is not False
        or decision['threshold_search'] is not False or decision['fallback_used'] is not False
        or decision['threshold'] != .5 or decision['candidate'] != 'first_linear'
        or any((reference/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])):
        raise ValueError('첫 판단 주간 모델의 이전 단일 판단·탈락·진단 차단 오류')
    snapshot, sources = reference/'code_snapshot', manifest['source_sha256']
    if (snapshot.is_symlink() or set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('첫 판단 주간 모델의 이전 코드 사본 오류')
    calibration = checked_files(reference/'calibration')
    if set(calibration) != LINEAR_CALIBRATION_FILES or sha256(reference/'calibration/files.json') != summary['calibration_files_sha256']:
        raise ValueError('첫 판단 주간 모델의 기존 아홉 정책 봉인 오류')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'verification': str(verification), 'verification_sha256': expected_sha256, 'files': files,
        'calibration_files': calibration, 'calibration_files_sha256': summary['calibration_files_sha256'], 'source_sha256': sources,
        'previous_verified_evidence_reused': True, 'previous_models_refitted': False, 'external_nested_runs_reverified': False}


def load_weekly_first_baseline(reference):
    folder = reference/'calibration'
    result = {name: json.loads((folder/f'{name}.json').read_text()) for name in ['metrics', 'first_metrics', 'probability_metrics',
        'breakdown', 'first_breakdown', 'probability_breakdown', 'first_probability_metrics']}
    result['positions'] = {name: pd.read_parquet(folder/f'positions_{name}.parquet') for name in LINEAR_POLICIES}
    result['predictions'] = pd.read_parquet(folder/'predictions.parquet')
    result['first_membership'] = pd.read_parquet(folder/'first_membership.parquet')
    result['calibration_decision'] = json.loads((reference/'decision.json').read_text())
    result['previous_first_decision'] = json.loads((reference/'previous_calibration_decision.json').read_text())
    result['earlier_regression_decision'] = json.loads((reference/'earlier_regression_decision.json').read_text())
    return result
