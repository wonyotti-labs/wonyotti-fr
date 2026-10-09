from __future__ import annotations

import json
import re
from pathlib import Path

from .close_threshold import THRESHOLD_SPLITS
from .common import sha256
from .retained_weight_close_reference import checked_files
from .stopping_regression_diagnostics import REGRESSION_INPUT_FILES
from .stopping_regression_evaluation import REGRESSION_NEW_POLICIES

FIRST_INPUT_FILES = {'early_training_used.parquet', 'early_training_weights.parquet', 'calibration_used.parquet',
    'calibration_weights.parquet', 'threshold_exclusion_ledger.parquet'}
REGRESSION_FAILURE_FILES = REGRESSION_INPUT_FILES | {'manifest.json', 'baseline_files.json', 'reference_parity.json',
    'model.json', 'training_support.json', 'regression_training_ledger.parquet', 'calibration_decision.json',
    'decision.json', 'summary.json', 'REPORT.md'}
REGRESSION_CALIBRATION_FILES = {'predictions.parquet', 'metrics.json', 'first_metrics.json', 'probability_metrics.json',
    'breakdown.json', 'first_breakdown.json', 'probability_breakdown.json',
    *[f'positions_{name}.parquet' for name in REGRESSION_NEW_POLICIES]}
REQUIRED_REVIEW_CHECKS = ['complete', 'all_original_inputs_and_prior79_outputs_exact',
    'whole_position_splits_original_weights_zero_targets_and64_new_trees_verified',
    'previous_future_first_cash_targets_and_availability_exact', 'all_six_calibration_policies_and_fixed_decision_verified',
    'failure_diagnosis_prediction_block_verified']


def checked_first_reference(reference, verification, expected_verification_sha256):
    if (type(expected_verification_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_verification_sha256) is None
        or verification.is_symlink() or sha256(verification) != expected_verification_sha256):
        raise ValueError('첫 적격 기회의 신뢰한 검산 기록 지문 불일치')
    review = json.loads(verification.read_text())
    if (any(review.get(key) is not True for key in REQUIRED_REVIEW_CHECKS)
        or review.get('calibration_passed') is not False or review.get('diagnosis_evaluated') is not False
        or review.get('profitability_accepted') is not False or review.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('첫 적격 기회의 검산 완료·참조 봉인 오류')
    files = checked_files(reference)
    if set(files) != REGRESSION_FAILURE_FILES:
        raise ValueError('첫 적격 기회의 기존 회귀 실패 파일 목록 오류')
    manifest, summary = (json.loads((reference/name).read_text()) for name in ['manifest.json', 'summary.json'])
    settings = manifest['settings']
    if (settings['candidate'] != 'stopping_regression' or settings['periods'] != THRESHOLD_SPLITS
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or summary['diagnosis_evaluated'] is not False or (reference/'predictions.parquet').exists()
        or (reference/'intervals.json').exists()):
        raise ValueError('첫 적격 기회의 기존 내부 실패·진단 차단 오류')
    snapshot = reference/'code_snapshot'
    sources = manifest['source_sha256']
    if (snapshot.is_symlink() or set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('첫 적격 기회의 기존 코드 사본 오류')
    calibration = checked_files(reference/'calibration')
    if (set(calibration) != REGRESSION_CALIBRATION_FILES
        or sha256(reference/'calibration/files.json') != summary['calibration_files_sha256']):
        raise ValueError('첫 적격 기회의 기존 보정 봉인 오류')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'verification': str(verification), 'verification_sha256': expected_verification_sha256,
        'calibration_files_sha256': summary['calibration_files_sha256'], 'files': files, 'calibration_files': calibration,
        'source_sha256': sources, 'previous_full_replay_and_independent_audit_reused': True,
        'previous_models_refitted': False, 'external_nested_runs_reverified': False}


def load_first_baseline(reference):
    folder = reference/'calibration'
    result = {name: json.loads((folder/f'{name}.json').read_text()) for name in
        ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown']}
    import pandas as pd
    result['predictions'] = pd.read_parquet(folder/'predictions.parquet')
    result['positions'] = {name: pd.read_parquet(folder/f'positions_{name}.parquet') for name in REGRESSION_NEW_POLICIES}
    result['calibration_decision'] = json.loads((reference/'calibration_decision.json').read_text())
    return result
