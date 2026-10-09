from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .common import sha256
from .first_opportunity_collection import EXPANDED_FIRST_FEATURES
from .first_opportunity_expansion import BASELINE_FILES, EXPANSION_PERIODS, sealed_phase
from .retained_weight_close_reference import checked_files
from .weekly_first_linear_evaluation import WEEKLY_FIRST_POLICIES
from .weekly_first_linear_reference import LINEAR_CALIBRATION_FILES, LINEAR_FAILURE_FILES

EXPANDED_REFERENCE_FILES = (LINEAR_FAILURE_FILES-{'model.json', 'training_support.json'}) | {
    'weekly_models.json', 'weekly_support.json', 'weekly_training_membership.parquet', 'prediction_routing.parquet', 'previous_first_decision.json'}
EXPANDED_CALIBRATION_FILES = LINEAR_CALIBRATION_FILES | {'positions_weekly_first_linear.parquet', 'positions_weekly_first_constant.parquet'}
EXPANDED_REFERENCE_CHECKS = ['complete', 'previous_verified_linear_inputs_and_calibration_reused',
    'all_first_rows_all_positions_and_equal_weights_verified', 'all_nine_weekly_models_maturity_memberships_and_scores_verified',
    'all_eleven_policies_nine_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']
EXPANSION_PHASE_CHECKS = ['complete', 'all_membership_first_choices_population_exact', 'all_74_features_independently_recomputed',
    'all_cash_cost_funding_natural_ends_exact', 'full_equity_trades_fills_state_exact']


def trusted_proof(path, expected):
    if (type(expected) is not str or re.fullmatch('[0-9a-f]{64}', expected) is None
        or path.is_symlink() or sha256(path) != expected):
        raise ValueError('과거 추가 학습의 신뢰한 검산 지문 오류')
    return json.loads(path.read_text())


def checked_expanded_baseline(reference, verification, expected):
    proof = trusted_proof(verification, expected)
    if (any(proof.get(key) is not True for key in EXPANDED_REFERENCE_CHECKS) or proof.get('synthetic_only') is not False
        or proof.get('calibration_passed') is not False or proof.get('profitability_accepted') is not False
        or proof.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('과거 추가 학습의 이전 검산 완료·참조 오류')
    files = checked_files(reference)
    if set(files) != EXPANDED_REFERENCE_FILES:
        raise ValueError('과거 추가 학습의 이전 파일 목록 오류')
    manifest, summary, decision = (json.loads((reference/name).read_text()) for name in ['manifest.json', 'summary.json', 'decision.json'])
    if (manifest['settings']['policies'] != WEEKLY_FIRST_POLICIES or manifest['settings']['score_rows'] != 'first_eligible_only'
        or summary['complete'] is not True or summary['profitability_accepted'] is not False or summary['diagnosis_evaluated'] is not False
        or decision['calibration_passed'] is not False or decision['threshold_search'] is not False or decision['fallback_used'] is not False
        or decision['threshold'] != .5 or decision['candidate'] != 'weekly_first_linear'
        or any((reference/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])):
        raise ValueError('과거 추가 학습의 이전 첫 판단·탈락·진단 차단 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (snapshot.is_symlink() or set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or (snapshot/name).is_symlink() or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('과거 추가 학습의 이전 생성 코드 오류')
    calibration = checked_files(reference/'calibration')
    if set(calibration) != EXPANDED_CALIBRATION_FILES or sha256(reference/'calibration/files.json') != summary['calibration_files_sha256']:
        raise ValueError('과거 추가 학습의 이전 열한 정책 봉인 오류')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'), 'verification': str(verification),
        'verification_sha256': expected, 'files': files, 'calibration_files': calibration, 'source_sha256': sources,
        'previous_verified_evidence_reused': True, 'previous_models_refitted': False, 'external_nested_runs_reverified': False}


def checked_expansion_source(expansion, verification, expected, reference):
    proof = trusted_proof(verification, expected)
    if (proof.get('complete') is not True or proof.get('source_files_sha256') != sha256(expansion/'files.json')
        or proof.get('profitability_accepted') is not False or proof.get('synthetic_only', False) is not False
        or proof.get('new_first_selection_features_cash_independently_verified') is not True
        or proof.get('original_files_unchanged') != 5 or set(proof.get('phases', {})) != set(EXPANSION_PERIODS)):
        raise ValueError('과거 추가 학습의 확장 검산 완료·출처 오류')
    files = checked_files(expansion)
    if set(files) != {'manifest.json', 'reference_evidence.json', 'summary.json'}:
        raise ValueError('과거 추가 학습의 확장 파일 목록 오류')
    manifest, summary = (json.loads((expansion/name).read_text()) for name in ['manifest.json', 'summary.json'])
    settings = manifest['settings']
    if (settings['periods'] != EXPANSION_PERIODS or settings['features_used'] != EXPANDED_FIRST_FEATURES
        or settings['new_models_fitted'] is not False or settings['whole_policy_historically_available_claimed'] is not False
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V84.md')) or summary['complete'] is not True
        or summary['new_models_fitted'] is not False or summary['forced_boundary_closes_as_labels'] is not False
        or summary['profitability_accepted'] is not False or set(summary['phases']) != set(EXPANSION_PERIODS)):
        raise ValueError('과거 추가 학습의 고정 확장 계획·기간 오류')
    source = expansion/'generation_source/wonyotti_fr'
    hashes = settings['implementation_sha256']
    if (source.is_symlink() or set(hashes) != {p.name for p in source.glob('*.py')}
        or any(Path(name).name != name or (source/name).is_symlink() or sha256(source/name) != value for name, value in hashes.items())):
        raise ValueError('과거 추가 학습의 확장 생성 코드 오류')
    previous = json.loads((reference/'manifest.json').read_text())['settings']
    if settings['diagnosis_files_sha256'] != previous['reference_files_sha256']:
        raise ValueError('과거 추가 학습의 기존 첫 기회 참조 교차')
    phases = {}
    for name in EXPANSION_PERIODS:
        folder = expansion/name
        mapping = checked_files(folder)
        expected_files = {'summary.json', 'first_opportunity_ledger.parquet', 'outcomes.sqlite', 'training_labels.parquet', 'input_verification.json', 'all_positions.parquet'}
        if name == 'expansion_2020':
            expected_files.add('baseline_source.json')
        if set(mapping) != expected_files or sha256(folder/'files.json') != summary['phases'][name]['files_sha256']:
            raise ValueError('과거 추가 학습의 단계 원장 봉인 오류')
        row = sealed_phase(folder)
        checked = proof['phases'][name]
        if (any(checked.get(key) is not True for key in EXPANSION_PHASE_CHECKS)
            or checked.get('first_eligible') != row['selected_first'] or checked.get('all_positions') != row['positions']
            or checked.get('closed') != row['closed'] or checked.get('management_rows') != row['counts']['management_rows']
            or set(row['baseline_sha256']) != set(BASELINE_FILES)
            or any((folder/('outcomes.sqlite'+suffix)).exists() for suffix in ['-wal', '-shm'])):
            raise ValueError('과거 추가 학습의 단계 전수 검산·원장 연결 오류')
        if name == 'parity_2021' and (row['reference_parity']['complete'] is not True
            or checked.get('previous_reference_parity') != row['reference_parity']['splits']):
            raise ValueError('과거 추가 학습의 기존 연도 동일성 오류')
        if name == 'expansion_2020' and not Path(row['baseline']).resolve().is_relative_to(folder.resolve()):
            raise ValueError('과거 추가 학습의 추가 기준 경로 오류')
        phases[name] = {'files': mapping, 'files_sha256': sha256(folder/'files.json'), 'summary': row}
    return {'expansion': str(expansion), 'source_files_sha256': sha256(expansion/'files.json'), 'verification': str(verification),
        'verification_sha256': expected, 'files': files, 'phases': phases, 'generation_source_sha256': hashes,
        'verified_full_generation_reused': True, 'generation_engine_rerun': False, 'new_first_labels_fitted': False}


def load_expanded_baseline(reference):
    folder = reference/'calibration'
    result = {name: json.loads((folder/f'{name}.json').read_text()) for name in ['metrics', 'first_metrics', 'probability_metrics',
        'breakdown', 'first_breakdown', 'probability_breakdown', 'first_probability_metrics']}
    result['positions'] = {name: pd.read_parquet(folder/f'positions_{name}.parquet') for name in WEEKLY_FIRST_POLICIES}
    for name in ['predictions', 'first_membership']:
        result[name] = pd.read_parquet(folder/f'{name}.parquet')
    for name, file in [('calibration_decision', 'decision'), ('previous_linear_decision', 'previous_calibration_decision'),
        ('previous_first_decision', 'previous_first_decision'), ('earlier_regression_decision', 'earlier_regression_decision')]:
        result[name] = json.loads((reference/f'{file}.json').read_text())
    return result
