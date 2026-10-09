from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_opportunity_close import (
    FirstOpportunityCloseModel,
    first_opportunity_rows,
    fit_first_opportunity,
)
from .first_opportunity_evaluation import FIRST_POLICIES, evaluate_first_opportunity
from .first_opportunity_reference import (
    FIRST_INPUT_FILES,
    checked_first_reference,
    load_first_baseline,
)
from .histogram_management import HISTOGRAM_SETTINGS
from .source_snapshot import copy_snapshot
from .stopping_close import validate_stopping_rows


def checked_first_inputs(reference):
    parts, weights = {}, {}
    assignment = pd.read_parquet(reference/'threshold_exclusion_ledger.parquet',
        columns=['decision_time', 'position_entry_time', 'split'])
    for name in ['early_training', 'calibration']:
        frame = pd.read_parquet(reference/f'{name}_used.parquet')
        validate_stopping_rows(frame)
        start, end = [pd.Timestamp(value, tz='UTC') for value in THRESHOLD_SPLITS[name]]
        if frame.position_entry_time.lt(start).any() or frame.decision_time.lt(start).any() or frame.label_end.ge(end).any():
            raise ValueError('첫 적격 기회의 학습·보정 시간 경계 오류')
        pd.testing.assert_frame_equal(frame[['decision_time', 'position_entry_time']],
            assignment.loc[assignment.split.eq(name), ['decision_time', 'position_entry_time']].reset_index(drop=True), check_exact=True)
        weight = pd.read_parquet(reference/f'{name}_weights.parquet')
        pd.testing.assert_frame_equal(weight.drop(columns='sample_weight'), frame[['decision_time', 'position_entry_time']], check_exact=True)
        np.testing.assert_array_equal(weight.sample_weight, position_weights(frame))
        parts[name], weights[name] = frame, weight
    if set(parts['early_training'].position_entry_time) & set(parts['calibration'].position_entry_time):
        raise ValueError('첫 적격 기회의 학습·보정 포지션 교차')
    return parts, weights


def run_first_opportunity_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_first_reference(reference, verification, verification_sha256)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V81.md')), 'features': FirstOpportunityCloseModel.features,
        'model_settings': HISTOGRAM_SETTINGS, 'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'policies': FIRST_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 1,
        'fit_weight_unit': 'one_per_first_eligible_position', 'score_rows': 'first_eligible_only',
        'target': 'natural_close_cash_difference_over_first_available_equity_bps', 'diagnosis_evaluated': False,
        'previous_full_replay_and_independent_audit_reused': True, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'first-opportunity-close-diagnosis', settings)
    print(f'첫 적격 기회의 단일 청산 판단: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        copy_snapshot(verification, out/'previous_verification.json')
        copy_snapshot(reference/'files.json', out/'previous_files.json')
        copy_snapshot(reference/'calibration_decision.json', out/'previous_calibration_decision.json')
        for name in FIRST_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        parts, weights = checked_first_inputs(out)
        training, calibration = parts['early_training'], parts['calibration']
        model, support = fit_first_opportunity(training, weights['early_training'], calibration, out)
        first, _ = first_opportunity_rows(calibration)
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        result = evaluate_first_opportunity(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, support['training_constant_score'], load_first_baseline(reference))
        calibration_folder = out/'calibration'
        calibration_folder.mkdir(mode=0o700)
        save_policy_results(calibration_folder, result)
        result['first_membership'].to_parquet(calibration_folder/'first_membership.parquet', index=False)
        save_json(calibration_folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(calibration_folder/'files.json', {p.name: sha256(p) for p in calibration_folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'first_opportunity', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if checked_first_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('첫 적격 기회 실행 중 기존 검산 근거 변경')
        for name in FIRST_INPUT_FILES:
            if sha256(out/name) != proof['files'][name]:
                raise ValueError('첫 적격 기회의 기존 전체 입력 사본 변경')
        for name, value in proof['calibration_files'].items():
            if sha256(baseline_folder/name) != value:
                raise ValueError('첫 적격 기회의 기존 보정 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V81.md')) != settings['protocol_sha256']):
            raise ValueError('첫 적격 기회 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1,
            'rows': {name: len(frame) for name, frame in parts.items()},
            'positions': {name: int(frame.position_entry_time.nunique()) for name, frame in parts.items()},
            'calibration_files_sha256': sha256(calibration_folder/'files.json'), 'previous_models_refitted': False,
            'external_nested_runs_reverified': False, 'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 첫 적격 기회의 단일 청산 판단\n\n'
            f'내부 보정 통과: {decision["calibration_passed"]}. '
            '각 포지션의 첫 적격 기회에서 한 번만 판단했다. 원래 전체 행·실패·기회 없음은 보존했다. '
            '이전 검산 근거를 지문으로 재사용했으며 이전 모델의 재학습이나 마지막 구간 평가는 수행하지 않았다. '
            '내부 보정 통과도 연속 계좌 수익성의 증거가 아니다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
