from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_linear_close import FIRST_LINEAR_SETTINGS
from .first_opportunity_diagnostics import checked_first_inputs
from .regularized_first_evaluation import REGULARIZED_POLICIES, evaluate_regularized_first
from .regularized_first_model import REGULARIZATION_STRENGTHS, RegularizedFirstModel
from .regularized_first_reference import (
    REGULARIZED_INPUT_FILES,
    checked_regularized_reference,
    load_regularized_baseline,
)
from .regularized_first_selection import REGULARIZATION_WINDOWS, fit_regularized_first
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot


def seal_regularization_selection(folder):
    folds = {}
    for number in range(3):
        name = f'fold-{number:02}'
        child = folder/name
        if child.is_symlink() or {p.name for p in child.iterdir()} != {'models.json', 'support.json', 'predictions.parquet'}:
            raise ValueError('시간순 정규화의 회차 출력 목록 오류')
        save_json(child/'files.json', {p.name: sha256(p) for p in child.iterdir() if p.is_file()})
        checked_files(child)
        folds[name] = sha256(child/'files.json')
    save_json(folder/'folds.json', folds)
    save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
    checked_files(folder)


def run_regularized_first_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_regularized_reference(reference, verification, verification_sha256)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V90.md')), 'features': RegularizedFirstModel.features,
        'base_model_settings': {name: value for name, value in FIRST_LINEAR_SETTINGS.items() if name != 'C'},
        'regularization_strengths': list(REGULARIZATION_STRENGTHS), 'internal_validation_windows': REGULARIZATION_WINDOWS,
        'training_purge_days': 2, 'selection_criterion': 'pooled_absolute_cost_log_loss', 'exact_tie_break': 'smaller_C',
        'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'added_period': ['2020-04-01', '2021-01-01'], 'policies': REGULARIZED_POLICIES, 'threshold': .5,
        'threshold_search': False, 'new_models': 13, 'selection_models': 12, 'final_models': 1,
        'score_rows': 'first_eligible_only', 'external_calibration_used_for_selection': False, 'diagnosis_evaluated': False,
        'whole_policy_historically_available_claimed': False, 'whole_system_periods_already_observed': True,
        'previous_verified_evidence_reused': True, 'market_and_parent_inputs_regenerated': False}
    out = new_run(output, 'regularized-first-linear-diagnosis', settings)
    print(f'앞 학습 시간순 정규화의 첫 판단 비교: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'), ('previous_model.json', reference/'model.json'),
            ('previous_training_support.json', reference/'training_support.json')]:
            copy_snapshot(origin, out/destination)
        for name in REGULARIZED_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline = out/'baseline_calibration'
        baseline.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline/name)
        model, support = fit_regularized_first(reference, out)
        seal_regularization_selection(out/'selection')
        parts, weights = checked_first_inputs(out)
        first = pd.read_parquet(out/'minute_first_calibration.parquet')
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        result = evaluate_regularized_first(parts['calibration'].assign(sample_weight=weights['calibration'].sample_weight),
            scores, load_regularized_baseline(reference))
        folder = out/'calibration'
        folder.mkdir(mode=0o700)
        save_policy_results(folder, result)
        result['first_membership'].to_parquet(folder/'first_membership.parquet', index=False)
        save_json(folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'regularized_first_linear',
            'selected_strength': support['selected_strength'], 'external_calibration_used_for_selection': False,
            'diagnosis_evaluated': False, 'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if checked_regularized_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('시간순 정규화 실행 중 이전 검산 근거 변경')
        if any(sha256(out/name) != proof['files'][name] for name in REGULARIZED_INPUT_FILES):
            raise ValueError('시간순 정규화 실행 중 기존 입력 사본 변경')
        if any(sha256(baseline/name) != value for name, value in proof['calibration_files'].items()):
            raise ValueError('시간순 정규화 실행 중 이전 열일곱 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V90.md')) != settings['protocol_sha256']):
            raise ValueError('시간순 정규화 실행 중 코드·계획 변경')
        checked_files(out/'selection')
        for name, digest in json.loads((out/'selection/folds.json').read_text()).items():
            if sha256(out/'selection'/name/'files.json') != digest:
                raise ValueError('시간순 정규화 실행 중 내부 회차 봉인 변경')
            checked_files(out/'selection'/name)
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 13, 'selection_models_fitted': 12, 'final_models_fitted': 1,
            'combined_first_positions': support['combined_first_positions'], 'phase_first_positions': support['phase_first_positions'],
            'calibration_files_sha256': sha256(folder/'files.json'), 'selection_files_sha256': sha256(out/'selection/files.json'),
            'previous_models_refitted': False, 'generation_engine_rerun': False, 'external_nested_runs_reverified': False,
            'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 앞 학습 시간순 정규화의 첫 판단 비교\n\n'
            f'내부 선택 C: {support["selected_strength"]}. 외부 보정 통과: {decision["calibration_passed"]}. '
            '같은 앞 학습의 세 시간 구간에서 네 강도를 비교하고 통합 비용 로그 손실로 강도를 고정했다. '
            '기존 원장·79개 입력·열일곱 정책을 보존했다. 전체 계좌 수익성이나 당시 가용 성과로 해석하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
