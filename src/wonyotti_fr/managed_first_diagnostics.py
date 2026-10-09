from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_linear_close import FIRST_LINEAR_SETTINGS
from .first_opportunity_diagnostics import checked_first_inputs
from .managed_first_evaluation import MANAGED_POLICIES, evaluate_managed_first
from .managed_first_fit import fit_managed_first
from .managed_first_model import ManagedFirstLinearModel
from .managed_first_parent import checked_manager_parent
from .managed_first_reference import (
    MANAGED_INPUT_FILES,
    checked_managed_reference,
    load_managed_baseline,
)
from .source_snapshot import copy_snapshot


def run_managed_first_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_managed_reference(reference, verification, verification_sha256)
    manager, parent = checked_manager_parent(reference)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V87.md')), 'features': ManagedFirstLinearModel.features,
        'model_settings': FIRST_LINEAR_SETTINGS, 'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'added_period': ['2020-04-01', '2021-01-01'], 'added_label_cutoff': '2020-12-31',
        'policies': MANAGED_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 1,
        'score_rows': 'first_eligible_only', 'target': 'unchanged_v85_expanded_first_natural_close_common_bps',
        'diagnosis_evaluated': False, 'whole_policy_historically_available_claimed': False,
        'whole_system_periods_already_observed': True, 'previous_verified_evidence_reused': True, 'parent': parent['parent'], 'parent_sha256': parent['parent_sha256']}
    out = new_run(output, 'managed-first-linear-diagnosis', settings)
    print(f'현재 관리 확률의 첫 선형 비교: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        save_json(out/'manager_evidence.json', parent)
        for name, item in parent['metadata'].items():
            copy_snapshot(Path(item['path']), out/name)
        copy_snapshot(Path(parent['parent'])/'frozen_selection.json', out/'manager_parent_selection.json')
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'), ('previous_model.json', reference/'model.json'),
            ('previous_training_support.json', reference/'training_support.json')]:
            copy_snapshot(origin, out/destination)
        for name in MANAGED_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        model, support = fit_managed_first(reference, manager, out)
        parts, weights = checked_first_inputs(out)
        calibration = parts['calibration']
        first = pd.read_parquet(out/'managed_first_calibration.parquet')
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        result = evaluate_managed_first(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, load_managed_baseline(reference))
        folder = out/'calibration'
        folder.mkdir(mode=0o700)
        save_policy_results(folder, result)
        result['first_membership'].to_parquet(folder/'first_membership.parquet', index=False)
        save_json(folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'managed_first_linear', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if checked_manager_parent(reference)[1] != parent:
            raise ValueError('관리 확률 첫 선형 실행 중 부모·메타데이터 변경')
        if any(sha256(out/name) != item['sha256'] for name, item in parent['metadata'].items()):
            raise ValueError('관리 확률 첫 선형 실행 중 부모 연결 사본 변경')
        if sha256(out/'manager_parent_selection.json') != parent['parent_sha256']:
            raise ValueError('관리 확률 첫 선형 실행 중 고정 부모 사본 변경')
        if checked_managed_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('관리 확률 첫 선형 실행 중 이전 검산 근거 변경')
        if any(sha256(out/name) != proof['files'][name] for name in MANAGED_INPUT_FILES):
            raise ValueError('관리 확률 첫 선형 실행 중 기존 원장 사본 변경')
        if any(sha256(baseline_folder/name) != value for name, value in proof['calibration_files'].items()):
            raise ValueError('관리 확률 첫 선형 실행 중 기존 열네 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V87.md')) != settings['protocol_sha256']):
            raise ValueError('관리 확률 첫 선형 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1,
            'combined_first_positions': support['combined_first_positions'], 'phase_first_positions': support['phase_first_positions'],
            'calibration_files_sha256': sha256(folder/'files.json'), 'previous_models_refitted': False,
            'generation_engine_rerun': False, 'external_nested_runs_reverified': False, 'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 현재 관리 확률의 첫 선형 비교\n\n'
            f'내부 보정 통과: {decision["calibration_passed"]}. '
            '같은 통합 첫 원장·현금·비용에 고정 관리 확률 3개를 붙여 같은 L2 선형 모형을 적합했다. '
            '기존 열네 정책의 저장된 점수로 산식을 다시 대조하고 실패와 전체 원장을 보존했다. '
            '과거 생성 정책의 당시 가용 성과나 연속 매매 수익성으로 해석하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
