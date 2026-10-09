from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_linear_close import FIRST_LINEAR_SETTINGS, FirstLinearCloseModel
from .first_linear_reference import LINEAR_INPUT_FILES
from .first_opportunity_diagnostics import checked_first_inputs
from .source_snapshot import copy_snapshot
from .weekly_first_linear import WEEKLY_FIRST_SETTINGS, fit_weekly_first_linear
from .weekly_first_linear_evaluation import WEEKLY_FIRST_POLICIES, evaluate_weekly_first_linear
from .weekly_first_linear_reference import (
    checked_weekly_first_reference,
    load_weekly_first_baseline,
)


def run_weekly_first_linear_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_weekly_first_reference(reference, verification, verification_sha256)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V83.md')), 'features': FirstLinearCloseModel.features,
        'model_settings': FIRST_LINEAR_SETTINGS, 'weekly_settings': WEEKLY_FIRST_SETTINGS,
        'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'policies': WEEKLY_FIRST_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 9,
        'score_rows': 'first_eligible_only', 'target': 'unchanged_v81_natural_close_common_bps',
        'diagnosis_evaluated': False, 'sequential_calibration_labels_used_only_after_maturity': True,
        'whole_system_periods_already_observed': True, 'previous_verified_evidence_reused': True}
    out = new_run(output, 'weekly-first-linear-diagnosis', settings)
    print(f'첫 판단 모델의 확정 결과 주간 갱신: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'), ('previous_model.json', reference/'model.json'),
            ('previous_training_support.json', reference/'training_support.json'),
            ('previous_first_decision.json', reference/'previous_calibration_decision.json'),
            ('earlier_regression_decision.json', reference/'earlier_regression_decision.json')]:
            copy_snapshot(origin, out/destination)
        for name in LINEAR_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        parts, weights = checked_first_inputs(out)
        training, calibration = parts['early_training'], parts['calibration']
        old_model = json.loads((out/'previous_model.json').read_text())
        old_support = json.loads((out/'previous_training_support.json').read_text())
        scores, constants = fit_weekly_first_linear(training, calibration,
            pd.read_parquet(out/'first_training_ledger.parquet'), old_model, old_support, out)
        result = evaluate_weekly_first_linear(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, constants, old_support['training_constant_score'], load_weekly_first_baseline(reference))
        calibration_folder = out/'calibration'
        calibration_folder.mkdir(mode=0o700)
        save_policy_results(calibration_folder, result)
        result['first_membership'].to_parquet(calibration_folder/'first_membership.parquet', index=False)
        save_json(calibration_folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(calibration_folder/'files.json', {p.name: sha256(p) for p in calibration_folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'weekly_first_linear', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if checked_weekly_first_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('첫 판단 주간 실행 중 기존 검산 근거 변경')
        if any(sha256(out/name) != proof['files'][name] for name in LINEAR_INPUT_FILES):
            raise ValueError('첫 판단 주간 실행 중 원래 입력 사본 변경')
        if any(sha256(baseline_folder/name) != value for name, value in proof['calibration_files'].items()):
            raise ValueError('첫 판단 주간 실행 중 기존 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V83.md')) != settings['protocol_sha256']):
            raise ValueError('첫 판단 주간 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 9, 'first_model_exact': True,
            'rows': {name: len(frame) for name, frame in parts.items()},
            'positions': {name: int(frame.position_entry_time.nunique()) for name, frame in parts.items()},
            'calibration_files_sha256': sha256(calibration_folder/'files.json'), 'previous_models_refitted': False,
            'external_nested_runs_reverified': False, 'sequential_calibration_labels_used_only_after_maturity': True, **decision})
        (out/'REPORT.md').write_text('# 첫 판단 모델의 확정 결과 주간 갱신\n\n'
            f'내부 순차 비교 통과: {decision["calibration_passed"]}. '
            '각 주 이틀 전까지 확정된 포지션의 첫 행으로 같은 선형 모델을 갱신했다. '
            '기존 아홉 정책과 실패를 보존했으며 마지막 진단·연속 매매 평가는 수행하지 않았다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
