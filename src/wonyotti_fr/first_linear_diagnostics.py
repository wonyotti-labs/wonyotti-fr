from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_linear_close import FIRST_LINEAR_SETTINGS, FirstLinearCloseModel, fit_first_linear
from .first_linear_evaluation import LINEAR_POLICIES, evaluate_first_linear
from .first_linear_reference import (
    LINEAR_INPUT_FILES,
    checked_linear_reference,
    load_linear_baseline,
)
from .first_opportunity_close import first_opportunity_rows
from .first_opportunity_diagnostics import checked_first_inputs
from .source_snapshot import copy_snapshot


def run_first_linear_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_linear_reference(reference, verification, verification_sha256)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V82.md')), 'features': FirstLinearCloseModel.features,
        'model_settings': FIRST_LINEAR_SETTINGS, 'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'policies': LINEAR_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 1,
        'score_rows': 'first_eligible_only', 'scaler_weight_unit': 'one_per_first_eligible_position_including_zero_cost',
        'target': 'unchanged_v81_natural_close_common_bps', 'diagnosis_evaluated': False,
        'previous_verified_evidence_reused': True, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'first-linear-close-diagnosis', settings)
    print(f'첫 기회 비용 판단의 선형 규제: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'), ('previous_model.json', reference/'model.json'),
            ('previous_training_support.json', reference/'training_support.json'),
            ('earlier_regression_decision.json', reference/'previous_calibration_decision.json')]:
            copy_snapshot(origin, out/destination)
        for name in LINEAR_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        parts, weights = checked_first_inputs(out)
        training, calibration = parts['early_training'], parts['calibration']
        model, support = fit_first_linear(training, weights['early_training'], calibration,
            pd.read_parquet(out/'first_training_ledger.parquet'), out)
        previous_support = json.loads((out/'previous_training_support.json').read_text())
        for name in ['eligible_positions', 'fit_positions', 'zero_effect_positions', 'normalizer', 'training_constant_score']:
            if support[name] != previous_support[name]:
                raise ValueError('첫 기회 선형 모델의 기존 표본·비용·상수 불일치')
        first, positions = first_opportunity_rows(training)
        pd.testing.assert_frame_equal(positions, pd.read_parquet(out/'all_training_positions.parquet'), check_exact=True)
        contribution = pd.read_parquet(out/'first_training_contribution.parquet')
        expected = training[['decision_time', 'position_entry_time']].copy()
        expected['original_weight'] = weights['early_training'].sample_weight
        expected['first_eligible'] = expected.index.isin(first.opportunity_index)
        expected['position_weight'] = expected.first_eligible.astype(float)
        pd.testing.assert_frame_equal(contribution, expected, check_exact=True)
        first, _ = first_opportunity_rows(calibration)
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        result = evaluate_first_linear(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, support['training_constant_score'], load_linear_baseline(reference))
        calibration_folder = out/'calibration'
        calibration_folder.mkdir(mode=0o700)
        save_policy_results(calibration_folder, result)
        result['first_membership'].to_parquet(calibration_folder/'first_membership.parquet', index=False)
        save_json(calibration_folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(calibration_folder/'files.json', {p.name: sha256(p) for p in calibration_folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'first_linear', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if checked_linear_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('첫 기회 선형 실행 중 기존 검산 근거 변경')
        for name in LINEAR_INPUT_FILES:
            if sha256(out/name) != proof['files'][name]:
                raise ValueError('첫 기회 선형 실행 중 기존 원장 사본 변경')
        for name, value in proof['calibration_files'].items():
            if sha256(baseline_folder/name) != value:
                raise ValueError('첫 기회 선형 실행 중 기존 여덟 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V82.md')) != settings['protocol_sha256']):
            raise ValueError('첫 기회 선형 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1, 'previous_first_ledger_exact': True,
            'rows': {name: len(frame) for name, frame in parts.items()},
            'positions': {name: int(frame.position_entry_time.nunique()) for name, frame in parts.items()},
            'calibration_files_sha256': sha256(calibration_folder/'files.json'), 'previous_models_refitted': False,
            'external_nested_runs_reverified': False, 'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 첫 기회 비용 판단의 선형 규제\n\n'
            f'내부 보정 통과: {decision["calibration_passed"]}. '
            '같은 첫 기회·현금·비중·비용과 단일 판단을 유지하고 표준화·L2 로지스틱 모델 하나를 적합했다. '
            '이전 여덟 정책과 실패를 보존했으며 기존 모델의 재학습·마지막 진단·연속 매매 평가는 수행하지 않았다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
