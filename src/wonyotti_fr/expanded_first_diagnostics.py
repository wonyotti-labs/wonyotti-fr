from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .expanded_first_evaluation import EXPANDED_FIRST_POLICIES, evaluate_expanded_first
from .expanded_first_linear import fit_expanded_first_linear
from .expanded_first_reference import (
    checked_expanded_baseline,
    checked_expansion_source,
    load_expanded_baseline,
)
from .first_linear_close import FIRST_LINEAR_SETTINGS, FirstLinearCloseModel
from .first_linear_reference import LINEAR_INPUT_FILES
from .first_opportunity_close import first_opportunity_rows
from .first_opportunity_diagnostics import checked_first_inputs
from .source_snapshot import copy_snapshot


def run_expanded_first_diagnosis(reference: Path, verification: Path, verification_sha256: str,
                                 expansion: Path, expansion_verification: Path, expansion_verification_sha256: str,
                                 output: Path) -> Path:
    proof = checked_expanded_baseline(reference, verification, verification_sha256)
    expanded_proof = checked_expansion_source(expansion, expansion_verification, expansion_verification_sha256, reference)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'expansion': str(expansion), 'expansion_files_sha256': expanded_proof['source_files_sha256'],
        'expansion_verification': str(expansion_verification), 'expansion_verification_sha256': expansion_verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V85.md')), 'features': FirstLinearCloseModel.features,
        'model_settings': FIRST_LINEAR_SETTINGS, 'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'added_period': ['2020-04-01', '2021-01-01'], 'added_label_cutoff': '2020-12-31',
        'policies': EXPANDED_FIRST_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 1,
        'score_rows': 'first_eligible_only', 'scaler_weight_unit': 'one_per_first_eligible_position_including_zero_cost',
        'diagnosis_evaluated': False, 'whole_policy_historically_available_claimed': False,
        'whole_system_periods_already_observed': True, 'previous_verified_evidence_reused': True}
    out = new_run(output, 'expanded-first-linear-diagnosis', settings)
    print(f'과거 첫 기회 추가 학습: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        save_json(out/'expansion_evidence.json', expanded_proof)
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('expansion_verification.json', expansion_verification), ('expansion_files.json', expansion/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'),
            ('previous_training_support.json', reference/'previous_training_support.json')]:
            copy_snapshot(origin, out/destination)
        for name in LINEAR_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        for name in ['first_opportunity_ledger.parquet', 'all_positions.parquet', 'training_labels.parquet']:
            copy_snapshot(expansion/'expansion_2020'/name, out/('added_'+name))
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        parts, weights = checked_first_inputs(out)
        training, calibration = parts['early_training'], parts['calibration']
        added = pd.read_parquet(out/'added_first_opportunity_ledger.parquet')
        positions = pd.read_parquet(out/'added_all_positions.parquet')
        pd.testing.assert_frame_equal(added[added.label_status.eq('closed')].reset_index(drop=True),
            pd.read_parquet(out/'added_training_labels.parquet'), check_exact=True)
        model, support = fit_expanded_first_linear(training, weights['early_training'], calibration,
            pd.read_parquet(out/'first_training_ledger.parquet'), added, positions, out)
        _, original_positions = first_opportunity_rows(training)
        pd.testing.assert_frame_equal(original_positions, pd.read_parquet(out/'all_training_positions.parquet'), check_exact=True)
        first, _ = first_opportunity_rows(calibration)
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        constant = json.loads((out/'previous_training_support.json').read_text())['training_constant_score']
        result = evaluate_expanded_first(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, support['training_constant_score'], constant, load_expanded_baseline(reference))
        folder = out/'calibration'
        folder.mkdir(mode=0o700)
        save_policy_results(folder, result)
        result['first_membership'].to_parquet(folder/'first_membership.parquet', index=False)
        save_json(folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'expanded_first_linear', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if (checked_expanded_baseline(reference, verification, verification_sha256) != proof
            or checked_expansion_source(expansion, expansion_verification, expansion_verification_sha256, reference) != expanded_proof):
            raise ValueError('과거 추가 학습 중 기존·확장 검산 근거 변경')
        for name in LINEAR_INPUT_FILES:
            if sha256(out/name) != proof['files'][name]:
                raise ValueError('과거 추가 학습 중 기존 원장 사본 변경')
        for name in ['first_opportunity_ledger.parquet', 'all_positions.parquet', 'training_labels.parquet']:
            if sha256(out/('added_'+name)) != expanded_proof['phases']['expansion_2020']['files'][name]:
                raise ValueError('과거 추가 학습 중 확장 원장 사본 변경')
        if any(sha256(baseline_folder/name) != value for name, value in proof['calibration_files'].items()):
            raise ValueError('과거 추가 학습 중 기존 열한 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V85.md')) != settings['protocol_sha256']):
            raise ValueError('과거 추가 학습 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1,
            'combined_first_positions': support['combined_first_positions'], 'phase_first_positions': support['phase_first_positions'],
            'calibration_files_sha256': sha256(folder/'files.json'), 'previous_models_refitted': False,
            'generation_engine_rerun': False, 'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 과거 첫 기회 추가 학습\n\n'
            f'내부 보정 통과: {decision["calibration_passed"]}. '
            '같은 선형 설정·입력·현금 정답에 과거 확정 첫 기회를 추가했다. '
            '기존 열한 정책·전체 원장·손실을 보존하고 새 모델과 상수를 비교했다. '
            '과거 생성 정책의 당시 가용 성과나 연속 매매 수익성으로 해석하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
