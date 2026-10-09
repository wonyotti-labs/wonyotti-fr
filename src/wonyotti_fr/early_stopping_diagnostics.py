from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_threshold import THRESHOLD_SPLITS, ThresholdCloseModel, threshold_splits
from .close_threshold_diagnostics import (
    THRESHOLD_INPUT_FILES,
    load_retained_evaluation,
    save_evaluation,
)
from .common import new_run, save_json, sha256
from .early_stopping_close import (
    EARLY_STOPPING_SETTINGS,
    EarlyStoppingCloseModel,
    fit_early_stopping_close,
)
from .early_stopping_evaluation import (
    EARLY_STOPPING_COMPARISONS,
    EARLY_STOPPING_POLICIES,
    evaluate_early_stopping_close,
    evaluate_new_policies,
)
from .early_stopping_reference import (
    checked_calibration,
    checked_retained_reproduction,
    reproduce_threshold_reference,
)
from .histogram_management import HISTOGRAM_SETTINGS
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot

EARLY_STOPPING_INPUT_FILES = THRESHOLD_INPUT_FILES | {'early_training_used.parquet', 'early_training_weights.parquet',
    'early_training_cost_ledger.parquet', 'calibration_used.parquet', 'calibration_weights.parquet', 'threshold_exclusion_ledger.parquet'}


def save_policy_results(output, result):
    save_evaluation(output, result)
    for name in ['probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown']:
        save_json(output/f'{name}.json', result[name])


def run_early_stopping_diagnosis(reference: Path, output: Path) -> Path:
    settings = {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V79.md')), 'periods': THRESHOLD_SPLITS,
        'features': EarlyStoppingCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'early_stopping_settings': EARLY_STOPPING_SETTINGS, 'teacher_models': 5, 'candidate_models': 1,
        'candidate': 'early_stopping', 'policies': EARLY_STOPPING_POLICIES, 'comparisons': EARLY_STOPPING_COMPARISONS,
        'score_kind': 'cost_weighted_current_vs_future_policy_close', 'score_threshold': .5,
        'evaluation_target': 'original_natural_close_advantage', 'evaluation_weights': 'original_full_minute_position_weights',
        'original_target_probability_metrics_are_descriptive_only': True,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'early-stopping-close-diagnosis', settings)
    print(f'분별 이후 정책 대비 현재 청산 학습: {out}', flush=True)
    try:
        reproduced = reproduce_threshold_reference(reference, out)
        print('기존 모델·일곱 문턱·선택 실패·진단 차단 재현 완료', flush=True)
        for name in sorted(EARLY_STOPPING_INPUT_FILES):
            copy_snapshot(reproduced/name, out/name)
        rows, assignment = threshold_splits(pd.read_parquet(out/'minute_ledger.parquet'))
        pd.testing.assert_frame_equal(assignment, pd.read_parquet(out/'threshold_exclusion_ledger.parquet'), check_exact=True)
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out/f'{name}_used.parquet'), check_exact=True)
        weights = pd.read_parquet(out/'early_training_weights.parquet')
        indices = np.flatnonzero(assignment.split.eq('early_training'))
        model, support = fit_early_stopping_close(rows['early_training'], weights, rows['calibration'], indices, out)
        natural = ThresholdCloseModel.from_dict(json.loads((reproduced/'model.json').read_text()))
        calibration_frame = rows['calibration'].assign(sample_weight=pd.read_parquet(out/'calibration_weights.parquet').sample_weight)
        original_calibration = pd.read_parquet(reproduced/'calibration/predictions.parquet')
        natural_scores = natural.probabilities(calibration_frame[natural.features].to_numpy(dtype=float))[:, 0]
        np.testing.assert_array_equal(natural_scores, original_calibration.early_score)
        scores = model.probabilities(calibration_frame[model.features].to_numpy(dtype=float))[:, 0]
        calibration = evaluate_new_policies(calibration_frame, scores, natural_scores, support['training_constant_score'])
        keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
        pd.testing.assert_frame_equal(calibration['predictions'][keys], original_calibration[keys], check_exact=True)
        pd.testing.assert_frame_equal(calibration['positions']['early_natural'],
            pd.read_parquet(reproduced/'calibration/positions_threshold_0.5.parquet'), check_exact=True)
        calibration_root = out/'calibration'
        calibration_root.mkdir(mode=0o700)
        save_policy_results(calibration_root, calibration)
        save_json(out/'calibration_decision.json', calibration['calibration_decision'])
        save_json(calibration_root/'files.json', {p.name: sha256(p) for p in calibration_root.iterdir() if p.is_file()})
        calibration_seal = sha256(calibration_root/'files.json')
        passed = calibration['calibration_decision']['calibration_passed']
        if passed:
            frame = rows['diagnosis'].assign(sample_weight=pd.read_parquet(out/'diagnosis_weights.parquet').sample_weight)
            scores = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
            natural_scores = natural.probabilities(frame[natural.features].to_numpy(dtype=float))[:, 0]
            retained = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
            result = evaluate_early_stopping_close(frame, load_retained_evaluation(retained), scores, natural_scores,
                support['training_constant_score'], calibration)
            save_policy_results(out, result)
            result['draws'].to_parquet(out/'block_draws.parquet', index=False)
            for name in EARLY_STOPPING_COMPARISONS:
                result['blocks'][name].to_parquet(out/f'{name}_blocks.parquet', index=False)
                result['replicates'][name].to_parquet(out/f'{name}_block_replicates.parquet', index=False)
            save_json(out/'intervals.json', result['intervals'])
            decision = result['decision']
        else:
            decision = {'checks': {'calibration_passed': False}, 'early_stopping_admitted': False,
                'candidate': None, 'trading_returns_evaluated': False, 'reason': 'fixed_candidate_failed_calibration'}
        save_json(out/'decision.json', decision)
        baseline = json.loads((out/'baseline_files.json').read_text())
        if (sha256(reference/'files.json') != settings['reference_files_sha256'] or checked_files(reference) != baseline
            or any(sha256(out/'baseline_source'/name) != value for name, value in baseline.items())
            or any(sha256(out/name) != sha256(reproduced/name) for name in EARLY_STOPPING_INPUT_FILES)
            or sha256(calibration_root/'files.json') != calibration_seal):
            raise ValueError('분별 이후 청산 학습 중 이전 자료·현재 보정 변경')
        checked_files(calibration_root)
        old_calibration = checked_calibration(reference)
        if old_calibration != checked_calibration(reproduced) or old_calibration != checked_files(out/'baseline_calibration'):
            raise ValueError('분별 이후 청산 학습 중 이전 보정 변경')
        proof = json.loads((out/'reference_parity.json').read_text())
        if (sha256(reproduced/'files.json') != proof['reproduction_files_sha256']
            or sha256(out/'baseline_calibration/files.json') != proof['calibration_files_sha256']):
            raise ValueError('분별 이후 청산 학습 중 이전 재현 봉인 변경')
        checked_files(reproduced)
        parent = Path(json.loads((reference/'manifest.json').read_text())['settings']['reference'])
        for key, folder in [('source', reference), ('reproduced', reproduced)]:
            if checked_retained_reproduction(folder, parent) != proof['previous_reproductions'][key]:
                raise ValueError('분별 이후 청산 학습 중 중첩 재현 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V79.md')) != settings['protocol_sha256']):
            raise ValueError('분별 이후 청산 학습 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'original_rows_weights_and_targets_preserved': True, 'whole_positions_disjoint': True,
            'teacher_models': 5, 'candidate_models': 1, 'policy_iterations': 1, 'threshold_search': False,
            'refit_after_calibration': False, 'diagnosis_evaluated': passed,
            'rows': {key: len(value) for key, value in rows.items()},
            'positions': {key: int(value.position_entry_time.nunique()) for key, value in rows.items()},
            'calibration_files_sha256': calibration_seal, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 분별 이후 정책 대비 현재 청산 학습\n\n'
            f'내부 보정 통과: {passed}. 마지막 진단 실행: {passed}. 사전 조건 통과: {decision["early_stopping_admitted"]}. '
            '현재보다 뒤에 있는 보조 정책의 첫 청산 현금과 비교한 정답으로 후보 하나를 적합했다. '
            '문턱은 0.5로 고정했고 보정 후 다시 적합하지 않았다. 기존 실패와 모든 원래 행을 보존했다. '
            '개발 경로의 조건부 효과이며 자체 상태의 연속 계좌 수익을 뜻하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
