from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_threshold import THRESHOLD_SPLITS, ThresholdCloseModel, threshold_splits
from .close_threshold_diagnostics import load_retained_evaluation
from .common import new_run, save_json, sha256
from .early_stopping_close import EarlyStoppingCloseModel
from .early_stopping_diagnostics import EARLY_STOPPING_INPUT_FILES, save_policy_results
from .entry_regression import REGRESSION_SETTINGS
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot
from .stopping_regression import StoppingRegressionModel, fit_stopping_regression
from .stopping_regression_evaluation import (
    REGRESSION_COMPARISONS,
    REGRESSION_POLICIES,
    evaluate_regression_policies,
    evaluate_stopping_regression,
)
from .stopping_regression_reference import (
    checked_stopping_calibration,
    checked_threshold_reproduction,
    reproduce_stopping_reference,
)

REGRESSION_INPUT_FILES = EARLY_STOPPING_INPUT_FILES | {'early_teacher_models.json', 'early_teacher_support.json',
    'early_teacher_training_membership.parquet', 'early_stopping_targets.parquet', 'stopping_training_cost_ledger.parquet'}


def compare_prior_positions(current, saved):
    left, right = current.copy(), saved.copy()
    # 전부 NaT인 열은 Parquet 왕복에서 단위가 달라지므로 시각만 정규화한다.
    for frame in [left, right]:
        for name in ['position_entry_time', 'first_available_time', 'first_selected_time', 'first_label_end']:
            frame[name] = pd.to_datetime(frame[name], utc=True).astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def run_stopping_regression_diagnosis(reference: Path, output: Path) -> Path:
    settings = {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V80.md')), 'periods': THRESHOLD_SPLITS,
        'features': StoppingRegressionModel.features, 'settings': REGRESSION_SETTINGS,
        'new_model_count': 1, 'existing_models_reproduced': True, 'candidate': 'stopping_regression',
        'policies': REGRESSION_POLICIES, 'comparisons': REGRESSION_COMPARISONS,
        'prediction_kind': 'current_vs_future_policy_close_bps', 'threshold_bps': 0.,
        'training_target': 'unchanged_v79_stopping_advantage_bps', 'training_weights': 'original_full_early_position_weights',
        'zero_targets_used_with_original_weight': True, 'evaluation_target': 'original_natural_close_advantage',
        'regression_predictions_are_probabilities': False, 'threshold_search': False, 'refit_after_calibration': False,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'stopping-regression-diagnosis', settings)
    print(f'이후 청산 대비 현금 효과의 직접 회귀: {out}', flush=True)
    try:
        reproduced = reproduce_stopping_reference(reference, out)
        print('기존 보조 정책·현금 정답·고정 분류·내부 탈락 재현 완료', flush=True)
        for name in sorted(REGRESSION_INPUT_FILES):
            copy_snapshot(reproduced/name, out/name)
        rows, assignment = threshold_splits(pd.read_parquet(out/'minute_ledger.parquet'))
        pd.testing.assert_frame_equal(assignment, pd.read_parquet(out/'threshold_exclusion_ledger.parquet'), check_exact=True)
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out/f'{name}_used.parquet'), check_exact=True)
        weights = pd.read_parquet(out/'early_training_weights.parquet')
        targets = pd.read_parquet(out/'early_stopping_targets.parquet')
        model, support = fit_stopping_regression(rows['early_training'], weights, targets, rows['calibration'],
            np.flatnonzero(assignment.split.eq('early_training')), out)
        stopping = EarlyStoppingCloseModel.from_dict(json.loads((reproduced/'model.json').read_text()))
        natural = ThresholdCloseModel.from_dict(json.loads((reproduced/'baseline_source/model.json').read_text()))
        frame = rows['calibration'].assign(sample_weight=pd.read_parquet(out/'calibration_weights.parquet').sample_weight)
        old_predictions = pd.read_parquet(reproduced/'calibration/predictions.parquet')
        values = frame[model.features].to_numpy(dtype=float)
        natural_scores, stopping_scores = natural.probabilities(values)[:, 0], stopping.probabilities(values)[:, 0]
        np.testing.assert_array_equal(natural_scores, old_predictions.early_natural_score)
        np.testing.assert_array_equal(stopping_scores, old_predictions.early_stopping_score)
        calibration = evaluate_regression_policies(frame, model.predict(values), stopping_scores, natural_scores, support['training_constant_bps'])
        keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
        pd.testing.assert_frame_equal(calibration['predictions'][keys], old_predictions[keys], check_exact=True)
        for name in ['early_natural', 'early_stopping', 'always_first', 'never_extra']:
            compare_prior_positions(calibration['positions'][name], pd.read_parquet(reproduced/f'calibration/positions_{name}.parquet'))
        calibration_root = out/'calibration'
        calibration_root.mkdir(mode=0o700)
        save_policy_results(calibration_root, calibration)
        save_json(out/'calibration_decision.json', calibration['calibration_decision'])
        save_json(calibration_root/'files.json', {p.name: sha256(p) for p in calibration_root.iterdir() if p.is_file()})
        calibration_seal = sha256(calibration_root/'files.json')
        passed = calibration['calibration_decision']['calibration_passed']
        if passed:
            frame = rows['diagnosis'].assign(sample_weight=pd.read_parquet(out/'diagnosis_weights.parquet').sample_weight)
            values = frame[model.features].to_numpy(dtype=float)
            threshold = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
            retained = Path(json.loads((threshold/'reference_parity.json').read_text())['reproduction'])
            result = evaluate_stopping_regression(frame, load_retained_evaluation(retained), model.predict(values),
                stopping.probabilities(values)[:, 0], natural.probabilities(values)[:, 0], support['training_constant_bps'], calibration)
            save_policy_results(out, result)
            result['draws'].to_parquet(out/'block_draws.parquet', index=False)
            for name in REGRESSION_COMPARISONS:
                result['blocks'][name].to_parquet(out/f'{name}_blocks.parquet', index=False)
                result['replicates'][name].to_parquet(out/f'{name}_block_replicates.parquet', index=False)
            save_json(out/'intervals.json', result['intervals'])
            decision = result['decision']
        else:
            decision = {'checks': {'calibration_passed': False}, 'stopping_regression_admitted': False,
                'candidate': None, 'trading_returns_evaluated': False, 'reason': 'fixed_regression_failed_calibration'}
        save_json(out/'decision.json', decision)
        baseline = json.loads((out/'baseline_files.json').read_text())
        if (sha256(reference/'files.json') != settings['reference_files_sha256'] or checked_files(reference) != baseline
            or any(sha256(out/'baseline_source'/name) != value for name, value in baseline.items())
            or any(sha256(out/name) != sha256(reproduced/name) for name in REGRESSION_INPUT_FILES)
            or sha256(calibration_root/'files.json') != calibration_seal):
            raise ValueError('이후 청산 회귀 중 기존 원장·정답·보정 변경')
        checked_files(calibration_root)
        old_calibration = checked_stopping_calibration(reference)
        if old_calibration != checked_stopping_calibration(reproduced) or old_calibration != checked_files(out/'baseline_calibration'):
            raise ValueError('이후 청산 회귀 중 기존 다섯 보정 정책 변경')
        proof = json.loads((out/'reference_parity.json').read_text())
        if (sha256(reproduced/'files.json') != proof['reproduction_files_sha256']
            or sha256(out/'baseline_calibration/files.json') != proof['calibration_files_sha256']):
            raise ValueError('이후 청산 회귀 중 기존 재현 봉인 변경')
        checked_files(reproduced)
        parent = Path(json.loads((reference/'manifest.json').read_text())['settings']['reference'])
        for key, folder in [('source', reference), ('reproduced', reproduced)]:
            if checked_threshold_reproduction(folder, parent) != proof['previous_reproductions'][key]:
                raise ValueError('이후 청산 회귀 중 중첩 재현 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V80.md')) != settings['protocol_sha256']):
            raise ValueError('이후 청산 회귀 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'original_rows_weights_and_stopping_targets_preserved': True, 'whole_positions_disjoint': True,
            'new_models_fitted': 1, 'zero_targets_used_with_original_weight': True, 'threshold_search': False,
            'refit_after_calibration': False, 'diagnosis_evaluated': passed,
            'rows': {key: len(value) for key, value in rows.items()},
            'positions': {key: int(value.position_entry_time.nunique()) for key, value in rows.items()},
            'calibration_files_sha256': calibration_seal, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 이후 청산 대비 현금 효과의 직접 회귀\n\n'
            f'내부 보정 통과: {passed}. 마지막 진단 실행: {passed}. 사전 조건 통과: {decision["stopping_regression_admitted"]}. '
            '이전 미래 첫 청산 현금 정답과 전체 행 비중을 유지하고 고정 회귀기 하나를 학습했다. '
            '0 정답도 원래 비중으로 참여하며 예측은 확률이 아닌 계좌 bp다. 문턱은 0bp이고 재학습하지 않았다. '
            '개발 경로의 조건부 효과이며 연속 계좌 수익을 대신하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
