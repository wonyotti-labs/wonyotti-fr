from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .addition_effect import position_weights
from .close_learning_inputs import close_learning_splits
from .close_threshold import (
    THRESHOLD_SETTINGS,
    THRESHOLD_SPLITS,
    ThresholdCloseModel,
    calibrate_threshold,
    fit_threshold_close,
    threshold_splits,
)
from .close_threshold_evaluation import (
    THRESHOLD_COMPARISONS,
    THRESHOLD_POLICIES,
    evaluate_threshold_close,
)
from .close_threshold_reference import checked_visited_reproduction, reproduce_retained_reference
from .common import new_run, save_json, sha256
from .histogram_management import HISTOGRAM_SETTINGS
from .retained_weight_close_diagnostics import RETAINED_INPUT_FILES
from .retained_weight_close_evaluation import RETAINED_POLICIES
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot

THRESHOLD_INPUT_FILES = RETAINED_INPUT_FILES | {'retained_training_used.parquet',
    'retained_training_weights.parquet', 'retained_training_cost_ledger.parquet', 'full_training_contribution.parquet'}


def load_retained_evaluation(reference):
    values = {name: json.loads((reference/f'{name}.json').read_text()) for name in
        ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown']}
    values['positions'] = {name: pd.read_parquet(reference/f'positions_{name}.parquet') for name in RETAINED_POLICIES}
    values['predictions'] = pd.read_parquet(reference/'predictions.parquet')
    values['draws'] = pd.read_parquet(reference/'block_draws.parquet')
    return values


def save_evaluation(output, result):
    result['predictions'].to_parquet(output/'predictions.parquet', index=False)
    for name, part in result['positions'].items():
        part.to_parquet(output/f'positions_{name}.parquet', index=False)
    for name in ['metrics', 'first_metrics']:
        save_json(output/f'{name}.json', result[name])


def run_close_threshold_diagnosis(reference: Path, output: Path) -> Path:
    settings = {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V78.md')), 'periods': THRESHOLD_SPLITS,
        'features': ThresholdCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'threshold_settings': THRESHOLD_SETTINGS, 'new_model_count': 1,
        'existing_models_reproduced': True, 'candidate': 'calibrated', 'policies': THRESHOLD_POLICIES,
        'comparisons': THRESHOLD_COMPARISONS, 'score_kind': 'cost_weighted_decision_score',
        'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'close-threshold-diagnosis', settings)
    print(f'최초 청산 효과에 따른 문턱 보정: {out}', flush=True)
    try:
        reproduced = reproduce_retained_reference(reference, out)
        print('기존 전체 원장·모델·아홉 정책·실패 재현 완료', flush=True)
        for name in sorted(THRESHOLD_INPUT_FILES):
            copy_snapshot(reproduced/name, out/name)
        ledger = pd.read_parquet(out/'minute_ledger.parquet')
        old_rows, old_assignment = close_learning_splits(ledger)
        pd.testing.assert_frame_equal(old_assignment, pd.read_parquet(out/'exclusion_ledger.parquet'), check_exact=True)
        for name, frame in old_rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out/f'{name}_used.parquet'), check_exact=True)
        rows, assignment = threshold_splits(ledger)
        pd.testing.assert_frame_equal(rows['diagnosis'], old_rows['diagnosis'], check_exact=True)
        assignment.to_parquet(out/'threshold_exclusion_ledger.parquet', index=False)
        for name in ['early_training', 'calibration']:
            rows[name].to_parquet(out/f'{name}_used.parquet', index=False)
        model, support = fit_threshold_close(rows['early_training'], rows['calibration'], out)
        calibration_frame = rows['calibration'].assign(sample_weight=position_weights(rows['calibration']))
        calibration_frame[['decision_time', 'position_entry_time', 'sample_weight']].to_parquet(out/'calibration_weights.parquet', index=False)
        calibration_score = model.probabilities(calibration_frame[model.features].to_numpy(dtype=float))[:, 0]
        calibration = calibrate_threshold(calibration_frame, calibration_score)
        calibration_root = out/'calibration'
        calibration_root.mkdir(mode=0o700)
        save_evaluation(calibration_root, calibration)
        save_json(out/'selection.json', calibration['selection'])
        save_json(calibration_root/'files.json', {p.name: sha256(p) for p in calibration_root.iterdir() if p.is_file()})
        calibration_seal = sha256(calibration_root/'files.json')
        passed = calibration['selection']['selection_passed']
        if passed:
            frame = rows['diagnosis'].assign(sample_weight=pd.read_parquet(out/'diagnosis_weights.parquet').sample_weight)
            score = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
            result = evaluate_threshold_close(frame, load_retained_evaluation(reproduced), score,
                support['training_constant_score'], calibration)
            save_evaluation(out, result)
            result['draws'].to_parquet(out/'block_draws.parquet', index=False)
            for name in THRESHOLD_COMPARISONS:
                result['blocks'][name].to_parquet(out/f'{name}_blocks.parquet', index=False)
                result['replicates'][name].to_parquet(out/f'{name}_block_replicates.parquet', index=False)
            for name in ['probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown', 'intervals']:
                save_json(out/f'{name}.json', result[name])
            decision = result['decision']
        else:
            decision = {'checks': {'calibration_selection_passed': False}, 'threshold_admitted': False,
                'candidate': None, 'trading_returns_evaluated': False, 'reason': 'no_eligible_calibration_threshold'}
        save_json(out/'decision.json', decision)
        baseline = json.loads((out/'baseline_files.json').read_text())
        if (sha256(reference/'files.json') != settings['reference_files_sha256'] or checked_files(reference) != baseline
            or any(sha256(out/'baseline_source'/name) != value for name, value in baseline.items())
            or any(sha256(out/name) != sha256(reproduced/name) for name in THRESHOLD_INPUT_FILES)
            or sha256(calibration_root/'files.json') != calibration_seal):
            raise ValueError('문턱 보정 중 입력·기준·보정 사본 변경')
        checked_files(calibration_root)
        proof = json.loads((out/'reference_parity.json').read_text())
        if sha256(reproduced/'files.json') != proof['reproduction_files_sha256']:
            raise ValueError('문턱 보정 중 기존 재현 봉인 변경')
        checked_files(reproduced)
        parent = Path(json.loads((reference/'manifest.json').read_text())['settings']['reference'])
        for key, folder in [('source', reference), ('reproduced', reproduced)]:
            if checked_visited_reproduction(folder, parent) != proof['previous_reproductions'][key]:
                raise ValueError('문턱 보정 중 이전 중첩 재현 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V78.md')) != settings['protocol_sha256']):
            raise ValueError('문턱 보정 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'original_full_training_and_diagnosis_preserved': True, 'whole_positions_disjoint': True,
            'new_models_fitted': 1, 'refit_after_selection': False, 'diagnosis_evaluated': passed,
            'selected_threshold': calibration['selection']['selected_threshold'],
            'rows': {key: len(value) for key, value in rows.items()},
            'positions': {key: int(value.position_entry_time.nunique()) for key, value in rows.items()},
            'calibration_files_sha256': calibration_seal, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 최초 청산 효과에 따른 문턱 보정\n\n'
            f'내부 선택 통과: {passed}. 마지막 진단 실행: {passed}. 사전 조건 통과: {decision["threshold_admitted"]}. '
            '앞 모델 하나와 별도 보정 구간으로 문턱을 선택했다. 선택 실패에는 대체 문턱이 없다. '
            '기존 원장·아홉 정책·실패를 보존했다. 개발 경로의 조건부 효과이며 연속 계좌 수익을 뜻하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
