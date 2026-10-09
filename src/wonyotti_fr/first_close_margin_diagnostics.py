from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_calibration import CALIBRATION_SPLITS
from .close_capacity import capacity_class
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .continuation_inputs import ContinuationCloseModel
from .entry_regression import REGRESSION_SETTINGS
from .first_close_diagnostics import first_close_metrics, first_close_positions, paired_week_blocks
from .first_close_margin import FIRST_MARGINS, margin_admission, select_first_margin
from .research_paths import reproduction_root
from .weekly_close import WEEKLY_SETTINGS
from .weekly_close_diagnostics import CAPACITY_FILES, run_weekly_close_diagnosis

WEEKLY_FILES = {'reference_parity.json', 'diagnosis_weights.parquet', 'first_metrics.json', 'manifest.json', 'blocks.parquet', 'first_breakdown.json', 'breakdown.json', 'weekly_support.json', 'block_draws.parquet', 'REPORT.md', 'decision.json', 'positions_weekly.parquet', 'weekly_training_membership.parquet', 'block_replicates.parquet', 'positions_continuation.parquet', 'metrics.json', 'weekly_models.json', 'predictions.parquet', 'prediction_routing.parquet', 'previous_models.json', 'training_weights.parquet', 'summary.json', 'block_intervals.json', 'exclusion_ledger.parquet', 'diagnosis_used.parquet', 'training_used.parquet'}


def reproduce_weekly(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != WEEKLY_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('최초 문턱의 주간 학습 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V67.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != ContinuationCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['margin_bps'] != 0
        or settings['weekly_settings'] != WEEKLY_SETTINGS or settings['weekly_model_count'] != 13 or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['profitability_accepted'] is not False or summary['trading_returns_evaluated'] is not False):
        raise ValueError('최초 문턱의 원본 진단 설정·완료 오류')
    reproduced = run_weekly_close_diagnosis(parent, output)
    same_outputs(reference, reproduced, WEEKLY_FILES-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('최초 문턱의 부모 재현 연결 오류')
        same_outputs(parent, child, CAPACITY_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def calibrate_first_margin(capacity, output):
    model_data = json.loads((capacity/'selection_models.json').read_text())['depth2_iter64']
    model = capacity_class('depth2_iter64').from_dict(model_data)
    frames = {}
    for name, source, period in [('half_training', 'training', 'training'), ('calibration', 'selection', 'calibration')]:
        frame = pd.read_parquet(capacity/f'selection_{source}_used.parquet')
        weights = pd.read_parquet(capacity/f'selection_{source}_weights.parquet')
        start, end = [pd.Timestamp(t, tz='UTC') for t in CALIBRATION_SPLITS[period]]
        if (frame.empty or not frame.label_status.eq('closed').all()
            or frame.position_entry_time.lt(start).any() or frame.decision_time.lt(start).any()
            or frame.label_end.ge(end).any()):
            raise ValueError('최초 문턱 보정의 고정 학습·선택 기간 오류')
        pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), frame[['decision_time', 'position_entry_time']], check_exact=True)
        np.testing.assert_array_equal(weights.sample_weight, position_weights(frame))
        frame.to_parquet(output/f'{name}_used.parquet', index=False)
        weights.to_parquet(output/f'{name}_weights.parquet', index=False)
        frames[name] = frame
    if (frames['half_training'].label_end.max() >= frames['calibration'].decision_time.min()
        or set(frames['half_training'].position_entry_time) & set(frames['calibration'].position_entry_time)):
        raise ValueError('최초 문턱 보정의 학습·선택 포지션 교차')
    frame = frames['calibration']
    prior = pd.read_parquet(capacity/'selection_selection_predictions.parquet')
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']
    pd.testing.assert_frame_equal(prior[keys], frame[keys], check_exact=True)
    prediction = model.predict(frame[model.features].to_numpy(dtype=float))
    np.testing.assert_array_equal(prediction, prior.predicted_depth2_iter64)
    np.testing.assert_array_equal(prior.sample_weight, position_weights(frame))
    scores = prior[keys+['sample_weight']].assign(predicted_half=prediction)
    scores.to_parquet(output/'calibration_predictions.parquet', index=False)
    save_json(output/'frozen_half_model.json', model.to_dict())
    selection, metrics, positions = select_first_margin(frame.assign(predicted_half=prediction))
    for name, data in positions.items():
        data.to_parquet(output/f'calibration_positions_{name}.parquet', index=False)
    save_json(output/'calibration_first_metrics.json', metrics)
    save_json(output/'calibration_row_metrics.json', [
        {'margin_bps': m, **close_metrics(scores, prediction, margin_bps=m)} for m in FIRST_MARGINS])
    save_json(output/'selection.json', selection)
    return model, selection


def evaluate_first_margin(reproduced, model, selection, output):
    diagnosis = pd.read_parquet(reproduced/'diagnosis_used.parquet')
    frame = pd.read_parquet(reproduced/'predictions.parquet')
    frame['predicted_half'] = model.predict(diagnosis[model.features].to_numpy(dtype=float))
    frame.to_parquet(output/'predictions.parquet', index=False)
    margin = selection['chosen_margin_bps']
    metrics = json.loads((reproduced/'metrics.json').read_text())
    metrics['half_zero'] = close_metrics(frame, frame.predicted_half)
    metrics['margin'] = close_metrics(frame, frame.predicted_half, margin_bps=margin)
    selection_frame = diagnosis.assign(predicted_half=frame.predicted_half)
    positions = {
        'margin': first_close_positions(selection_frame, 'predicted_half', margin_bps=margin),
        'half_zero': first_close_positions(selection_frame, 'predicted_half'),
        'continuation': pd.read_parquet(reproduced/'positions_continuation.parquet'),
        'weekly': pd.read_parquet(reproduced/'positions_weekly.parquet')}
    first, first_details = {}, []
    for name, part in positions.items():
        part.to_parquet(output/f'positions_{name}.parquet', index=False)
        first[name] = first_close_metrics(part)
        groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
        groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
        for kind, group, f in groups:
            first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
    blocks, draws, replicates, intervals = paired_week_blocks(
        {n: positions[n] for n in ['margin', 'half_zero']}, model_names=['margin', 'half_zero'])
    for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
        data.to_parquet(output/f'{name}.parquet', index=False)
    decision, details = margin_admission(selection, metrics, first, intervals), []
    groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
    groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
    for kind, group, part in groups:
        for name in metrics:
            column = 'predicted_half' if name in {'margin', 'half_zero'} else 'predicted_'+name
            details.append({'kind': kind, 'group': group, 'model': name,
                **close_metrics(part, part[column], margin_bps=margin if name == 'margin' else 0.)})
    for name, value in [('metrics', metrics), ('breakdown', details), ('first_metrics', first),
        ('first_breakdown', first_details), ('block_intervals', intervals)]:
        save_json(output/f'{name}.json', value)
    return decision


def run_first_close_margin_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'first-close-margin-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V68.md')),
        'periods': CALIBRATION_SPLITS, 'features': ContinuationCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'fixed_half_model': 'depth2_iter64', 'margins_bps': FIRST_MARGINS,
        'new_models_fitted': False, 'existing_models_reproduced': True,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'최초 청산 선택의 실행 문턱 보정: {out}', flush=True)
    try:
        reproduced = reproduce_weekly(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        capacity = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
        model, selection = calibrate_first_margin(capacity, out)
        for source, name in [('previous_models.json', 'previous_models.json'),
            ('weekly_models.json', 'previous_weekly_models.json'), ('metrics.json', 'previous_metrics.json')]:
            save_json(out/name, json.loads((reproduced/source).read_text()))
        for source, name in [('predictions.parquet', 'previous_predictions.parquet'),
            ('exclusion_ledger.parquet', 'exclusion_ledger.parquet'),
            ('diagnosis_used.parquet', 'diagnosis_used.parquet'), ('diagnosis_weights.parquet', 'diagnosis_weights.parquet')]:
            pd.read_parquet(reproduced/source).to_parquet(out/name, index=False)
        # 보정 실패 시 새 반년 모델의 마지막 구간 예측을 생성하지 않는다.
        decision = evaluate_first_margin(reproduced, model, selection, out) if selection['selection_passed'] else margin_admission(selection)
        save_json(out/'decision.json', decision)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('최초 문턱 보정 중 원본 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'frozen_half_model_unchanged': True, 'previous_predictions_and_weights_preserved': True,
            'new_models_fitted': False, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 최초 청산 선택의 실행 문턱 보정\n\n'
            f'내부 선택 통과: {selection["selection_passed"]}. 마지막 진단 실행: {decision["final_diagnosis_evaluated"]}. '
            f'최종 조건 통과: {decision["margin_admitted"]}. '
            '고정 반년 모델의 점수를 유지하고 앞 구간 최초 효과로 실행 문턱만 선택했다. '
            '회귀 오차 개선이나 연속 매매 수익성으로 해석하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
