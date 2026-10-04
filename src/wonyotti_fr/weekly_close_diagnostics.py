from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_calibration import CALIBRATION_SPLITS
from .close_capacity import CAPACITY_GRID
from .close_capacity_diagnostics import EXPOSURE_FILES, run_close_capacity_diagnosis
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .continuation_inputs import ContinuationCloseModel
from .entry_regression import REGRESSION_SETTINGS
from .first_close_diagnostics import first_close_metrics, first_close_positions, paired_week_blocks
from .weekly_close import WEEKLY_SETTINGS, fit_weekly_close

CAPACITY_FILES = {'summary.json', 'manifest.json', 'block_replicates.parquet', 'previous_models.json', 'exclusion_ledger.parquet', 'breakdown.json', 'selection_exclusion_ledger.parquet', 'positions_capacity.parquet', 'selection_training_weights.parquet', 'first_metrics.json', 'selection_support.json', 'block_intervals.json', 'training_weights.parquet', 'reference_parity.json', 'selection_models.json', 'decision.json', 'blocks.parquet', 'selection_selection_predictions.parquet', 'diagnosis_weights.parquet', 'block_draws.parquet', 'selection_metrics.json', 'model.json', 'training_support.json', 'training_used.parquet', 'metrics.json', 'selection.json', 'REPORT.md', 'selection_selection_used.parquet', 'selection_training_used.parquet', 'first_breakdown.json', 'diagnosis_used.parquet', 'positions_continuation.parquet', 'selection_selection_weights.parquet', 'predictions.parquet', 'selection_training_predictions.parquet'}


def reproduce_capacity(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != CAPACITY_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('주간 청산의 복잡도 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V66.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != ContinuationCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['margin_bps'] != 0
        or settings['capacity_grid'] != CAPACITY_GRID or settings['selection_periods'] != CALIBRATION_SPLITS
        or settings['selection_model_count'] != 4 or settings['refit_model_count'] != 1 or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['profitability_accepted'] is not False or summary['trading_returns_evaluated'] is not False):
        raise ValueError('주간 청산의 원본 진단 설정·완료 오류')
    reproduced = run_close_capacity_diagnosis(parent, output)
    same_outputs(reference, reproduced, CAPACITY_FILES-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('주간 청산의 부모 재현 연결 오류')
        same_outputs(parent, child, EXPOSURE_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def weekly_admission(metrics, first):
    references = ['capacity', 'exposure', 'continuation', 'economic', 'boosted', 'ridge', 'constant']
    candidate = metrics['weekly']
    if (set(metrics) != {*references, 'weekly'}
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or first['positions'] != candidate['positions'] or first['selected_positions'] != candidate['selected_positions']
        or not np.isfinite(first['all_position_mean_common_bps'])):
        raise ValueError('주간 청산의 대조·최초 포지션 수 오류')
    checks = {f'weighted_mse_vs_{k}': candidate['weighted_mse'] < metrics[k]['weighted_mse']*.99 for k in references}
    checks.update({f'unweighted_mse_vs_{k}': candidate['mse'] <= metrics[k]['mse']+1e-9 for k in references[:-1]})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        positive_first_choice_mean=first['all_position_mean_common_bps'] > 0)
    return {'checks': checks, 'weekly_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_weekly_close_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'weekly-close-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V67.md')),
        'periods': CLOSE_SPLITS, 'features': ContinuationCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'weekly_settings': WEEKLY_SETTINGS, 'margin_bps': 0, 'weekly_model_count': 13,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'확정 결과의 주간 누적 청산 학습 진단: {out}', flush=True)
    try:
        reproduced = reproduce_capacity(reference, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        rows, weights = {}, {}
        for name in ['training', 'diagnosis']:
            rows[name] = pd.read_parquet(reproduced/f'{name}_used.parquet')
            original_weights = pd.read_parquet(reproduced/f'{name}_weights.parquet')
            pd.testing.assert_frame_equal(rows[name][['decision_time', 'position_entry_time']], original_weights.drop(columns='sample_weight'), check_exact=True)
            weights[name] = original_weights.sample_weight.to_numpy()
            rows[name].to_parquet(out/f'{name}_used.parquet', index=False)
            original_weights.to_parquet(out/f'{name}_weights.parquet', index=False)
        pd.read_parquet(reproduced/'exclusion_ledger.parquet').to_parquet(out/'exclusion_ledger.parquet', index=False)
        exposure = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
        continuation = Path(json.loads((exposure/'reference_parity.json').read_text())['reproduction'])
        ledger = pd.read_parquet(continuation/'continuation_ledger.parquet')
        original_model = json.loads((continuation/'model.json').read_text())
        prediction = fit_weekly_close(ledger, rows['diagnosis'], rows['training'], weights['training'], original_model, out)
        previous = json.loads((reproduced/'previous_models.json').read_text())
        previous['capacity'] = json.loads((reproduced/'model.json').read_text())
        save_json(out/'previous_models.json', previous)
        frame = pd.read_parquet(reproduced/'predictions.parquet')
        frame['predicted_weekly'] = prediction
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((reproduced/'metrics.json').read_text())
        metrics['weekly'] = close_metrics(frame, frame.predicted_weekly)
        positions = {'weekly': first_close_positions(rows['diagnosis'].assign(predicted_weekly=frame.predicted_weekly), 'predicted_weekly'),
            'continuation': pd.read_parquet(reproduced/'positions_continuation.parquet')}
        first_metrics, first_details = {}, []
        for name, part in positions.items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
            first_metrics[name] = first_close_metrics(part)
            groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
            groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
            for kind, group, f in groups:
                first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
        blocks, draws, replicates, intervals = paired_week_blocks(positions, model_names=['weekly', 'continuation'])
        for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
            data.to_parquet(out/f'{name}.parquet', index=False)
        decision, details = weekly_admission(metrics, first_metrics['weekly']), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision),
            ('first_metrics', first_metrics), ('first_breakdown', first_details), ('block_intervals', intervals)]:
            save_json(out/f'{name}.json', content)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('주간 청산 학습 중 원본 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 확정 결과의 주간 누적 청산 학습 진단\n\n'
            f'사전 조건 통과: {decision["weekly_admitted"]}. '
            '각 주 시작 이틀 전에 확정된 결과만 누적해 같은 고정 모델을 재학습했다. '
            '평가에는 원래 정답·포지션 비중을 유지했으며 최초 선택과 주간 비교도 보존했다. '
            '갱신 시점에 미확정인 결과는 학습하지 않았으며 연속 매매 수익을 입증하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
