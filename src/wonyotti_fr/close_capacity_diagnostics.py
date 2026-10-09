from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_calibration import CALIBRATION_SPLITS, calibration_splits
from .close_capacity import CAPACITY_GRID, capacity_class, fit_capacity_selection
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .continuation_inputs import ContinuationCloseModel
from .entry_regression import REGRESSION_SETTINGS
from .exposure_close import EXPOSURE_TRANSFORM
from .exposure_close_diagnostics import CONTINUATION_FILES, run_exposure_close_diagnosis
from .first_close_diagnostics import first_close_metrics, first_close_positions, paired_week_blocks
from .research_paths import reproduction_root

EXPOSURE_FILES = {'manifest.json', 'training_support.json', 'REPORT.md', 'decision.json', 'first_breakdown.json', 'exclusion_ledger.parquet', 'positions_exposure.parquet', 'positions_continuation.parquet', 'training_weights.parquet', 'blocks.parquet', 'training_used.parquet', 'block_replicates.parquet', 'diagnosis_used.parquet', 'reference_parity.json', 'model.json', 'breakdown.json', 'diagnosis_weights.parquet', 'block_draws.parquet', 'previous_models.json', 'training_transformation.parquet', 'predictions.parquet', 'first_metrics.json', 'block_intervals.json', 'metrics.json', 'summary.json'}


def reproduce_exposure(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != EXPOSURE_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('복잡도 회귀의 노출 변환 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V65.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != ContinuationCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['margin_bps'] != 0
        or settings['transform'] != EXPOSURE_TRANSFORM or settings['new_model_count'] != 1 or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['profitability_accepted'] is not False or summary['trading_returns_evaluated'] is not False):
        raise ValueError('복잡도 회귀의 원본 진단 설정·완료 오류')
    reproduced = run_exposure_close_diagnosis(parent, output)
    same_outputs(reference, reproduced, EXPOSURE_FILES-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('복잡도 회귀의 부모 재현 연결 오류')
        same_outputs(parent, child, CONTINUATION_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def capacity_admission(metrics, first):
    references = ['exposure', 'continuation', 'economic', 'boosted', 'ridge', 'constant']
    candidate = metrics['capacity']
    if (set(metrics) != {*references, 'capacity'}
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or first['positions'] != candidate['positions'] or first['selected_positions'] != candidate['selected_positions']
        or not np.isfinite(first['all_position_mean_common_bps'])):
        raise ValueError('복잡도 회귀의 대조·최초 포지션 수 오류')
    checks = {f'weighted_mse_vs_{k}': candidate['weighted_mse'] < metrics[k]['weighted_mse']*.99 for k in references}
    checks.update({f'unweighted_mse_vs_{k}': candidate['mse'] <= metrics[k]['mse']+1e-9 for k in references[:-1]})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        positive_first_choice_mean=first['all_position_mean_common_bps'] > 0)
    return {'checks': checks, 'capacity_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_close_capacity_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-capacity-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V66.md')),
        'periods': CLOSE_SPLITS, 'features': ContinuationCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'capacity_grid': CAPACITY_GRID, 'selection_periods': CALIBRATION_SPLITS, 'margin_bps': 0, 'selection_model_count': 4, 'refit_model_count': 1,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'시간순 복잡도 선택 청산 진단: {out}', flush=True)
    try:
        reproduced = reproduce_exposure(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        rows, weights, values = {}, {}, {}
        for name in ['training', 'diagnosis']:
            rows[name] = pd.read_parquet(reproduced/f'{name}_used.parquet')
            original_weights = pd.read_parquet(reproduced/f'{name}_weights.parquet')
            pd.testing.assert_frame_equal(rows[name][['decision_time', 'position_entry_time']], original_weights.drop(columns='sample_weight'), check_exact=True)
            weights[name] = original_weights.sample_weight.to_numpy()
            values[name] = rows[name][ContinuationCloseModel.features].to_numpy(dtype=float)
            rows[name].to_parquet(out/f'{name}_used.parquet', index=False)
            original_weights.to_parquet(out/f'{name}_weights.parquet', index=False)
        pd.read_parquet(reproduced/'exclusion_ledger.parquet').to_parquet(out/'exclusion_ledger.parquet', index=False)
        target = rows['training'].close_advantage_bps.to_numpy()
        continuation = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
        ledger = pd.read_parquet(continuation/'continuation_ledger.parquet')
        internal, assignments = calibration_splits(ledger)
        pd.testing.assert_frame_equal(internal['diagnosis'], rows['diagnosis'], check_exact=True)
        assignments.to_parquet(out/'selection_exclusion_ledger.parquet', index=False)
        selected = fit_capacity_selection(internal['training'], internal['calibration'], out)
        model, support = capacity_class(selected).fit(values['training'], target, weights['training'], values['diagnosis'])
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        previous = json.loads((reproduced/'previous_models.json').read_text())
        previous['exposure'] = json.loads((reproduced/'model.json').read_text())
        save_json(out/'previous_models.json', previous)
        frame = pd.read_parquet(reproduced/'predictions.parquet')
        frame['predicted_capacity'] = model.predict(values['diagnosis'])
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((reproduced/'metrics.json').read_text())
        metrics['capacity'] = close_metrics(frame, frame.predicted_capacity)
        positions = {'capacity': first_close_positions(rows['diagnosis'].assign(predicted_capacity=frame.predicted_capacity), 'predicted_capacity'),
            'continuation': pd.read_parquet(reproduced/'positions_continuation.parquet')}
        first_metrics, first_details = {}, []
        for name, part in positions.items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
            first_metrics[name] = first_close_metrics(part)
            groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
            groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
            for kind, group, f in groups:
                first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
        blocks, draws, replicates, intervals = paired_week_blocks(positions, model_names=['capacity', 'continuation'])
        for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
            data.to_parquet(out/f'{name}.parquet', index=False)
        decision, details = capacity_admission(metrics, first_metrics['capacity']), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision),
            ('first_metrics', first_metrics), ('first_breakdown', first_details), ('block_intervals', intervals)]:
            save_json(out/f'{name}.json', content)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('복잡도 회귀 중 원본 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 시간순 복잡도 선택 청산 진단\n\n'
            f'사전 조건 통과: {decision["capacity_admitted"]}. '
            '앞 학습 내부의 시간순 선택으로 정한 설정 하나를 전체 앞 학습에 재적합했다. '
            '평가에는 원래 정답·포지션 비중을 유지했으며 최초 선택과 주간 비교도 보존했다. '
            '마지막 진단은 후보 선택에 사용하지 않았으며 연속 매매 수익을 입증하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
