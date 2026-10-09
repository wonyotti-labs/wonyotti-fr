from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .continuation_diagnostics import FIRST_CLOSE_FILES, run_continuation_diagnosis, same_outputs
from .entry_regression import REGRESSION_SETTINGS
from .exposure_close import (
    EXPOSURE_TRANSFORM,
    ExposureCloseModel,
    current_exposure,
    exposure_training,
)
from .first_close_diagnostics import first_close_metrics, first_close_positions, paired_week_blocks
from .research_paths import reproduction_root

CONTINUATION_FILES = {'manifest.json', 'reference_parity.json', 'continuation_ledger.parquet',
    'exclusion_ledger.parquet', 'training_used.parquet', 'diagnosis_used.parquet',
    'training_weights.parquet', 'diagnosis_weights.parquet', 'input_verification.json', 'model.json',
    'training_support.json', 'previous_models.json', 'predictions.parquet', 'metrics.json', 'breakdown.json',
    'decision.json', 'first_metrics.json', 'first_breakdown.json', 'block_intervals.json',
    'positions_continuation.parquet', 'positions_economic.parquet', 'blocks.parquet', 'block_draws.parquet',
    'block_replicates.parquet', 'summary.json', 'REPORT.md'}


def reproduce_continuation(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != CONTINUATION_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('노출 회귀의 관리 의도 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V64.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != ExposureCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['margin_bps'] != 0
        or settings['new_model_count'] != 1 or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_weights_preserved'] is not True
        or summary['profitability_accepted'] is not False or summary['trading_returns_evaluated'] is not False):
        raise ValueError('노출 회귀의 원본 진단 설정·완료 오류')
    reproduced = run_continuation_diagnosis(parent, output)
    same_outputs(reference, reproduced, CONTINUATION_FILES-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('노출 회귀의 부모 재현 연결 오류')
        same_outputs(parent, child, FIRST_CLOSE_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def exposure_admission(metrics, first):
    references = ['continuation', 'economic', 'boosted', 'ridge', 'constant']
    candidate = metrics['exposure']
    if (set(metrics) != {*references, 'exposure'}
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or first['positions'] != candidate['positions'] or first['selected_positions'] != candidate['selected_positions']
        or not np.isfinite(first['all_position_mean_common_bps'])):
        raise ValueError('노출 회귀의 대조·최초 포지션 수 오류')
    checks = {f'weighted_mse_vs_{k}': candidate['weighted_mse'] < metrics[k]['weighted_mse']*.99 for k in references}
    checks.update({f'unweighted_mse_vs_{k}': candidate['mse'] <= metrics[k]['mse']+1e-9 for k in references[:-1]})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        positive_first_choice_mean=first['all_position_mean_common_bps'] > 0)
    return {'checks': checks, 'exposure_scaling_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_exposure_close_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'exposure-close-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V65.md')),
        'periods': CLOSE_SPLITS, 'features': ExposureCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'transform': EXPOSURE_TRANSFORM, 'margin_bps': 0, 'new_model_count': 1,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'현재 노출을 반영한 청산 가치 진단: {out}', flush=True)
    try:
        reproduced = reproduce_continuation(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        rows, weights, values = {}, {}, {}
        for name in ['training', 'diagnosis']:
            rows[name] = pd.read_parquet(reproduced/f'{name}_used.parquet')
            original_weights = pd.read_parquet(reproduced/f'{name}_weights.parquet')
            pd.testing.assert_frame_equal(rows[name][['decision_time', 'position_entry_time']], original_weights.drop(columns='sample_weight'), check_exact=True)
            weights[name] = original_weights.sample_weight.to_numpy()
            values[name] = rows[name][ExposureCloseModel.features].to_numpy(dtype=float)
            current_exposure(values[name])
            rows[name].to_parquet(out/f'{name}_used.parquet', index=False)
            original_weights.to_parquet(out/f'{name}_weights.parquet', index=False)
        pd.read_parquet(reproduced/'exclusion_ledger.parquet').to_parquet(out/'exclusion_ledger.parquet', index=False)
        target = rows['training'].close_advantage_bps.to_numpy()
        unit_target, unit_weight, normalizer = exposure_training(values['training'], target, weights['training'])
        transformation = rows['training'][['decision_time', 'position_entry_time', 'current_gross_exposure', 'close_advantage_bps']].copy()
        transformation['original_weight'], transformation['unit_target_bps'] = weights['training'], unit_target
        transformation['unit_training_weight'] = unit_weight
        transformation.to_parquet(out/'training_transformation.parquet', index=False)
        model, support = ExposureCloseModel.fit(values['training'], target, weights['training'], values['diagnosis'])
        if support['training_weight_normalizer'] != normalizer:
            raise ValueError('노출 회귀의 정규화 재계산 불일치')
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        previous = json.loads((reproduced/'previous_models.json').read_text())
        previous['continuation'] = json.loads((reproduced/'model.json').read_text())
        save_json(out/'previous_models.json', previous)
        frame = pd.read_parquet(reproduced/'predictions.parquet')
        frame['predicted_exposure'] = model.predict(values['diagnosis'])
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((reproduced/'metrics.json').read_text())
        metrics['exposure'] = close_metrics(frame, frame.predicted_exposure)
        positions = {'exposure': first_close_positions(rows['diagnosis'].assign(predicted_exposure=frame.predicted_exposure), 'predicted_exposure'),
            'continuation': pd.read_parquet(reproduced/'positions_continuation.parquet')}
        first_metrics, first_details = {}, []
        for name, part in positions.items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
            first_metrics[name] = first_close_metrics(part)
            groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
            groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
            for kind, group, f in groups:
                first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
        blocks, draws, replicates, intervals = paired_week_blocks(positions, model_names=['exposure', 'continuation'])
        for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
            data.to_parquet(out/f'{name}.parquet', index=False)
        decision, details = exposure_admission(metrics, first_metrics['exposure']), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision),
            ('first_metrics', first_metrics), ('first_breakdown', first_details), ('block_intervals', intervals)]:
            save_json(out/f'{name}.json', content)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('노출 회귀 중 원본 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 현재 노출을 반영한 청산 가치 진단\n\n'
            f'사전 조건 통과: {decision["exposure_scaling_admitted"]}. '
            '단위 노출 정답과 노출 제곱 가중치를 학습에 사용하고 계좌 기준 예측을 복원했다. '
            '평가에는 원래 정답·포지션 비중을 유지했으며 최초 선택과 주간 비교도 보존했다. '
            '노출 변환은 새로운 시장 정보가 아니며 연속 매매 수익을 입증하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
