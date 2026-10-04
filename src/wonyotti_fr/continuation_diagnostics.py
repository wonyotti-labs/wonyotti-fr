from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .continuation_inputs import PENDING_FEATURES, ContinuationCloseModel, load_pending_inputs
from .entry_regression import REGRESSION_SETTINGS
from .first_close_diagnostics import (
    ECONOMIC_FILES,
    FIRST_MODELS,
    first_close_metrics,
    first_close_positions,
    paired_week_blocks,
    run_first_close_diagnosis,
)

FIRST_CLOSE_FILES = {'manifest.json', 'reference_parity.json', 'opportunity_ledger.parquet', 'exclusion_ledger.parquet',
    'positions_economic.parquet', 'positions_boosted.parquet', 'blocks.parquet', 'block_draws.parquet',
    'block_replicates.parquet', 'metrics.json', 'breakdown.json', 'block_intervals.json',
    'opportunity_metrics.json', 'summary.json', 'REPORT.md'}


def same_outputs(left, right, names):
    for name in sorted(names):
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(left/name), pd.read_parquet(right/name), check_exact=True)
        elif name.endswith('.json'):
            if json.loads((left/name).read_text()) != json.loads((right/name).read_text()):
                raise ValueError('현재 관리 의도 진단의 이전 출력 재현 불일치: '+name)
        elif (left/name).read_bytes() != (right/name).read_bytes():
            raise ValueError('현재 관리 의도 진단의 이전 보고서 재현 불일치')


def reproduce_first_close(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != FIRST_CLOSE_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('현재 관리 의도 진단의 최초 선택 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V63.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['period'] != CLOSE_SPLITS['diagnosis'] or settings['models'] != FIRST_MODELS
        or settings['margin_bps'] != 0 or settings['new_models_fitted'] is not False
        or settings['existing_models_reproduced'] is not True or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True or summary['complete'] is not True
        or any(summary[k] is not False for k in ['profitability_accepted', 'trading_returns_evaluated', 'admission_decision_made'])):
        raise ValueError('현재 관리 의도 진단의 최초 선택 가정·완료 오류')
    reproduced = run_first_close_diagnosis(parent, output)
    same_outputs(reference, reproduced, FIRST_CLOSE_FILES-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('현재 관리 의도 진단의 재현 경로 연결 오류')
        same_outputs(parent, child, ECONOMIC_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def continuation_admission(metrics, first_metrics):
    references = ['economic', 'boosted', 'ridge', 'constant']
    candidate = metrics['continuation']
    if (set(metrics) != {*references, 'continuation'}
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or first_metrics['positions'] != candidate['positions']
        or first_metrics['selected_positions'] != candidate['selected_positions']
        or not np.isfinite(first_metrics['all_position_mean_common_bps'])):
        raise ValueError('관리 의도 진단의 대조·포지션·최초 선택 연결 오류')
    checks = {f'weighted_mse_vs_{k}': candidate['weighted_mse'] < metrics[k]['weighted_mse']*.99 for k in references}
    checks.update({f'unweighted_mse_vs_{k}': candidate['mse'] <= metrics[k]['mse']+1e-9 for k in references[:-1]})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        positive_first_choice_mean=first_metrics['all_position_mean_common_bps'] > 0)
    return {'checks': checks, 'continuation_inputs_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_continuation_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'continuation-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V64.md')),
        'periods': CLOSE_SPLITS, 'features': ContinuationCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'margin_bps': 0, 'new_model_count': 1, 'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'현재 관리 의도별 청산 순효과 진단: {out}', flush=True)
    try:
        reproduced = reproduce_first_close(reference, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        economic = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
        previous = Path(json.loads((economic/'manifest.json').read_text())['settings']['reference'])
        labels = Path(json.loads((previous/'manifest.json').read_text())['settings']['labels'])
        augmented, verification = load_pending_inputs(labels, pd.read_parquet(economic/'economic_ledger.parquet'))
        augmented.to_parquet(out/'continuation_ledger.parquet', index=False)
        save_json(out/'input_verification.json', verification)
        rows, assignments = close_learning_splits(augmented)
        assignments.to_parquet(out/'exclusion_ledger.parquet', index=False)
        pd.testing.assert_frame_equal(assignments.drop(columns=PENDING_FEATURES), pd.read_parquet(economic/'exclusion_ledger.parquet'), check_exact=True)
        values, weights = {}, {}
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame.drop(columns=PENDING_FEATURES), pd.read_parquet(economic/f'{name}_used.parquet'), check_exact=True)
            weights[name] = position_weights(frame)
            saved = pd.read_parquet(economic/f'{name}_weights.parquet')
            pd.testing.assert_frame_equal(frame[['decision_time', 'position_entry_time']].assign(sample_weight=weights[name]), saved, check_exact=True)
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
            saved.to_parquet(out/f'{name}_weights.parquet', index=False)
            values[name] = frame[ContinuationCloseModel.features].to_numpy(dtype=float)
        model, support = ContinuationCloseModel.fit(values['training'], rows['training'].close_advantage_bps,
            weights['training'], values['diagnosis'])
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        previous_models = json.loads((economic/'previous_models.json').read_text())
        previous_models['economic'] = json.loads((economic/'model.json').read_text())
        save_json(out/'previous_models.json', previous_models)
        frame = pd.read_parquet(economic/'predictions.parquet')
        frame['predicted_continuation'] = model.predict(values['diagnosis'])
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((economic/'metrics.json').read_text())
        metrics['continuation'] = close_metrics(frame, frame.predicted_continuation)
        selection = rows['diagnosis'].assign(predicted_continuation=frame.predicted_continuation)
        positions = {'continuation': first_close_positions(selection, 'predicted_continuation'),
            'economic': pd.read_parquet(reproduced/'positions_economic.parquet')}
        first_metrics, first_details = {}, []
        for name, part in positions.items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
            first_metrics[name] = first_close_metrics(part)
            groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
            groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
            for kind, group, f in groups:
                first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
        blocks, draws, replicates, intervals = paired_week_blocks(positions, model_names=['continuation', 'economic'])
        for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
            data.to_parquet(out/f'{name}.parquet', index=False)
        decision, details = continuation_admission(metrics, first_metrics['continuation']), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision),
            ('first_metrics', first_metrics), ('first_breakdown', first_details), ('block_intervals', intervals)]:
            save_json(out/f'{name}.json', content)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_weights_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 현재 관리 의도에 따른 청산 가치 진단\n\n'
            f'사전 조건 통과: {decision["continuation_inputs_admitted"]}. '
            '현재 봇이 이미 결정한 다음 관리 요청만 명시하고 기존 입력·행·가중치·대조를 보존했다. '
            '최초 선택의 공통 기준 효과와 주간 불확실성도 기록했다. 미래 체결이나 연속 계좌 수익을 입력·성과로 사용하지 않았다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
