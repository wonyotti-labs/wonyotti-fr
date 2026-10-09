from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_context import ContextCloseModel
from .close_context_diagnostics import (
    UTILITY_COMMON_FILES,
    UTILITY_MARGIN_FILES,
    evaluate_context,
    run_close_context_diagnosis,
)
from .close_flow import (
    FLOW_WINDOWS,
    FlowCloseModel,
    attach_close_flow,
    close_flow_source,
    flow_close_admission,
)
from .close_learning_inputs import CLOSE_SPLITS
from .close_utility_diagnostics import fit_utility
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root

CONTEXT_COMMON_FILES = {'metrics.json', 'diagnosis_used.parquet', 'positions_utility.parquet', 'block_intervals.json',
    'block_replicates.parquet', 'exclusion_ledger.parquet', 'utility_block_intervals.json', 'summary.json', 'model.json',
    'blocks.parquet', 'utility_block_replicates.parquet', 'training_support.json', 'positions_context.parquet',
    'previous_decision.json', 'reference_parity.json', 'diagnosis_weights.parquet', 'REPORT.md', 'positions_continuation.parquet',
    'context_source.json', 'utility_blocks.parquet', 'first_metrics.json', 'probability_breakdown.json', 'probability_metrics.json',
    'utility_block_draws.parquet', 'previous_predictions.parquet', 'positions_training_constant.parquet', 'previous_training_support.json',
    'manifest.json', 'training_used.parquet', 'previous_model.json', 'decision.json', 'positions_weekly.parquet',
    'training_cost_ledger.parquet', 'first_breakdown.json', 'breakdown.json', 'block_draws.parquet', 'predictions.parquet', 'training_weights.parquet'}
CONTEXT_MARGIN_FILES = {'positions_half_zero.parquet', 'positions_margin.parquet'}


def reproduce_context(reference, output):
    files = json.loads((reference/'files.json').read_text())
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    parent = Path(settings['reference'])
    if settings['reference_files_sha256'] != sha256(parent/'files.json'):
        raise ValueError('청산 체결 방향의 이전 비용 결과 연결 오류')
    selection = json.loads((parent/'previous_selection.json').read_text())
    if type(selection['selection_passed']) is not bool:
        raise ValueError('청산 체결 방향의 이전 문턱 선택 상태 오류')
    expected = CONTEXT_COMMON_FILES | (CONTEXT_MARGIN_FILES if selection['selection_passed'] else set())
    if (set(files) != expected or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('청산 체결 방향의 다일 원본 파일·지문 오류')
    summary = json.loads((reference/'summary.json').read_text())
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V70.md'))
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != ContextCloseModel.features
        or settings['settings'] != HISTOGRAM_SETTINGS or settings['new_model_count'] != 1
        or settings['score_threshold'] != .5 or settings['score_kind'] != 'cost_weighted_decision_score'
        or settings['cost_weight'] != 'original_position_weight_times_absolute_effect'
        or settings['zero_effect_policy'] != 'zero_fit_contribution_preserved_in_evaluation'
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['original_cost_ledger_and_prior_preserved'] is not True
        or summary['zero_effect_rows_preserved'] is not True or summary['profitability_accepted'] is not False):
        raise ValueError('청산 체결 방향의 원래 다일 설정·완료 오류')
    reproduced = run_close_context_diagnosis(parent, output)
    same_outputs(reference, reproduced, expected-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('청산 체결 방향의 다일 부모 연결 오류')
        utility_files = UTILITY_COMMON_FILES | (UTILITY_MARGIN_FILES if selection['selection_passed'] else set())
        same_outputs(parent, child, utility_files-{'manifest.json', 'reference_parity.json'})
    return reproduced


def run_close_flow_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-flow-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V71.md')),
        'periods': CLOSE_SPLITS, 'features': FlowCloseModel.features, 'settings': HISTOGRAM_SETTINGS, 'flow_windows': FLOW_WINDOWS,
        'new_model_count': 1, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'cost_weight': 'original_position_weight_times_absolute_effect', 'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'체결 방향의 청산 비용 학습 진단: {out}', flush=True)
    try:
        reproduced = reproduce_context(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        bars, source = close_flow_source(reference)
        save_json(out/'flow_source.json', source)
        rows = attach_close_flow({name: pd.read_parquet(reproduced/f'{name}_used.parquet') for name in ['training', 'diagnosis']}, bars)
        for name, frame in rows.items():
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
        for name in ['training_weights', 'diagnosis_weights', 'exclusion_ledger']:
            pd.read_parquet(reproduced/f'{name}.parquet').to_parquet(out/f'{name}.parquet', index=False)
        weights = pd.read_parquet(out/'training_weights.parquet')
        model, support, costs = fit_utility(rows['training'], weights, rows['diagnosis'], model_class=FlowCloseModel)
        pd.testing.assert_frame_equal(costs, pd.read_parquet(reproduced/'training_cost_ledger.parquet'), check_exact=True)
        previous_support = json.loads((reproduced/'training_support.json').read_text())
        if {k: v for k, v in support.items() if k != 'export'} != {k: v for k, v in previous_support.items() if k != 'export'}:
            raise ValueError('청산 체결 방향의 원래 비용 정규화·학습 상수 불일치')
        costs.to_parquet(out/'training_cost_ledger.parquet', index=False)
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        for name in ['model', 'training_support', 'decision']:
            save_json(out/f'previous_{name}.json', json.loads((reproduced/f'{name}.json').read_text()))
        pd.read_parquet(reproduced/'predictions.parquet').to_parquet(out/'previous_predictions.parquet', index=False)
        decision = evaluate_context(reproduced, rows['diagnosis'], model, out, candidate_name='flow',
            comparisons=('continuation', 'utility', 'context'), admission=flow_close_admission)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('청산 체결 방향 진단 중 원본 지문 변경')
        _, final_source = close_flow_source(reference)
        if source != final_source:
            raise ValueError('청산 체결 방향 진단 중 시세 원본 변경')
        save_json(out/'decision.json', decision)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'original_cost_ledger_and_prior_preserved': True,
            'zero_effect_rows_preserved': True, 'training_rows': len(rows['training']), 'fit_rows': support['fit_rows'],
            'diagnosis_rows': len(rows['diagnosis']), 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 체결 방향의 청산 비용 학습 진단\n\n'
            f'사전 조건 통과: {decision["flow_admitted"]}. '
            '원래 64개 입력에 판단 당시 확정된 매수 체결 불균형과 방향별 값 열 개를 추가했다. '
            '원래 행·가중치·손익 비용·학습 상수·정답과 고정 점수 0.5 기준을 유지했다. '
            '비용 가중 선택 점수는 수익 확률이나 예상 손익이 아니다. 연속 계좌 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
