from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_context import attach_close_context
from .close_context_diagnostics import evaluate_close_scores
from .close_flow import FLOW_WINDOWS, FlowCloseModel, attach_close_flow, close_flow_source
from .close_flow_diagnostics import (
    CONTEXT_COMMON_FILES,
    CONTEXT_MARGIN_FILES,
    run_close_flow_diagnosis,
)
from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .continuation_inputs import ContinuationCloseModel
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .weekly_close import WEEKLY_SETTINGS
from .weekly_flow import fit_weekly_flow, weekly_flow_admission

FLOW_COMMON_FILES = (CONTEXT_COMMON_FILES-{'context_source.json'}) | {'flow_source.json', 'positions_flow.parquet',
    'context_blocks.parquet', 'context_block_draws.parquet', 'context_block_replicates.parquet', 'context_block_intervals.json'}


def reproduce_flow(reference, output):
    files = json.loads((reference/'files.json').read_text())
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    parent = Path(settings['reference'])
    if settings['reference_files_sha256'] != sha256(parent/'files.json'):
        raise ValueError('주간 비용 학습의 이전 체결 결과 연결 오류')
    parent_files = json.loads((parent/'files.json').read_text())
    optional = set(parent_files) & CONTEXT_MARGIN_FILES
    if optional and optional != CONTEXT_MARGIN_FILES:
        raise ValueError('주간 비용 학습의 이전 문턱 분기 누락')
    expected = FLOW_COMMON_FILES | optional
    if (set(files) != expected or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('주간 비용 학습의 체결 원본 파일·지문 오류')
    summary = json.loads((reference/'summary.json').read_text())
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V71.md'))
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != FlowCloseModel.features
        or settings['settings'] != HISTOGRAM_SETTINGS or settings['new_model_count'] != 1
        or settings['flow_windows'] != FLOW_WINDOWS or settings['score_threshold'] != .5
        or settings['score_kind'] != 'cost_weighted_decision_score'
        or settings['cost_weight'] != 'original_position_weight_times_absolute_effect'
        or settings['zero_effect_policy'] != 'zero_fit_contribution_preserved_in_evaluation'
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['original_cost_ledger_and_prior_preserved'] is not True
        or summary['zero_effect_rows_preserved'] is not True or summary['profitability_accepted'] is not False):
        raise ValueError('주간 비용 학습의 원래 체결 설정·완료 오류')
    reproduced = run_close_flow_diagnosis(parent, output)
    same_outputs(reference, reproduced, expected-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('주간 비용 학습의 체결 부모 연결 오류')
        same_outputs(parent, child, (CONTEXT_COMMON_FILES | optional)-{'manifest.json', 'reference_parity.json'})
    return reproduced


def load_full_flow_ledger(reference):
    current, seen = Path(reference), set()
    for _ in range(16):
        if current.resolve() in seen:
            raise ValueError('주간 비용 학습의 전체 원장 참조 순환')
        seen.add(current.resolve())
        files = json.loads((current/'files.json').read_text())
        manifest = current/'manifest.json'
        if manifest.is_symlink() or sha256(manifest) != files['manifest.json']:
            raise ValueError('주간 비용 학습의 전체 원장 참조 지문 오류')
        settings = json.loads(manifest.read_text())['settings']
        if 'continuation_ledger.parquet' in files:
            path = current/'continuation_ledger.parquet'
            if (path.is_symlink() or sha256(path) != files[path.name]
                or settings['features'] != ContinuationCloseModel.features
                or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V64.md'))):
                raise ValueError('주간 비용 학습의 원래 전체 원장 오류')
            original = pd.read_parquet(path)
            bars, source = close_flow_source(reference)
            context = attach_close_context({'ledger': original}, bars)
            ledger = attach_close_flow(context, bars)['ledger']
            return ledger, {'continuation': str(current), 'continuation_files_sha256': sha256(current/'files.json'),
                'original_ledger_sha256': sha256(path), 'rows': len(ledger), 'market': source}
        parent = Path(settings['reference'])
        if settings['reference_files_sha256'] != sha256(parent/'files.json'):
            raise ValueError('주간 비용 학습의 전체 원장 부모 연결 오류')
        current = parent
    raise ValueError('주간 비용 학습의 전체 원장 참조 깊이 초과')


def run_weekly_flow_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'weekly-flow-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V72.md')),
        'periods': CLOSE_SPLITS, 'features': FlowCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'flow_windows': FLOW_WINDOWS, 'weekly_settings': WEEKLY_SETTINGS, 'weekly_model_count': 13,
        'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'cost_weight': 'original_position_weight_times_absolute_effect', 'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'weekly_constant_control': True, 'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'확정 결과의 주간 청산 비용 재학습 진단: {out}', flush=True)
    try:
        reproduced = reproduce_flow(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        ledger, source = load_full_flow_ledger(reference)
        ledger.to_parquet(out/'flow_ledger.parquet', index=False)
        save_json(out/'ledger_source.json', source)
        rows, _ = close_learning_splits(ledger)
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(reproduced/f'{name}_used.parquet'), check_exact=True)
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
        for name in ['training_weights', 'diagnosis_weights', 'exclusion_ledger', 'training_cost_ledger']:
            pd.read_parquet(reproduced/f'{name}.parquet').to_parquet(out/f'{name}.parquet', index=False)
        weights = pd.read_parquet(out/'training_weights.parquet')
        original = {name: json.loads((reproduced/f'{name}.json').read_text()) for name in ['model', 'training_support', 'decision']}
        for name, value in original.items():
            save_json(out/f'previous_{name}.json', value)
        pd.read_parquet(reproduced/'predictions.parquet').to_parquet(out/'previous_predictions.parquet', index=False)
        score, constant = fit_weekly_flow(ledger, rows['diagnosis'], rows['training'], weights, original['model'],
            pd.read_parquet(out/'training_cost_ledger.parquet'), original['training_support'], out)
        decision = evaluate_close_scores(reproduced, rows['diagnosis'], {'weekly_flow': score, 'weekly_constant': constant}, out,
            candidate_name='weekly_flow', comparisons=('continuation', 'utility', 'context', 'flow'), admission=weekly_flow_admission)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('주간 비용 학습 중 원본 지문 변경')
        final_ledger, final_source = load_full_flow_ledger(reference)
        if final_source != source:
            raise ValueError('주간 비용 학습 중 전체 원장·시세 원본 변경')
        pd.testing.assert_frame_equal(final_ledger, ledger, check_exact=True)
        save_json(out/'decision.json', decision)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'first_week_model_costs_and_prior_exact': True,
            'zero_effect_rows_preserved': True, 'weekly_models': 13, 'ledger_rows': len(ledger),
            'diagnosis_rows': len(rows['diagnosis']), 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 확정 결과의 주간 청산 비용 재학습 진단\n\n'
            f'사전 조건 통과: {decision["weekly_flow_admitted"]}. '
            '매주 시작 이틀 전에 종료된 거래만 누적해 같은 비용 분류기를 갱신했다. '
            '첫 모델·비용·상수는 원래 모델과 같으며 원래 행·평가 비중과 모든 과거 결과를 보존했다. '
            '같은 과거 비용의 주간 상수 대조를 함께 기록했다. 연속 계좌 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
