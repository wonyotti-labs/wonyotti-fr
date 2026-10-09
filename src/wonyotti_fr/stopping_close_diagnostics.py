from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_context_diagnostics import evaluate_close_scores
from .close_flow import FLOW_WINDOWS, FlowCloseModel
from .close_flow_diagnostics import CONTEXT_MARGIN_FILES
from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .stopping_close import (
    STOPPING_SETTINGS,
    StoppingCloseModel,
    fit_stopping_close,
    stopping_admission,
)
from .weekly_close import WEEKLY_SETTINGS
from .weekly_flow_diagnostics import FLOW_COMMON_FILES, run_weekly_flow_diagnosis

WEEKLY_FLOW_COMMON_FILES = (FLOW_COMMON_FILES-{'model.json', 'training_support.json', 'flow_source.json'}) | {
    'ledger_source.json', 'flow_ledger.parquet', 'weekly_models.json', 'weekly_support.json',
    'weekly_training_membership.parquet', 'prediction_routing.parquet', 'positions_weekly_flow.parquet',
    'positions_weekly_constant.parquet', 'flow_blocks.parquet', 'flow_block_draws.parquet',
    'flow_block_replicates.parquet', 'flow_block_intervals.json'}


def reproduce_weekly_flow(reference, output):
    files = json.loads((reference/'files.json').read_text())
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    parent = Path(settings['reference'])
    if settings['reference_files_sha256'] != sha256(parent/'files.json'):
        raise ValueError('이후 청산의 주간 비용 부모 연결 오류')
    parent_files = json.loads((parent/'files.json').read_text())
    optional = set(parent_files) & CONTEXT_MARGIN_FILES
    if optional and optional != CONTEXT_MARGIN_FILES:
        raise ValueError('이후 청산의 이전 문턱 분기 누락')
    expected = WEEKLY_FLOW_COMMON_FILES | optional
    if (set(files) != expected or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('이후 청산의 주간 비용 파일·지문 오류')
    summary = json.loads((reference/'summary.json').read_text())
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V72.md'))
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != FlowCloseModel.features
        or settings['settings'] != HISTOGRAM_SETTINGS or settings['flow_windows'] != FLOW_WINDOWS
        or settings['weekly_settings'] != WEEKLY_SETTINGS or settings['weekly_model_count'] != 13
        or settings['score_threshold'] != .5 or settings['score_kind'] != 'cost_weighted_decision_score'
        or settings['cost_weight'] != 'original_position_weight_times_absolute_effect'
        or settings['zero_effect_policy'] != 'zero_fit_contribution_preserved_in_evaluation'
        or settings['weekly_constant_control'] is not True or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True or summary['complete'] is not True
        or summary['all_previous_outputs_reproduced'] is not True or summary['weekly_models'] != 13
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['first_week_model_costs_and_prior_exact'] is not True
        or summary['zero_effect_rows_preserved'] is not True or summary['profitability_accepted'] is not False):
        raise ValueError('이후 청산의 주간 비용 설정·완료 오류')
    reproduced = run_weekly_flow_diagnosis(parent, output)
    same_outputs(reference, reproduced, expected-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('이후 청산의 주간 부모 재현 연결 오류')
        same_outputs(parent, child, (FLOW_COMMON_FILES | optional)-{'manifest.json', 'reference_parity.json'})
    return reproduced


def run_stopping_close_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'stopping-close-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V73.md')),
        'periods': CLOSE_SPLITS, 'features': StoppingCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'stopping_settings': STOPPING_SETTINGS, 'teacher_models': 5, 'candidate_models': 1,
        'score_kind': 'stopping_cost_weighted_decision_score', 'score_threshold': .5,
        'training_target': 'immediate_close_minus_crossfit_future_first_close', 'evaluation_target': 'original_natural_close_advantage',
        'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation', 'weekly_candidate_refit': False,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'이후 청산 정책을 반영한 정답 진단: {out}', flush=True)
    try:
        reproduced = reproduce_weekly_flow(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        ledger = pd.read_parquet(reproduced/'flow_ledger.parquet')
        rows, _ = close_learning_splits(ledger)
        ledger.to_parquet(out/'flow_ledger.parquet', index=False)
        save_json(out/'ledger_source.json', json.loads((reproduced/'ledger_source.json').read_text()))
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(reproduced/f'{name}_used.parquet'), check_exact=True)
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
        for name in ['training_weights', 'diagnosis_weights', 'training_cost_ledger', 'exclusion_ledger']:
            pd.read_parquet(reproduced/f'{name}.parquet').to_parquet(out/f'{name}.parquet', index=False)
        for old, new in [('weekly_models', 'previous_weekly_models'), ('weekly_support', 'previous_weekly_support'),
            ('previous_model', 'fixed_reference_model'), ('previous_training_support', 'fixed_reference_training_support'), ('decision', 'previous_decision')]:
            save_json(out/f'{new}.json', json.loads((reproduced/f'{old}.json').read_text()))
        pd.read_parquet(reproduced/'predictions.parquet').to_parquet(out/'previous_predictions.parquet', index=False)
        indices = np.flatnonzero(ledger.decision_time.isin(rows['training'].decision_time))
        pd.testing.assert_frame_equal(ledger.iloc[indices].reset_index(drop=True), rows['training'], check_exact=True)
        model, support = fit_stopping_close(rows['training'], pd.read_parquet(out/'training_weights.parquet'), rows['diagnosis'], indices, out)
        scores = {'stopping_flow': model.probabilities(rows['diagnosis'][model.features].to_numpy(dtype=float))[:, 0]}
        decision = evaluate_close_scores(reproduced, rows['diagnosis'], scores, out, candidate_name='stopping_flow',
            comparisons=('continuation', 'utility', 'context', 'flow', 'weekly_flow'), admission=stopping_admission)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('이후 청산 진단 중 원본 지문 변경')
        if any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in json.loads((reference/'files.json').read_text()).items()):
            raise ValueError('이후 청산 진단 중 원본 출력 변경')
        save_json(out/'decision.json', decision)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'original_training_cost_ledger_preserved': True,
            'teacher_whole_position_exclusion': True, 'future_fields_used_only_in_targets': True,
            'zero_effect_rows_preserved': True, 'teacher_models': 5, 'candidate_models': 1,
            'training_rows': len(rows['training']), 'fit_rows': support['fit_rows'],
            'diagnosis_rows': len(rows['diagnosis']), 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 이후 청산 정책을 반영한 정답 진단\n\n'
            f'사전 조건 통과: {decision["stopping_flow_admitted"]}. '
            '같은 포지션을 학습하지 않은 다섯 보조 정책의 미래 첫 청산 현금으로 정답을 구성했다. '
            '최종 후보 하나를 원래 고정 학습에 적합했고, 기존 자연 종료 대비 손익과 서른일곱 조건으로 평가했다. '
            '원래 비용 분류 지표는 새 목표의 확률 보정 지표가 아니며 보수적 대조 조건이다. '
            '미래 값은 실행 입력에 없고 모든 기존 결과와 손실·선택 없음을 보존했다. 연속 매매 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
