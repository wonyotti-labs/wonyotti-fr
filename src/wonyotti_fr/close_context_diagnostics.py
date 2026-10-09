from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_context import (
    ContextCloseModel,
    attach_close_context,
    close_context_source,
    context_close_admission,
)
from .close_learning_inputs import CLOSE_SPLITS
from .close_utility import (
    UtilityCloseModel,
    cost_probability_metrics,
    first_cost_positions,
    policy_effect_metrics,
)
from .close_utility_diagnostics import (
    MARGIN_COMMON_FILES,
    MARGIN_FINAL_FILES,
    fit_utility,
    run_close_utility_diagnosis,
)
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .histogram_management import HISTOGRAM_SETTINGS

UTILITY_COMMON_FILES = {'REPORT.md', 'block_draws.parquet', 'block_intervals.json', 'block_replicates.parquet', 'blocks.parquet',
    'breakdown.json', 'decision.json', 'diagnosis_used.parquet', 'diagnosis_weights.parquet', 'exclusion_ledger.parquet',
    'first_breakdown.json', 'first_metrics.json', 'manifest.json', 'metrics.json', 'model.json', 'positions_continuation.parquet',
    'positions_training_constant.parquet', 'positions_utility.parquet', 'positions_weekly.parquet', 'predictions.parquet',
    'previous_frozen_half_model.json', 'previous_metrics.json', 'previous_models.json', 'previous_predictions.parquet',
    'previous_selection.json', 'previous_weekly_models.json', 'probability_breakdown.json', 'probability_metrics.json',
    'reference_parity.json', 'summary.json', 'training_cost_ledger.parquet', 'training_support.json', 'training_used.parquet',
    'training_weights.parquet'}
UTILITY_MARGIN_FILES = {'positions_half_zero.parquet', 'positions_margin.parquet', 'previous_margin_metrics.json'}


def reproduce_utility(reference, output):
    files = json.loads((reference/'files.json').read_text())
    selection = json.loads((reference/'previous_selection.json').read_text())
    if type(selection['selection_passed']) is not bool:
        raise ValueError('청산 시장 맥락의 이전 문턱 선택 상태 오류')
    expected = UTILITY_COMMON_FILES | (UTILITY_MARGIN_FILES if selection['selection_passed'] else set())
    if (set(files) != expected or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('청산 시장 맥락의 비용 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V69.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != UtilityCloseModel.features
        or settings['settings'] != HISTOGRAM_SETTINGS or settings['new_model_count'] != 1
        or settings['score_threshold'] != .5 or settings['score_kind'] != 'cost_weighted_decision_score'
        or settings['cost_weight'] != 'original_position_weight_times_absolute_effect'
        or settings['zero_effect_policy'] != 'zero_fit_contribution_preserved_in_evaluation'
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['all_original_rows_and_evaluation_weights_preserved'] is not True
        or summary['zero_effect_rows_preserved'] is not True or summary['profitability_accepted'] is not False):
        raise ValueError('청산 시장 맥락의 원래 비용 설정·완료 오류')
    reproduced = run_close_utility_diagnosis(parent, output)
    same_outputs(reference, reproduced, expected-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('청산 시장 맥락의 비용 부모 연결 오류')
        margin_files = MARGIN_COMMON_FILES | (MARGIN_FINAL_FILES if selection['selection_passed'] else set())
        same_outputs(parent, child, margin_files-{'manifest.json', 'reference_parity.json'})
    return reproduced


def evaluate_context(reference, diagnosis, model, output, *, candidate_name='context',
                     comparisons=('continuation', 'utility'), admission=context_close_admission):
    if comparisons != {'context': ('continuation', 'utility'), 'flow': ('continuation', 'utility', 'context')}.get(candidate_name):
        raise ValueError('청산 비용 확장의 후보·비교 이름 오류')
    score = candidate_name+'_score'
    frame = pd.read_parquet(reference/'predictions.parquet')
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']
    pd.testing.assert_frame_equal(frame[keys], diagnosis[keys], check_exact=True)
    if score in frame:
        raise ValueError('청산 비용 확장의 기존 점수 덮어쓰기 거부')
    frame[score] = model.probabilities(diagnosis[model.features].to_numpy(dtype=float))[:, 0]
    frame.to_parquet(output/'predictions.parquet', index=False)
    action = frame[score].gt(.5).to_numpy() & frame.original_intent.ne('exit').to_numpy()
    values = {name: json.loads((reference/f'{name}.json').read_text()) for name in
        ['metrics', 'probability_metrics', 'first_metrics', 'breakdown', 'probability_breakdown', 'first_breakdown']}
    values['metrics'][candidate_name] = policy_effect_metrics(frame, action)
    values['probability_metrics'][candidate_name] = cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, frame[score])
    positions = {name: pd.read_parquet(reference/f'positions_{name}.parquet') for name in values['first_metrics']}
    positions[candidate_name] = first_cost_positions(diagnosis, frame[score])
    values['first_metrics'][candidate_name] = first_close_metrics(positions[candidate_name])
    for name, part in positions.items():
        part.to_parquet(output/f'positions_{name}.parquet', index=False)
    groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
    groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
    for kind, group, part in groups:
        values['breakdown'].append({'kind': kind, 'group': group, 'model': candidate_name, **policy_effect_metrics(part, action[part.index])})
        values['probability_breakdown'].append({'kind': kind, 'group': group, 'model': candidate_name,
            **cost_probability_metrics(part.close_advantage_bps, part.sample_weight, part[score])})
    first = positions[candidate_name]
    groups = [('direction', str(k), part) for k, part in first.groupby('direction')]
    groups += [('entry_month', k, part) for k, part in first.groupby(first.position_entry_time.dt.strftime('%Y-%m'))]
    for kind, group, part in groups:
        values['first_breakdown'].append({'kind': kind, 'group': group, 'model': candidate_name, **first_close_metrics(part)})
    intervals = {}
    previous_draws = pd.read_parquet(reference/'block_draws.parquet')
    for other in comparisons:
        prefix = '' if other == 'continuation' else other+'_'
        blocks, draws, replicates, interval = paired_week_blocks(
            {name: positions[name] for name in [candidate_name, other]}, model_names=[candidate_name, other])
        pd.testing.assert_frame_equal(draws, previous_draws, check_exact=True)
        for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
            data.to_parquet(output/f'{prefix}{name}.parquet', index=False)
        save_json(output/f'{prefix}block_intervals.json', interval)
        intervals[other] = interval
    for name, value in values.items():
        save_json(output/f'{name}.json', value)
    return admission(values['metrics'], values['probability_metrics'], values['first_metrics'],
        *[intervals[name] for name in comparisons])


def run_close_context_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-context-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V70.md')),
        'periods': CLOSE_SPLITS, 'features': ContextCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'new_model_count': 1, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'cost_weight': 'original_position_weight_times_absolute_effect', 'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'다일 시장 상태의 청산 비용 학습 진단: {out}', flush=True)
    try:
        reproduced = reproduce_utility(reference, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        bars, source = close_context_source(reference)
        save_json(out/'context_source.json', source)
        rows = attach_close_context({name: pd.read_parquet(reproduced/f'{name}_used.parquet') for name in ['training', 'diagnosis']}, bars)
        for name, frame in rows.items():
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
        for name in ['training_weights', 'diagnosis_weights', 'exclusion_ledger']:
            pd.read_parquet(reproduced/f'{name}.parquet').to_parquet(out/f'{name}.parquet', index=False)
        weights = pd.read_parquet(out/'training_weights.parquet')
        model, support, costs = fit_utility(rows['training'], weights, rows['diagnosis'], model_class=ContextCloseModel)
        pd.testing.assert_frame_equal(costs, pd.read_parquet(reproduced/'training_cost_ledger.parquet'), check_exact=True)
        previous_support = json.loads((reproduced/'training_support.json').read_text())
        if {k: v for k, v in support.items() if k != 'export'} != {k: v for k, v in previous_support.items() if k != 'export'}:
            raise ValueError('청산 시장 맥락의 원래 비용 정규화·학습 상수 불일치')
        costs.to_parquet(out/'training_cost_ledger.parquet', index=False)
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        for name in ['model', 'training_support', 'decision']:
            save_json(out/f'previous_{name}.json', json.loads((reproduced/f'{name}.json').read_text()))
        pd.read_parquet(reproduced/'predictions.parquet').to_parquet(out/'previous_predictions.parquet', index=False)
        decision = evaluate_context(reproduced, rows['diagnosis'], model, out)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('청산 시장 맥락 진단 중 원본 지문 변경')
        _, final_source = close_context_source(reference)
        if source != final_source:
            raise ValueError('청산 시장 맥락 진단 중 시세 원본 변경')
        save_json(out/'decision.json', decision)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'original_cost_ledger_and_prior_preserved': True,
            'zero_effect_rows_preserved': True, 'training_rows': len(rows['training']), 'fit_rows': support['fit_rows'],
            'diagnosis_rows': len(rows['diagnosis']), 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 다일 시장 상태의 청산 비용 학습 진단\n\n'
            f'사전 조건 통과: {decision["context_admitted"]}. '
            '원래 56개 입력에 판단 당시 확정된 다일 시장 상태 여덟 개를 추가했다. '
            '원래 행·가중치·손익 비용·학습 상수·정답과 고정 점수 0.5 기준을 유지했다. '
            '비용 가중 선택 점수는 수익 확률이나 예상 손익이 아니다. 연속 계좌 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
