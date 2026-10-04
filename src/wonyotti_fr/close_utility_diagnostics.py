from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_calibration import CALIBRATION_SPLITS
from .close_learning_inputs import CLOSE_SPLITS
from .close_utility import (
    UtilityCloseModel,
    cost_probability_metrics,
    first_cost_positions,
    policy_effect_metrics,
    utility_admission,
)
from .common import new_run, save_json, sha256
from .continuation_diagnostics import same_outputs
from .entry_regression import REGRESSION_SETTINGS
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .first_close_margin import FIRST_MARGINS
from .first_close_margin_diagnostics import WEEKLY_FILES, run_first_close_margin_diagnosis
from .histogram_management import HISTOGRAM_SETTINGS

MARGIN_COMMON_FILES = {'manifest.json', 'reference_parity.json', 'half_training_used.parquet', 'half_training_weights.parquet',
    'calibration_used.parquet', 'calibration_weights.parquet', 'calibration_predictions.parquet', 'frozen_half_model.json',
    'calibration_first_metrics.json', 'calibration_row_metrics.json', 'selection.json', 'previous_models.json',
    'previous_weekly_models.json', 'previous_metrics.json', 'previous_predictions.parquet', 'exclusion_ledger.parquet',
    'diagnosis_used.parquet', 'diagnosis_weights.parquet', 'decision.json', 'summary.json', 'REPORT.md',
    *[f'calibration_positions_candidate-{i:02}.parquet' for i in range(6)]}
MARGIN_FINAL_FILES = {'predictions.parquet', 'metrics.json', 'breakdown.json', 'positions_margin.parquet',
    'positions_half_zero.parquet', 'positions_continuation.parquet', 'positions_weekly.parquet', 'first_metrics.json',
    'first_breakdown.json', 'blocks.parquet', 'block_draws.parquet', 'block_replicates.parquet', 'block_intervals.json'}


def reproduce_margin(reference, output):
    files = json.loads((reference/'files.json').read_text())
    selection = json.loads((reference/'selection.json').read_text())
    if type(selection['selection_passed']) is not bool:
        raise ValueError('청산 비용 진단의 원래 문턱 선택 상태 오류')
    expected = MARGIN_COMMON_FILES | (MARGIN_FINAL_FILES if selection['selection_passed'] else set())
    if (set(files) != expected or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('청산 비용 진단의 문턱 원본 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    parent = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V68.md'))
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CALIBRATION_SPLITS or settings['features'] != UtilityCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['fixed_half_model'] != 'depth2_iter64'
        or settings['margins_bps'] != FIRST_MARGINS or settings['new_models_fitted'] is not False
        or settings['existing_models_reproduced'] is not True or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_previous_outputs_reproduced'] is not True
        or summary['frozen_half_model_unchanged'] is not True or summary['previous_predictions_and_weights_preserved'] is not True
        or summary['new_models_fitted'] is not False or summary['profitability_accepted'] is not False
        or summary['final_diagnosis_evaluated'] != selection['selection_passed']):
        raise ValueError('청산 비용 진단의 원래 문턱 설정·완료 오류')
    reproduced = run_first_close_margin_diagnosis(parent, output)
    same_outputs(reference, reproduced, expected-{'manifest.json', 'reference_parity.json'})
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_outputs_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('청산 비용 진단의 주간 부모 연결 오류')
        same_outputs(parent, child, WEEKLY_FILES-{'manifest.json', 'reference_parity.json'})
    return reproduced


def fit_utility(training, weights, diagnosis):
    pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
    model, support, costs = UtilityCloseModel.fit(training[UtilityCloseModel.features].to_numpy(dtype=float),
        training.close_advantage_bps, weights.sample_weight, diagnosis[UtilityCloseModel.features].to_numpy(dtype=float))
    ledger = pd.concat([training[['decision_time', 'position_entry_time']].reset_index(drop=True), costs], axis=1)
    return model, support, ledger


def evaluate_utility(weekly, margin, model, constant, output):
    diagnosis = pd.read_parquet(weekly/'diagnosis_used.parquet')
    margin_selection = json.loads((margin/'selection.json').read_text())
    prior = margin/'predictions.parquet' if margin_selection['selection_passed'] else weekly/'predictions.parquet'
    frame = pd.read_parquet(prior)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']
    pd.testing.assert_frame_equal(frame[keys], diagnosis[keys], check_exact=True)
    frame['utility_score'] = model.probabilities(diagnosis[model.features].to_numpy(dtype=float))[:, 0]
    frame['training_constant_score'] = constant
    frame.to_parquet(output/'predictions.parquet', index=False)
    allowed = frame.original_intent.ne('exit').to_numpy()
    policies = {name.removeprefix('predicted_'): frame[name].gt(0).to_numpy() & allowed
        for name in frame if name.startswith('predicted_') and name != 'predicted_half'}
    if margin_selection['selection_passed']:
        policies['half_zero'] = frame.predicted_half.gt(0).to_numpy() & allowed
        policies['margin'] = frame.predicted_half.gt(margin_selection['chosen_margin_bps']).to_numpy() & allowed
    policies['utility'] = frame.utility_score.gt(.5).to_numpy() & allowed
    policies['training_constant'] = frame.training_constant_score.gt(.5).to_numpy() & allowed
    metrics = {name: policy_effect_metrics(frame, action) for name, action in policies.items()}
    probability = {name: cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, frame[name+'_score'])
        for name in ['utility', 'training_constant']}
    positions = {'utility': first_cost_positions(diagnosis, frame.utility_score),
        'training_constant': first_cost_positions(diagnosis, frame.training_constant_score),
        'continuation': pd.read_parquet(weekly/'positions_continuation.parquet'),
        'weekly': pd.read_parquet(weekly/'positions_weekly.parquet')}
    if margin_selection['selection_passed']:
        for name in ['margin', 'half_zero']:
            positions[name] = pd.read_parquet(margin/f'positions_{name}.parquet')
    first, first_details = {}, []
    for name, part in positions.items():
        part.to_parquet(output/f'positions_{name}.parquet', index=False)
        first[name] = first_close_metrics(part)
        groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
        groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
        for kind, group, f in groups:
            first_details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
    blocks, draws, replicates, intervals = paired_week_blocks(
        {name: positions[name] for name in ['utility', 'continuation']}, model_names=['utility', 'continuation'])
    for name, data in [('blocks', blocks), ('block_draws', draws), ('block_replicates', replicates)]:
        data.to_parquet(output/f'{name}.parquet', index=False)
    details, probability_details = [], []
    groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
    groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
    for kind, group, part in groups:
        for name, action in policies.items():
            details.append({'kind': kind, 'group': group, 'model': name, **policy_effect_metrics(part, action[part.index])})
        for name in probability:
            probability_details.append({'kind': kind, 'group': group, 'model': name,
                **cost_probability_metrics(part.close_advantage_bps, part.sample_weight, part[name+'_score'])})
    for name, value in [('metrics', metrics), ('probability_metrics', probability), ('breakdown', details),
        ('probability_breakdown', probability_details), ('first_metrics', first), ('first_breakdown', first_details), ('block_intervals', intervals)]:
        save_json(output/f'{name}.json', value)
    return utility_admission(metrics, probability, first, intervals)


def run_close_utility_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-utility-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V69.md')),
        'periods': CLOSE_SPLITS, 'features': UtilityCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'new_model_count': 1, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'cost_weight': 'original_position_weight_times_absolute_effect', 'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'청산 오류의 손익 비용 학습 진단: {out}', flush=True)
    try:
        reproduced = reproduce_margin(reference, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        weekly = Path(json.loads((reproduced/'reference_parity.json').read_text())['reproduction'])
        for name in ['training_used', 'training_weights', 'diagnosis_used', 'diagnosis_weights', 'exclusion_ledger']:
            pd.read_parquet(weekly/f'{name}.parquet').to_parquet(out/f'{name}.parquet', index=False)
        training, diagnosis = (pd.read_parquet(out/f'{name}_used.parquet') for name in ['training', 'diagnosis'])
        weights = pd.read_parquet(out/'training_weights.parquet')
        model, support, costs = fit_utility(training, weights, diagnosis)
        costs.to_parquet(out/'training_cost_ledger.parquet', index=False)
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        for name in ['previous_models', 'previous_weekly_models', 'previous_metrics', 'frozen_half_model', 'selection']:
            save_json(out/('previous_'+name+'.json' if name in {'frozen_half_model', 'selection'} else name+'.json'),
                json.loads((reproduced/f'{name}.json').read_text()))
        if (reproduced/'metrics.json').exists():
            save_json(out/'previous_margin_metrics.json', json.loads((reproduced/'metrics.json').read_text()))
        pd.read_parquet(reproduced/'previous_predictions.parquet').to_parquet(out/'previous_predictions.parquet', index=False)
        decision = evaluate_utility(weekly, reproduced, model, support['training_constant_score'], out)
        save_json(out/'decision.json', decision)
        if sha256(reference/'files.json') != json.loads((out/'manifest.json').read_text())['settings']['reference_files_sha256']:
            raise ValueError('청산 비용 진단 중 원본 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_evaluation_weights_preserved': True, 'zero_effect_rows_preserved': True,
            'training_rows': len(training), 'fit_rows': support['fit_rows'], 'diagnosis_rows': len(diagnosis),
            'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 청산 오류의 손익 비용 학습 진단\n\n'
            f'사전 조건 통과: {decision["utility_admitted"]}. '
            '원래 손익 효과의 절댓값을 분류 오류 비용으로 반영했다. 0 효과 행의 학습 기여만 0이며 원장과 평가는 보존했다. '
            '선택 점수는 일반적인 수익 확률이나 예상 손익이 아니다. 연속 계좌 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
