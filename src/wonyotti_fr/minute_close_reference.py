from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_flow import FlowCloseModel
from .close_learning_inputs import close_learning_splits
from .close_utility import cost_probability_metrics, first_cost_positions, policy_effect_metrics
from .close_utility_diagnostics import fit_utility
from .common import save_json, sha256
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .source_snapshot import copy_snapshot
from .weekly_flow_diagnostics import load_full_flow_ledger, validate_flow_reference


def verify_minute_baseline(reference, labels, ledger, output):
    validate_flow_reference(reference)
    before = sha256(reference/'files.json')
    files = json.loads((reference/'files.json').read_text())
    legacy, source = load_full_flow_ledger(reference)
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    if (Path(source['market']['labels']).resolve() != Path(settings['legacy_labels']).resolve()
        or source['market']['labels_files_sha256'] != settings['legacy_files_sha256']):
        raise ValueError('분별 청산 기준의 기존 정답 연결 오류')
    grid = ledger.decision_time.astype('datetime64[ns, UTC]').array.asi8 % (300*10**9) == 0
    pd.testing.assert_frame_equal(ledger.loc[grid].reset_index(drop=True), legacy, check_exact=True)
    rows, assignments = close_learning_splits(legacy)
    old_exclusions = pd.read_parquet(reference/'exclusion_ledger.parquet')
    pd.testing.assert_frame_equal(assignments[old_exclusions.columns], old_exclusions, check_exact=True)
    weights = {}
    for name, frame in rows.items():
        pd.testing.assert_frame_equal(frame, pd.read_parquet(reference/f'{name}_used.parquet'), check_exact=True)
        weights[name] = frame[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(frame))
        pd.testing.assert_frame_equal(weights[name], pd.read_parquet(reference/f'{name}_weights.parquet'), check_exact=True)
    model, support, costs = fit_utility(rows['training'], weights['training'], rows['diagnosis'], model_class=FlowCloseModel)
    if model.to_dict() != json.loads((reference/'model.json').read_text()) or support != json.loads((reference/'training_support.json').read_text()):
        raise ValueError('분별 청산 기준의 기존 모델·학습 지원 재현 불일치')
    pd.testing.assert_frame_equal(costs, pd.read_parquet(reference/'training_cost_ledger.parquet'), check_exact=True)
    frame = rows['diagnosis'].assign(sample_weight=weights['diagnosis'].sample_weight)
    score = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
    previous = pd.read_parquet(reference/'predictions.parquet')
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    pd.testing.assert_frame_equal(previous[keys], frame[keys], check_exact=True)
    np.testing.assert_array_equal(score, previous.flow_score)
    actions = (score > .5) & frame.original_intent.ne('exit').to_numpy()
    positions = first_cost_positions(frame, score)
    pd.testing.assert_frame_equal(positions, pd.read_parquet(reference/'positions_flow.parquet'), check_exact=True)
    values = {'metrics': policy_effect_metrics(frame, actions), 'first_metrics': first_close_metrics(positions),
        'probability_metrics': cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, score)}
    for name, result in values.items():
        if result != json.loads((reference/f'{name}.json').read_text())['flow']:
            raise ValueError('분별 청산 기준의 기존 지표 재현 불일치: '+name)
    breakdown, probability_breakdown, first_breakdown = [], [], []
    groups = [('direction', str(key), group) for key, group in frame.groupby('direction')]
    groups += [('month', key, group) for key, group in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
    for kind, key, group in groups:
        breakdown.append({'kind': kind, 'group': key, 'model': 'flow', **policy_effect_metrics(group, actions[group.index])})
        probability_breakdown.append({'kind': kind, 'group': key, 'model': 'flow',
            **cost_probability_metrics(group.close_advantage_bps, group.sample_weight, score[group.index])})
    groups = [('direction', str(key), group) for key, group in positions.groupby('direction')]
    groups += [('entry_month', key, group) for key, group in positions.groupby(positions.position_entry_time.dt.strftime('%Y-%m'))]
    for kind, key, group in groups:
        first_breakdown.append({'kind': kind, 'group': key, 'model': 'flow', **first_close_metrics(group)})
    for name, actual in [('breakdown', breakdown), ('probability_breakdown', probability_breakdown), ('first_breakdown', first_breakdown)]:
        expected = [item for item in json.loads((reference/f'{name}.json').read_text()) if item['model'] == 'flow']
        if actual != expected:
            raise ValueError('분별 청산 기준의 기존 그룹 지표 재현 불일치: '+name)
    for other in ['continuation', 'utility', 'context']:
        compare = pd.read_parquet(reference/f'positions_{other}.parquet')
        result = paired_week_blocks({'flow': positions, other: compare}, model_names=['flow', other])
        prefix = '' if other == 'continuation' else other+'_'
        for name, data in zip(['blocks', 'block_draws', 'block_replicates'], result[:3], strict=True):
            pd.testing.assert_frame_equal(data, pd.read_parquet(reference/f'{prefix}{name}.parquet'), check_exact=True)
        if result[3] != json.loads((reference/f'{prefix}block_intervals.json').read_text()):
            raise ValueError('분별 청산 기준의 기존 주간 구간 재현 불일치')
    target = output/'baseline_source'
    target.mkdir(mode=0o700)
    for name, expected in files.items():
        if sha256(reference/name) != expected:
            raise ValueError('분별 청산 기준 재현 중 원본 변경')
        copy_snapshot(reference/name, target/name)
        if sha256(target/name) != expected:
            raise ValueError('분별 청산 기준 사본 지문 불일치')
    if sha256(reference/'files.json') != before:
        raise ValueError('분별 청산 기준의 봉인 변경')
    save_json(output/'baseline_files.json', files)
    verification = {'complete': True, 'reference': str(reference), 'reference_files_sha256': before,
        'all_legacy_rows_and_74_features_exact': True, 'legacy_rows': len(legacy),
        'all_legacy_training_weights_costs_model_and_scores_exact': True,
        'legacy_first_choices_metrics_breakdowns_and_intervals_exact': True,
        'all_previous_source_files_preserved': True, 'original_ledger_source': source,
        'original_decision': json.loads((reference/'decision.json').read_text())}
    save_json(output/'baseline_verification.json', verification)
    return model, verification
