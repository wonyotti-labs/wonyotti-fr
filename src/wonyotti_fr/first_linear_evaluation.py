from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .close_utility import cost_probability_metrics, policy_effect_metrics
from .early_stopping_evaluation import positive
from .first_close_diagnostics import first_close_metrics
from .first_opportunity_close import first_opportunity_policy, first_opportunity_rows
from .first_opportunity_evaluation import (
    FIRST_POLICIES,
    evaluate_first_opportunity,
    first_calibration_decision,
)
from .stopping_regression_diagnostics import compare_prior_positions
from .stopping_regression_evaluation import REGRESSION_NEW_POLICIES

LINEAR_POLICIES = [*FIRST_POLICIES, 'first_linear']
LINEAR_COMPARISONS = ['first_opportunity', 'stopping_regression', 'early_stopping', 'early_natural', 'always_first']


def linear_calibration_decision(metrics, first):
    if (set(metrics) != set(LINEAR_POLICIES) or set(first) != set(LINEAR_POLICIES)
        or len({(row['rows'], row['positions']) for row in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in LINEAR_POLICIES)):
        raise ValueError('첫 기회 선형 보정의 정책·동일 모집단 오류')
    item = first['first_linear']
    value = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(value)}
    for name in LINEAR_COMPARISONS:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(value) in (int, float) and type(other) in (int, float)
            and np.isfinite(value) and np.isfinite(other) and value > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def verify_previous_first_policies(frame, baseline, constant):
    first, _ = first_opportunity_rows(frame)
    old = {name: copy.deepcopy(baseline[name]) for name in
        ['predictions', 'probability_metrics', 'probability_breakdown', 'breakdown', 'first_breakdown']}
    for name in ['metrics', 'first_metrics', 'positions']:
        old[name] = {key: copy.deepcopy(baseline[name][key]) for key in REGRESSION_NEW_POLICIES}
    columns = [name for name in old['predictions'] if name not in
        ['first_opportunity_score', 'first_constant_score', 'selected_first_opportunity', 'selected_first_constant']]
    old['predictions'] = old['predictions'][columns]
    for name in ['breakdown', 'first_breakdown']:
        old[name] = [row for row in old[name] if row['policy'] in REGRESSION_NEW_POLICIES]
    old['calibration_decision'] = baseline['earlier_regression_decision']
    scores = baseline['predictions'].first_opportunity_score.iloc[first.opportunity_index].to_numpy()
    reproduced = evaluate_first_opportunity(frame, scores, constant, old)
    decision = first_calibration_decision(baseline['metrics'], baseline['first_metrics'])
    if any(baseline['calibration_decision'][key] != value for key, value in decision.items()):
        raise ValueError('첫 기회 선형 보정의 이전 여섯 판정 불일치')
    for name in ['metrics', 'first_metrics', 'probability_metrics', 'probability_breakdown', 'breakdown', 'first_breakdown', 'first_probability_metrics']:
        if reproduced[name] != baseline[name]:
            raise ValueError('첫 기회 선형 보정의 이전 정책 재계산 불일치: '+name)
    pd.testing.assert_frame_equal(reproduced['predictions'], baseline['predictions'], check_exact=True)
    pd.testing.assert_frame_equal(reproduced['first_membership'], baseline['first_membership'], check_exact=True)
    for name in FIRST_POLICIES:
        compare_prior_positions(reproduced['positions'][name], baseline['positions'][name])


def evaluate_first_linear(frame, scores, constant, baseline):
    frame = frame.reset_index(drop=True)
    verify_previous_first_policies(frame, baseline, constant)
    result = copy.deepcopy(baseline)
    result.pop('earlier_regression_decision')
    first, _ = first_opportunity_rows(frame)
    action, part, full_scores = first_opportunity_policy(frame, scores)
    result['predictions']['first_linear_score'] = full_scores
    result['predictions']['selected_first_linear'] = action
    result['positions']['first_linear'] = part
    result['metrics']['first_linear'] = policy_effect_metrics(frame, action)
    result['first_metrics']['first_linear'] = first_close_metrics(part)
    result['first_probability_metrics']['first_linear'] = cost_probability_metrics(
        first.first_target_common_bps, np.ones(len(first)), scores)
    for kind, groups in [('direction', frame.groupby('direction')),
        ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
        for key, group in groups:
            result['breakdown'].append({'policy': 'first_linear', 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
    for kind, groups in [('direction', part.groupby('direction')),
        ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
        for key, group in groups:
            result['first_breakdown'].append({'policy': 'first_linear', 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    result['calibration_decision'] = linear_calibration_decision(result['metrics'], result['first_metrics'])
    return result
