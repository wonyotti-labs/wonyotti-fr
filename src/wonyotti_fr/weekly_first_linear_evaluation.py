from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .close_utility import cost_probability_metrics, policy_effect_metrics
from .early_stopping_evaluation import positive
from .first_close_diagnostics import first_close_metrics
from .first_linear_evaluation import (
    LINEAR_POLICIES,
    evaluate_first_linear,
    linear_calibration_decision,
)
from .first_opportunity_close import first_opportunity_policy, first_opportunity_rows
from .first_opportunity_evaluation import FIRST_POLICIES
from .stopping_regression_diagnostics import compare_prior_positions

WEEKLY_FIRST_POLICIES = [*LINEAR_POLICIES, 'weekly_first_linear', 'weekly_first_constant']
WEEKLY_FIRST_COMPARISONS = ['first_linear', 'first_opportunity', 'stopping_regression',
    'early_stopping', 'early_natural', 'always_first', 'weekly_first_constant']


def weekly_first_decision(metrics, first):
    if (set(metrics) != set(WEEKLY_FIRST_POLICIES) or set(first) != set(WEEKLY_FIRST_POLICIES)
        or len({(row['rows'], row['positions']) for row in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in WEEKLY_FIRST_POLICIES)):
        raise ValueError('첫 판단 주간 보정의 정책·모집단 오류')
    item = first['weekly_first_linear']
    value = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(value)}
    for name in WEEKLY_FIRST_COMPARISONS:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(value) in (int, float) and type(other) in (int, float)
            and np.isfinite(value) and np.isfinite(other) and value > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def verify_previous_linear_policies(frame, baseline, constant):
    old = copy.deepcopy(baseline)
    old['calibration_decision'] = old.pop('previous_first_decision')
    for field in ['metrics', 'first_metrics', 'positions']:
        old[field] = {name: old[field][name] for name in FIRST_POLICIES}
    for field in ['breakdown', 'first_breakdown']:
        old[field] = [row for row in old[field] if row['policy'] in FIRST_POLICIES]
    old['first_probability_metrics'].pop('first_linear')
    old['predictions'] = old['predictions'].drop(columns=['first_linear_score', 'selected_first_linear'])
    first, _ = first_opportunity_rows(frame)
    scores = baseline['predictions'].first_linear_score.iloc[first.opportunity_index].to_numpy()
    reproduced = evaluate_first_linear(frame, scores, constant, old)
    for field in ['metrics', 'first_metrics', 'probability_metrics', 'probability_breakdown', 'breakdown', 'first_breakdown', 'first_probability_metrics']:
        if reproduced[field] != baseline[field]:
            raise ValueError('첫 판단 주간 비교의 기존 아홉 정책 재계산 불일치: '+field)
    decision = linear_calibration_decision(baseline['metrics'], baseline['first_metrics'])
    if any(baseline['calibration_decision'][key] != value for key, value in decision.items()):
        raise ValueError('첫 판단 주간 비교의 기존 일곱 판정 불일치')
    for field in ['predictions', 'first_membership']:
        pd.testing.assert_frame_equal(reproduced[field], baseline[field], check_exact=True)
    for name in LINEAR_POLICIES:
        compare_prior_positions(reproduced['positions'][name], baseline['positions'][name])


def evaluate_weekly_first_linear(frame, scores, constants, original_constant, baseline):
    frame = frame.reset_index(drop=True)
    verify_previous_linear_policies(frame, baseline, original_constant)
    result = copy.deepcopy(baseline)
    result.pop('earlier_regression_decision')
    result.pop('previous_first_decision')
    first, _ = first_opportunity_rows(frame)
    for name, values in [('weekly_first_linear', scores), ('weekly_first_constant', constants)]:
        action, part, full_scores = first_opportunity_policy(frame, values)
        result['predictions'][name+'_score'] = full_scores
        result['predictions']['selected_'+name] = action
        result['positions'][name] = part
        result['metrics'][name] = policy_effect_metrics(frame, action)
        result['first_metrics'][name] = first_close_metrics(part)
        result['first_probability_metrics'][name] = cost_probability_metrics(first.first_target_common_bps, np.ones(len(first)), values)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                result['breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                result['first_breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    result['calibration_decision'] = weekly_first_decision(result['metrics'], result['first_metrics'])
    return result
