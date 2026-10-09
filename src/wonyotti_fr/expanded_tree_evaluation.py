from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .close_utility import cost_probability_metrics, policy_effect_metrics
from .early_stopping_evaluation import positive
from .expanded_first_evaluation import (
    EXPANDED_FIRST_COMPARISONS,
    EXPANDED_FIRST_POLICIES,
    expanded_first_decision,
)
from .first_close_diagnostics import first_close_metrics
from .first_opportunity_close import first_opportunity_policy, first_opportunity_rows
from .stopping_regression_diagnostics import compare_prior_positions
from .stopping_regression_evaluation import REGRESSION_NEW_POLICIES, evaluate_regression_policies

EXPANDED_TREE_POLICIES = [*EXPANDED_FIRST_POLICIES, 'expanded_first_tree']
EXPANDED_TREE_COMPARISONS = ['expanded_first_linear', *EXPANDED_FIRST_COMPARISONS]


def expanded_tree_decision(metrics, first):
    if (set(metrics) != set(EXPANDED_TREE_POLICIES) or set(first) != set(EXPANDED_TREE_POLICIES)
        or len({(row['rows'], row['positions']) for row in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in EXPANDED_TREE_POLICIES)):
        raise ValueError('확장 첫 트리 보정의 정책·동일 모집단 오류')
    item = first['expanded_first_tree']
    value = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(value)}
    for name in EXPANDED_TREE_COMPARISONS:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(value) in (int, float) and type(other) in (int, float)
            and np.isfinite(value) and np.isfinite(other) and value > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def append_first_policy(result, frame, name, values):
    first, _ = first_opportunity_rows(frame)
    action, part, full_scores = first_opportunity_policy(frame, values)
    result['predictions'][name+'_score'] = full_scores
    result['predictions']['selected_'+name] = action
    result['positions'][name] = part
    result['metrics'][name] = policy_effect_metrics(frame, action)
    result['first_metrics'][name] = first_close_metrics(part)
    result['first_probability_metrics'][name] = cost_probability_metrics(first.first_target_common_bps, np.ones(len(first)), values)
    for kind, groups in [('direction', frame.groupby('direction')), ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
        for key, group in groups:
            result['breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
    for kind, groups in [('direction', part.groupby('direction')), ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
        for key, group in groups:
            result['first_breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})


def verify_thirteen_saved_policies(frame, baseline):
    pred = baseline['predictions']
    constant = pred.regression_constant_effect_bps.iloc[0]
    if not pred.regression_constant_effect_bps.eq(constant).all():
        raise ValueError('확장 첫 트리 비교의 이전 회귀 상수 변경')
    rebuilt = evaluate_regression_policies(frame, pred.stopping_regression_effect_bps.to_numpy(),
        pred.early_stopping_score.to_numpy(), pred.early_natural_score.to_numpy(), float(constant))
    first, positions = first_opportunity_rows(frame)
    rebuilt['first_membership'] = positions
    rebuilt['first_probability_metrics'] = {}
    for name in EXPANDED_FIRST_POLICIES:
        if name not in REGRESSION_NEW_POLICIES:
            append_first_policy(rebuilt, frame, name, pred[name+'_score'].iloc[first.opportunity_index].to_numpy())
    pd.testing.assert_frame_equal(rebuilt['predictions'], pred, check_exact=True)
    pd.testing.assert_frame_equal(positions, baseline['first_membership'], check_exact=True)
    for name in EXPANDED_FIRST_POLICIES:
        compare_prior_positions(rebuilt['positions'][name], baseline['positions'][name])
    for field in ['metrics', 'first_metrics', 'probability_metrics', 'probability_breakdown', 'breakdown', 'first_breakdown', 'first_probability_metrics']:
        if rebuilt[field] != baseline[field]:
            raise ValueError('확장 첫 트리 비교의 기존 열세 정책 재계산 불일치: '+field)
    decision = expanded_first_decision(baseline['metrics'], baseline['first_metrics'])
    if any(baseline['calibration_decision'][key] != value for key, value in decision.items()):
        raise ValueError('확장 첫 트리 비교의 이전 열 판정 불일치')


def evaluate_expanded_tree(frame, scores, baseline):
    frame = frame.reset_index(drop=True)
    verify_thirteen_saved_policies(frame, baseline)
    result = copy.deepcopy(baseline)
    append_first_policy(result, frame, 'expanded_first_tree', scores)
    result['calibration_decision'] = expanded_tree_decision(result['metrics'], result['first_metrics'])
    return result
