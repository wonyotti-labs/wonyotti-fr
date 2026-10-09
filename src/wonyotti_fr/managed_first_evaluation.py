from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .early_stopping_evaluation import positive
from .expanded_first_evaluation import expanded_first_decision
from .expanded_tree_evaluation import (
    EXPANDED_TREE_COMPARISONS,
    EXPANDED_TREE_POLICIES,
    append_first_policy,
    expanded_tree_decision,
    verify_thirteen_saved_policies,
)
from .first_opportunity_close import first_opportunity_rows
from .stopping_regression_diagnostics import compare_prior_positions

MANAGED_POLICIES = [*EXPANDED_TREE_POLICIES, 'managed_first_linear']
MANAGED_COMPARISONS = ['expanded_first_tree', *EXPANDED_TREE_COMPARISONS]


def managed_decision(metrics, first):
    if (set(metrics) != set(MANAGED_POLICIES) or set(first) != set(MANAGED_POLICIES)
        or len({(row['rows'], row['positions']) for row in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in MANAGED_POLICIES)):
        raise ValueError('관리 확률 첫 선형 보정의 정책·동일 모집단 오류')
    item = first['managed_first_linear']
    value = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(value)}
    for name in MANAGED_COMPARISONS:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(value) in (int, float) and type(other) in (int, float)
            and np.isfinite(value) and np.isfinite(other) and value > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def verify_fourteen_saved_policies(frame, baseline):
    reduced = copy.deepcopy(baseline)
    name = 'expanded_first_tree'
    reduced['predictions'] = reduced['predictions'].drop(columns=[name+'_score', 'selected_'+name])
    for field in ['positions', 'metrics', 'first_metrics', 'first_probability_metrics']:
        reduced[field].pop(name)
    for field in ['breakdown', 'first_breakdown']:
        reduced[field] = [row for row in reduced[field] if row['policy'] != name]
    reduced['calibration_decision'] = expanded_first_decision(reduced['metrics'], reduced['first_metrics'])
    verify_thirteen_saved_policies(frame, reduced)
    first, _ = first_opportunity_rows(frame)
    append_first_policy(reduced, frame, name, baseline['predictions'][name+'_score'].iloc[first.opportunity_index].to_numpy())
    pd.testing.assert_frame_equal(reduced['predictions'], baseline['predictions'], check_exact=True)
    compare_prior_positions(reduced['positions'][name], baseline['positions'][name])
    for field in ['metrics', 'first_metrics', 'first_probability_metrics', 'breakdown', 'first_breakdown']:
        if reduced[field] != baseline[field]:
            raise ValueError('관리 확률 첫 선형 비교의 이전 열네 정책 재계산 불일치: '+field)
    decision = expanded_tree_decision(reduced['metrics'], reduced['first_metrics'])
    if any(baseline['calibration_decision'][key] != value for key, value in decision.items()):
        raise ValueError('관리 확률 첫 선형 비교의 이전 열한 판정 불일치')


def evaluate_managed_first(frame, scores, baseline):
    frame = frame.reset_index(drop=True)
    verify_fourteen_saved_policies(frame, baseline)
    result = copy.deepcopy(baseline)
    append_first_policy(result, frame, 'managed_first_linear', scores)
    result['calibration_decision'] = managed_decision(result['metrics'], result['first_metrics'])
    return result
