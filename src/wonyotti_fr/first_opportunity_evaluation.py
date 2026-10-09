from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_utility import cost_probability_metrics, policy_effect_metrics
from .early_stopping_evaluation import positive
from .first_close_diagnostics import first_close_metrics, first_close_positions
from .first_opportunity_close import first_opportunity_policy, first_opportunity_rows
from .stopping_regression_diagnostics import compare_prior_positions
from .stopping_regression_evaluation import REGRESSION_NEW_POLICIES, regression_calibration_decision

FIRST_NEW_POLICIES = ['first_opportunity', 'first_constant']
FIRST_POLICIES = [*REGRESSION_NEW_POLICIES, *FIRST_NEW_POLICIES]
FIRST_COMPARISONS = ['stopping_regression', 'early_stopping', 'early_natural', 'always_first']


def first_calibration_decision(metrics, first):
    if (set(metrics) != set(FIRST_POLICIES) or set(first) != set(FIRST_POLICIES)
        or len({(row['rows'], row['positions']) for row in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in FIRST_POLICIES)):
        raise ValueError('첫 적격 기회 보정의 정책·동일 모집단 오류')
    item = first['first_opportunity']
    value = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(value)}
    for name in FIRST_COMPARISONS:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(value) in (int, float) and type(other) in (int, float)
            and np.isfinite(value) and np.isfinite(other) and value > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def evaluate_first_opportunity(frame, scores, constant, baseline):
    if type(constant) not in (int, float) or not np.isfinite(constant) or not 0 < constant < 1:
        raise ValueError('첫 적격 기회의 학습 상수 오류')
    frame = frame.reset_index(drop=True)
    np.testing.assert_array_equal(frame.sample_weight, position_weights(frame))
    if any(set(baseline[name]) != set(REGRESSION_NEW_POLICIES) for name in ['metrics', 'first_metrics', 'positions']):
        raise ValueError('첫 적격 기회의 기존 여섯 정책 누락')
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    pd.testing.assert_frame_equal(frame[keys], baseline['predictions'][keys], check_exact=True)
    old_decision = regression_calibration_decision(baseline['metrics'], baseline['first_metrics'])
    if old_decision != baseline['calibration_decision']:
        raise ValueError('첫 적격 기회의 기존 내부 실패 변경')
    for name in REGRESSION_NEW_POLICIES:
        actions = baseline['predictions']['selected_'+name].to_numpy()
        if policy_effect_metrics(frame, actions) != baseline['metrics'][name]:
            raise ValueError('첫 적격 기회의 이전 선택 효과 불일치')
        part = first_close_positions(frame.assign(_choice=actions.astype(float)), '_choice').drop(columns='first_prediction_bps')
        previous = baseline['positions'][name].drop(columns=['first_cost_score', 'first_prediction_bps'], errors='ignore')
        compare_prior_positions(part, previous)
        if first_close_metrics(part) != baseline['first_metrics'][name]:
            raise ValueError('첫 적격 기회의 이전 최초 효과 불일치')
    result = copy.deepcopy(baseline)
    first, membership = first_opportunity_rows(frame)
    if first.empty:
        raise ValueError('첫 적격 기회 보정의 숫자 검증 입력 누락')
    first_probability = {}
    for name, values in [('first_opportunity', scores), ('first_constant', np.full(len(first), constant))]:
        action, part, full_score = first_opportunity_policy(frame, values)
        result['predictions'][name+'_score'] = full_score
        result['predictions']['selected_'+name] = action
        result['positions'][name] = part
        result['metrics'][name] = policy_effect_metrics(frame, action)
        result['first_metrics'][name] = first_close_metrics(part)
        first_probability[name] = cost_probability_metrics(first.first_target_common_bps, np.ones(len(first)), values)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                result['breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                result['first_breakdown'].append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    counterpart = 'always_first' if constant > .5 else 'never_extra'
    fields = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity', 'chosen',
        'first_selected_time', 'first_label_end', 'first_effect_pnl', 'first_effect_common_bps']
    compare_prior_positions(result['positions']['first_constant'][fields], result['positions'][counterpart][fields])
    result['first_probability_metrics'], result['first_membership'] = first_probability, membership
    result['calibration_decision'] = first_calibration_decision(result['metrics'], result['first_metrics'])
    return result
