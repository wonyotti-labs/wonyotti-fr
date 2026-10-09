from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_utility import (
    cost_probability_metrics,
    cost_scores,
    first_cost_positions,
    policy_effect_metrics,
)
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .minute_close_learning import minute_grid_actions
from .visited_close_evaluation import VISITED_POLICIES

RETAINED_POLICIES = [*VISITED_POLICIES, 'retained_1m', 'retained_constant']
RETAINED_COMPARISONS = ['legacy_1m', 'legacy_5m', 'minute_1m', 'minute_5m', 'visited_1m', 'retained_constant']


def retained_weight_admission(metrics, probability, first, intervals):
    if (set(metrics) != set(RETAINED_POLICIES) or set(first) != set(RETAINED_POLICIES)
        or set(probability) != {'legacy_1m', 'minute_1m', 'training_constant', 'visited_1m', 'visited_constant', 'retained_1m', 'retained_constant'}
        or set(intervals) != set(RETAINED_COMPARISONS)
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or any(p['rows'] != metrics['retained_1m']['rows'] for p in probability.values())
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in RETAINED_POLICIES)):
        raise ValueError('방문 구간 원래 비중 판정의 동일 모집단·기존 대조 누락 오류')

    def finite(value):
        return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
            and bool(np.isfinite(value)))

    def positive(value):
        return finite(value) and value > 0

    def count(value, minimum):
        return finite(value) and isinstance(value, (int, np.integer)) and value >= minimum

    def improved(left, right, factor=1.):
        return finite(left) and finite(right) and left < factor*right

    candidate, earliest, p = metrics['retained_1m'], first['retained_1m'], probability['retained_1m']
    checks = {'at_least_100_selected': count(candidate['selected'], 100),
        'at_least_30_selected_positions': count(candidate['selected_positions'], 30),
        'positive_selected_weighted_mean': positive(candidate['selected_weighted_mean_bps']),
        'positive_selected_mean': positive(candidate['selected_mean_bps']),
        'positive_chosen_first_mean': positive(earliest['selected_mean_common_bps']),
        'positive_all_first_mean': positive(earliest['all_position_mean_common_bps']),
        'positive_first_interval_lower': positive(intervals['legacy_1m']['intervals']['retained_1m']['lower'])}
    for reference in ['legacy_1m', 'legacy_5m', 'minute_1m', 'visited_1m', 'retained_constant']:
        checks['positive_paired_interval_vs_'+reference] = positive(intervals[reference]['intervals']['paired_difference']['lower'])
    for reference in ['legacy_1m', 'minute_1m', 'visited_1m', 'retained_constant']:
        checks['weighted_regret_vs_'+reference] = improved(candidate['weighted_regret_bps'], metrics[reference]['weighted_regret_bps'])
        checks['cost_log_loss_vs_'+reference] = improved(p['cost_log_loss'], probability[reference]['cost_log_loss'], .99)
        left, right = p['cost_brier'], probability[reference]['cost_brier']
        checks['cost_brier_vs_'+reference] = finite(left) and finite(right) and left <= right+1e-12
    return {'checks': {key: bool(value) for key, value in checks.items()}, 'retained_weight_admitted': all(checks.values()),
        'candidate': 'retained_1m', 'trading_returns_evaluated': False}


def evaluate_retained_weight_close(frame, previous, scores, constant):
    if type(constant) not in (float, int) or not np.isfinite(constant) or not 0 < constant < 1:
        raise ValueError('방문 구간 원래 비중의 학습 상수 오류')
    np.testing.assert_allclose(frame.sample_weight, position_weights(frame), rtol=0, atol=1e-12)
    frame = frame.reset_index(drop=True)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    pd.testing.assert_frame_equal(previous['predictions'][keys], frame[keys], check_exact=True)
    for key in ['metrics', 'first_metrics', 'positions']:
        if set(previous[key]) != set(VISITED_POLICIES):
            raise ValueError('방문 구간 원래 비중 평가의 기존 정책 누락')
    predictions = previous['predictions'].copy()
    raw = {'retained_1m': cost_scores(scores, len(frame)), 'retained_constant': np.full(len(frame), constant)}
    predictions['retained_score'], predictions['retained_constant_score'] = raw['retained_1m'], raw['retained_constant']
    metrics, first, probability, positions = [dict(previous[key]) for key in ['metrics', 'first_metrics', 'probability_metrics', 'positions']]
    details, first_details, probability_details = [list(previous[key]) for key in ['breakdown', 'first_breakdown', 'probability_breakdown']]
    for name, values in raw.items():
        action = minute_grid_actions(frame, values, 60)
        predictions['selected_'+name] = action
        metrics[name] = policy_effect_metrics(frame, action)
        probability[name] = cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, values)
        part = first_cost_positions(frame, values)
        common = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity']
        pd.testing.assert_frame_equal(part[common], positions['legacy_5m'][common], check_exact=True)
        positions[name], first[name] = part, first_close_metrics(part)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                details.append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
                probability_details.append({'model': name, 'kind': kind, 'group': str(key),
                    **cost_probability_metrics(group.close_advantage_bps, group.sample_weight, values[group.index])})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                first_details.append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    blocks, replicates, intervals = {}, {}, {}
    common_draws = previous['draws']
    for reference in RETAINED_COMPARISONS:
        names = ['retained_1m', reference]
        block, draws, replica, interval = paired_week_blocks({name: positions[name] for name in names}, model_names=names)
        pd.testing.assert_frame_equal(draws, common_draws, check_exact=True)
        blocks[reference], replicates[reference], intervals[reference] = block, replica, interval
    decision = retained_weight_admission(metrics, probability, first, intervals)
    return {'predictions': predictions, 'metrics': metrics, 'first_metrics': first, 'probability_metrics': probability,
        'positions': positions, 'breakdown': details, 'first_breakdown': first_details, 'probability_breakdown': probability_details,
        'blocks': blocks, 'replicates': replicates, 'intervals': intervals, 'draws': common_draws, 'decision': decision}
