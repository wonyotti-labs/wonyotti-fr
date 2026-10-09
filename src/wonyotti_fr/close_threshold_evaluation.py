from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_threshold import THRESHOLDS, choose_threshold, threshold_policy
from .close_utility import cost_probability_metrics, cost_scores, policy_effect_metrics
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .retained_weight_close_evaluation import RETAINED_POLICIES

THRESHOLD_POLICIES = [*RETAINED_POLICIES, 'early_default', 'calibrated', 'always_first', 'never_extra']
THRESHOLD_COMPARISONS = ['early_default', 'minute_1m', 'retained_1m', 'always_first', 'never_extra']
THRESHOLD_PROBABILITIES = {'legacy_1m', 'minute_1m', 'training_constant', 'visited_1m',
    'visited_constant', 'retained_1m', 'retained_constant', 'early_score', 'early_constant'}


def threshold_admission(metrics, probability, first, intervals, selection):
    if (set(metrics) != set(THRESHOLD_POLICIES) or set(first) != set(THRESHOLD_POLICIES)
        or set(probability) != THRESHOLD_PROBABILITIES or set(intervals) != set(THRESHOLD_COMPARISONS)
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or any(p['rows'] != metrics['calibrated']['rows'] for p in probability.values())
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in THRESHOLD_POLICIES)):
        raise ValueError('문턱 보정 평가의 동일 모집단·기존 대조 누락 오류')

    def finite(value):
        return type(value) in (int, float) and bool(np.isfinite(value))

    def positive(value):
        return finite(value) and value > 0

    candidate, earliest = metrics['calibrated'], first['calibrated']
    checks = {'calibration_selection_passed': selection['selection_passed'] is True,
        'at_least_100_selected': type(candidate['selected']) is int and candidate['selected'] >= 100,
        'at_least_30_selected_positions': type(candidate['selected_positions']) is int and candidate['selected_positions'] >= 30,
        'positive_selected_weighted_mean': positive(candidate['selected_weighted_mean_bps']),
        'positive_selected_mean': positive(candidate['selected_mean_bps']),
        'positive_chosen_first_mean': positive(earliest['selected_mean_common_bps']),
        'positive_all_first_mean': positive(earliest['all_position_mean_common_bps']),
        'positive_first_interval_lower': positive(intervals['early_default']['intervals']['calibrated']['lower'])}
    for reference in THRESHOLD_COMPARISONS[:-1]:
        checks['positive_paired_interval_vs_'+reference] = positive(intervals[reference]['intervals']['paired_difference']['lower'])
    return {'checks': {key: bool(value) for key, value in checks.items()}, 'threshold_admitted': all(checks.values()),
        'candidate': 'calibrated', 'trading_returns_evaluated': False}


def evaluate_threshold_close(frame, previous, scores, constant, calibration):
    selection = choose_threshold(calibration['first_metrics'])
    if selection != calibration['selection'] or selection['selection_passed'] is not True:
        raise ValueError('적격 문턱 선택 없이 마지막 진단 실행 불가')
    threshold = selection['selected_threshold']
    if threshold not in THRESHOLDS or type(constant) not in (float, int) or not np.isfinite(constant) or not 0 < constant < 1:
        raise ValueError('문턱 보정의 선택·학습 상수 오류')
    frame = frame.reset_index(drop=True)
    np.testing.assert_array_equal(frame.sample_weight, position_weights(frame))
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    pd.testing.assert_frame_equal(previous['predictions'][keys], frame[keys], check_exact=True)
    for key in ['metrics', 'first_metrics', 'positions']:
        if set(previous[key]) != set(RETAINED_POLICIES):
            raise ValueError('문턱 보정 평가의 이전 아홉 정책 누락')
    score = cost_scores(scores, len(frame))
    predictions = previous['predictions'].assign(early_score=score, early_constant_score=constant)
    metrics, first, probability, positions = [dict(previous[key]) for key in ['metrics', 'first_metrics', 'probability_metrics', 'positions']]
    details, first_details, probability_details = [list(previous[key]) for key in ['breakdown', 'first_breakdown', 'probability_breakdown']]
    policies = [('early_default', score, .5), ('calibrated', score, threshold),
        ('always_first', np.ones(len(frame)), .5), ('never_extra', np.zeros(len(frame)), .5)]
    for name, values, cutoff in policies:
        action, part = threshold_policy(frame, values, cutoff)
        predictions['selected_'+name] = action
        metrics[name], positions[name], first[name] = policy_effect_metrics(frame, action), part, first_close_metrics(part)
        common = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity']
        pd.testing.assert_frame_equal(part[common], positions['legacy_5m'][common], check_exact=True)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                details.append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                first_details.append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    for name, values in [('early_score', score), ('early_constant', np.full(len(frame), constant))]:
        probability[name] = cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, values)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                probability_details.append({'model': name, 'kind': kind, 'group': str(key),
                    **cost_probability_metrics(group.close_advantage_bps, group.sample_weight, values[group.index])})
    blocks, replicates, intervals = {}, {}, {}
    for reference in THRESHOLD_COMPARISONS:
        names = ['calibrated', reference]
        block, draws, replica, interval = paired_week_blocks({name: positions[name] for name in names}, model_names=names)
        pd.testing.assert_frame_equal(draws, previous['draws'], check_exact=True)
        blocks[reference], replicates[reference], intervals[reference] = block, replica, interval
    np.testing.assert_array_equal(replicates['never_extra'].calibrated, replicates['never_extra'].paired_difference)
    decision = threshold_admission(metrics, probability, first, intervals, selection)
    return {'predictions': predictions, 'metrics': metrics, 'first_metrics': first, 'probability_metrics': probability,
        'positions': positions, 'breakdown': details, 'first_breakdown': first_details, 'probability_breakdown': probability_details,
        'blocks': blocks, 'replicates': replicates, 'intervals': intervals, 'draws': previous['draws'], 'decision': decision}
