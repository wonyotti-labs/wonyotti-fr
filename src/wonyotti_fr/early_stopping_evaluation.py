from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_threshold import threshold_policy
from .close_utility import cost_probability_metrics, cost_scores, policy_effect_metrics
from .first_close_diagnostics import first_close_metrics, paired_week_blocks
from .retained_weight_close_evaluation import RETAINED_POLICIES

EARLY_STOPPING_NEW_POLICIES = ['early_natural', 'early_stopping', 'always_first', 'never_extra', 'stopping_constant']
EARLY_STOPPING_POLICIES = [*RETAINED_POLICIES, *EARLY_STOPPING_NEW_POLICIES]
EARLY_STOPPING_COMPARISONS = ['early_natural', 'minute_1m', 'retained_1m', 'always_first', 'never_extra', 'stopping_constant']
EARLY_STOPPING_PROBABILITIES = {'legacy_1m', 'minute_1m', 'training_constant', 'visited_1m',
    'visited_constant', 'retained_1m', 'retained_constant', 'early_natural', 'early_stopping', 'stopping_constant'}


def positive(value):
    return type(value) in (int, float) and bool(np.isfinite(value)) and value > 0


def calibration_decision(metrics, first):
    if (set(metrics) != set(EARLY_STOPPING_NEW_POLICIES) or set(first) != set(EARLY_STOPPING_NEW_POLICIES)
        or len({(item['rows'], item['positions']) for item in metrics.values()}) != 1
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in EARLY_STOPPING_NEW_POLICIES)):
        raise ValueError('분별 이후 청산 보정의 정책·동일 모집단 오류')
    item = first['early_stopping']
    effect = item['all_position_mean_common_bps']
    checks = {'at_least_30_selected_positions': type(item['selected_positions']) is int and item['selected_positions'] >= 30,
        'positive_all_first_mean': positive(effect)}
    for name in ['early_natural', 'always_first']:
        other = first[name]['all_position_mean_common_bps']
        checks['better_than_'+name] = (type(effect) in (int, float) and type(other) in (int, float)
            and np.isfinite(effect) and np.isfinite(other) and effect > other)
    checks = {key: bool(value) for key, value in checks.items()}
    return {'checks': checks, 'calibration_passed': all(checks.values()), 'threshold': .5,
        'threshold_search': False, 'fallback_used': False, 'reasons': [key for key, value in checks.items() if not value]}


def evaluate_new_policies(frame, scores, natural_scores, constant):
    if type(constant) not in (float, int) or not np.isfinite(constant) or not 0 < constant < 1:
        raise ValueError('분별 이후 청산의 학습 상수 오류')
    frame = frame.reset_index(drop=True)
    np.testing.assert_array_equal(frame.sample_weight, position_weights(frame))
    raw = {'early_natural': cost_scores(natural_scores, len(frame)), 'early_stopping': cost_scores(scores, len(frame)),
        'stopping_constant': np.full(len(frame), constant)}
    values = {**raw, 'always_first': np.ones(len(frame)), 'never_extra': np.zeros(len(frame))}
    predictions = frame[['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent',
        'close_advantage_bps', 'sample_weight']].copy()
    for name, score in raw.items():
        predictions[name+'_score'] = score
    metrics, first, probability, positions, details, first_details, probability_details = {}, {}, {}, {}, [], [], []
    for name in EARLY_STOPPING_NEW_POLICIES:
        action, part = threshold_policy(frame, values[name], .5)
        predictions['selected_'+name] = action
        metrics[name], first[name], positions[name] = policy_effect_metrics(frame, action), first_close_metrics(part), part
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                details.append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, action[group.index])})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                first_details.append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    for name, score in raw.items():
        probability[name] = cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, score)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                probability_details.append({'model': name, 'kind': kind, 'group': str(key),
                    **cost_probability_metrics(group.close_advantage_bps, group.sample_weight, score[group.index])})
    # 상수의 행동은 점수에 따라 항상 청산 또는 추가 청산 없음과 같아야 한다.
    counterpart = 'always_first' if constant > .5 else 'never_extra'
    pd.testing.assert_frame_equal(positions['stopping_constant'].drop(columns='first_cost_score'),
        positions[counterpart].drop(columns='first_cost_score'), check_exact=True)
    return {'predictions': predictions, 'metrics': metrics, 'first_metrics': first, 'probability_metrics': probability,
        'positions': positions, 'breakdown': details, 'first_breakdown': first_details, 'probability_breakdown': probability_details,
        'calibration_decision': calibration_decision(metrics, first)}


def early_stopping_admission(metrics, probability, first, intervals, calibration):
    if (set(metrics) != set(EARLY_STOPPING_POLICIES) or set(first) != set(EARLY_STOPPING_POLICIES)
        or set(probability) != EARLY_STOPPING_PROBABILITIES or set(intervals) != set(EARLY_STOPPING_COMPARISONS)
        or len({(item['rows'], item['positions']) for item in metrics.values()}) != 1
        or any(item['rows'] != metrics['early_stopping']['rows'] for item in probability.values())
        or any(first[name]['positions'] != metrics[name]['positions']
            or first[name]['selected_positions'] != metrics[name]['selected_positions'] for name in EARLY_STOPPING_POLICIES)):
        raise ValueError('분별 이후 청산 판정의 동일 모집단·이전 대조 누락 오류')
    candidate, earliest = metrics['early_stopping'], first['early_stopping']
    checks = {'calibration_passed': calibration['calibration_passed'] is True,
        'at_least_100_selected': type(candidate['selected']) is int and candidate['selected'] >= 100,
        'at_least_30_selected_positions': type(candidate['selected_positions']) is int and candidate['selected_positions'] >= 30,
        'positive_selected_weighted_mean': positive(candidate['selected_weighted_mean_bps']),
        'positive_selected_mean': positive(candidate['selected_mean_bps']),
        'positive_chosen_first_mean': positive(earliest['selected_mean_common_bps']),
        'positive_all_first_mean': positive(earliest['all_position_mean_common_bps']),
        'positive_first_interval_lower': positive(intervals['early_natural']['intervals']['early_stopping']['lower'])}
    for reference in EARLY_STOPPING_COMPARISONS[:4]:
        checks['positive_paired_interval_vs_'+reference] = positive(intervals[reference]['intervals']['paired_difference']['lower'])
    return {'checks': {key: bool(value) for key, value in checks.items()}, 'early_stopping_admitted': all(checks.values()),
        'candidate': 'early_stopping', 'trading_returns_evaluated': False}


def evaluate_early_stopping_close(frame, previous, scores, natural_scores, constant, calibration):
    decision = calibration_decision(calibration['metrics'], calibration['first_metrics'])
    if decision != calibration['calibration_decision'] or decision['calibration_passed'] is not True:
        raise ValueError('분별 이후 청산의 내부 보정 실패 뒤 진단 실행 불가')
    result = evaluate_new_policies(frame, scores, natural_scores, constant)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    pd.testing.assert_frame_equal(previous['predictions'][keys], result['predictions'][keys], check_exact=True)
    for key in ['metrics', 'first_metrics', 'positions']:
        if set(previous[key]) != set(RETAINED_POLICIES):
            raise ValueError('분별 이후 청산의 이전 아홉 정책 누락')
    for part in result['positions'].values():
        common = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity']
        pd.testing.assert_frame_equal(part[common], previous['positions']['legacy_5m'][common], check_exact=True)
    prediction = previous['predictions'].copy()
    for name in result['predictions'].columns.difference(keys):
        prediction[name] = result['predictions'][name]
    result['predictions'] = prediction
    for key in ['metrics', 'first_metrics', 'probability_metrics', 'positions']:
        result[key] = {**previous[key], **result[key]}
    for key in ['breakdown', 'first_breakdown', 'probability_breakdown']:
        result[key] = [*previous[key], *result[key]]
    blocks, replicates, intervals = {}, {}, {}
    for reference in EARLY_STOPPING_COMPARISONS:
        names = ['early_stopping', reference]
        block, draws, replica, interval = paired_week_blocks({name: result['positions'][name] for name in names}, model_names=names)
        pd.testing.assert_frame_equal(draws, previous['draws'], check_exact=True)
        blocks[reference], replicates[reference], intervals[reference] = block, replica, interval
    np.testing.assert_array_equal(replicates['never_extra'].early_stopping, replicates['never_extra'].paired_difference)
    result.update(blocks=blocks, replicates=replicates, intervals=intervals, draws=previous['draws'])
    result['decision'] = early_stopping_admission(result['metrics'], result['probability_metrics'], result['first_metrics'], intervals, decision)
    # 마지막 구간의 통계로 보정 판정을 다시 만들지 않는다.
    result['calibration_decision'] = decision
    return result
