from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_flow import FlowCloseModel
from .close_utility import cost_probability_metrics, cost_scores, policy_effect_metrics
from .first_close_diagnostics import first_close_metrics, first_close_positions, paired_week_blocks

MINUTE_POLICIES = ['legacy_5m', 'legacy_1m', 'minute_5m', 'minute_1m', 'training_constant']
MINUTE_COMPARISONS = {'legacy_1m': ['minute_1m', 'legacy_1m'],
    'legacy_5m': ['minute_1m', 'legacy_5m'], 'minute_5m': ['minute_1m', 'minute_5m'],
    'legacy_frequency': ['legacy_1m', 'legacy_5m']}


class MinuteCloseModel(FlowCloseModel):
    format = 'minute_cost_weighted_close_v1'


def minute_grid_actions(frame, scores, seconds):
    if (frame.empty or type(seconds) is not int or seconds not in (60, 300)
        or not isinstance(frame.decision_time.dtype, pd.DatetimeTZDtype)
        or str(frame.decision_time.dt.tz) != 'UTC' or frame.decision_time.isna().any()
        or frame.decision_time.duplicated().any() or not frame.decision_time.is_monotonic_increasing
        or not frame.original_intent.isin(['hold', 'exit', 'reduce', 'increase']).all()):
        raise ValueError('분별 청산 정책의 간격·시각·의도 오류')
    times = frame.decision_time.astype('datetime64[ns, UTC]').array.asi8
    if (times % (60*10**9)).any():
        raise ValueError('분별 청산 정책의 분 경계 오류')
    return (cost_scores(scores, len(frame)) > .5) & (times % (seconds*10**9) == 0) & frame.original_intent.ne('exit').to_numpy()


def minute_close_admission(metrics, probability, first, intervals):
    names = set(MINUTE_POLICIES)
    if (set(metrics) != names or set(first) != names
        or set(probability) != {'legacy_1m', 'minute_1m', 'training_constant'}
        or set(intervals) != set(MINUTE_COMPARISONS)
        or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or any(p['rows'] != metrics['minute_1m']['rows'] for p in probability.values())
        or any(first[k]['positions'] != metrics[k]['positions']
            or first[k]['selected_positions'] != metrics[k]['selected_positions'] for k in names)):
        raise ValueError('분별 청산 판정의 동일 모집단·대조 누락 오류')

    def finite(value):
        return (value is not None and not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
            and isinstance(value, (int, float, np.number)) and bool(np.isfinite(value)))

    def positive(value):
        return finite(value) and value > 0

    def improved(left, right, factor=1.):
        return finite(left) and finite(right) and left < right*factor

    candidate, earliest, p = metrics['minute_1m'], first['minute_1m'], probability['minute_1m']
    checks = {'at_least_100_selected': candidate['selected'] >= 100,
        'at_least_30_selected_positions': candidate['selected_positions'] >= 30,
        'positive_selected_weighted_mean': positive(candidate['selected_weighted_mean_bps']),
        'positive_selected_mean': positive(candidate['selected_mean_bps']),
        'positive_chosen_first_mean': positive(earliest['selected_mean_common_bps']),
        'positive_all_first_mean': positive(earliest['all_position_mean_common_bps']),
        'positive_first_interval_lower': positive(intervals['legacy_1m']['intervals']['minute_1m']['lower'])}
    for reference in ['legacy_1m', 'legacy_5m']:
        checks['positive_paired_interval_vs_'+reference] = positive(intervals[reference]['intervals']['paired_difference']['lower'])
    for reference in ['legacy_1m', 'training_constant']:
        checks['weighted_regret_vs_'+reference] = improved(candidate['weighted_regret_bps'], metrics[reference]['weighted_regret_bps'])
        checks['cost_log_loss_vs_'+reference] = improved(p['cost_log_loss'], probability[reference]['cost_log_loss'], .99)
        left, right = p['cost_brier'], probability[reference]['cost_brier']
        checks['cost_brier_vs_'+reference] = finite(left) and finite(right) and left <= right+1e-12
    return {'checks': {k: bool(v) for k, v in checks.items()}, 'minute_close_admitted': all(checks.values()),
        'candidate': 'minute_1m', 'trading_returns_evaluated': False}


def evaluate_minute_close(frame, legacy_scores, minute_scores, constant):
    if type(constant) not in (float, int) or not np.isfinite(constant) or not 0 < constant < 1:
        raise ValueError('분별 청산의 학습 상수 오류')
    np.testing.assert_allclose(frame.sample_weight, position_weights(frame), rtol=0, atol=1e-12)
    raw = {'legacy': cost_scores(legacy_scores, len(frame)), 'minute': cost_scores(minute_scores, len(frame)),
        'training_constant': np.full(len(frame), constant)}
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    predictions = frame[keys].copy()
    for name, values in raw.items():
        predictions[name+'_score'] = values
    metrics, first, positions, details, first_details = {}, {}, {}, [], []
    for name in MINUTE_POLICIES:
        family = name.split('_')[0] if name != 'training_constant' else name
        scores = raw[family]
        actions = minute_grid_actions(frame, scores, 300 if name.endswith('_5m') else 60)
        predictions['selected_'+name] = actions
        metrics[name] = policy_effect_metrics(frame, actions)
        # 모든 정책의 기준 순자산은 같은 분별 원장의 첫 기회다.
        part = first_close_positions(frame.assign(_selected=actions.astype(float)), '_selected').drop(columns='first_prediction_bps')
        part['first_cost_score'] = part.first_selected_time.map(pd.Series(scores, index=frame.decision_time))
        positions[name], first[name] = part, first_close_metrics(part)
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                selected = predictions.set_index('decision_time').loc[group.decision_time, 'selected_'+name].to_numpy()
                details.append({'policy': name, 'kind': kind, 'group': str(key), **policy_effect_metrics(group, selected)})
        for kind, groups in [('direction', part.groupby('direction')),
            ('entry_month', part.groupby(part.position_entry_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                first_details.append({'policy': name, 'kind': kind, 'group': str(key), **first_close_metrics(group)})
    probability, probability_details = {}, []
    for name, family in [('legacy_1m', 'legacy'), ('minute_1m', 'minute'), ('training_constant', 'training_constant')]:
        probability[name] = cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, raw[family])
        for kind, groups in [('direction', frame.groupby('direction')),
            ('decision_month', frame.groupby(frame.decision_time.dt.strftime('%Y-%m')))]:
            for key, group in groups:
                scores = pd.Series(raw[family], index=frame.decision_time).loc[group.decision_time]
                probability_details.append({'model': name, 'kind': kind, 'group': str(key),
                    **cost_probability_metrics(group.close_advantage_bps, group.sample_weight, scores)})
    intervals, blocks, replicates, common_draws = {}, {}, {}, None
    for key, names in MINUTE_COMPARISONS.items():
        block, draws, replica, interval = paired_week_blocks({name: positions[name] for name in names}, model_names=names)
        if common_draws is None:
            common_draws = draws
        else:
            pd.testing.assert_frame_equal(draws, common_draws, check_exact=True)
        intervals[key], blocks[key], replicates[key] = interval, block, replica
    decision = minute_close_admission(metrics, probability, first, intervals)
    return {'predictions': predictions, 'positions': positions, 'metrics': metrics, 'first_metrics': first,
        'probability_metrics': probability, 'breakdown': details, 'first_breakdown': first_details,
        'probability_breakdown': probability_details, 'intervals': intervals, 'blocks': blocks,
        'replicates': replicates, 'draws': common_draws, 'decision': decision}
