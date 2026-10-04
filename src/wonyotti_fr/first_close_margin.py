from __future__ import annotations

import numpy as np

from .first_close_diagnostics import first_close_metrics, first_close_positions

FIRST_MARGINS = [0., .5, 1., 2., 4., 8.]


def select_first_margin(frame, prediction='predicted_half'):
    positions, metrics = {}, []
    for number, margin in enumerate(FIRST_MARGINS):
        key = f'candidate-{number:02}'
        positions[key] = first_close_positions(frame, prediction, margin_bps=margin)
        value = first_close_metrics(positions[key])
        metrics.append({'candidate': key, 'margin_bps': margin, **value,
            'eligible': value['selected_positions'] >= 30 and value['all_position_mean_common_bps'] > 0})
    eligible = [m for m in metrics if m['eligible']]
    best = max(eligible, key=lambda m: (m['all_position_mean_common_bps'], m['margin_bps'])) if eligible else None
    improved = best is not None and best['all_position_mean_common_bps'] > metrics[0]['all_position_mean_common_bps']
    reason = 'no_supported_positive_candidate' if best is None else ('no_first_effect_improvement' if not improved else None)
    return {'selection_passed': improved, 'chosen_margin_bps': best['margin_bps'] if improved else None,
        'chosen_candidate': best['candidate'] if improved else None,
        'best_eligible_margin_bps': best['margin_bps'] if best is not None else None,
        'reason': reason, 'criterion': 'maximum_all_position_first_mean', 'tie_break': 'higher_margin',
        'final_diagnosis_used_for_selection': False}, metrics, positions


def margin_admission(selection, metrics=None, first=None, intervals=None):
    if selection['selection_passed'] is not True:
        return {'checks': {'selection_supported': False}, 'margin_admitted': False,
            'final_diagnosis_evaluated': False, 'trading_returns_evaluated': False}
    if (selection['chosen_margin_bps'] not in FIRST_MARGINS or selection['chosen_margin_bps'] == 0
        or any(v is None for v in [metrics, first, intervals])):
        raise ValueError('최초 문턱 보정의 선택·마지막 진단 연결 오류')
    candidate, zero = metrics['margin'], metrics['half_zero']
    if (len({(m['rows'], m['positions']) for m in metrics.values()}) != 1
        or any(candidate[k] != zero[k] for k in ['weighted_mse', 'mse', 'mean_predicted_bps', 'mean_actual_bps'])
        or any(m['positions'] != candidate['positions'] for m in first.values())
        or first['margin']['selected_positions'] != candidate['selected_positions']):
        raise ValueError('최초 문턱 보정의 고정 점수·행·포지션 불일치')
    low = intervals['intervals']['margin']['lower']
    paired = intervals['intervals']['paired_difference']['lower']
    mean = first['margin']['all_position_mean_common_bps']
    checks = {'selection_supported': True, 'at_least_100_selected': candidate['selected'] >= 100,
        'at_least_30_selected_positions': candidate['selected_positions'] >= 30,
        'positive_selected_weighted_mean': candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        'positive_selected_mean': candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        'positive_first_choice_mean': mean > 0,
        'first_mean_vs_half_zero': mean > first['half_zero']['all_position_mean_common_bps'],
        'first_mean_vs_continuation': mean > first['continuation']['all_position_mean_common_bps'],
        'positive_first_interval_lower': bool(low is not None and np.isfinite(low) and low > 0),
        'positive_paired_interval_lower': bool(paired is not None and np.isfinite(paired) and paired > 0)}
    return {'checks': checks, 'margin_admitted': all(checks.values()),
        'final_diagnosis_evaluated': True, 'trading_returns_evaluated': False}
