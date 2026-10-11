from __future__ import annotations

import numpy as np


def position_boost_admission(metrics, monthly):
    if set(metrics) != {'position', 'constant', 'boosted'}:
        raise ValueError('보유 방향 부스팅의 비교 모형 누락')
    if [row['month'] for row in monthly] != [f'2021-{month:02}' for month in range(1, 13)]:
        raise ValueError('보유 방향 부스팅의 월별 순서·범위 오류')
    losses = [value['log_loss'] for value in metrics.values()]
    precisions = [value[side]['average_precision'] for value in metrics.values() for side in ['short', 'long']]
    month_losses = [row[name + '_log_loss'] for row in monthly for name in metrics]
    if (not np.isfinite(losses + precisions + month_losses).all()
        or min(losses + month_losses) < 0 or not all(0 <= value <= 1 for value in precisions)):
        raise ValueError('보유 방향 부스팅의 지표 범위 오류')
    candidate = metrics['boosted']
    comparisons = {}
    for name in ['position', 'constant']:
        baseline = metrics[name]
        gain = 1 - candidate['log_loss'] / baseline['log_loss'] if baseline['log_loss'] > 0 else 0.
        comparisons[name] = {'relative_log_loss_gain': gain,
            'gain_passed': bool(gain > .01 and not np.isclose(gain, .01, atol=1e-12, rtol=0)),
            'short_precision_preserved': candidate['short']['average_precision'] >= baseline['short']['average_precision'],
            'long_precision_preserved': candidate['long']['average_precision'] >= baseline['long']['average_precision']}
    improved = sum(row['boosted_log_loss'] < row['constant_log_loss'] for row in monthly)
    passed = improved >= 8 and all(all(value[key] for key in
        ['gain_passed', 'short_precision_preserved', 'long_precision_preserved']) for value in comparisons.values())
    return {'position_boost_admitted': bool(passed), 'comparisons': comparisons,
        'improved_months': improved, 'required_improved_months': 8, 'monthly_consistency': improved >= 8,
        'profitability_accepted': False, 'trading_returns_evaluated': False,
        'all_source_periods_already_observed': True}
