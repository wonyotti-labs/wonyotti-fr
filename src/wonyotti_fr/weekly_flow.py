from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_flow import FlowCloseModel, flow_close_admission
from .common import save_json
from .weekly_close import WEEKLY_SETTINGS, weekly_boundaries, weekly_training_rows


def fit_flow_week(ledger, fit_time, validation_values):
    training, indices = weekly_training_rows(ledger, fit_time)
    weights = position_weights(training)
    values = training[FlowCloseModel.features].to_numpy(dtype=float)
    validation = np.asarray(validation_values, dtype=float)
    if validation.ndim != 2 or validation.shape[1] != len(FlowCloseModel.features):
        raise ValueError('주간 비용 학습의 예측 입력 차원 오류')
    model, support, costs = FlowCloseModel.fit(values, training.close_advantage_bps, weights,
        validation if len(validation) else values[:1])
    return training, indices, weights, model, support, costs


def fit_weekly_flow(ledger, diagnosis, original_training, original_weights, original_model,
                    original_costs, original_support, output):
    if (diagnosis.empty or diagnosis.decision_time.isna().any() or diagnosis.decision_time.duplicated().any()
        or not diagnosis.decision_time.is_monotonic_increasing):
        raise ValueError('주간 비용 예측의 진단 시각 오류')
    score, constant = np.full(len(diagnosis), np.nan), np.full(len(diagnosis), np.nan)
    models, support, memberships, routing = {}, {}, [], []
    for number, (start, end) in enumerate(weekly_boundaries()):
        key = f'week-{number:02}'
        selected = diagnosis.decision_time.ge(start) & diagnosis.decision_time.lt(end)
        week = diagnosis.loc[selected]
        values = week[FlowCloseModel.features].to_numpy(dtype=float)
        training, indices, weights, model, details, costs = fit_flow_week(ledger, start, values)
        if set(training.position_entry_time) & set(week.position_entry_time):
            raise ValueError('주간 비용 학습과 해당 주 예측의 포지션 교차')
        if number == 0:
            pd.testing.assert_frame_equal(training, original_training, check_exact=True)
            pd.testing.assert_frame_equal(original_weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
            np.testing.assert_array_equal(weights, original_weights.sample_weight)
            expected = pd.concat([training[['decision_time', 'position_entry_time']].reset_index(drop=True), costs], axis=1)
            pd.testing.assert_frame_equal(expected, original_costs, check_exact=True)
            if model.to_dict() != original_model:
                raise ValueError('주간 비용 학습의 첫 모델이 기존 모델과 불일치')
            if {k: v for k, v in details.items() if k != 'export'} != {k: v for k, v in original_support.items() if k != 'export'}:
                raise ValueError('주간 비용 학습의 첫 비용 정규화·상수 불일치')
        if len(week):
            score[selected.to_numpy()] = model.probabilities(values)[:, 0]
            constant[selected.to_numpy()] = details['training_constant_score']
        models[key] = model.to_dict()
        support[key] = {'fit_time': start, 'prediction_end': end,
            'label_cutoff': start-pd.Timedelta(days=WEEKLY_SETTINGS['label_gap_days']),
            'positions': int(training.position_entry_time.nunique()), 'last_label_end': training.label_end.max(),
            'prediction_rows': len(week), **details}
        membership = training[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_bps']].reset_index(drop=True)
        membership['opportunity_index'], membership['sample_weight'], membership['model_key'] = indices, weights, key
        memberships.append(pd.concat([membership, costs], axis=1))
        routing.append(week[['decision_time', 'position_entry_time']].assign(model_key=key))
    if not np.isfinite(score).all() or not np.isfinite(constant).all():
        raise ValueError('주간 비용 예측의 누락·비유한 값')
    pd.concat(memberships, ignore_index=True).to_parquet(output/'weekly_training_membership.parquet', index=False)
    pd.concat(routing, ignore_index=True).to_parquet(output/'prediction_routing.parquet', index=False)
    save_json(output/'weekly_models.json', models)
    save_json(output/'weekly_support.json', support)
    return score, constant


def weekly_flow_admission(metrics, probability, first, intervals, utility_intervals, context_intervals, flow_intervals, *, candidate_name='weekly_flow'):
    base = flow_close_admission(metrics, probability, first, intervals, utility_intervals, context_intervals,
        candidate_name=candidate_name)
    checks = dict(base['checks'])
    candidate, previous, constant = probability[candidate_name], probability['flow'], probability['weekly_constant']
    low = flow_intervals['intervals']['paired_difference']['lower']
    checks.update(cost_log_loss_vs_flow=candidate['cost_log_loss'] is not None and previous['cost_log_loss'] is not None and candidate['cost_log_loss'] < previous['cost_log_loss']*.99,
        cost_brier_vs_flow=candidate['cost_brier'] is not None and previous['cost_brier'] is not None and candidate['cost_brier'] <= previous['cost_brier']+1e-12,
        weighted_regret_vs_flow=metrics[candidate_name]['weighted_regret_bps'] < metrics['flow']['weighted_regret_bps'],
        first_mean_vs_flow=first[candidate_name]['all_position_mean_common_bps'] > first['flow']['all_position_mean_common_bps'],
        positive_flow_paired_interval_lower=bool(low is not None and np.isfinite(low) and low > 0),
        cost_log_loss_vs_weekly_constant=candidate['cost_log_loss'] is not None and constant['cost_log_loss'] is not None and candidate['cost_log_loss'] < constant['cost_log_loss']*.99,
        cost_brier_vs_weekly_constant=candidate['cost_brier'] is not None and constant['cost_brier'] is not None and candidate['cost_brier'] <= constant['cost_brier']+1e-12,
        weighted_regret_vs_weekly_constant=metrics[candidate_name]['weighted_regret_bps'] < metrics['weekly_constant']['weighted_regret_bps'])
    return {'checks': checks, 'weekly_flow_admitted': all(checks.values()), 'trading_returns_evaluated': False}
