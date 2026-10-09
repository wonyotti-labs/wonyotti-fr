from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .common import save_json
from .minute_close_learning import MinuteCloseModel
from .visited_close import VISITATION_SETTINGS, visitation_rows
from .weekly_close import weekly_training_rows

RETAINED_WEIGHT_SETTINGS = {'visitation_prefix': 'unchanged_v76',
    'weighting': 'original_full_position_weights_restricted_and_global_mean_one',
    'position_rebalance_after_restriction': False, 'teacher_models_updated': False,
    'training_target': 'original_natural_close_advantage', 'score_threshold': .5}


class RetainedWeightCloseModel(MinuteCloseModel):
    format = 'retained_original_weight_minute_cost_weighted_close_v1'


def retained_training_rows(training, original_weights, visitation):
    pd.testing.assert_frame_equal(original_weights.drop(columns='sample_weight'),
        training[['decision_time', 'position_entry_time']], check_exact=True)
    np.testing.assert_array_equal(original_weights.sample_weight, position_weights(training))
    expected = visitation_rows(training, visitation.teacher_score, visitation.opportunity_index)
    pd.testing.assert_frame_equal(expected, visitation, check_exact=True)
    selected = visitation.visited_prefix.to_numpy()
    retained = training.loc[selected].reset_index(drop=True)
    checked, _ = weekly_training_rows(retained, pd.Timestamp(VISITATION_SETTINGS['fit_time']))
    pd.testing.assert_frame_equal(checked, retained, check_exact=True)
    weight = original_weights.loc[selected, 'sample_weight'].to_numpy(dtype=float)
    mean = float(weight.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('방문 구간 원래 행 비중의 평균 오류')
    # 유지한 모든 행에 같은 상수를 적용해 원래 상대 비중을 보존한다.
    weight = weight/mean
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError('방문 구간 원래 행 비중의 정규화 오류')
    weights = retained[['decision_time', 'position_entry_time']].assign(sample_weight=weight)
    return retained, weights, selected, mean


def fit_retained_weight_close(training, original_weights, visitation, diagnosis, output):
    retained, weights, selected, mean = retained_training_rows(training, original_weights, visitation)
    model, support, costs = RetainedWeightCloseModel.fit(
        retained[RetainedWeightCloseModel.features].to_numpy(dtype=float), retained.close_advantage_bps,
        weights.sample_weight, diagnosis[RetainedWeightCloseModel.features].to_numpy(dtype=float))
    retained.to_parquet(output/'retained_training_used.parquet', index=False)
    weights.to_parquet(output/'retained_training_weights.parquet', index=False)
    pd.concat([retained[['decision_time', 'position_entry_time']], costs], axis=1).to_parquet(
        output/'retained_training_cost_ledger.parquet', index=False)
    audit = visitation.copy()
    audit['retained_original_weight'], audit['fit_used'], audit['cost_weight'], audit['fit_weight'] = 0., False, 0., 0.
    audit.loc[selected, 'retained_original_weight'] = weights.sample_weight.to_numpy()
    for name in ['fit_used', 'cost_weight', 'fit_weight']:
        audit.loc[selected, name] = costs[name].to_numpy()
    audit.loc[selected, 'reason'] = costs.reason.to_numpy()
    audit.to_parquet(output/'full_training_contribution.parquet', index=False)
    support = {**support, 'original_rows': len(training), 'retained_rows': len(retained),
        'after_first_teacher_close_rows': int((~selected).sum()), 'original_prefix_weight_mean': mean,
        'original_positions': int(training.position_entry_time.nunique()),
        'retained_positions': int(retained.position_entry_time.nunique())}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
