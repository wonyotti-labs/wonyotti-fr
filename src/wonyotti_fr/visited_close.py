from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_utility import cost_scores
from .common import save_json
from .minute_close_learning import MinuteCloseModel, minute_grid_actions
from .stopping_close import checked_opportunity_indices, position_folds, validate_stopping_rows
from .weekly_close import weekly_training_rows

VISITATION_SETTINGS = {'folds': 5, 'fold_rule': 'sha256_utc_entry_nanoseconds_decimal_big_endian_mod_5',
    'fit_time': '2021-10-02T00:00:00+00:00', 'label_cutoff': '2021-09-30T00:00:00+00:00',
    'policy_iterations': 1, 'score_threshold': .5, 'decision_seconds': 60,
    'prefix': 'through_first_teacher_close_inclusive', 'fallback': 'all_original_rows',
    'target': 'original_natural_close_advantage', 'weighting': 'equal_position_within_visited_prefix',
    'teacher_features_added': False}


class VisitedCloseModel(MinuteCloseModel):
    format = 'visited_minute_cost_weighted_close_v1'


def visitation_rows(training, scores, opportunity_indices):
    validate_stopping_rows(training)
    frame = training.reset_index(drop=True)
    score = cost_scores(scores, len(frame))
    indices = checked_opportunity_indices(opportunity_indices, len(frame))
    actions = minute_grid_actions(frame, score, 60)
    visited = np.zeros(len(frame), dtype=bool)
    first_indices = np.full(len(frame), -1, dtype=np.int64)
    first_times = pd.Series(pd.NaT, index=frame.index, dtype=frame.decision_time.dtype)
    for group in frame.groupby('position_entry_time', sort=False).indices.values():
        selected = group[actions[group]]
        if len(selected):
            first = selected[0]
            # 청산을 선택한 현재 행까지 학습하고 이후 원래 경로를 구분한다.
            visited[group[group <= first]] = True
            first_indices[group] = indices[first]
            first_times.iloc[group] = frame.decision_time.iloc[first]
        else:
            visited[group] = True
    retained = frame.loc[visited].reset_index(drop=True)
    if set(retained.position_entry_time) != set(frame.position_entry_time):
        raise ValueError('방문 청산 학습의 포지션 누락')
    new_weights = np.zeros(len(frame), dtype=float)
    new_weights[visited] = position_weights(retained)
    result = frame[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_bps']].copy()
    result['opportunity_index'], result['teacher_fold'], result['teacher_score'] = indices, position_folds(frame), score
    result['teacher_selected'], result['first_teacher_opportunity_index'] = actions, first_indices
    result['first_teacher_close_time'], result['visited_prefix'] = first_times, visited
    result['original_position_weight'], result['visited_position_weight'] = position_weights(frame), new_weights
    result['reason'] = np.where(visited, 'visited_prefix', 'after_first_teacher_close')
    return result


def fit_visitation_teachers(training, opportunity_indices, output):
    validate_stopping_rows(training)
    indices = checked_opportunity_indices(opportunity_indices, len(training))
    checked, _ = weekly_training_rows(training, pd.Timestamp(VISITATION_SETTINGS['fit_time']))
    pd.testing.assert_frame_equal(checked, training, check_exact=True)
    minute_grid_actions(training, np.zeros(len(training)), 60)
    folds = position_folds(training)
    scores = np.full(len(training), np.nan)
    models, supports, members = {}, {}, []
    for fold in range(5):
        key, selected = f'fold-{fold:02}', folds == fold
        heldout, rows = training.loc[selected], training.loc[~selected].reset_index(drop=True)
        if heldout.empty or set(rows.position_entry_time) & set(heldout.position_entry_time):
            raise ValueError('방문 청산의 보조 학습·제외 포지션 오류')
        checked, _ = weekly_training_rows(rows, pd.Timestamp(VISITATION_SETTINGS['fit_time']))
        pd.testing.assert_frame_equal(checked, rows, check_exact=True)
        weights = position_weights(rows)
        model, support, costs = MinuteCloseModel.fit(rows[MinuteCloseModel.features].to_numpy(dtype=float),
            rows.close_advantage_bps, weights, heldout[MinuteCloseModel.features].to_numpy(dtype=float))
        scores[selected] = model.probabilities(heldout[MinuteCloseModel.features].to_numpy(dtype=float))[:, 0]
        models[key] = model.to_dict()
        supports[key] = {'fit_time': VISITATION_SETTINGS['fit_time'], 'label_cutoff': VISITATION_SETTINGS['label_cutoff'],
            'positions': int(rows.position_entry_time.nunique()), 'last_label_end': rows.label_end.max(),
            'heldout_positions': int(heldout.position_entry_time.nunique()), 'prediction_rows': len(heldout), **support}
        member = rows[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_bps']].copy()
        member['opportunity_index'], member['sample_weight'], member['model_key'] = indices[~selected], weights, key
        members.append(pd.concat([member, costs], axis=1))
    membership = visitation_rows(training, scores, indices)
    save_json(output/'teacher_models.json', models)
    save_json(output/'teacher_support.json', supports)
    pd.concat(members, ignore_index=True).to_parquet(output/'teacher_training_membership.parquet', index=False)
    membership.to_parquet(output/'visitation_ledger.parquet', index=False)
    return membership


def fit_visited_close(training, weights, diagnosis, opportunity_indices, output):
    pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    membership = fit_visitation_teachers(training, opportunity_indices, output)
    selected = membership.visited_prefix.to_numpy()
    retained = training.loc[selected].reset_index(drop=True)
    checked, _ = weekly_training_rows(retained, pd.Timestamp(VISITATION_SETTINGS['fit_time']))
    pd.testing.assert_frame_equal(checked, retained, check_exact=True)
    visited_weights = membership.loc[selected, ['decision_time', 'position_entry_time', 'visited_position_weight']].reset_index(drop=True)
    visited_weights = visited_weights.rename(columns={'visited_position_weight': 'sample_weight'})
    np.testing.assert_array_equal(visited_weights.sample_weight, position_weights(retained))
    model, support, costs = VisitedCloseModel.fit(retained[VisitedCloseModel.features].to_numpy(dtype=float),
        retained.close_advantage_bps, visited_weights.sample_weight, diagnosis[VisitedCloseModel.features].to_numpy(dtype=float))
    retained.to_parquet(output/'visited_training_used.parquet', index=False)
    visited_weights.to_parquet(output/'visited_training_weights.parquet', index=False)
    ledger = pd.concat([retained[['decision_time', 'position_entry_time']].reset_index(drop=True), costs], axis=1)
    ledger.to_parquet(output/'visited_training_cost_ledger.parquet', index=False)
    audit = membership.copy()
    audit['fit_used'], audit['cost_weight'], audit['fit_weight'] = False, 0., 0.
    for name in ['fit_used', 'cost_weight', 'fit_weight']:
        audit.loc[selected, name] = costs[name].to_numpy()
    audit.loc[selected, 'reason'] = costs.reason.to_numpy()
    audit.to_parquet(output/'full_training_contribution.parquet', index=False)
    support = {**support, 'original_rows': len(training), 'visited_rows': len(retained),
        'after_first_teacher_close_rows': int((~selected).sum()),
        'original_positions': int(training.position_entry_time.nunique()),
        'visited_positions': int(retained.position_entry_time.nunique())}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
