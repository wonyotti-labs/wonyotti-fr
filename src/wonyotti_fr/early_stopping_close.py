from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .common import save_json
from .minute_close_learning import MinuteCloseModel
from .stopping_close import (
    checked_opportunity_indices,
    position_folds,
    stopping_targets,
    validate_stopping_rows,
)
from .weekly_close import WEEKLY_SETTINGS

EARLY_STOPPING_SETTINGS = {'folds': 5, 'fold_rule': 'sha256_utc_entry_nanoseconds_decimal_big_endian_mod_5',
    'fit_time': '2021-08-02T00:00:00+00:00', 'label_cutoff': '2021-07-31T00:00:00+00:00',
    'policy_iterations': 1, 'score_threshold': .5, 'future_choice': 'strictly_later_first_teacher_choice',
    'fallback': 'original_natural_close', 'teacher_features_added': False,
    'threshold_search': False, 'refit_after_calibration': False}


class EarlyStoppingCloseModel(MinuteCloseModel):
    format = 'early_minute_stopping_cost_weighted_close_v1'


def validate_early_stopping_training(frame):
    validate_stopping_rows(frame)
    directions = frame.groupby('direction').position_entry_time.nunique()
    if (frame.label_end.ge(pd.Timestamp(EARLY_STOPPING_SETTINGS['label_cutoff'])).any()
        or len(frame) < WEEKLY_SETTINGS['minimum_rows']
        or frame.position_entry_time.nunique() < WEEKLY_SETTINGS['minimum_positions']
        or any(directions.get(side, 0) < WEEKLY_SETTINGS['minimum_positions_per_direction'] for side in [-1, 1])
        or frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=WEEKLY_SETTINGS['minimum_span_days'])
        or not np.isfinite(frame[MinuteCloseModel.features+['close_advantage_bps']]).all().all()
        or (frame.decision_time.astype('datetime64[ns, UTC]').array.asi8 % (60*10**9)).any()):
        raise ValueError('분별 이후 청산의 앞 학습 경계·지원·입력 오류')


def fit_early_stopping_teachers(training, opportunity_indices, output):
    validate_early_stopping_training(training)
    indices = checked_opportunity_indices(opportunity_indices, len(training))
    folds = position_folds(training)
    scores = np.full(len(training), np.nan)
    models, supports, members = {}, {}, []
    for fold in range(5):
        key, heldout_mask = f'fold-{fold:02}', folds == fold
        heldout, rows = training.loc[heldout_mask], training.loc[~heldout_mask].reset_index(drop=True)
        if heldout.empty or set(rows.position_entry_time) & set(heldout.position_entry_time):
            raise ValueError('분별 이후 청산의 보조 학습·제외 포지션 오류')
        validate_early_stopping_training(rows)
        weight = position_weights(rows)
        model, support, costs = MinuteCloseModel.fit(rows[MinuteCloseModel.features].to_numpy(dtype=float),
            rows.close_advantage_bps, weight, heldout[MinuteCloseModel.features].to_numpy(dtype=float))
        scores[heldout_mask] = model.probabilities(heldout[MinuteCloseModel.features].to_numpy(dtype=float))[:, 0]
        models[key] = model.to_dict()
        supports[key] = {'fit_time': EARLY_STOPPING_SETTINGS['fit_time'], 'label_cutoff': EARLY_STOPPING_SETTINGS['label_cutoff'],
            'positions': int(rows.position_entry_time.nunique()), 'last_label_end': rows.label_end.max(),
            'heldout_positions': int(heldout.position_entry_time.nunique()), 'prediction_rows': len(heldout), **support}
        member = rows[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_bps']].copy()
        member['opportunity_index'], member['sample_weight'], member['model_key'] = indices[~heldout_mask], weight, key
        members.append(pd.concat([member, costs], axis=1))
    # 기존 미래 첫 선택 계산에 이번 앞 학습의 더 이른 확정 경계를 추가한다.
    targets = stopping_targets(training, scores, indices)
    if targets.target_available_time.ge(pd.Timestamp(EARLY_STOPPING_SETTINGS['label_cutoff'])).any():
        raise ValueError('분별 이후 청산의 새 정답 확정 경계 오류')
    save_json(output/'early_teacher_models.json', models)
    save_json(output/'early_teacher_support.json', supports)
    pd.concat(members, ignore_index=True).to_parquet(output/'early_teacher_training_membership.parquet', index=False)
    targets.to_parquet(output/'early_stopping_targets.parquet', index=False)
    return targets


def fit_early_stopping_close(training, weights, calibration, opportunity_indices, output):
    pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    targets = fit_early_stopping_teachers(training, opportunity_indices, output)
    model, support, costs = EarlyStoppingCloseModel.fit(training[EarlyStoppingCloseModel.features].to_numpy(dtype=float),
        targets.stopping_advantage_bps, weights.sample_weight, calibration[EarlyStoppingCloseModel.features].to_numpy(dtype=float))
    pd.concat([training[['decision_time', 'position_entry_time']].reset_index(drop=True), costs], axis=1).to_parquet(
        output/'stopping_training_cost_ledger.parquet', index=False)
    support = {**support, 'positions': int(training.position_entry_time.nunique()),
        'fit_time': EARLY_STOPPING_SETTINGS['fit_time'], 'last_label_end': training.label_end.max(),
        'export_validation_period': 'calibration', 'diagnosis_used_for_export': False}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
