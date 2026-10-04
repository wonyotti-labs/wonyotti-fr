from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_learning_inputs import CLOSE_SPLITS
from .common import save_json
from .continuation_inputs import ContinuationCloseModel

WEEKLY_SETTINGS = {'cadence_days': 7, 'label_gap_days': 2,
    'training_start': '2021-01-01', 'window': 'expanding', 'minimum_rows': 1000,
    'minimum_positions': 100, 'minimum_positions_per_direction': 20, 'minimum_span_days': 180}


def weekly_boundaries():
    start, end = [pd.Timestamp(t, tz='UTC') for t in CLOSE_SPLITS['diagnosis']]
    return [(t, min(t+pd.Timedelta(days=7), end)) for t in pd.date_range(start, end, freq='7D', inclusive='left')]


def weekly_training_rows(ledger, fit_time):
    fit_time = pd.Timestamp(fit_time)
    if fit_time not in [t for t, _end in weekly_boundaries()]:
        raise ValueError('주간 청산 학습의 고정 갱신 시각 오류')
    closed = ledger.label_status.eq('closed')
    if (ledger.decision_time.isna().any() or ledger.decision_time.duplicated().any()
        or not ledger.decision_time.is_monotonic_increasing
        or ledger.loc[closed, ['position_entry_time', 'label_end']].isna().any().any()
        or ledger.loc[closed, 'position_entry_time'].ge(ledger.loc[closed, 'decision_time']).any()
        or ledger.loc[closed, 'decision_time'].ge(ledger.loc[closed, 'label_end']).any()):
        raise ValueError('주간 청산 학습의 원장 시각·종료 연결 오류')
    start = pd.Timestamp(WEEKLY_SETTINGS['training_start'], tz='UTC')
    cutoff = fit_time-pd.Timedelta(days=WEEKLY_SETTINGS['label_gap_days'])
    mask = (closed & ledger.position_entry_time.ge(start)
        & ledger.decision_time.ge(start) & ledger.label_end.lt(cutoff))
    frame = ledger.loc[mask].reset_index(drop=True)
    positions = frame.groupby('direction').position_entry_time.nunique()
    if (len(frame) < WEEKLY_SETTINGS['minimum_rows']
        or frame.position_entry_time.nunique() < WEEKLY_SETTINGS['minimum_positions']
        or any(positions.get(side, 0) < WEEKLY_SETTINGS['minimum_positions_per_direction'] for side in [-1, 1])
        or frame.groupby('position_entry_time').direction.nunique().gt(1).any()
        or not np.isfinite(frame[ContinuationCloseModel.features+['close_advantage_bps']]).all().all()
        or frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=WEEKLY_SETTINGS['minimum_span_days'])):
        raise ValueError('주간 청산 학습의 표본·기간·방향·숫자 지원 부족')
    return frame, np.flatnonzero(mask.to_numpy())


def fit_weekly_close(ledger, diagnosis, original_training, original_weights, original_model, output):
    if (not len(diagnosis) or diagnosis.decision_time.isna().any() or diagnosis.decision_time.duplicated().any()
        or not diagnosis.decision_time.is_monotonic_increasing):
        raise ValueError('주간 청산 예측의 진단 시각 오류')
    prediction = np.full(len(diagnosis), np.nan)
    models, support, memberships, routing = {}, {}, [], []
    for number, (start, end) in enumerate(weekly_boundaries()):
        key = f'week-{number:02}'
        training, indices = weekly_training_rows(ledger, start)
        weight = position_weights(training)
        selected = diagnosis.decision_time.ge(start) & diagnosis.decision_time.lt(end)
        week = diagnosis.loc[selected]
        if set(training.position_entry_time) & set(week.position_entry_time):
            raise ValueError('주간 청산 학습과 해당 주 예측의 포지션 교차')
        x = training[ContinuationCloseModel.features].to_numpy(dtype=float)
        vx = week[ContinuationCloseModel.features].to_numpy(dtype=float)
        # 판단이 없는 주에도 모델을 저장하고 학습 행 하나로 숫자 내보내기만 대조한다.
        model, export = ContinuationCloseModel.fit(x, training.close_advantage_bps, weight, vx if len(vx) else x[:1])
        if number == 0:
            pd.testing.assert_frame_equal(training, original_training, check_exact=True)
            np.testing.assert_array_equal(weight, np.asarray(original_weights))
            if model.to_dict() != original_model:
                raise ValueError('주간 청산의 첫 학습 모델이 기존 모델과 불일치')
        if len(week):
            prediction[selected.to_numpy()] = model.predict(vx)
        models[key] = model.to_dict()
        support[key] = {'fit_time': start, 'prediction_end': end, 'label_cutoff': start-pd.Timedelta(days=2),
            'rows': len(training), 'positions': int(training.position_entry_time.nunique()),
            'last_label_end': training.label_end.max(), 'prediction_rows': len(week), **export}
        membership = training[['decision_time', 'position_entry_time', 'label_end']].copy()
        membership['opportunity_index'], membership['sample_weight'], membership['model_key'] = indices, weight, key
        memberships.append(membership)
        routing.append(week[['decision_time', 'position_entry_time']].assign(model_key=key))
    if not np.isfinite(prediction).all():
        raise ValueError('주간 청산 예측의 누락·비유한 값')
    pd.concat(memberships, ignore_index=True).to_parquet(output/'weekly_training_membership.parquet', index=False)
    pd.concat(routing, ignore_index=True).to_parquet(output/'prediction_routing.parquet', index=False)
    save_json(output/'weekly_models.json', models)
    save_json(output/'weekly_support.json', support)
    return prediction
