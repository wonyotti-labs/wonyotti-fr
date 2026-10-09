from __future__ import annotations

import numpy as np
import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import save_json
from .early_stopping_close import validate_early_stopping_training
from .first_linear_close import FirstLinearCloseModel
from .first_opportunity_close import first_cost_training, first_opportunity_rows

WEEKLY_FIRST_SETTINGS = {'start': '2021-08-02', 'end': '2021-09-30', 'step_days': 7, 'label_gap_days': 2}
FIRST_LEDGER_KEYS = ['opportunity_index', 'decision_time', 'position_entry_time', 'label_end', 'direction',
    'decision_equity', 'reference_equity', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']


def weekly_first_boundaries():
    start, end = [pd.Timestamp(WEEKLY_FIRST_SETTINGS[key], tz='UTC') for key in ['start', 'end']]
    step = pd.Timedelta(days=WEEKLY_FIRST_SETTINGS['step_days'])
    return [(point, min(point+step, end)) for point in pd.date_range(start, end, freq=step, inclusive='left')]


def weekly_first_inputs(training, calibration, previous_ledger):
    validate_early_stopping_training(training)
    firsts, populations = [], []
    for name, frame in [('early_training', training), ('calibration', calibration)]:
        start, end = [pd.Timestamp(value, tz='UTC') for value in THRESHOLD_SPLITS[name]]
        if (not isinstance(frame.label_end.dtype, pd.DatetimeTZDtype) or str(frame.label_end.dt.tz) != 'UTC'
            or frame.label_end.isna().any() or frame.position_entry_time.lt(start).any() or frame.label_end.ge(end).any()
            or frame.decision_time.ge(frame.label_end).any() or frame.groupby('position_entry_time').label_end.nunique().ne(1).any()):
            raise ValueError('첫 판단 주간 학습의 전체 포지션·정답 종료 경계 오류')
        first, positions = first_opportunity_rows(frame)
        FirstLinearCloseModel.matrix(first[FirstLinearCloseModel.features].to_numpy())
        firsts.append(first.assign(source_split=name))
        populations.append(set(positions.position_entry_time))
    if populations[0] & populations[1]:
        raise ValueError('첫 판단 주간 원래 학습·보정 포지션 교차')
    costs, _ = first_cost_training(firsts[0].first_target_common_bps)
    pd.testing.assert_frame_equal(pd.concat([firsts[0][FIRST_LEDGER_KEYS], costs], axis=1), previous_ledger, check_exact=True)
    return pd.concat(firsts, ignore_index=True), firsts[1]


def fit_weekly_first_linear(training, calibration, previous_ledger, previous_model, previous_support, output):
    pool, first = weekly_first_inputs(training, calibration, previous_ledger)
    scores, constants = np.full(len(first), np.nan), np.full(len(first), np.nan)
    models, supports, membership, routing = {}, {}, [], []
    for number, (start, end) in enumerate(weekly_first_boundaries()):
        key = f'week-{number:02}'
        cutoff = start-pd.Timedelta(days=WEEKLY_FIRST_SETTINGS['label_gap_days'])
        learned = pool[pool.label_end.lt(cutoff)].reset_index(drop=True)
        chosen = first.decision_time.ge(start) & first.decision_time.lt(end)
        prediction = first.loc[chosen]
        directions = learned.groupby('direction').position_entry_time.nunique()
        if (len(learned) < 500 or any(directions.get(side, 0) < 20 for side in [-1, 1])
            or learned.decision_time.max()-learned.decision_time.min() < pd.Timedelta(days=180)
            or set(learned.position_entry_time) & set(prediction.position_entry_time)):
            raise ValueError('첫 판단 주간 학습의 지원·예측 포지션 교차 오류')
        x = learned[FirstLinearCloseModel.features].to_numpy()
        vx = prediction[FirstLinearCloseModel.features].to_numpy()
        # 판단이 없는 주도 과거 자료만으로 적합하고 학습 첫 행으로 내보내기를 검사한다.
        model, support, costs = FirstLinearCloseModel.fit(x, learned.first_target_common_bps, vx if len(vx) else x[:1])
        if number == 0:
            pd.testing.assert_frame_equal(learned[FIRST_LEDGER_KEYS], previous_ledger[FIRST_LEDGER_KEYS], check_exact=True)
            pd.testing.assert_frame_equal(costs, previous_ledger[costs.columns], check_exact=True)
            if model.to_dict() != previous_model or any(support[name] != previous_support[name] for name in
                ['normalizer', 'training_constant_score', 'eligible_positions', 'fit_positions', 'zero_effect_positions', 'iterations']):
                raise ValueError('첫 판단 주간 최초 모델·비용·상수의 기존 선형 모델 불일치')
        if len(vx):
            scores[chosen.to_numpy()] = model.probabilities(vx)[:, 0]
            constants[chosen.to_numpy()] = support['training_constant_score']
        models[key] = model.to_dict()
        supports[key] = {**support, 'fit_time': start, 'prediction_end': end, 'label_cutoff': cutoff,
            'last_label_end': learned.label_end.max(), 'prediction_rows': len(prediction),
            'newly_available_calibration_positions': int(learned.source_split.eq('calibration').sum())}
        membership.append(pd.concat([learned[FIRST_LEDGER_KEYS+['source_split']].assign(model_key=key), costs], axis=1))
        routing.append(prediction[['opportunity_index', 'decision_time', 'position_entry_time']].assign(model_key=key))
    if not np.isfinite(scores).all() or not np.isfinite(constants).all():
        raise ValueError('첫 판단 주간 모델 연결·점수 누락')
    save_json(output/'weekly_models.json', models)
    save_json(output/'weekly_support.json', supports)
    pd.concat(membership, ignore_index=True).to_parquet(output/'weekly_training_membership.parquet', index=False)
    pd.concat(routing, ignore_index=True).to_parquet(output/'prediction_routing.parquet', index=False)
    return scores, constants
