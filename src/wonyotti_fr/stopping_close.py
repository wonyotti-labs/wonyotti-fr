from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_flow import FlowCloseModel
from .close_utility import cost_scores
from .common import save_json
from .weekly_close import weekly_training_rows
from .weekly_flow import weekly_flow_admission

STOPPING_SETTINGS = {'folds': 5, 'fold_rule': 'sha256_utc_entry_nanoseconds_decimal_big_endian_mod_5',
    'fit_time': '2021-10-02T00:00:00+00:00', 'label_cutoff': '2021-09-30T00:00:00+00:00',
    'policy_iterations': 1, 'score_threshold': .5, 'future_choice': 'strictly_later_first_teacher_choice',
    'fallback': 'original_natural_close', 'teacher_features_added': False}


class StoppingCloseModel(FlowCloseModel):
    format = 'stopping_cost_weighted_close_v1'


def position_folds(frame):
    if (frame.empty or frame.position_entry_time.isna().any()
        or not isinstance(frame.position_entry_time.dtype, pd.DatetimeTZDtype) or str(frame.position_entry_time.dt.tz) != 'UTC'):
        raise ValueError('이후 청산의 포지션 분할 입력 오류')
    mapping = {time: int.from_bytes(hashlib.sha256(str(pd.Timestamp(time).value).encode('ascii')).digest(), 'big') % 5
        for time in frame.position_entry_time.unique()}
    return frame.position_entry_time.map(mapping).to_numpy(dtype=np.int64)


def checked_opportunity_indices(values, rows):
    indices = np.asarray(values)
    if (indices.shape != (rows,) or not np.issubdtype(indices.dtype, np.integer)
        or (indices < 0).any() or (indices > np.iinfo(np.int64).max).any() or not np.all(indices[1:] > indices[:-1])):
        raise ValueError('이후 청산의 원래 행 번호 오류')
    return indices.astype(np.int64, copy=False)


def validate_stopping_rows(frame):
    times = ['decision_time', 'position_entry_time', 'label_end', 'continue_end']
    if (frame.empty or any(not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or str(frame[name].dt.tz) != 'UTC' for name in times)
        or frame[times].isna().any().any() or frame.decision_time.duplicated().any()
        or not frame.decision_time.is_monotonic_increasing or not frame.label_status.eq('closed').all()
        or frame.position_entry_time.ge(frame.decision_time).any() or frame.decision_time.ge(frame.continue_end).any()
        or frame.continue_end.gt(frame.label_end).any() or frame.label_end.ge(pd.Timestamp(STOPPING_SETTINGS['label_cutoff'])).any()
        or frame.position_entry_time.lt(pd.Timestamp('2021-01-01', tz='UTC')).any()
        or not frame.original_intent.isin(['hold', 'exit', 'reduce', 'increase']).all() or not frame.direction.isin([-1, 1]).all()):
        raise ValueError('이후 청산의 확정 시각·포지션·원래 의도 오류')
    fields = ['close_cash', 'continue_cash', 'close_advantage_pnl', 'close_advantage_bps', 'decision_equity']
    if (not np.isfinite(frame[fields]).all().all() or frame.decision_equity.le(0).any()
        or frame.groupby('position_entry_time')[['continue_cash', 'continue_end', 'direction']].nunique().ne(1).any().any()
        or not np.allclose(frame.close_cash-frame.continue_cash, frame.close_advantage_pnl, rtol=0, atol=1e-10)
        or not np.allclose(frame.close_advantage_pnl/frame.decision_equity*10000, frame.close_advantage_bps, rtol=0, atol=1e-10)):
        raise ValueError('이후 청산의 자연 종료 현금·순자산 환산 오류')


def stopping_targets(training, scores, opportunity_indices):
    validate_stopping_rows(training)
    frame = training.reset_index(drop=True)
    score = cost_scores(scores, len(frame))
    indices = checked_opportunity_indices(opportunity_indices, len(frame))
    future = np.full(len(frame), -1, dtype=np.int64)
    allowed = (score > .5) & frame.original_intent.ne('exit').to_numpy()
    for group in frame.groupby('position_entry_time', sort=False).indices.values():
        next_choice = -1
        for index in reversed(group):
            # 현재 행을 기록한 뒤 후보를 갱신해 자기 자신을 미래 청산으로 쓰지 않는다.
            future[index] = next_choice
            if allowed[index]:
                next_choice = index
    chosen = future >= 0
    take = np.maximum(future, 0)
    cash = np.where(chosen, frame.close_cash.to_numpy()[take], frame.continue_cash.to_numpy())
    available = frame.label_end.copy()
    later_end = frame.label_end.iloc[take].reset_index(drop=True)
    available = available.where(~chosen | available.ge(later_end), later_end)
    with np.errstate(over='ignore', invalid='ignore'):
        pnl = frame.close_cash.to_numpy()-cash
        target = pnl/frame.decision_equity.to_numpy()*10000
    if not np.isfinite(target).all() or available.ge(pd.Timestamp(STOPPING_SETTINGS['label_cutoff'])).any():
        raise ValueError('이후 청산의 정답 넘침·미확정 결과 오류')
    result = frame[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_pnl', 'close_advantage_bps']].copy()
    result['opportunity_index'], result['teacher_fold'], result['teacher_score'] = indices, position_folds(frame), score
    result['future_opportunity_index'] = np.where(chosen, indices[take], -1)
    result['future_decision_time'] = frame.decision_time.iloc[take].reset_index(drop=True).where(chosen)
    result['future_teacher_score'] = np.where(chosen, score[take], np.nan)
    result['continuation_policy_cash'], result['natural_close_fallback'] = cash, ~chosen
    result['stopping_advantage_pnl'], result['stopping_advantage_bps'], result['target_available_time'] = pnl, target, available
    return result


def fit_stopping_teachers(training, opportunity_indices, output):
    validate_stopping_rows(training)
    opportunity_indices = checked_opportunity_indices(opportunity_indices, len(training))
    checked, _ = weekly_training_rows(training, pd.Timestamp(STOPPING_SETTINGS['fit_time']))
    pd.testing.assert_frame_equal(checked, training, check_exact=True)
    folds = position_folds(training)
    scores = np.full(len(training), np.nan)
    models, supports, members = {}, {}, []
    for fold in range(5):
        key, selected = f'fold-{fold:02}', folds == fold
        heldout, rows = training.loc[selected], training.loc[~selected].reset_index(drop=True)
        if heldout.empty or set(rows.position_entry_time) & set(heldout.position_entry_time):
            raise ValueError('이후 청산의 보조 학습·제외 포지션 오류')
        checked, _ = weekly_training_rows(rows, pd.Timestamp(STOPPING_SETTINGS['fit_time']))
        pd.testing.assert_frame_equal(checked, rows, check_exact=True)
        weight = position_weights(rows)
        model, support, costs = FlowCloseModel.fit(rows[FlowCloseModel.features].to_numpy(dtype=float),
            rows.close_advantage_bps, weight, heldout[FlowCloseModel.features].to_numpy(dtype=float))
        scores[selected] = model.probabilities(heldout[FlowCloseModel.features].to_numpy(dtype=float))[:, 0]
        models[key] = model.to_dict()
        supports[key] = {'fit_time': STOPPING_SETTINGS['fit_time'], 'label_cutoff': STOPPING_SETTINGS['label_cutoff'],
            'positions': int(rows.position_entry_time.nunique()), 'last_label_end': rows.label_end.max(),
            'heldout_positions': int(heldout.position_entry_time.nunique()), 'prediction_rows': len(heldout), **support}
        member = rows[['decision_time', 'position_entry_time', 'label_end', 'close_advantage_bps']].copy()
        member['opportunity_index'], member['sample_weight'], member['model_key'] = np.asarray(opportunity_indices)[~selected], weight, key
        members.append(pd.concat([member, costs], axis=1))
    targets = stopping_targets(training, scores, opportunity_indices)
    save_json(output/'teacher_models.json', models)
    save_json(output/'teacher_support.json', supports)
    pd.concat(members, ignore_index=True).to_parquet(output/'teacher_training_membership.parquet', index=False)
    targets.to_parquet(output/'stopping_targets.parquet', index=False)
    return targets


def fit_stopping_close(training, weights, diagnosis, opportunity_indices, output):
    pd.testing.assert_frame_equal(weights.drop(columns='sample_weight'), training[['decision_time', 'position_entry_time']], check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    targets = fit_stopping_teachers(training, opportunity_indices, output)
    model, support, costs = StoppingCloseModel.fit(training[StoppingCloseModel.features].to_numpy(dtype=float),
        targets.stopping_advantage_bps, weights.sample_weight, diagnosis[StoppingCloseModel.features].to_numpy(dtype=float))
    ledger = pd.concat([training[['decision_time', 'position_entry_time']].reset_index(drop=True), costs], axis=1)
    ledger.to_parquet(output/'stopping_training_cost_ledger.parquet', index=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support


def stopping_admission(metrics, probability, first, intervals, utility_intervals, context_intervals, flow_intervals, weekly_intervals):
    base = weekly_flow_admission(metrics, probability, first, intervals, utility_intervals, context_intervals, flow_intervals,
        candidate_name='stopping_flow')
    checks = dict(base['checks'])
    candidate, previous = probability['stopping_flow'], probability['weekly_flow']
    low = weekly_intervals['intervals']['paired_difference']['lower']
    checks.update(cost_log_loss_vs_weekly_flow=candidate['cost_log_loss'] is not None and previous['cost_log_loss'] is not None and candidate['cost_log_loss'] < previous['cost_log_loss']*.99,
        cost_brier_vs_weekly_flow=candidate['cost_brier'] is not None and previous['cost_brier'] is not None and candidate['cost_brier'] <= previous['cost_brier']+1e-12,
        weighted_regret_vs_weekly_flow=metrics['stopping_flow']['weighted_regret_bps'] < metrics['weekly_flow']['weighted_regret_bps'],
        first_mean_vs_weekly_flow=first['stopping_flow']['all_position_mean_common_bps'] > first['weekly_flow']['all_position_mean_common_bps'],
        positive_weekly_flow_paired_interval_lower=bool(low is not None and np.isfinite(low) and low > 0))
    return {'checks': checks, 'stopping_flow_admitted': all(checks.values()), 'trading_returns_evaluated': False}
