from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .addition_effect import position_weights
from .close_flow import validate_flow_matrix
from .close_utility import cost_scores
from .common import save_json
from .continuation_inputs import PENDING_FEATURES, validate_pending_matrix
from .early_stopping_close import validate_early_stopping_training
from .first_close_diagnostics import first_close_positions
from .histogram_management import HISTOGRAM_SETTINGS
from .minute_close_learning import MinuteCloseModel, minute_grid_actions


def first_opportunity_rows(frame):
    frame = frame.reset_index(drop=True)
    minute_grid_actions(frame, np.zeros(len(frame)), 60)
    cash = ['decision_equity', 'close_advantage_pnl', 'close_advantage_bps']
    if (not isinstance(frame.position_entry_time.dtype, pd.DatetimeTZDtype)
        or str(frame.position_entry_time.dt.tz) != 'UTC' or frame.position_entry_time.isna().any()
        or frame.position_entry_time.ge(frame.decision_time).any() or not frame.direction.isin([-1, 1]).all()
        or frame.groupby('position_entry_time').direction.nunique().ne(1).any()
        or not np.isfinite(frame[cash]).all().all() or frame.decision_equity.le(0).any()
        or not np.allclose(frame.close_advantage_pnl/frame.decision_equity*10000, frame.close_advantage_bps, rtol=0, atol=1e-10)):
        raise ValueError('첫 적격 기회의 포지션·현금·순자산 오류')
    initial = frame.drop_duplicates('position_entry_time').set_index('position_entry_time')
    selected = frame[frame.original_intent.ne('exit')].drop_duplicates('position_entry_time').copy()
    selected['opportunity_index'] = selected.index.to_numpy()
    selected['reference_equity'] = selected.position_entry_time.map(initial.decision_equity)
    selected['first_target_common_bps'] = selected.close_advantage_pnl/selected.reference_equity*10000
    if not np.isfinite(selected.first_target_common_bps).all():
        raise ValueError('첫 적격 기회의 공통 현금 효과 넘침')
    positions = initial[['direction', 'decision_time', 'decision_equity']].rename(columns={
        'decision_time': 'first_available_time', 'decision_equity': 'reference_equity'})
    lookup = selected.set_index('position_entry_time')
    positions['has_eligible_opportunity'] = positions.index.isin(lookup.index)
    positions['first_eligible_time'] = lookup.decision_time.reindex(positions.index)
    positions['first_opportunity_index'] = lookup.opportunity_index.reindex(positions.index).fillna(-1).astype('int64')
    positions['first_target_common_bps'] = lookup.first_target_common_bps.reindex(positions.index).fillna(0.)
    positions['reason'] = np.where(positions.has_eligible_opportunity, 'first_eligible', 'no_eligible_opportunity')
    return selected.reset_index(drop=True), positions.reset_index()


def first_cost_training(target):
    y = np.asarray(target, dtype=float)
    if y.ndim != 1 or len(y) < 500 or not np.isfinite(y).all() or min((y > 0).sum(), (y < 0).sum()) < 64:
        raise ValueError('첫 적격 기회 비용의 포지션·양쪽 지원 부족')
    cost = np.abs(y)
    fit = cost > 0
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        normalizer = float(cost[fit].mean())
        weight = cost/normalizer
    if (not np.isfinite(normalizer) or normalizer <= 0 or not np.isfinite(weight).all()
        or (weight[fit] <= 0).any() or not np.isclose(weight[fit].mean(), 1., rtol=0, atol=1e-12)):
        raise ValueError('첫 적격 기회의 비용 정규화 넘침·소실')
    prior = float(np.average(y[fit] > 0, weights=weight[fit]))
    if not 0 < prior < 1:
        raise ValueError('첫 적격 기회의 학습 상수 지원 부족')
    ledger = pd.DataFrame({'position_weight': np.ones(len(y)), 'positive_effect': y > 0,
        'absolute_effect_bps': cost, 'fit_weight': weight, 'fit_used': fit,
        'reason': np.where(fit, 'positive_cost', 'zero_effect')})
    return ledger, {'eligible_positions': len(y), 'fit_positions': int(fit.sum()), 'zero_effect_positions': int((~fit).sum()),
        'normalizer': normalizer, 'training_constant_score': prior}


class FirstOpportunityCloseModel(MinuteCloseModel):
    format = 'first_opportunity_cost_weighted_close_v1'

    @classmethod
    def fit(cls, values, target, validation_values):
        x, vx = (np.asarray(value, dtype=float) for value in [values, validation_values])
        for matrix in [x, vx]:
            if matrix.ndim != 2 or matrix.shape[1] != len(cls.features) or not len(matrix) or not np.isfinite(matrix).all():
                raise ValueError('첫 적격 기회 모델의 현재 입력 오류')
            validate_flow_matrix(matrix, cls.features)
            validate_pending_matrix(matrix[:, [cls.features.index(name) for name in PENDING_FEATURES]])
        ledger, support = first_cost_training(target)
        if len(x) != len(ledger):
            raise ValueError('첫 적격 기회의 입력·정답 행 수 오류')
        fit = ledger.fit_used.to_numpy()
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(
                x[fit], ledger.positive_effect.to_numpy()[fit], sample_weight=ledger.fit_weight.to_numpy()[fit])
            expected, validation_expected = learner.predict_proba(x)[:, 1], learner.predict_proba(vx)[:, 1]
        if (learner.n_iter_ != 64 or learner._baseline_prediction.shape != (1, 1)
            or len(learner._predictors) != 64 or any(len(part) != 1 for part in learner._predictors)):
            raise ValueError('첫 적격 기회의 트리·초기 점수 구조 오류')
        trees = []
        required = {'value', 'count', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right',
            'gain', 'depth', 'is_leaf', 'bin_threshold', 'is_categorical', 'bitset_idx'}
        for part in learner._predictors:
            nodes = part[0].nodes
            if set(nodes.dtype.names) != required or nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                raise ValueError('첫 적격 기회의 지원하지 않는 트리 구조')
            leaf = nodes['is_leaf'].astype(bool)
            trees.append({'left': np.where(leaf, -1, nodes['left'].astype(int)).tolist(),
                'right': np.where(leaf, -1, nodes['right'].astype(int)).tolist(),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)).tolist(),
                'threshold': np.where(leaf, 0., nodes['num_threshold']).tolist(),
                'value': np.where(leaf, nodes['value'], 0.).tolist()})
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': HISTOGRAM_SETTINGS,
            'models': [{'action': cls.actions[0], 'baseline': float(learner._baseline_prediction[0, 0]), 'trees': trees}]})
        error = float(np.max(np.abs(model.probabilities(x)[:, 0]-expected)))
        validation_error = float(np.max(np.abs(model.probabilities(vx)[:, 0]-validation_expected)))
        if error > 1e-12 or validation_error > 1e-12:
            raise ValueError('첫 적격 기회의 숫자 내보내기 불일치')
        return model, {**support, 'export_max_error': error, 'validation_export_max_error': validation_error}, ledger


def fit_first_opportunity(training, weights, calibration, output):
    validate_early_stopping_training(training)
    pd.testing.assert_frame_equal(training[['decision_time', 'position_entry_time']], weights.drop(columns='sample_weight'), check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    training, weights = training.reset_index(drop=True), weights.reset_index(drop=True)
    first, positions = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    directions = first.groupby('direction').position_entry_time.nunique()
    if (len(first) < 500 or any(directions.get(side, 0) < 20 for side in [-1, 1])
        or first.decision_time.max()-first.decision_time.min() < pd.Timedelta(days=180)):
        raise ValueError('첫 적격 기회의 학습 포지션·방향·기간 지원 부족')
    model, support, ledger = FirstOpportunityCloseModel.fit(first[FirstOpportunityCloseModel.features].to_numpy(),
        first.first_target_common_bps.to_numpy(), validation[FirstOpportunityCloseModel.features].to_numpy())
    keys = ['opportunity_index', 'decision_time', 'position_entry_time', 'label_end', 'direction',
        'decision_equity', 'reference_equity', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']
    pd.concat([first[keys], ledger], axis=1).to_parquet(output/'first_training_ledger.parquet', index=False)
    positions.to_parquet(output/'all_training_positions.parquet', index=False)
    contribution = training[['decision_time', 'position_entry_time']].copy()
    contribution['original_weight'] = weights.sample_weight
    contribution['first_eligible'] = contribution.index.isin(first.opportunity_index)
    contribution['position_weight'] = contribution.first_eligible.astype(float)
    contribution.to_parquet(output/'first_training_contribution.parquet', index=False)
    support = {**support, 'original_rows': len(training), 'all_positions': len(positions),
        'no_eligible_positions': int((~positions.has_eligible_opportunity).sum()),
        'fit_time': '2021-08-02T00:00:00+00:00', 'last_label_end': training.label_end.max(),
        'diagnosis_used_for_export': False, 'export_validation_period': 'first_calibration_opportunities',
        'new_models_fitted': 1, 'refit_after_calibration': False}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support


def first_opportunity_policy(frame, scores):
    first, _ = first_opportunity_rows(frame)
    values = cost_scores(scores, len(first))
    action, full_scores = np.zeros(len(frame), dtype=bool), np.full(len(frame), np.nan)
    indices = first.opportunity_index.to_numpy()
    action[indices], full_scores[indices] = values > .5, values
    # 거절한 첫 기회 뒤에는 다시 예측하거나 나중의 양수 점수로 청산하지 않는다.
    positions = first_close_positions(frame.assign(_first_action=action.astype(float)), '_first_action').drop(columns='first_prediction_bps')
    positions['first_selected_time'] = pd.to_datetime(positions.first_selected_time, utc=True).astype('datetime64[ns, UTC]')
    positions['first_cost_score'] = positions.first_selected_time.map(pd.Series(full_scores, index=frame.decision_time))
    return action, positions, full_scores
