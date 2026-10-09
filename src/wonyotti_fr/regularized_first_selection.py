from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .close_utility import cost_probability_metrics
from .common import save_json
from .first_opportunity_close import first_cost_training
from .regularized_first_model import REGULARIZATION_STRENGTHS, RegularizedFirstModel

REGULARIZATION_WINDOWS = [('2021-03-01', '2021-05-01'), ('2021-05-01', '2021-07-01'), ('2021-07-01', '2021-07-31')]
FIRST_KEYS = ['source_phase', 'source_opportunity_index', 'decision_time', 'position_entry_time', 'label_end']


def validate_first_table(frame):
    RegularizedFirstModel.matrix(frame[RegularizedFirstModel.features].to_numpy())
    for name in ['decision_time', 'position_entry_time', 'label_end']:
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or str(frame[name].dt.tz) != 'UTC' or frame[name].isna().any():
            raise ValueError('시간순 정규화 첫 원장의 UTC 시각 오류')
    if (frame.empty or frame.position_entry_time.duplicated().any() or frame.decision_time.duplicated().any()
        or not frame.decision_time.is_monotonic_increasing or frame.position_entry_time.ge(frame.decision_time).any()
        or frame.label_end.le(frame.decision_time).any()
        or not np.isfinite(frame[['close_cash', 'continue_cash', 'close_advantage_pnl', 'reference_equity', 'first_target_common_bps']]).all().all()
        or frame.reference_equity.le(0).any()):
        raise ValueError('시간순 정규화 첫 원장의 중복·순서·현금 오류')
    np.testing.assert_allclose(frame.close_cash-frame.continue_cash, frame.close_advantage_pnl, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frame.close_advantage_pnl/frame.reference_equity*10000, frame.first_target_common_bps, rtol=0, atol=1e-10)


def training_support(frame):
    costs, support = first_cost_training(frame.first_target_common_bps)
    counts = frame.groupby('direction').position_entry_time.nunique()
    if (any(counts.get(side, 0) < 20 for side in [-1, 1])
        or frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=180)):
        raise ValueError('시간순 정규화 학습의 방향·기간 지원 부족')
    return costs, support


def regularization_membership(training):
    validate_first_table(training)
    if (training.position_entry_time.lt(pd.Timestamp('2020-04-01', tz='UTC')).any()
        or training.label_end.ge(pd.Timestamp('2021-07-31', tz='UTC')).any()):
        raise ValueError('시간순 정규화 앞 학습의 고정 기간 오류')
    members, splits = [], []
    for number, bounds in enumerate(REGULARIZATION_WINDOWS):
        start, end = [pd.Timestamp(value, tz='UTC') for value in bounds]
        cutoff = start-pd.Timedelta(days=2)
        train = training.position_entry_time.lt(cutoff) & training.decision_time.lt(cutoff) & training.label_end.lt(cutoff)
        validation = (training.position_entry_time.ge(start) & training.position_entry_time.lt(end)
            & training.decision_time.ge(start) & training.decision_time.lt(end) & training.label_end.lt(end))
        if (train & validation).any() or validation.sum() < 30 or training.loc[validation, 'first_target_common_bps'].abs().sum() <= 0:
            raise ValueError('시간순 정규화의 교차·검증 지원 부족')
        costs, support = training_support(training[train])
        role = np.where(train, 'training', np.where(validation, 'validation', 'excluded'))
        reason = np.select([train, validation,
            training.position_entry_time.lt(cutoff) & training.label_end.ge(cutoff),
            training.position_entry_time.ge(cutoff) & training.position_entry_time.lt(start),
            training.position_entry_time.ge(start) & training.position_entry_time.lt(end) & training.label_end.ge(end)],
            ['training', 'validation', 'training_label_not_available', 'purge_gap', 'validation_label_crosses_end'], default='outside_fold')
        member = training[FIRST_KEYS].copy()
        member['input_row'] = np.arange(len(training))
        member['fold'] = f'fold-{number:02}'
        member['role'], member['reason'] = role, reason
        member['fit_used'] = False
        member['fit_weight'] = 0.
        member.loc[train, 'fit_used'] = costs.fit_used.to_numpy()
        member.loc[train, 'fit_weight'] = costs.fit_weight.to_numpy()
        members.append(member)
        splits.append({'fold': f'fold-{number:02}', 'validation_start': start, 'validation_end': end,
            'training_label_cutoff': cutoff, 'training_rows': np.flatnonzero(train), 'validation_rows': np.flatnonzero(validation), 'support': support})
    return pd.concat(members, ignore_index=True), splits


def score_column(strength):
    return 'score_C_'+str(float(strength))


def choose_regularization(predictions):
    if (predictions.empty or predictions.input_row.duplicated().any() or predictions.position_entry_time.duplicated().any()
        or set(predictions.fold) != {'fold-00', 'fold-01', 'fold-02'} or predictions.groupby('fold').size().min() < 30):
        raise ValueError('시간순 정규화의 통합 검증 행 오류')
    target, weights = predictions.first_target_common_bps.to_numpy(), np.ones(len(predictions))
    metrics = {}
    for strength in REGULARIZATION_STRENGTHS:
        scores = predictions[score_column(strength)].to_numpy()
        probability = cost_probability_metrics(target, weights, scores)
        if probability['cost_log_loss'] is None:
            raise ValueError('시간순 정규화 검증의 비용 합 부족')
        metrics[str(strength)] = {**probability, 'selected_positions': int((scores > .5).sum()),
            'all_first_mean_common_bps': float(np.mean(np.where(scores > .5, target, 0.)))}
    selected = min(REGULARIZATION_STRENGTHS, key=lambda value: (metrics[str(value)]['cost_log_loss'], value))
    return {'selected_strength': selected, 'criterion': 'pooled_absolute_cost_log_loss', 'exact_tie_break': 'smaller_C',
        'validation_rows': len(predictions), 'candidates': metrics,
        'training_constant': cost_probability_metrics(target, weights, predictions.training_constant_score.to_numpy()),
        'external_calibration_used_for_selection': False}


def fit_regularization_selection(training, output):
    members, splits = regularization_membership(training)
    members.to_parquet(output/'selection_membership.parquet', index=False)
    all_predictions, all_supports = [], {}
    for split in splits:
        folder = output/split['fold']
        folder.mkdir(mode=0o700)
        train, valid = (training.iloc[split[key]].reset_index(drop=True) for key in ['training_rows', 'validation_rows'])
        expected_costs, _ = training_support(train)
        predictions = valid[[*FIRST_KEYS, 'direction', 'first_target_common_bps']].copy()
        predictions['input_row'] = split['validation_rows']
        predictions['fold'] = split['fold']
        predictions['training_constant_score'] = split['support']['training_constant_score']
        models, supports, metrics = {}, {}, {}
        for strength in REGULARIZATION_STRENGTHS:
            model, support, costs = RegularizedFirstModel.fit(train[model_features := RegularizedFirstModel.features].to_numpy(),
                train.first_target_common_bps.to_numpy(), valid[model_features].to_numpy(), strength=strength)
            pd.testing.assert_frame_equal(costs, expected_costs, check_exact=True)
            scores = model.probabilities(valid[model_features].to_numpy())[:, 0]
            predictions[score_column(strength)] = scores
            models[str(strength)], supports[str(strength)] = model.to_dict(), support
            metrics[str(strength)] = {**cost_probability_metrics(valid.first_target_common_bps, np.ones(len(valid)), scores),
                'selected_positions': int((scores > .5).sum()),
                'all_first_mean_common_bps': float(np.mean(np.where(scores > .5, valid.first_target_common_bps, 0.)))}
        metrics['training_constant'] = cost_probability_metrics(valid.first_target_common_bps, np.ones(len(valid)),
            predictions.training_constant_score.to_numpy())
        save_json(folder/'models.json', models)
        save_json(folder/'support.json', {**split, 'training_rows': split['training_rows'].tolist(),
            'validation_rows': split['validation_rows'].tolist(), 'models': supports, 'validation_metrics': metrics})
        predictions.to_parquet(folder/'predictions.parquet', index=False)
        all_predictions.append(predictions)
        all_supports[split['fold']] = {'training_positions': len(train), 'validation_positions': len(valid), **split['support']}
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(output/'selection_predictions.parquet', index=False)
    decision = choose_regularization(predictions)
    save_json(output/'selection.json', {**decision, 'folds': all_supports, 'models_fitted': 12})
    return decision


def fit_regularized_first(reference, output):
    training = pd.read_parquet(reference/'minute_first_training.parquet')
    costs, support = training_support(training)
    pd.testing.assert_frame_equal(pd.concat([training[FIRST_KEYS], costs], axis=1),
        pd.read_parquet(reference/'combined_training_costs.parquet'), check_exact=True)
    previous = json.loads((reference/'training_support.json').read_text())
    for name in ['eligible_positions', 'fit_positions', 'zero_effect_positions', 'normalizer', 'training_constant_score']:
        if previous[name] != support[name]:
            raise ValueError('시간순 정규화의 기존 전체 비용 불일치')
    selection = output/'selection'
    selection.mkdir(mode=0o700)
    decision = fit_regularization_selection(training, selection)
    # 외부 보정 입력은 강도를 확정한 뒤에만 최종 숫자 검증에 제공한다.
    calibration = pd.read_parquet(reference/'minute_first_calibration.parquet')
    validate_first_table(calibration)
    if (calibration.position_entry_time.lt(pd.Timestamp('2021-08-02', tz='UTC')).any()
        or calibration.label_end.ge(pd.Timestamp('2021-09-30', tz='UTC')).any()
        or set(training.position_entry_time) & set(calibration.position_entry_time)):
        raise ValueError('시간순 정규화의 외부 보정 경계·교차 오류')
    model, fitted_support, fitted_costs = RegularizedFirstModel.fit(training[RegularizedFirstModel.features].to_numpy(),
        training.first_target_common_bps.to_numpy(), calibration[RegularizedFirstModel.features].to_numpy(), strength=decision['selected_strength'])
    pd.testing.assert_frame_equal(costs, fitted_costs, check_exact=True)
    if decision['selected_strength'] == .1:
        prior = json.loads((reference/'model.json').read_text())
        if any(prior[key] != value for key, value in model.to_dict().items() if key != 'format'):
            raise ValueError('시간순 정규화 C=0.1의 기존 모델 불일치')
    fitted_support.update(new_models_fitted=13, selection_models_fitted=12, final_models_fitted=1,
        combined_first_positions=len(training), phase_first_positions=training.source_phase.value_counts().to_dict(),
        selected_strength=decision['selected_strength'], external_calibration_used_for_selection=False,
        final_costs_and_constant_unchanged=True, refit_after_calibration=False, diagnosis_used_for_export=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', fitted_support)
    return model, fitted_support
