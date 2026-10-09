from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_utility import cost_scores, policy_effect_metrics
from .common import save_json
from .first_close_diagnostics import first_close_metrics, first_close_positions
from .minute_close_learning import MinuteCloseModel, minute_grid_actions
from .weekly_close import WEEKLY_SETTINGS

THRESHOLDS = [.2, .3, .4, .5, .6, .7, .8]
THRESHOLD_SPLITS = {'early_training': ['2021-01-01', '2021-07-31'],
    'calibration': ['2021-08-02', '2021-09-30'], 'diagnosis': ['2021-10-02', '2021-12-31']}
THRESHOLD_SETTINGS = {'fit_time': '2021-08-02', 'thresholds': THRESHOLDS,
    'minimum_selected_positions': 30, 'score_comparison': 'strictly_greater',
    'objective': 'all_position_mean_common_bps', 'tie_break': 'higher_threshold',
    'refit_after_selection': False, 'selection_failure_fallback': None}


class ThresholdCloseModel(MinuteCloseModel):
    format = 'early_training_cost_weighted_close_v1'


def threshold_splits(ledger):
    for name in ['decision_time', 'position_entry_time', 'label_end']:
        if not isinstance(ledger[name].dtype, pd.DatetimeTZDtype) or str(ledger[name].dt.tz) != 'UTC':
            raise ValueError('문턱 보정 원장의 UTC 시각 오류')
    closed = ledger.label_status.eq('closed')
    if (ledger.empty or ledger.decision_time.isna().any() or ledger.position_entry_time.isna().any()
        or ledger.decision_time.duplicated().any() or not ledger.decision_time.is_monotonic_increasing
        or ledger.loc[closed, 'label_end'].isna().any()
        or ledger.position_entry_time.ge(ledger.decision_time).any()
        or ledger.loc[closed, 'decision_time'].ge(ledger.loc[closed, 'label_end']).any()
        or not ledger.direction.isin([-1, 1]).all()
        or ledger.groupby('position_entry_time').direction.nunique().gt(1).any()
        or ledger.groupby('position_entry_time').label_end.nunique().gt(1).any()):
        raise ValueError('문턱 보정 원장의 순서·포지션·종료 오류')
    assignment = ledger.copy()
    assignment['opportunity_index'] = np.arange(len(ledger))
    assignment['split'] = np.where(closed, 'excluded_boundary', 'excluded_not_closed')
    rows, populations = {}, []
    for name, bounds in THRESHOLD_SPLITS.items():
        start, end = [pd.Timestamp(value, tz='UTC') for value in bounds]
        valid = closed & ledger.position_entry_time.ge(start) & ledger.decision_time.ge(start) & ledger.label_end.lt(end)
        # 경계를 넘는 포지션의 일부만 학습하거나 평가하지 않는다.
        mask = valid.groupby(ledger.position_entry_time).transform('all')
        frame = ledger.loc[mask].reset_index(drop=True)
        if frame.empty or not np.isfinite(frame[ThresholdCloseModel.features+['close_advantage_bps']]).all().all():
            raise ValueError('문턱 보정 구간의 빈 원장·비유한 입력 오류')
        population = set(frame.position_entry_time)
        if any(population & prior for prior in populations):
            raise ValueError('문턱 보정 구간의 포지션 교차')
        populations.append(population)
        rows[name] = frame
        assignment.loc[mask, 'split'] = name
    training = rows['early_training']
    directions = training.groupby('direction').position_entry_time.nunique()
    if (len(training) < WEEKLY_SETTINGS['minimum_rows']
        or training.position_entry_time.nunique() < WEEKLY_SETTINGS['minimum_positions']
        or any(directions.get(side, 0) < WEEKLY_SETTINGS['minimum_positions_per_direction'] for side in [-1, 1])
        or training.decision_time.max()-training.decision_time.min() < pd.Timedelta(days=WEEKLY_SETTINGS['minimum_span_days'])):
        raise ValueError('문턱 앞 학습의 표본·기간·방향 지원 부족')
    return rows, assignment


def fit_threshold_close(training, calibration, output):
    weights = position_weights(training)
    model, support, costs = ThresholdCloseModel.fit(training[ThresholdCloseModel.features].to_numpy(dtype=float),
        training.close_advantage_bps, weights, calibration[ThresholdCloseModel.features].to_numpy(dtype=float))
    training[['decision_time', 'position_entry_time']].assign(sample_weight=weights).to_parquet(
        output/'early_training_weights.parquet', index=False)
    pd.concat([training[['decision_time', 'position_entry_time']], costs], axis=1).to_parquet(
        output/'early_training_cost_ledger.parquet', index=False)
    support = {**support, 'positions': int(training.position_entry_time.nunique()),
        'fit_time': THRESHOLD_SETTINGS['fit_time'], 'last_label_end': training.label_end.max(),
        'export_validation_period': 'calibration', 'diagnosis_used_for_export': False}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support


def threshold_policy(frame, scores, threshold):
    if type(threshold) not in (float, int) or threshold not in THRESHOLDS:
        raise ValueError('고정 비용 점수 문턱 오류')
    score = cost_scores(scores, len(frame))
    action = minute_grid_actions(frame, (score > threshold).astype(float), 60)
    part = first_close_positions(frame.assign(_selected=action.astype(float)), '_selected').drop(columns='first_prediction_bps')
    part['first_selected_time'] = pd.to_datetime(part.first_selected_time, utc=True).astype('datetime64[ns, UTC]')
    part['first_cost_score'] = part.first_selected_time.map(pd.Series(score, index=frame.decision_time))
    return action, part


def choose_threshold(first):
    expected = {f'threshold_{value:.1f}' for value in THRESHOLDS} | {'always_first', 'never_extra'}
    if set(first) != expected or len({item['positions'] for item in first.values()}) != 1:
        raise ValueError('문턱 선택의 후보·대조·전체 모집단 오류')
    baseline = first['threshold_0.5']['all_position_mean_common_bps']
    always = first['always_first']['all_position_mean_common_bps']
    candidates, eligible = {}, []
    for threshold in THRESHOLDS:
        key = f'threshold_{threshold:.1f}'
        item = first[key]
        effect, count = item['all_position_mean_common_bps'], item['selected_positions']
        finite = all(type(v) in (int, float) and np.isfinite(v) for v in [effect, baseline, always])
        checks = {'at_least_30_selected_positions': type(count) is int and count >= 30,
            'positive_all_first_mean': finite and effect > 0,
            'better_than_default': finite and effect > baseline, 'better_than_always_first': finite and effect > always}
        passed = all(checks.values())
        candidates[key] = {'threshold': threshold, 'checks': {k: bool(v) for k, v in checks.items()},
            'eligible': bool(passed), 'all_position_mean_common_bps': effect,
            'reasons': [key for key, value in checks.items() if not value]}
        if passed:
            eligible.append((effect, threshold))
    selected = max(eligible)[1] if eligible else None
    return {'selection_passed': selected is not None, 'selected_threshold': selected, 'candidates': candidates,
        'diagnosis_allowed': selected is not None, 'fallback_used': False}


def calibrate_threshold(frame, scores):
    frame = frame.reset_index(drop=True)
    np.testing.assert_array_equal(frame.sample_weight, position_weights(frame))
    score = cost_scores(scores, len(frame))
    predictions = frame[['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent',
        'close_advantage_bps', 'sample_weight']].assign(early_score=score)
    metrics, first, positions = {}, {}, {}
    policies = [(f'threshold_{value:.1f}', score, value) for value in THRESHOLDS]
    policies += [('always_first', np.ones(len(frame)), .5), ('never_extra', np.zeros(len(frame)), .5)]
    for key, values, threshold in policies:
        action, part = threshold_policy(frame, values, threshold)
        predictions['selected_'+key] = action
        metrics[key], first[key], positions[key] = policy_effect_metrics(frame, action), first_close_metrics(part), part
    return {'predictions': predictions, 'metrics': metrics, 'first_metrics': first, 'positions': positions,
        'selection': choose_threshold(first)}
