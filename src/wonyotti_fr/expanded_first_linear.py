from __future__ import annotations

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .common import save_json
from .early_stopping_close import validate_early_stopping_training
from .first_linear_close import FirstLinearCloseModel
from .first_opportunity_close import first_cost_training, first_opportunity_rows

EXPANDED_FIRST_KEYS = ['decision_time', 'position_entry_time', 'label_end', 'original_intent',
    'decision_equity', 'reference_equity', 'close_cash', 'continue_cash', 'close_advantage_pnl', 'close_advantage_bps',
    'first_target_common_bps', *FirstLinearCloseModel.features]
EXPANDED_FIRST_KEYS = list(dict.fromkeys(EXPANDED_FIRST_KEYS))


def expanded_first_inputs(training, weights, calibration, previous_ledger, added, positions):
    validate_early_stopping_training(training)
    pd.testing.assert_frame_equal(training[['decision_time', 'position_entry_time']], weights.drop(columns='sample_weight'), check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    original, original_positions = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    original_costs, _ = first_cost_training(original.first_target_common_bps)
    fields = ['opportunity_index', 'decision_time', 'position_entry_time', 'label_end', 'direction',
        'decision_equity', 'reference_equity', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']
    pd.testing.assert_frame_equal(pd.concat([original[fields], original_costs], axis=1), previous_ledger, check_exact=True)
    if (added.empty or positions.empty or added.position_entry_time.duplicated().any() or positions.position_entry_time.duplicated().any()
        or not added.decision_time.is_monotonic_increasing or added.decision_time.duplicated().any()
        or not set(added.label_status) <= {'closed', 'right_censored', 'outside_training_boundary'}):
        raise ValueError('과거 확장 첫 기회의 전체 원장·중복·상태 오류')
    for frame, columns in [(added, ['decision_time', 'position_entry_time', 'label_end', 'first_available_time']),
        (positions, ['position_entry_time', 'first_available_time', 'first_eligible_time', 'natural_exit_time'])]:
        for name in columns:
            if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or str(frame[name].dt.tz) != 'UTC':
                raise ValueError('과거 확장 첫 기회의 UTC 시각 오류')
    start, end, cutoff = [pd.Timestamp(value, tz='UTC') for value in ['2020-04-01', '2021-01-01', '2020-12-31']]
    if (positions.position_entry_time.lt(start).any() or positions.position_entry_time.ge(end).any()
        or added.position_entry_time.ge(added.decision_time).any() or added.decision_time.ge(end).any()
        or added.first_available_time.gt(added.decision_time).any() or added.original_intent.eq('exit').any()
        or not np.isfinite(added.reference_equity).all() or added.reference_equity.le(0).any()):
        raise ValueError('과거 확장 첫 기회의 기간·첫 시각·순자산 오류')
    selected = positions[positions.has_eligible_opportunity].reset_index(drop=True)
    for first_name, position_name in [('position_entry_time', 'position_entry_time'), ('decision_time', 'first_eligible_time'),
        ('first_available_time', 'first_available_time'), ('reference_equity', 'reference_equity'),
        ('opportunity_index', 'first_opportunity_index'), ('label_status', 'reason'), ('first_target_common_bps', 'first_target_common_bps')]:
        pd.testing.assert_series_equal(added[first_name].reset_index(drop=True), selected[position_name], check_names=False, check_exact=True)
    FirstLinearCloseModel.matrix(added[FirstLinearCloseModel.features].to_numpy())
    closed = added[added.label_status.eq('closed')].reset_index(drop=True)
    if (closed.empty or closed.label_end.isna().any() or closed.label_end.le(closed.decision_time).any() or closed.label_end.ge(cutoff).any()
        or not np.isfinite(closed[['decision_equity', 'close_cash', 'continue_cash', 'close_advantage_pnl', 'first_target_common_bps']]).all().all()
        or closed.decision_equity.le(0).any() or selected[selected.reason.eq('closed')].natural_exit_reason.eq('end_of_test').any()):
        raise ValueError('과거 확장 첫 기회의 자연 종료·경계·현금 오류')
    np.testing.assert_allclose(closed.close_cash-closed.continue_cash, closed.close_advantage_pnl, rtol=0, atol=1e-9)
    np.testing.assert_allclose(closed.close_advantage_pnl/closed.reference_equity*10000, closed.first_target_common_bps, rtol=0, atol=1e-10)
    np.testing.assert_allclose(closed.close_advantage_pnl/closed.decision_equity*10000, closed.close_advantage_bps, rtol=0, atol=1e-10)
    pieces = []
    for phase, part in [('expansion_2020', closed), ('original_2021', original)]:
        piece = part[EXPANDED_FIRST_KEYS].copy()
        piece['source_phase'] = phase
        piece['source_opportunity_index'] = part.opportunity_index.to_numpy()
        for name in ['decision_time', 'position_entry_time', 'label_end']:
            piece[name] = piece[name].astype('datetime64[ns, UTC]')
        pieces.append(piece)
    combined = pd.concat(pieces, ignore_index=True).sort_values('decision_time', kind='stable').reset_index(drop=True)
    directions = combined.groupby('direction').position_entry_time.nunique()
    if (combined.position_entry_time.duplicated().any() or combined.decision_time.duplicated().any()
        or len(combined) < 500 or any(directions.get(side, 0) < 20 for side in [-1, 1])
        or combined.decision_time.max()-combined.decision_time.min() < pd.Timedelta(days=180)
        or combined.label_end.ge(pd.Timestamp('2021-07-31', tz='UTC')).any()
        or set(combined.position_entry_time) & set(calibration.position_entry_time)):
        raise ValueError('과거 확장 첫 기회의 통합 지원·시간 경계·보정 교차 오류')
    return combined, validation, original_positions


def fit_expanded_first_linear(training, weights, calibration, previous_ledger, added, positions, output):
    combined, validation, original_positions = expanded_first_inputs(training, weights, calibration, previous_ledger, added, positions)
    model, support, costs = FirstLinearCloseModel.fit(combined[model_features := FirstLinearCloseModel.features].to_numpy(),
        combined.first_target_common_bps.to_numpy(), validation[model_features].to_numpy())
    combined.to_parquet(output/'combined_first_training.parquet', index=False)
    pd.concat([combined[['source_phase', 'source_opportunity_index', 'decision_time', 'position_entry_time', 'label_end']], costs], axis=1).to_parquet(
        output/'combined_training_costs.parquet', index=False)
    original_positions.to_parquet(output/'original_training_positions.parquet', index=False)
    membership = positions.copy()
    membership['fit_eligible'] = membership.reason.eq('closed') & membership.has_eligible_opportunity
    membership['fit_exclusion_reason'] = np.where(membership.fit_eligible, 'included_first_closed', membership.reason)
    membership.to_parquet(output/'added_training_membership.parquet', index=False)
    support.update(phase_first_positions=combined.source_phase.value_counts().to_dict(),
        added_all_positions=len(positions), original_all_positions=len(original_positions),
        combined_first_positions=len(combined), fit_time='2021-08-02T00:00:00+00:00', last_label_end=combined.label_end.max(),
        old_first_rows_and_costs_exact=True, new_models_fitted=1, refit_after_calibration=False,
        whole_policy_historically_available_claimed=False, diagnosis_used_for_export=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
