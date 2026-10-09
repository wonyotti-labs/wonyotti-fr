from __future__ import annotations

import json

import pandas as pd

from .common import save_json
from .expanded_first_linear import expanded_first_inputs
from .first_opportunity_close import first_cost_training
from .first_opportunity_diagnostics import checked_first_inputs
from .managed_first_model import augment_manager_inputs
from .minute_first_inputs import MinuteFirstLinearModel, attach_first_minute_flow


def fit_minute_first(reference, manager, bars, output):
    parts, weights = checked_first_inputs(reference)
    combined, validation, positions = expanded_first_inputs(parts['early_training'], weights['early_training'], parts['calibration'],
        pd.read_parquet(reference/'first_training_ledger.parquet'), pd.read_parquet(reference/'added_first_opportunity_ledger.parquet'),
        pd.read_parquet(reference/'added_all_positions.parquet'))
    pd.testing.assert_frame_equal(combined, pd.read_parquet(reference/'combined_first_training.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(positions, pd.read_parquet(reference/'original_training_positions.parquet'), check_exact=True)
    costs, original_support = first_cost_training(combined.first_target_common_bps)
    keys = ['source_phase', 'source_opportunity_index', 'decision_time', 'position_entry_time', 'label_end']
    pd.testing.assert_frame_equal(pd.concat([combined[keys], costs], axis=1),
        pd.read_parquet(reference/'combined_training_costs.parquet'), check_exact=True)
    previous = json.loads((reference/'training_support.json').read_text())
    for name in ['eligible_positions', 'fit_positions', 'zero_effect_positions', 'normalizer', 'training_constant_score']:
        if original_support[name] != previous[name]:
            raise ValueError('확정 분봉 첫 선형의 기존 포지션·비용·상수 불일치')
    combined = augment_manager_inputs(combined, manager)
    validation = augment_manager_inputs(validation, manager)
    pd.testing.assert_frame_equal(combined, pd.read_parquet(reference/'managed_first_training.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(validation, pd.read_parquet(reference/'managed_first_calibration.parquet'), check_exact=True)
    for name, first in [('training', combined), ('calibration', validation)]:
        augmented, selected = attach_first_minute_flow(first, bars)
        augmented.to_parquet(output/('minute_first_'+name+'.parquet'), index=False)
        selected.to_parquet(output/('minute_bars_'+name+'.parquet'), index=False)
        if name == 'training':
            combined = augmented
        else:
            validation = augmented
    model, support, fitted_costs = MinuteFirstLinearModel.fit(combined[MinuteFirstLinearModel.features].to_numpy(),
        combined.first_target_common_bps.to_numpy(), validation[MinuteFirstLinearModel.features].to_numpy())
    pd.testing.assert_frame_equal(fitted_costs, costs, check_exact=True)
    support.update(phase_first_positions=combined.source_phase.value_counts().to_dict(), combined_first_positions=len(combined),
        old_combined_rows_and_costs_exact=True, fit_time='2021-08-02T00:00:00+00:00', last_label_end=combined.label_end.max(),
        new_models_fitted=1, previous_models_refitted=False, refit_after_calibration=False,
        whole_policy_historically_available_claimed=False, diagnosis_used_for_export=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
