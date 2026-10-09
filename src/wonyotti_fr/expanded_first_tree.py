from __future__ import annotations

import json

import pandas as pd

from .common import save_json
from .expanded_first_linear import expanded_first_inputs
from .first_opportunity_close import FirstOpportunityCloseModel, first_cost_training
from .first_opportunity_diagnostics import checked_first_inputs


def fit_expanded_first_tree(reference, output):
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
            raise ValueError('확장 첫 트리의 기존 포지션·비용·상수 불일치')
    model, support, fitted_costs = FirstOpportunityCloseModel.fit(combined[FirstOpportunityCloseModel.features].to_numpy(),
        combined.first_target_common_bps.to_numpy(), validation[FirstOpportunityCloseModel.features].to_numpy())
    pd.testing.assert_frame_equal(fitted_costs, costs, check_exact=True)
    support.update(phase_first_positions=combined.source_phase.value_counts().to_dict(), combined_first_positions=len(combined),
        old_combined_rows_and_costs_exact=True, fit_time='2021-08-02T00:00:00+00:00', last_label_end=combined.label_end.max(),
        new_models_fitted=1, previous_models_refitted=False, refit_after_calibration=False,
        whole_policy_historically_available_claimed=False, diagnosis_used_for_export=False)
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
