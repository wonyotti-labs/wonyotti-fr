import json

import numpy as np
import pandas as pd
import pytest
from test_close_utility_diagnostics import (
    test_evaluation_preserves_all_old_scores_and_handles_both_previous_margin_branches as make_utility,
)
from test_continuation_diagnostics import fixture
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_context import ContextCloseModel
from wonyotti_fr.close_context_diagnostics import (
    evaluate_context,
    reproduce_utility,
    run_close_context_diagnosis,
)
from wonyotti_fr.close_utility_diagnostics import run_close_utility_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.context_position import CONTEXT_FEATURES
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_margin_diagnostics import run_first_close_margin_diagnosis
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis


def synthetic_context(rows, _bars):
    result = {}
    for name, frame in rows.items():
        result[name] = frame.copy()
        for i, column in enumerate(CONTEXT_FEATURES):
            result[name][column] = frame.ret_5m * (i + 1)
    return result


@pytest.mark.parametrize('margin_passed', [False, True])
def test_context_evaluation_preserves_old_scores_losses_and_both_paired_draws(tmp_path, margin_passed):
    make_utility(tmp_path, margin_passed)
    reference, out = tmp_path/'out', tmp_path/'context'
    out.mkdir()
    original = pd.read_parquet(tmp_path/'weekly'/'diagnosis_used.parquet')
    diagnosis = synthetic_context({'diagnosis': original}, None)['diagnosis']

    class FixedScore:
        features = ContextCloseModel.features

        def probabilities(self, values):
            return np.where((values[:, 0] < 1) & (values[:, self.features.index('direction')] > 0), .8, .5)[:, None]

    decision = evaluate_context(reference, diagnosis, FixedScore(), out)
    assert len(decision['checks']) == 19 and not decision['trading_returns_evaluated']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='context_score'),
        pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    for name in ['metrics', 'probability_metrics', 'first_metrics']:
        current, prior = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert {k: v for k, v in current.items() if k != 'context'} == prior
    for name in ['breakdown', 'probability_breakdown', 'first_breakdown']:
        current, prior = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert [v for v in current if v['model'] != 'context'] == prior
    for prior in reference.glob('positions_*.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(prior), pd.read_parquet(out/prior.name), check_exact=True)
    for name in ['block_draws', 'utility_block_draws']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'),
            pd.read_parquet(reference/'block_draws.parquet'), check_exact=True)
    metrics = json.loads((out/'first_metrics.json').read_text())['context']
    assert metrics['selected_positions'] > 0 and metrics['selected_negative'] > 0
    assert metrics['positions'] > metrics['selected_positions'] and not decision['context_admitted']
    positions = pd.read_parquet(out/'positions_context.parquet')
    assert positions.loc[positions.chosen, 'first_cost_score'].eq(.8).all()
    assert positions.loc[~positions.chosen, 'first_effect_pnl'].eq(0).all()


def test_full_context_pipeline_keeps_cost_prior_rows_and_rejects_resigned_old_score(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    exposure = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    capacity = run_close_capacity_diagnosis(exposure, tmp_path/'capacity')
    weekly = run_weekly_close_diagnosis(capacity, tmp_path/'weekly')
    margin = run_first_close_margin_diagnosis(weekly, tmp_path/'margin')
    utility = run_close_utility_diagnosis(margin, tmp_path/'utility')
    # 합성 원장의 시장값은 실제 시세가 아니며 시세 연결은 별도 검사에서 검증한다.
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.close_context_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.attach_close_context', synthetic_context)
    out = run_close_context_diagnosis(utility, tmp_path/'context')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced']
    assert summary['original_cost_ledger_and_prior_preserved'] and not summary['profitability_accepted']
    assert len(summary['checks']) == 19
    for name in ['training', 'diagnosis']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_used.parquet').drop(columns=CONTEXT_FEATURES),
            pd.read_parquet(utility/f'{name}_used.parquet'), check_exact=True)
    for name in ['training_cost_ledger', 'training_weights', 'diagnosis_weights', 'exclusion_ledger']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'),
            pd.read_parquet(utility/f'{name}.parquet'), check_exact=True)
    for name in ['model', 'training_support', 'decision']:
        assert json.loads((out/f'previous_{name}.json').read_text()) == json.loads((utility/f'{name}.json').read_text())
    damaged = pd.read_parquet(utility/'predictions.parquet')
    damaged.loc[0, 'utility_score'] = 1 - damaged.loc[0, 'utility_score']
    damaged.to_parquet(utility/'predictions.parquet', index=False)
    with pytest.raises(ValueError, match='지문'):
        reproduce_utility(utility, tmp_path/'invalid_hash')
    files = json.loads((utility/'files.json').read_text())
    files['predictions.parquet'] = sha256(utility/'predictions.parquet')
    save_json(utility/'files.json', files)
    with pytest.raises(AssertionError):
        reproduce_utility(utility, tmp_path/'resigned')
