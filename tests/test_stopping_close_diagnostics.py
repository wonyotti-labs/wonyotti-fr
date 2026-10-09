import json

import numpy as np
import pandas as pd
import pytest
from test_close_context_diagnostics import synthetic_context
from test_close_flow_diagnostics import synthetic_flow
from test_continuation_diagnostics import fixture
from test_first_close_diagnostics import financial_examples
from test_first_state import fixed_budget  # noqa: F401
from test_weekly_flow_diagnostics import (
    test_weekly_scores_preserve_every_old_result_and_share_all_four_block_draws as make_weekly,
)

from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_context_diagnostics import evaluate_close_scores, run_close_context_diagnosis
from wonyotti_fr.close_flow_diagnostics import run_close_flow_diagnosis
from wonyotti_fr.close_utility_diagnostics import run_close_utility_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_margin_diagnostics import run_first_close_margin_diagnosis
from wonyotti_fr.stopping_close import stopping_admission
from wonyotti_fr.stopping_close_diagnostics import (
    reproduce_weekly_flow,
    run_stopping_close_diagnosis,
)
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis
from wonyotti_fr.weekly_flow_diagnostics import run_weekly_flow_diagnosis


@pytest.mark.parametrize('margin_passed', [False, True])
def test_stopping_scores_keep_all_old_rows_losses_and_five_identical_block_draws(tmp_path, margin_passed):
    make_weekly(tmp_path, margin_passed)
    reference, out = tmp_path/'weekly_flow', tmp_path/'stopping'
    out.mkdir()
    diagnosis = pd.read_parquet(tmp_path/'weekly'/'diagnosis_used.parquet')
    scores = {'stopping_flow': np.where(diagnosis.direction < 0, .8, .5)}
    options = {'candidate_name': 'stopping_flow', 'comparisons': ('continuation', 'utility', 'context', 'flow', 'weekly_flow'), 'admission': stopping_admission}
    decision = evaluate_close_scores(reference, diagnosis, scores, out, **options)
    assert len(decision['checks']) == 37 and not decision['trading_returns_evaluated']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='stopping_flow_score'),
        pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    for name in ['metrics', 'probability_metrics', 'first_metrics']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert {k: v for k, v in new.items() if k != 'stopping_flow'} == old
    for name in ['breakdown', 'probability_breakdown', 'first_breakdown']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert [v for v in new if v['model'] != 'stopping_flow'] == old
    for prior in reference.glob('positions_*.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(prior), pd.read_parquet(out/prior.name), check_exact=True)
    for name in ['block_draws', 'utility_block_draws', 'context_block_draws', 'flow_block_draws', 'weekly_flow_block_draws']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/'block_draws.parquet'), check_exact=True)
    metrics = json.loads((out/'first_metrics.json').read_text())['stopping_flow']
    assert metrics['selected_positions'] == metrics['selected_negative'] == 20
    assert metrics['positions'] == 40 and not decision['stopping_flow_admitted']
    with pytest.raises(ValueError, match='덮어쓰기'):
        evaluate_close_scores(out, diagnosis, scores, out, **options)


def cash_examples():
    frame = financial_examples()
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    return frame


def test_full_stopping_pipeline_preserves_original_costs_and_rejects_resigned_weekly_score(tmp_path, monkeypatch):
    monkeypatch.setattr('test_continuation_diagnostics.financial_examples', cash_examples)
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    exposure = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    capacity = run_close_capacity_diagnosis(exposure, tmp_path/'capacity')
    weekly = run_weekly_close_diagnosis(capacity, tmp_path/'weekly')
    margin = run_first_close_margin_diagnosis(weekly, tmp_path/'margin')
    utility = run_close_utility_diagnosis(margin, tmp_path/'utility')
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.close_context_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.attach_close_context', synthetic_context)
    context = run_close_context_diagnosis(utility, tmp_path/'context')
    monkeypatch.setattr('wonyotti_fr.close_flow_diagnostics.close_flow_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.close_flow_diagnostics.attach_close_flow', synthetic_flow)
    flow = run_close_flow_diagnosis(context, tmp_path/'flow')
    monkeypatch.setattr('wonyotti_fr.weekly_flow_diagnostics.close_flow_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.weekly_flow_diagnostics.attach_close_context', synthetic_context)
    monkeypatch.setattr('wonyotti_fr.weekly_flow_diagnostics.attach_close_flow', synthetic_flow)
    reference = run_weekly_flow_diagnosis(flow, tmp_path/'weekly_flow')
    out = run_stopping_close_diagnosis(reference, tmp_path/'new')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced'] and not summary['profitability_accepted']
    assert summary['original_training_cost_ledger_preserved'] and summary['teacher_whole_position_exclusion']
    assert summary['teacher_models'] == 5 and summary['candidate_models'] == 1 and len(summary['checks']) == 37
    for name in ['flow_ledger', 'training_used', 'diagnosis_used', 'training_weights', 'diagnosis_weights', 'training_cost_ledger', 'exclusion_ledger']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/f'{name}.parquet'), check_exact=True)
    assert json.loads((out/'previous_weekly_models.json').read_text()) == json.loads((reference/'weekly_models.json').read_text())
    targets = pd.read_parquet(out/'stopping_targets.parquet')
    source = pd.read_parquet(out/'flow_ledger.parquet').iloc[targets.opportunity_index].reset_index(drop=True)
    pd.testing.assert_frame_equal(targets[['decision_time', 'position_entry_time', 'close_advantage_bps']], source[['decision_time', 'position_entry_time', 'close_advantage_bps']], check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='stopping_flow_score'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    damaged = pd.read_parquet(reference/'predictions.parquet')
    damaged.loc[0, 'weekly_flow_score'] = 1-damaged.loc[0, 'weekly_flow_score']
    damaged.to_parquet(reference/'predictions.parquet', index=False)
    with pytest.raises(ValueError, match='지문'):
        reproduce_weekly_flow(reference, tmp_path/'invalid_hash')
    files = json.loads((reference/'files.json').read_text())
    files['predictions.parquet'] = sha256(reference/'predictions.parquet')
    save_json(reference/'files.json', files)
    with pytest.raises(AssertionError):
        reproduce_weekly_flow(reference, tmp_path/'resigned')
