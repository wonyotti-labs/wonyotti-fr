import json

import numpy as np
import pandas as pd
import pytest
from test_close_context_diagnostics import synthetic_context
from test_close_flow_diagnostics import synthetic_flow
from test_close_flow_diagnostics import (
    test_flow_evaluation_keeps_all_previous_scores_and_losses_with_three_identical_draws as make_flow,
)
from test_continuation_diagnostics import fixture
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_context_diagnostics import evaluate_close_scores, run_close_context_diagnosis
from wonyotti_fr.close_flow_diagnostics import run_close_flow_diagnosis
from wonyotti_fr.close_utility_diagnostics import run_close_utility_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_margin_diagnostics import run_first_close_margin_diagnosis
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis
from wonyotti_fr.weekly_flow import weekly_flow_admission
from wonyotti_fr.weekly_flow_diagnostics import (
    load_full_flow_ledger,
    reproduce_flow,
    run_weekly_flow_diagnosis,
)


@pytest.mark.parametrize('margin_passed', [False, True])
def test_weekly_scores_preserve_every_old_result_and_share_all_four_block_draws(tmp_path, margin_passed):
    make_flow(tmp_path, margin_passed)
    reference, out = tmp_path/'flow', tmp_path/'weekly_flow'
    out.mkdir()
    diagnosis = pd.read_parquet(tmp_path/'weekly'/'diagnosis_used.parquet')
    scores = {'weekly_flow': np.where(diagnosis.direction < 0, .8, .5), 'weekly_constant': np.full(len(diagnosis), .4)}
    options = {'candidate_name': 'weekly_flow', 'comparisons': ('continuation', 'utility', 'context', 'flow'), 'admission': weekly_flow_admission}
    decision = evaluate_close_scores(reference, diagnosis, scores, out, **options)
    assert len(decision['checks']) == 32 and not decision['trading_returns_evaluated']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns=['weekly_flow_score', 'weekly_constant_score']),
        pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    for name in ['metrics', 'probability_metrics', 'first_metrics']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert {k: v for k, v in new.items() if k not in scores} == old
    for name in ['breakdown', 'probability_breakdown', 'first_breakdown']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert [v for v in new if v['model'] not in scores] == old
    for prior in reference.glob('positions_*.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(prior), pd.read_parquet(out/prior.name), check_exact=True)
    for name in ['block_draws', 'utility_block_draws', 'context_block_draws', 'flow_block_draws']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/'block_draws.parquet'), check_exact=True)
    metrics = json.loads((out/'first_metrics.json').read_text())
    assert metrics['weekly_flow']['selected_positions'] == metrics['weekly_flow']['selected_negative'] == 20
    assert metrics['weekly_flow']['positions'] == metrics['weekly_constant']['positions'] == 40
    assert metrics['weekly_constant']['selected_positions'] == 0
    assert not decision['weekly_flow_admitted']
    with pytest.raises(ValueError, match='덮어쓰기'):
        evaluate_close_scores(out, diagnosis, scores, out, **options)
    with pytest.raises(ValueError, match='이름'):
        evaluate_close_scores(reference, diagnosis, {'weekly_flow': scores['weekly_flow']}, out, **options)


def test_full_weekly_pipeline_reconstructs_original_ledger_and_rejects_resigned_flow_score(tmp_path, monkeypatch):
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
    out = run_weekly_flow_diagnosis(flow, tmp_path/'new')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced']
    assert summary['first_week_model_costs_and_prior_exact'] and not summary['profitability_accepted']
    assert summary['weekly_models'] == 13 and len(summary['checks']) == 32
    original = pd.read_parquet(continuation/'continuation_ledger.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(out/'flow_ledger.parquet')[original.columns], original, check_exact=True)
    for name in ['training_used', 'diagnosis_used', 'training_weights', 'diagnosis_weights', 'training_cost_ledger', 'exclusion_ledger']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(flow/f'{name}.parquet'), check_exact=True)
    assert json.loads((out/'weekly_models.json').read_text())['week-00'] == json.loads((flow/'model.json').read_text())
    source = json.loads((out/'ledger_source.json').read_text())
    assert source['continuation'] == str(continuation) and source['rows'] == len(original)
    damaged = pd.read_parquet(flow/'predictions.parquet')
    damaged.loc[0, 'flow_score'] = 1-damaged.loc[0, 'flow_score']
    damaged.to_parquet(flow/'predictions.parquet', index=False)
    with pytest.raises(ValueError, match='지문'):
        reproduce_flow(flow, tmp_path/'invalid_hash')
    files = json.loads((flow/'files.json').read_text())
    files['predictions.parquet'] = sha256(flow/'predictions.parquet')
    save_json(flow/'files.json', files)
    with pytest.raises(AssertionError):
        reproduce_flow(flow, tmp_path/'resigned')
    original.loc[0, 'favorable_move'] += 1
    original.to_parquet(continuation/'continuation_ledger.parquet', index=False)
    with pytest.raises(ValueError, match='원장 오류'):
        load_full_flow_ledger(flow)
