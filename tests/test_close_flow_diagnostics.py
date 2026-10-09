import json

import numpy as np
import pandas as pd
import pytest
from test_close_context_diagnostics import (
    synthetic_context,
)
from test_close_context_diagnostics import (
    test_context_evaluation_preserves_old_scores_losses_and_both_paired_draws as make_context,
)
from test_continuation_diagnostics import fixture
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_context_diagnostics import evaluate_context, run_close_context_diagnosis
from wonyotti_fr.close_flow import FLOW_FEATURES, FLOW_WINDOWS, FlowCloseModel, flow_close_admission
from wonyotti_fr.close_flow_diagnostics import reproduce_context, run_close_flow_diagnosis
from wonyotti_fr.close_utility_diagnostics import run_close_utility_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_margin_diagnostics import run_first_close_margin_diagnosis
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis


def synthetic_flow(rows, _bars):
    output = {}
    for name, frame in rows.items():
        part = frame.copy()
        for i, feature in enumerate(FLOW_WINDOWS):
            part[feature] = np.tanh(frame.ret_5m/(i+1))
        for feature in FLOW_WINDOWS:
            part['directional_'+feature] = part[feature]*part.direction
        output[name] = part
    return output


@pytest.mark.parametrize('margin_passed', [False, True])
def test_flow_evaluation_keeps_all_previous_scores_and_losses_with_three_identical_draws(tmp_path, margin_passed):
    make_context(tmp_path, margin_passed)
    reference, out = tmp_path/'context', tmp_path/'flow'
    out.mkdir()
    original = pd.read_parquet(tmp_path/'weekly'/'diagnosis_used.parquet')
    diagnosis = synthetic_flow(synthetic_context({'diagnosis': original}, None), None)['diagnosis']

    class FixedScore:
        features = FlowCloseModel.features

        def probabilities(self, values):
            return np.where((values[:, 0] < 1) & (values[:, self.features.index('direction')] < 0), .8, .5)[:, None]

    decision = evaluate_context(reference, diagnosis, FixedScore(), out, candidate_name='flow',
        comparisons=('continuation', 'utility', 'context'), admission=flow_close_admission)
    assert len(decision['checks']) == 24 and not decision['trading_returns_evaluated']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='flow_score'),
        pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    for name in ['metrics', 'probability_metrics', 'first_metrics']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert {k: v for k, v in new.items() if k != 'flow'} == old
    for name in ['breakdown', 'probability_breakdown', 'first_breakdown']:
        new, old = (json.loads((folder/f'{name}.json').read_text()) for folder in [out, reference])
        assert [v for v in new if v['model'] != 'flow'] == old
    for prior in reference.glob('positions_*.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(prior), pd.read_parquet(out/prior.name), check_exact=True)
    for name in ['block_draws', 'utility_block_draws', 'context_block_draws']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'),
            pd.read_parquet(reference/'block_draws.parquet'), check_exact=True)
    metrics = json.loads((out/'first_metrics.json').read_text())['flow']
    assert metrics['selected_positions'] == metrics['selected_negative'] == 20
    assert metrics['positions'] == 40 and not decision['flow_admitted']
    with pytest.raises(ValueError, match='덮어쓰기'):
        evaluate_context(out, diagnosis, FixedScore(), out, candidate_name='flow',
            comparisons=('continuation', 'utility', 'context'), admission=flow_close_admission)


def test_full_flow_pipeline_preserves_original64_costs_and_rejects_resigned_context_score(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    exposure = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    capacity = run_close_capacity_diagnosis(exposure, tmp_path/'capacity')
    weekly = run_weekly_close_diagnosis(capacity, tmp_path/'weekly')
    margin = run_first_close_margin_diagnosis(weekly, tmp_path/'margin')
    utility = run_close_utility_diagnosis(margin, tmp_path/'utility')
    # 합성 시세 연결을 분리하고 원래 전체 모델과 판정 재현을 검사한다.
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.close_context_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.close_context_diagnostics.attach_close_context', synthetic_context)
    context = run_close_context_diagnosis(utility, tmp_path/'context')
    monkeypatch.setattr('wonyotti_fr.close_flow_diagnostics.close_flow_source', lambda _r: (None, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.close_flow_diagnostics.attach_close_flow', synthetic_flow)
    out = run_close_flow_diagnosis(context, tmp_path/'flow')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced']
    assert summary['original_cost_ledger_and_prior_preserved'] and not summary['profitability_accepted']
    assert len(summary['checks']) == 24
    for name in ['training', 'diagnosis']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_used.parquet').drop(columns=FLOW_FEATURES),
            pd.read_parquet(context/f'{name}_used.parquet'), check_exact=True)
    for name in ['training_cost_ledger', 'training_weights', 'diagnosis_weights', 'exclusion_ledger']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'),
            pd.read_parquet(context/f'{name}.parquet'), check_exact=True)
    for name in ['model', 'training_support', 'decision']:
        assert json.loads((out/f'previous_{name}.json').read_text()) == json.loads((context/f'{name}.json').read_text())
    damaged = pd.read_parquet(context/'predictions.parquet')
    damaged.loc[0, 'context_score'] = 1-damaged.loc[0, 'context_score']
    damaged.to_parquet(context/'predictions.parquet', index=False)
    with pytest.raises(ValueError, match='지문'):
        reproduce_context(context, tmp_path/'invalid_hash')
    files = json.loads((context/'files.json').read_text())
    files['predictions.parquet'] = sha256(context/'predictions.parquet')
    save_json(context/'files.json', files)
    with pytest.raises(AssertionError):
        reproduce_context(context, tmp_path/'resigned')
