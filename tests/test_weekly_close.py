import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_close_capacity import internal_frames
from test_continuation_diagnostics import fixture
from test_first_state import fixed_budget  # noqa: F401
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_learning_inputs import close_learning_splits
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.continuation_inputs import ContinuationCloseModel
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.weekly_close import fit_weekly_close, weekly_boundaries, weekly_training_rows
from wonyotti_fr.weekly_close_diagnostics import (
    reproduce_capacity,
    run_weekly_close_diagnosis,
    weekly_admission,
)


def setup():
    ledger = internal_frames()
    rows, _ = close_learning_splits(ledger)
    weight = position_weights(rows['training'])
    model, _ = ContinuationCloseModel.fit(rows['training'][ContinuationCloseModel.features],
        rows['training'].close_advantage_bps, weight, rows['diagnosis'][ContinuationCloseModel.features])
    return ledger, rows, weight, model


def test_weekly_calendar_and_strict_maturity_keep_unfinished_and_boundary_cases_out():
    ledger, rows, _, _ = setup()
    boundaries = weekly_boundaries()
    assert len(boundaries) == 13 and boundaries[0][0] == pd.Timestamp('2021-10-02', tz='UTC')
    assert boundaries[-1][1]-boundaries[-1][0] == pd.Timedelta(days=6)
    first, indices = weekly_training_rows(ledger, boundaries[0][0])
    pd.testing.assert_frame_equal(first, rows['training'], check_exact=True)
    altered = ledger.copy()
    altered['label_end'] = altered.label_end.astype('datetime64[ns, UTC]')
    index = indices[-1]
    altered.loc[index, 'label_end'] = boundaries[0][0]-pd.Timedelta(days=2)
    _, modified = weekly_training_rows(altered, boundaries[0][0])
    assert index not in modified
    altered.loc[index, 'label_end'] -= pd.Timedelta(nanoseconds=1)
    _, modified = weekly_training_rows(altered, boundaries[0][0])
    assert index in modified
    altered.loc[index, 'label_status'] = 'right_censored'
    _, modified = weekly_training_rows(altered, boundaries[0][0])
    assert index not in modified
    with pytest.raises(ValueError):
        weekly_training_rows(ledger, pd.Timestamp('2021-10-03', tz='UTC'))


def test_unavailable_future_changes_leave_current_training_weights_model_and_prediction_unchanged():
    ledger, _, _, _ = setup()
    time = weekly_boundaries()[3][0]
    before, indices = weekly_training_rows(ledger, time)
    changed = ledger.copy()
    future = ~changed.index.isin(indices)
    changed.loc[future, 'close_advantage_bps'] *= -100
    changed.loc[future, 'favorable_move'] += 10
    after, other_indices = weekly_training_rows(changed, time)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    np.testing.assert_array_equal(indices, other_indices)
    weight, other_weight = position_weights(before), position_weights(after)
    np.testing.assert_array_equal(weight, other_weight)
    features = ContinuationCloseModel.features
    a, _ = ContinuationCloseModel.fit(before[features], before.close_advantage_bps, weight, before[features].iloc[:20])
    b, _ = ContinuationCloseModel.fit(after[features], after.close_advantage_bps, other_weight, after[features].iloc[:20])
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.predict(before[features]), b.predict(before[features]))


def test_all_thirteen_refits_routing_first_model_and_weights_match_independent_calculation(tmp_path):
    ledger, rows, weight, model = setup()
    prediction = fit_weekly_close(ledger, rows['diagnosis'], rows['training'], weight, model.to_dict(), tmp_path)
    models = json.loads((tmp_path/'weekly_models.json').read_text())
    support = json.loads((tmp_path/'weekly_support.json').read_text())
    membership = pd.read_parquet(tmp_path/'weekly_training_membership.parquet')
    routing = pd.read_parquet(tmp_path/'prediction_routing.parquet')
    assert len(models) == len(support) == 13 and models['week-00'] == model.to_dict()
    pd.testing.assert_frame_equal(routing.drop(columns='model_key'), rows['diagnosis'][['decision_time', 'position_entry_time']], check_exact=True)
    for i, (start, end) in enumerate(weekly_boundaries()):
        key = f'week-{i:02}'
        mask = ledger.label_status.eq('closed') & ledger.label_end.lt(start-pd.Timedelta(days=2))
        training = ledger[mask]
        saved = membership[membership.model_key.eq(key)]
        np.testing.assert_array_equal(saved.opportunity_index, np.flatnonzero(mask))
        counts = training.position_entry_time.value_counts()
        independent_weight = np.array([len(training)/(len(counts)*counts[t]) for t in training.position_entry_time])
        np.testing.assert_allclose(saved.sample_weight, independent_weight, rtol=0, atol=1e-12)
        selected = rows['diagnosis'].decision_time.ge(start) & rows['diagnosis'].decision_time.lt(end)
        assert routing.loc[selected, 'model_key'].eq(key).all()
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(training[model.features].to_numpy(),
                training.close_advantage_bps, sample_weight=saved.sample_weight.to_numpy())
            if selected.any():
                expected = learner.predict(rows['diagnosis'].loc[selected, model.features].to_numpy())
                np.testing.assert_allclose(prediction[selected], expected, rtol=0, atol=1e-10)
    assert any(v['prediction_rows'] == 0 for v in support.values())


def test_unsupported_training_and_changed_first_model_are_rejected(tmp_path):
    ledger, rows, weight, model = setup()
    with pytest.raises(ValueError, match='지원 부족'):
        weekly_training_rows(ledger.iloc[:100], weekly_boundaries()[0][0])
    damaged = model.to_dict()
    damaged['baseline'] += 1
    with pytest.raises(ValueError, match='첫 학습 모델'):
        fit_weekly_close(ledger, rows['diagnosis'], rows['training'], weight, damaged, tmp_path)


def test_all_eighteen_conditions_include_every_reference_and_first_choice():
    base = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    refs = ['capacity', 'exposure', 'continuation', 'economic', 'boosted', 'ridge', 'constant']
    metrics = {k: {**base, 'weighted_mse': 100., 'mse': 100.} for k in refs}
    metrics['weekly'] = base
    first = {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.}
    result = weekly_admission(metrics, first)
    assert result['weekly_admitted'] and len(result['checks']) == 18
    cases = [(k, 'weighted_mse', 50.) for k in refs]+[(k, 'mse', 49.) for k in refs[:-1]]
    cases += [('weekly', 'selected', 99), ('weekly', 'selected_positions', 29),
        ('weekly', 'selected_weighted_mean_bps', 0.), ('weekly', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        altered, f = copy.deepcopy(metrics), dict(first)
        altered[name][field] = value
        if field == 'selected_positions':
            f[field] = value
        assert not weekly_admission(altered, f)['weekly_admitted']
    assert not weekly_admission(metrics, {**first, 'all_position_mean_common_bps': 0.})['weekly_admitted']


def test_full_pipeline_preserves_prior_outputs_and_rejects_resigned_selection_tampering(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    exposure = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    reference = run_close_capacity_diagnosis(exposure, tmp_path/'capacity')
    out = run_weekly_close_diagnosis(reference, tmp_path/'weekly')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and not summary['profitability_accepted'] and len(summary['checks']) == 18
    for name in ['training_used', 'diagnosis_used', 'training_weights', 'diagnosis_weights', 'exclusion_ledger', 'positions_continuation']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/f'{name}.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='predicted_weekly'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    selection = json.loads((reference/'selection.json').read_text())
    selection['candidate'] = 'depth4_iter256' if selection['candidate'] != 'depth4_iter256' else 'depth2_iter64'
    save_json(reference/'selection.json', selection)
    hashes = json.loads((reference/'files.json').read_text())
    hashes['selection.json'] = sha256(reference/'selection.json')
    save_json(reference/'files.json', hashes)
    with pytest.raises(ValueError, match='재현 불일치'):
        reproduce_capacity(reference, tmp_path/'tampered')
