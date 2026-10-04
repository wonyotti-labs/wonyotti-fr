import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_continuation_diagnostics import fixture
from test_first_close_diagnostics import financial_examples
from test_first_state import fixed_budget  # noqa: F401
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_calibration import calibration_splits
from wonyotti_fr.close_capacity import (
    CAPACITY_GRID,
    capacity_class,
    capacity_from_dict,
    choose_capacity,
    fit_capacity_selection,
)
from wonyotti_fr.close_capacity_diagnostics import (
    capacity_admission,
    reproduce_exposure,
    run_close_capacity_diagnosis,
)
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.continuation_inputs import PENDING_FEATURES, ContinuationCloseModel, pending_values
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis


def values():
    rng = np.random.default_rng(66)
    x = np.c_[rng.normal(size=(1600, 52)), np.tile(np.eye(4), (400, 1))]
    y = 20*x[:, 0]**2+10*x[:, 1]**2+x[:, 2]*x[:, 3]*5
    return x, y, np.linspace(.5, 1.5, len(x))


@pytest.mark.parametrize('option', CAPACITY_GRID)
def test_all_four_numeric_models_match_library_and_enforce_their_fixed_structure(option):
    x, y, w = values()
    cls = capacity_class(option['name'])
    model, _ = cls.fit(x, y, w, x[:200])
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingRegressor(**option['settings']).fit(x, y, sample_weight=w)
        np.testing.assert_allclose(model.predict(x), learner.predict(x), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(capacity_from_dict(model.to_dict()).predict(x), model.predict(x))
    assert len(model.data['trees']) == option['settings']['max_iter']
    if option['settings']['max_depth'] == 4:
        assert any(len(t['left']) > 7 for t in model.data['trees'])
    bad = model.to_dict()
    bad['settings']['max_depth'] = 3
    with pytest.raises(ValueError):
        capacity_from_dict(bad)
    bad = model.to_dict()
    bad['trees'].pop()
    with pytest.raises(ValueError):
        capacity_from_dict(bad)


def test_original_default_model_keeps_identical_trees_predictions_and_fixed_format():
    x, y, w = values()
    old, _ = ContinuationCloseModel.fit(x, y, w, x[:200])
    base, _ = capacity_class('depth2_iter64').fit(x, y, w, x[:200])
    assert old.data['trees'] == base.data['trees'] and old.data['baseline'] == base.data['baseline']
    np.testing.assert_array_equal(old.predict(x), base.predict(x))
    assert old.data['format'] == 'close_continuation_histogram_v1'
    with pytest.raises(ValueError):
        ContinuationCloseModel.from_dict(base.to_dict())
    with pytest.raises(ValueError):
        capacity_class('depth8_iter1024')


def test_selection_uses_only_fixed_weighted_error_with_declared_tie_order():
    names = [r['name'] for r in CAPACITY_GRID]
    metrics = {n: {'weighted_mse': 5., 'selected_mean_bps': i*100.} for i, n in enumerate(names)}
    assert choose_capacity(metrics) == names[0]
    metrics[names[-1]]['weighted_mse'] = 4.
    assert choose_capacity(metrics) == names[-1]
    metrics[names[-1]]['weighted_mse'] = np.nan
    with pytest.raises(ValueError):
        choose_capacity(metrics)
    with pytest.raises(ValueError):
        choose_capacity({})


def internal_frames():
    ledger = financial_examples()
    ledger['current_gross_exposure'] = .25
    ledger['current_exit_net_bps'] = ledger.favorable_move*10000
    ledger[PENDING_FEATURES] = pending_values(ledger.original_intent)
    return ledger


def test_future_diagnosis_does_not_change_internal_models_selection_or_final_refit(tmp_path):
    ledger = internal_frames()
    rows, _ = calibration_splits(ledger)
    original = tmp_path/'original'
    original.mkdir()
    selected = fit_capacity_selection(rows['training'], rows['calibration'], original)
    changed = ledger.copy()
    future = changed.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC'))
    changed.loc[future, 'close_advantage_bps'] *= -100
    changed.loc[future, 'favorable_move'] += 20
    other, _ = calibration_splits(changed)
    destination = tmp_path/'changed'
    destination.mkdir()
    assert selected == fit_capacity_selection(other['training'], other['calibration'], destination)
    for name in ['selection_models', 'selection', 'selection_support', 'selection_metrics']:
        assert json.loads((original/f'{name}.json').read_text()) == json.loads((destination/f'{name}.json').read_text())
    x, y, w = values()
    cls = capacity_class(selected)
    a, _ = cls.fit(x, y, w, rows['diagnosis'][cls.features])
    b, _ = cls.fit(x, y, w, other['diagnosis'][cls.features])
    assert a.to_dict() == b.to_dict()
    different = rows['calibration'].copy()
    different['close_advantage_bps'] *= -100
    different['favorable_move'] += 10
    third = tmp_path/'selection_changed'
    third.mkdir()
    fit_capacity_selection(rows['training'], different, third)
    assert json.loads((original/'selection_models.json').read_text()) == json.loads((third/'selection_models.json').read_text())
    with pytest.raises(ValueError):
        fit_capacity_selection(rows['training'], rows['training'], tmp_path)


def test_all_sixteen_conditions_retain_account_effect_and_first_choice_requirements():
    base = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    refs = ['exposure', 'continuation', 'economic', 'boosted', 'ridge', 'constant']
    metrics = {k: {**base, 'weighted_mse': 100., 'mse': 100.} for k in refs}
    metrics['capacity'] = base
    first = {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.}
    result = capacity_admission(metrics, first)
    assert result['capacity_admitted'] and len(result['checks']) == 16
    cases = [(k, 'weighted_mse', 50.) for k in refs]+[(k, 'mse', 49.) for k in refs[:-1]]
    cases += [('capacity', 'selected', 99), ('capacity', 'selected_positions', 29),
        ('capacity', 'selected_weighted_mean_bps', 0.), ('capacity', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        altered, f = copy.deepcopy(metrics), dict(first)
        altered[name][field] = value
        if field == 'selected_positions':
            f[field] = value
        assert not capacity_admission(altered, f)['capacity_admitted']
    assert not capacity_admission(metrics, {**first, 'all_position_mean_common_bps': 0.})['capacity_admitted']


def test_full_pipeline_preserves_all_prior_outputs_and_refingerprinted_tampering_is_rejected(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    reference = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    out = run_close_capacity_diagnosis(reference, tmp_path/'capacity')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and not summary['profitability_accepted'] and len(summary['checks']) == 16
    for name in ['training_used', 'diagnosis_used', 'training_weights', 'diagnosis_weights', 'exclusion_ledger', 'positions_continuation']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/f'{name}.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='predicted_capacity'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    selected = json.loads((out/'selection.json').read_text())
    assert not selected['final_diagnosis_used_for_selection']
    assert capacity_from_dict(json.loads((out/'model.json').read_text())).data['settings'] == capacity_class(selected['candidate']).settings
    metrics = json.loads((reference/'metrics.json').read_text())
    metrics['exposure']['weighted_mse'] += 1
    save_json(reference/'metrics.json', metrics)
    hashes = json.loads((reference/'files.json').read_text())
    hashes['metrics.json'] = sha256(reference/'metrics.json')
    save_json(reference/'files.json', hashes)
    with pytest.raises(ValueError, match='재현 불일치'):
        reproduce_exposure(reference, tmp_path/'tampered')
