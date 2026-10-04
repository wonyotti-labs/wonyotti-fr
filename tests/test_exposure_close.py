import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_continuation_diagnostics import fixture
from test_first_state import fixed_budget  # noqa: F401
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.exposure_close import (
    EXPOSURE_FEATURE,
    ExposureCloseModel,
    current_exposure,
    exposure_training,
)
from wonyotti_fr.exposure_close_diagnostics import (
    exposure_admission,
    reproduce_continuation,
    run_exposure_close_diagnosis,
)


def examples():
    rng = np.random.default_rng(65)
    x = np.c_[rng.normal(size=(600, 52)), np.tile(np.eye(4), (150, 1))]
    x[:, ExposureCloseModel.features.index(EXPOSURE_FEATURE)] = np.geomspace(.002, .25, len(x))
    exposure = current_exposure(x)
    y = exposure*(20*x[:, 0]+10*x[:, -1])+rng.normal(size=len(x))
    return x, y, np.linspace(.5, 1.5, len(x))


def test_unit_transform_preserves_original_account_error_for_arbitrary_predictions():
    x, y, w = examples()
    exposure = current_exposure(x)
    unit_target, unit_weight, norm = exposure_training(x, y, w)
    np.testing.assert_array_equal(unit_target, y/exposure)
    np.testing.assert_allclose(unit_weight, w*exposure*exposure/np.mean(w*exposure*exposure), rtol=1e-15)
    assert unit_weight.mean() == pytest.approx(1.)
    for unit_prediction in [np.zeros(len(x)), np.linspace(-100, 100, len(x)), y/exposure]:
        np.testing.assert_allclose(w*(y-exposure*unit_prediction)**2,
            norm*unit_weight*(unit_target-unit_prediction)**2, rtol=2e-14, atol=1e-12)
    assert (unit_target < 0).sum() == (y < 0).sum()


def test_model_matches_independent_weighted_unit_refit_and_account_predictions():
    x, y, w = examples()
    exposure = current_exposure(x)
    model, support = ExposureCloseModel.fit(x, y, w, x[:200])
    raw = w*exposure**2
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(x, y/exposure, sample_weight=raw/raw.mean())
        np.testing.assert_allclose(model.unit_model.predict(x), learner.predict(x), rtol=0, atol=1e-10)
        np.testing.assert_allclose(model.predict(x), learner.predict(x)*exposure, rtol=0, atol=1e-10)
    assert support['training_weight_normalizer'] == raw.mean()
    np.testing.assert_array_equal(ExposureCloseModel.from_dict(model.to_dict()).predict(x), model.predict(x))
    damaged = model.to_dict()
    damaged['transform']['prediction'] = 'unit_prediction'
    with pytest.raises(ValueError):
        ExposureCloseModel.from_dict(damaged)


def test_diagnosis_changes_do_not_fit_scale_normalizer_or_unit_model():
    x, y, w = examples()
    model, support = ExposureCloseModel.fit(x, y, w, x[:200])
    changed = x[:200].copy()
    changed[:, :52] *= -500
    changed[:, ExposureCloseModel.features.index(EXPOSURE_FEATURE)] = .9
    changed[:, -4:] = np.roll(changed[:, -4:], 1, axis=1)
    other, other_support = ExposureCloseModel.fit(x, y, w, changed)
    assert other.to_dict() == model.to_dict()
    assert other_support['training_weight_normalizer'] == support['training_weight_normalizer']
    frame = pd.DataFrame(x, columns=model.features)
    before = current_exposure(frame[model.features])
    frame['future_pnl'], frame['future_price'], frame['future_exposure'] = -1e8, 1e8, 1000.
    np.testing.assert_array_equal(current_exposure(frame[model.features]), before)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf, 1e-250, 1e250])
def test_nonpositive_exposure_and_conversion_underflow_or_overflow_fail_closed(bad):
    x, y, w = examples()
    x[0, ExposureCloseModel.features.index(EXPOSURE_FEATURE)] = bad
    with pytest.raises(ValueError):
        exposure_training(x, y, w)


def test_bad_weights_targets_and_prediction_overflow_fail_closed():
    x, y, w = examples()
    for bad_y, bad_w in [(y[:-1], w), (y*np.nan, w), (y, w*2), (y, w*0), (y, w*np.inf)]:
        with pytest.raises(ValueError):
            exposure_training(x, bad_y, bad_w)
    model, _ = ExposureCloseModel.fit(x, y, w, x[:200])
    bad = model.to_dict()
    bad['unit_model']['baseline'] = 1e308
    model = ExposureCloseModel.from_dict(bad)
    x[:, ExposureCloseModel.features.index(EXPOSURE_FEATURE)] = 10.
    with pytest.raises(ValueError, match='예측 넘침'):
        model.predict(x)


def test_all_fourteen_conditions_apply_to_original_account_and_first_position_metrics():
    base = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    references = ['continuation', 'economic', 'boosted', 'ridge', 'constant']
    metrics = {k: {**base, 'weighted_mse': 100., 'mse': 100.} for k in references}
    metrics['exposure'] = base
    first = {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.}
    result = exposure_admission(metrics, first)
    assert result['exposure_scaling_admitted'] and len(result['checks']) == 14
    cases = [(k, 'weighted_mse', 50.) for k in references]+[(k, 'mse', 49.) for k in references[:-1]]
    cases += [('exposure', 'selected', 99), ('exposure', 'selected_positions', 29),
        ('exposure', 'selected_weighted_mean_bps', 0.), ('exposure', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        altered, f = copy.deepcopy(metrics), dict(first)
        altered[name][field] = value
        if field == 'selected_positions':
            f[field] = value
        assert not exposure_admission(altered, f)['exposure_scaling_admitted']
    assert not exposure_admission(metrics, {**first, 'all_position_mean_common_bps': 0.})['exposure_scaling_admitted']
    with pytest.raises(ValueError):
        exposure_admission(metrics, {**first, 'selected_positions': 29})


def test_full_pipeline_reproduces_all_previous_results_and_preserves_original_evaluation(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    reference = run_continuation_diagnosis(first, tmp_path/'continuation')
    out = run_exposure_close_diagnosis(reference, tmp_path/'scaled')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_original_rows_and_evaluation_weights_preserved']
    assert not summary['profitability_accepted'] and not summary['trading_returns_evaluated']
    assert len(summary['checks']) == 14
    for name in ['training_used', 'diagnosis_used', 'training_weights', 'diagnosis_weights', 'exclusion_ledger', 'positions_continuation']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/f'{name}.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='predicted_exposure'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    previous = json.loads((reference/'previous_models.json').read_text())
    previous['continuation'] = json.loads((reference/'model.json').read_text())
    assert previous == json.loads((out/'previous_models.json').read_text())
    transform = pd.read_parquet(out/'training_transformation.parquet')
    np.testing.assert_array_equal(transform.unit_target_bps, transform.close_advantage_bps/transform.current_gross_exposure)
    before = pd.read_parquet(reference/'predictions.parquet')
    before.loc[0, 'predicted_continuation'] += 1
    before.to_parquet(reference/'predictions.parquet', index=False)
    mapping = json.loads((reference/'files.json').read_text())
    mapping['predictions.parquet'] = sha256(reference/'predictions.parquet')
    save_json(reference/'files.json', mapping)
    with pytest.raises(AssertionError):
        reproduce_continuation(reference, tmp_path/'tampered')
