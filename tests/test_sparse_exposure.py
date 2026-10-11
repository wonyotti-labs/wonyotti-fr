import copy
import json

import numpy as np
import pytest

from wonyotti_fr.sparse_exposure import (
    ExecutionMagnitudeModel,
    SparseExposureModel,
    action_labels,
    choose_actions,
    linear_baseline,
)
from wonyotti_fr.sparse_exposure_evaluation import diagnostic_metrics
from wonyotti_fr.sparse_exposure_reference import (
    independent_refit,
    raw_forest,
    reference_metrics,
    verify_metrics,
)


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(49)
    x, vx = rng.normal(size=(1800, 15)), rng.normal(size=(95, 15))
    y = np.where(x[:, 0] < -1, -np.abs(.2+x[:, 1]/10), np.where(x[:, 0] > 1, np.abs(.3+x[:, 2]/10), 0.))
    model, support = SparseExposureModel.fit(x, y, vx)
    linear, probability = linear_baseline(x, action_labels(y), vx)
    return x, y, vx, model, support, linear, probability


def test_full_model_independent_refit_roundtrip_and_single_state(fitted):
    x, y, vx, model, _, linear, linear_p = fitted
    restored = SparseExposureModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    expected = independent_refit(restored.to_dict(), linear, x, y, vx)
    np.testing.assert_allclose(expected['linear'], linear_p, atol=1e-12, rtol=0)
    batch = restored.predict(vx)
    for key, value in batch.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0)
    for i, row in enumerate(vx):
        single = restored.predict_one(row[:-1], float(row[-1]))
        for key, value in single.items():
            np.testing.assert_allclose(value, batch['actions' if key == 'action' else key][i], atol=1e-12, rtol=0)
    for side in ['negative', 'positive']:
        np.testing.assert_allclose(np.maximum(raw_forest(restored.to_dict()[side], vx), 0), batch[side+'_size'], atol=1e-12, rtol=0)


def test_future_inputs_do_not_change_training(fitted):
    x, y, vx, model, _, linear, _ = fitted
    before_x, before_y = x.copy(), y.copy()
    changed, _ = SparseExposureModel.fit(x, y, vx*37+23)
    changed_linear, _ = linear_baseline(x, action_labels(y), vx*37+23)
    assert changed.to_dict() == model.to_dict() and changed_linear == linear
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


def test_zero_boundary_and_exact_probability_ties():
    np.testing.assert_array_equal(action_labels([-2e-12, -1e-12, 0, 1e-12, 2e-12]), [0, 1, 1, 1, 2])
    np.testing.assert_array_equal(choose_actions([[.5, 0, .5], [.4, .4, .2], [.2, .4, .4], [.6, .2, .2], [.2, .2, .6]]), [0, 0, 0, -1, 1])


@pytest.mark.parametrize('probability', [[[0, 0, 0]], [[.5, .5, np.nan]], [[-1, 1, 1]], [[.5, .5]], [1, 0, 0]])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError):
        choose_actions(probability)


@pytest.mark.parametrize('case', ['extra', 'missing', 'format', 'bool', 'cycle', 'index', 'nonfinite', 'settings', 'size'])
def test_bad_numeric_models_rejected(fitted, case):
    raw = copy.deepcopy(fitted[3].to_dict())
    tree = raw['classifier']['models'][0]['trees'][0]
    if case == 'extra':
        tree['extra'] = []
    elif case == 'missing':
        del raw['negative']['trees']
    elif case == 'format':
        raw['format'] = 'other'
    elif case == 'bool':
        tree['value'][0] = True
    elif case == 'cycle':
        tree['left'][0] = 0
    elif case == 'index':
        tree['feature'][0] = 15
    elif case == 'nonfinite':
        raw['positive']['baseline'] = float('nan')
    elif case == 'settings':
        raw['classifier']['settings']['class_weight'] = 'balanced'
    elif case == 'size':
        raw['positive']['trees'].pop()
    with pytest.raises(ValueError):
        SparseExposureModel.from_dict(raw)


def test_directional_minimum_support_and_zero_size_clipping(fitted):
    x, _, vx, model, *_ = fitted
    with pytest.raises(ValueError):
        ExecutionMagnitudeModel.fit(x[:127], np.ones(127), vx)
    with pytest.raises(ValueError):
        ExecutionMagnitudeModel.fit(x[:128], np.zeros(128), vx)
    supported, details = ExecutionMagnitudeModel.fit(x[:128], np.ones(128), vx)
    assert details['training_rows'] == 128
    np.testing.assert_array_equal(supported.predict(vx), np.ones(len(vx)))
    raw = model.to_dict()
    for side in ['negative', 'positive']:
        raw[side]['baseline'] = -100
    restored = SparseExposureModel.from_dict(raw)
    for key in ['negative_size', 'positive_size', 'expected_change', 'decision_change']:
        np.testing.assert_array_equal(restored.predict(vx)[key], np.zeros(len(vx)))
    with pytest.raises(ValueError):
        SparseExposureModel.fit(x, np.zeros(len(x)), vx)


def test_metrics_with_ties_and_absent_direction():
    y = np.array([-.2, 0, .3, 0, .8, -.5])
    p = np.array([[.6, .2, .2], [.2, .6, .2], [.2, .6, .2], [.5, 0, .5], [.2, .2, .6], [.2, .2, .6]])
    probabilities = {'candidate': p, 'frequency': np.tile([.2, .6, .2], (len(y), 1))}
    changes = {'zero': np.zeros(len(y)), 'candidate': np.arange(len(y))/10}
    sizes = {0: {'median': np.full(len(y), .2)}, 2: {'median': np.full(len(y), .3)}}
    for mask in [np.ones(len(y), dtype=bool), y >= 0, y == 0]:
        args = (y[mask], {k: v[mask] for k, v in probabilities.items()}, {k: v[mask] for k, v in changes.items()},
            {k: {n: v[mask] for n, v in part.items()} for k, part in sizes.items()})
        verify_metrics(diagnostic_metrics(*args), reference_metrics(*args))


@pytest.mark.parametrize('bad', [np.full((1, 15), np.nan), np.zeros((1, 14)), np.zeros(15)])
def test_invalid_account_inputs_rejected(fitted, bad):
    with pytest.raises(ValueError):
        fitted[3].predict(bad)
