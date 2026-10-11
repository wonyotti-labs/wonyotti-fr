from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.position_boost import PositionBoostModel
from wonyotti_fr.position_boost_evaluation import position_boost_admission


@pytest.fixture(scope='module')
def inputs():
    rng = np.random.default_rng(94)
    x = rng.normal(size=(1500, len(PositionBoostModel.features)))
    y = np.where(x[:, 0] * x[:, 1] > .3, 2, np.where(x[:, 2] > -.2, 1, 0))
    return x, y, rng.normal(size=(101, x.shape[1]))


@pytest.fixture(scope='module')
def fitted(inputs):
    return PositionBoostModel.fit(*inputs)


def test_independent_multiclass_library_and_all_trees(inputs, fitted):
    x, y, validation = inputs
    model, support = fitted
    with threadpool_limits(limits=1):
        reference = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y)
        expected = reference.predict_proba(validation)
    np.testing.assert_allclose(model.probabilities(validation), expected, rtol=0, atol=1e-12)
    assert support['trees_per_iteration'] == 3 and support['iterations'] == 64
    for index, item in enumerate(model.to_dict()['models']):
        assert item['baseline'] == reference._baseline_prediction[0, index]
        for iteration, tree in enumerate(item['trees']):
            nodes = reference._predictors[iteration][index].nodes
            leaf = nodes['is_leaf'].astype(bool)
            for key, expected_values in {
                'left': [-1 if end else int(value) for end, value in zip(leaf, nodes['left'], strict=True)],
                'right': [-1 if end else int(value) for end, value in zip(leaf, nodes['right'], strict=True)],
                'feature': [-2 if end else int(value) for end, value in zip(leaf, nodes['feature_idx'], strict=True)],
                'threshold': np.where(leaf, 0., nodes['num_threshold']),
                'value': np.where(leaf, nodes['value'], 0.),
            }.items():
                np.testing.assert_array_equal(tree[key], expected_values)


def test_export_copy_and_single_batch(fitted, inputs):
    model, _ = fitted
    payload = model.to_dict()
    restored = PositionBoostModel.from_dict(json.loads(json.dumps(payload)))
    matrix = inputs[2]
    np.testing.assert_array_equal(model.probabilities(matrix), restored.probabilities(matrix))
    np.testing.assert_array_equal(model.probabilities(matrix), np.vstack([model.probabilities(row[None])[0] for row in matrix]))
    payload['models'][0]['baseline'] += 7
    assert model.to_dict() != payload


def test_future_validation_inputs_do_not_change_fit(inputs, fitted):
    x, y, validation = inputs
    changed, _ = PositionBoostModel.fit(x, y, validation * -100)
    assert changed.to_dict() == fitted[0].to_dict()


def test_nonfinite_inputs_and_empty(fitted, inputs):
    model, _ = fitted
    matrix = inputs[2][:4].copy()
    matrix[1, 0], matrix[2, 1], matrix[3, 2] = np.nan, np.inf, -np.inf
    predictions = model.probabilities(matrix)
    assert np.isfinite(predictions[0]).all() and np.isnan(predictions[1:]).all()
    assert model.probabilities(matrix[:0]).shape == (0, 3)
    with pytest.raises(ValueError):
        model.probabilities(matrix[:, :-1])


@pytest.mark.parametrize('change', ['few_rows', 'few_classes', 'float_labels', 'nan_training', 'nan_validation', 'wrong_width'])
def test_fit_input_support_rejected(inputs, change):
    x, y, validation = (value.copy() for value in inputs)
    if change == 'few_rows':
        x, y = x[:999], y[:999]
    elif change == 'few_classes':
        y[y == 2] = 1
    elif change == 'float_labels':
        y = y.astype(float)
    elif change == 'nan_training':
        x[0, 0] = np.nan
    elif change == 'nan_validation':
        validation[0, 0] = np.nan
    else:
        x = x[:, :-1]
    with pytest.raises(ValueError):
        PositionBoostModel.fit(x, y, validation)


@pytest.mark.parametrize('change', ['format', 'class_order', 'settings', 'missing_tree', 'cycle', 'shared_child', 'bad_leaf', 'nonfinite'])
def test_serialized_model_tampering_rejected(fitted, change):
    data = fitted[0].to_dict()
    tree = data['models'][0]['trees'][0]
    if change == 'format':
        data['format'] = 'histogram_management_v1'
    elif change == 'class_order':
        data['models'][0], data['models'][1] = data['models'][1], data['models'][0]
    elif change == 'settings':
        data['settings']['max_depth'] = 3
    elif change == 'missing_tree':
        data['models'][0]['trees'].pop()
    elif change == 'cycle':
        tree['left'][0] = 0
    elif change == 'shared_child':
        tree['right'][0] = tree['left'][0]
    elif change == 'bad_leaf':
        tree['feature'][-1] = 0
    else:
        data['models'][0]['baseline'] = np.inf
    with pytest.raises(ValueError):
        PositionBoostModel.from_dict(data)


def test_branch_boundary_and_stable_softmax(fitted):
    data = fitted[0].to_dict()
    leaf = {'left': [-1], 'right': [-1], 'feature': [-2], 'threshold': [0.], 'value': [0.]}
    for item in data['models']:
        item['baseline'] = 1000
        item['trees'] = [copy.deepcopy(leaf) for _ in range(64)]
    data['models'][0]['trees'][0] = {'left': [1, -1, -1], 'right': [2, -1, -1],
        'feature': [0, -2, -2], 'threshold': [1., 0., 0.], 'value': [0., 2., -2.]}
    model = PositionBoostModel.from_dict(data)
    matrix = np.zeros((3, len(model.features)))
    matrix[:, 0] = [np.nextafter(1., -np.inf), 1., np.nextafter(1., np.inf)]
    expected = np.exp(np.array([[2., 0., 0.], [2., 0., 0.], [-2., 0., 0.]]))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.probabilities(matrix), expected, atol=1e-15, rtol=0)
    data['models'][0]['baseline'] = 1.7e308
    data['models'][0]['trees'][0]['value'][1] = 1.7e308
    assert np.isnan(PositionBoostModel.from_dict(data).probabilities(matrix[:1])).all()


def evidence():
    metrics = {name: {'log_loss': loss, 'short': {'average_precision': .6},
        'long': {'average_precision': .6}} for name, loss in [('position', .8), ('constant', .9), ('boosted', .7)]}
    months = [{'month': f'2021-{month:02}', 'position_log_loss': .8, 'constant_log_loss': .9,
        'boosted_log_loss': .7 if month <= 8 else 1.} for month in range(1, 13)]
    return metrics, months


def test_admission_does_not_declare_profitability():
    result = position_boost_admission(*evidence())
    assert result['position_boost_admitted'] and not result['profitability_accepted']
    assert result['improved_months'] == 8 and not result['trading_returns_evaluated']


@pytest.mark.parametrize('change', ['gain_boundary', 'linear_loss', 'short_precision', 'long_precision', 'seven_months'])
def test_every_admission_condition_is_required(change):
    metrics, monthly = evidence()
    if change == 'gain_boundary':
        metrics['boosted']['log_loss'] = metrics['position']['log_loss'] * .99
    elif change == 'linear_loss':
        metrics['position']['log_loss'] = .69
    elif change.endswith('_precision'):
        metrics['boosted'][change.split('_')[0]]['average_precision'] = .599
    else:
        monthly[7]['boosted_log_loss'] = 1.
    assert not position_boost_admission(metrics, monthly)['position_boost_admitted']


@pytest.mark.parametrize('change', ['missing_model', 'nan', 'negative', 'month_order'])
def test_admission_invalid_evidence_rejected(change):
    metrics, monthly = evidence()
    if change == 'missing_model':
        metrics.pop('position')
    elif change == 'month_order':
        monthly.reverse()
    else:
        metrics['boosted']['log_loss'] = np.nan if change == 'nan' else -1.
    with pytest.raises(ValueError):
        position_boost_admission(metrics, monthly)
