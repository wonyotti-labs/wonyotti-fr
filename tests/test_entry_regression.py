import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_first_state import fixed_budget  # noqa: F401
from test_label_weighting import training
from test_policy_entry import fixture as policy_fixture
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS, EntryRegressionModel
from wonyotti_fr.entry_regression_diagnostics import (
    regression_admission,
    regression_metrics,
    regression_splits,
    run_entry_regression_diagnosis,
)
from wonyotti_fr.net_edge_model import NET_FEATURES


def regression_data():
    rng = np.random.default_rng(57)
    values = rng.normal(size=(500, len(NET_FEATURES)))
    target = np.where(values[:, 0] > 0, 100., -100.)+values[:, 1]*30
    weights = rng.uniform(.1, 2., len(values))
    return values, target, weights/weights.mean()


def test_weighted_numeric_export_matches_library_batch_single_and_future_input_invariance():
    x, y, w = regression_data()
    model, support = EntryRegressionModel.fit(x, y, w, x[:100])
    with threadpool_limits(limits=1):
        expected = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(x, y, sample_weight=w)
        predictions = expected.predict(x)
    np.testing.assert_allclose(model.predict(x), predictions, rtol=0, atol=1e-10)
    np.testing.assert_allclose([model.predict(row[None, :])[0] for row in x[:10]], predictions[:10], rtol=0, atol=1e-10)
    assert model.data['baseline'] == pytest.approx(np.average(y, weights=w))
    changed, _ = EntryRegressionModel.fit(x, y, w, x[:100]*100)
    assert model.to_dict() == changed.to_dict() and support['early_stopping'] is False
    damaged = x[:2].copy()
    damaged[0, 0] = np.nan
    damaged[1, 1] = np.inf
    assert np.isnan(model.predict(damaged)).all()
    serialized = model.to_dict()
    restored = EntryRegressionModel.from_dict(serialized)
    serialized['baseline'] += 1
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


@pytest.mark.parametrize('damage', ['cycle', 'shared', 'range', 'leaf', 'feature', 'nan', 'count', 'settings'])
def test_numeric_model_rejects_unsafe_nodes_and_parameter_changes(damage):
    x, y, w = regression_data()
    model, _ = EntryRegressionModel.fit(x, y, w, x[:2])
    data = model.to_dict()
    tree = data['trees'][0]
    if damage == 'cycle':
        tree['left'][0] = 0
    elif damage == 'shared':
        tree['left'][0] = tree['right'][0]
    elif damage == 'range':
        tree['left'][0] = 99
    elif damage == 'leaf':
        i = tree['left'].index(-1)
        tree['feature'][i] = 0
    elif damage == 'feature':
        tree['feature'][0] = len(NET_FEATURES)
    elif damage == 'nan':
        tree['value'][0] = float('nan')
    elif damage == 'count':
        data['trees'].pop()
    else:
        data['settings']['early_stopping'] = True
    with pytest.raises(ValueError):
        EntryRegressionModel.from_dict(data)


@pytest.mark.parametrize('damage', ['zero', 'negative', 'size', 'nan', 'normalization'])
def test_weights_cannot_remove_losses_or_change_total_fit_strength(damage):
    x, y, w = regression_data()
    if damage == 'zero':
        w[0] = 0
    elif damage == 'negative':
        w[0] = -1
    elif damage == 'size':
        w = w[:-1]
    elif damage == 'nan':
        w[0] = np.nan
    else:
        w *= 2
    with pytest.raises(ValueError):
        EntryRegressionModel.fit(x, y, w, x[:2])


def combined_training():
    frame = training()
    future = frame.iloc[:100].copy()
    future['decision_time'] = pd.date_range('2021-10-02', periods=100, freq='h', tz='UTC')
    future['label_end'] = future.decision_time+pd.Timedelta(hours=1)
    boundary = frame.iloc[:2].copy()
    boundary['decision_time'] = pd.to_datetime(['2021-09-29T23:45Z', '2021-10-01T00:10Z'])
    boundary['label_end'] = boundary.decision_time+pd.Timedelta(hours=1)
    return pd.concat([frame, boundary, future], ignore_index=True)


def test_time_split_purges_crossing_boundaries_and_preserves_all_exclusions():
    frame = combined_training().assign(label_status='closed')
    tail = frame.iloc[-1:].copy()
    tail['decision_time'] += pd.Timedelta(hours=1)
    tail['label_end'] += pd.Timedelta(hours=1)
    tail['label_status'] = 'right_censored'
    frame = pd.concat([frame, tail], ignore_index=True)
    rows, ledger = regression_splits(frame)
    assert len(rows['training']) == 200 and len(rows['diagnosis']) == 100
    assert ledger.split.value_counts().to_dict() == {'training': 200, 'diagnosis': 100, 'excluded_boundary': 2, 'excluded_not_closed': 1}


def test_metrics_and_all_admission_conditions_are_required():
    actual = np.tile([20., 40., -10.], 20)
    metrics = regression_metrics(actual, np.full(60, 10.), np.linspace(.5, 1.5, 60))
    assert metrics['mse'] == pytest.approx(np.mean((actual-10)**2))
    assert metrics['weighted_mse'] == pytest.approx(np.average((actual-10)**2, weights=np.linspace(.5, 1.5, 60)))
    assert metrics['selected'] == 60 and metrics['selected_mean_bps'] > 0
    candidate = {'rows': 60, 'weighted_mse': 50., 'mse': 50., 'selected': 30,
                 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    values = {'boosted': candidate, 'ridge': {**candidate, 'weighted_mse': 100., 'mse': 100.},
              'constant': {**candidate, 'weighted_mse': 100., 'mse': 100.}}
    assert regression_admission(values)['boosted_admitted']
    for field, value in [('weighted_mse', 99.), ('mse', 101.), ('selected', 29),
                         ('selected_weighted_mean_bps', 0.), ('selected_mean_bps', 0.)]:
        damaged = copy.deepcopy(values)
        damaged['boosted'][field] = value
        assert not regression_admission(damaged)['boosted_admitted']
    assert regression_metrics(actual, np.zeros(60), np.ones(60))['selected_mean_bps'] is None


def test_complete_diagnosis_and_future_targets_do_not_change_training_or_models(tmp_path, monkeypatch):
    monkeypatch.setattr('test_policy_entry.training', combined_training)
    root, parent, labels, _, _ = policy_fixture(tmp_path, monkeypatch)
    save_json(root/'manifest.json', {'settings': {'labels': str(labels), 'reference': str(parent),
        'labels_files_sha256': sha256(labels/'files.json')}})
    out = run_entry_regression_diagnosis(root, tmp_path/'runs')
    saved = json.loads((out/'summary.json').read_text())
    assert saved['complete'] and saved['training_rows'] == 200 and saved['diagnosis_rows'] == 100
    original = regression_splits
    def changed_future(ledger):
        rows, assignments = original(ledger)
        rows['diagnosis']['net_bps'] *= -10
        return rows, assignments
    monkeypatch.setattr('wonyotti_fr.entry_regression_diagnostics.regression_splits', changed_future)
    altered = run_entry_regression_diagnosis(root, tmp_path/'future')
    for name in ['training_used.parquet', 'training_weights.parquet', 'models.json']:
        assert (out/name).read_bytes() == (altered/name).read_bytes()
    for name in ['training', 'diagnosis']:
        used = pd.read_parquet(out/f'{name}_used.parquet')
        assert used.net_bps.lt(0).any()
    assert len(pd.read_parquet(out/'exclusion_ledger.parquet')) == len(pd.read_parquet(labels/'opportunity_ledger.parquet'))
