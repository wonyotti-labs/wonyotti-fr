import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_inventory_recent import source_frame
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS, HistogramManagementModels
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.management_diagnostics import (
    management_admission,
    management_metrics,
    management_split,
    run_management_diagnostics,
)
from wonyotti_fr.minute_inventory import CONTEXT_FEATURES, MinuteInventoryModels
from wonyotti_fr.minute_management import ACTIONS, purged_window


def source():
    frame, _ = source_frame()
    rng = np.random.default_rng(91)
    for name in CONTEXT_FEATURES:
        frame[name] = rng.normal(size=len(frame))
    return frame


@pytest.fixture(scope='module')
def learned():
    frame = source()
    train, validation = frame.iloc[:1100], frame.iloc[1100:]
    x = train[MinuteInventoryModels.features].to_numpy()
    y = train[[f'y_{a}' for a in ACTIONS]].to_numpy()
    vx = validation[MinuteInventoryModels.features].to_numpy()
    model, support = HistogramManagementModels.fit(x, y, vx)
    return model, support, x, y, vx


def test_histogram_export_validation_and_single_row_predictions(learned):
    model, support, x, y, vx = learned
    assert support['export_max_error'] < 1e-12 and support['validation_export_max_error'] < 1e-12
    assert support['early_stopping'] is False
    expected = []
    for i in range(3):
        with threadpool_limits(limits=1):
            estimator = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y[:, i])
            expected.append(estimator.predict_proba(vx)[:, 1])
    np.testing.assert_allclose(model.probabilities(vx), np.column_stack(expected), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(model.probabilities(vx[:8]), np.vstack([model.probabilities(row[None, :]) for row in vx[:8]]))
    changed, _ = HistogramManagementModels.fit(x, y, vx * 17 + 2)
    assert changed.to_dict() == model.to_dict()
    invalid = vx[:2].copy()
    invalid[0, 0] = np.nan
    assert np.isnan(model.probabilities(invalid)[0]).all()
    with pytest.raises(ValueError):
        model.probabilities(vx[:, :-1])


@pytest.mark.parametrize('damage', ['cycle', 'unreachable', 'depth', 'feature', 'trees', 'number', 'settings'])
def test_histogram_numeric_loader_rejects_invalid_graphs_and_settings(learned, damage):
    model = copy.deepcopy(learned[0].to_dict())
    tree = model['models'][0]['trees'][0]
    if damage == 'cycle':
        tree['left'][0] = 0
    elif damage == 'unreachable':
        tree['left'][0] = tree['right'][0]
    elif damage == 'depth':
        model['models'][0]['trees'][0] = {'left': [1, 2, 3, -1, -1, -1, -1],
            'right': [6, 5, 4, -1, -1, -1, -1], 'feature': [0, 0, 0, -2, -2, -2, -2],
            'threshold': [0.]*7, 'value': [0.]*7}
    elif damage == 'feature':
        tree['feature'][0] = 44
    elif damage == 'trees':
        model['models'][0]['trees'].pop()
    elif damage == 'number':
        model['models'][0]['baseline'] = float('nan')
    else:
        model['settings']['early_stopping'] = True
    with pytest.raises(ValueError):
        HistogramManagementModels.from_dict(model)


@pytest.mark.parametrize('damage', ['few', 'class', 'nan'])
def test_histogram_training_support_is_required(learned, damage):
    _, _, x, y, vx = learned
    x, y = x.copy(), y.copy()
    if damage == 'few':
        x, y = x[:999], y[:999]
    elif damage == 'class':
        y[:, 0] = 0
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        HistogramManagementModels.fit(x, y, vx)


def test_all_action_admission_requires_gain_constant_and_average_precision():
    metrics = {a: {kind: {'log_loss': loss, 'average_precision': ap} for kind, loss, ap in
        [('logistic', .6, .4), ('histogram', .5, .5), ('constant', .69, .1)]} for a in ACTIONS}
    assert management_admission(metrics)['histogram_admitted']
    for loss, ap in [(.594, .5), (.7, .5), (.5, .39)]:
        changed = copy.deepcopy(metrics)
        changed['reduce']['histogram'] = {'log_loss': loss, 'average_precision': ap}
        assert not management_admission(changed)['histogram_admitted']
    scores = management_metrics([0]*30 + [1]*30, [.1]*30 + [.9]*30)
    assert scores['average_precision'] == scores['roc_auc'] == 1.
    with pytest.raises(ValueError):
        management_metrics([0]*19+[1]*30, [.5]*49)


def test_management_split_keeps_crossing_episode_out_and_rejects_changed_rows(tmp_path):
    frame = source()
    crossing = frame.iloc[:1].copy()
    crossing['end'] = pd.Timestamp('2021-07-01', tz='UTC')
    crossing['label_end'] = crossing.end + pd.Timedelta(minutes=1)
    frame = pd.concat([frame, crossing]).sort_values('end').reset_index(drop=True)
    for name, period in [('training_used', TRAINING_PERIODS[1]), ('calibration_used', CALIBRATION_PERIODS[1])]:
        purged_window(frame, *period).to_parquet(tmp_path / f'{name}.parquet', index=False)
    train, validation = management_split(frame, tmp_path)
    assert 1 not in set(train.episode_id) | set(validation.episode_id)
    assert len(train) == 1099 and len(validation) == 300
    changed = frame.copy()
    changed.loc[1, MinuteInventoryModels.features[0]] += 1
    with pytest.raises(AssertionError):
        management_split(changed, tmp_path)


def test_full_management_diagnosis_reproduces_existing_model_and_preserves_outputs(tmp_path, monkeypatch):
    reference, labels = tmp_path / 'reference', tmp_path / 'labels'
    reference.mkdir()
    labels.mkdir()
    (labels / 'files.json').write_text('{}')
    (reference / 'frozen_selection.json').write_text('{}')
    save_json(reference / 'manifest.json', {'settings': {'files_sha256': sha256(labels / 'files.json')}})
    frame = source()
    train = purged_window(frame, *TRAINING_PERIODS[1]).reset_index(drop=True)
    validation = purged_window(frame, *CALIBRATION_PERIODS[1]).reset_index(drop=True)
    for name, rows in [('training_used', train), ('calibration_used', validation)]:
        rows.to_parquet(reference / f'{name}.parquet', index=False)
    model, thresholds, _ = MinuteInventoryModels.fit(train, validation, 'logistic')
    monkeypatch.setattr('wonyotti_fr.management_diagnostics.load_selection',
        lambda _: ({'protocol': 'minute_inventory_micro_v21', 'inventory_thresholds': thresholds}, SimpleNamespace(manager=model)))
    monkeypatch.setattr('wonyotti_fr.management_diagnostics.load_minute_inventory_labels', lambda _: (frame, None))
    out = run_management_diagnostics(reference, labels, tmp_path / 'runs')
    metrics = json.loads((out / 'metrics.json').read_text())
    predictions = pd.read_parquet(out / 'predictions.parquet')
    for action in ACTIONS:
        for name in ['logistic', 'histogram', 'constant']:
            assert metrics[action][name] == management_metrics(predictions[f'y_{action}'], predictions[f'{action}_{name}'])
    assert json.loads((out / 'decision.json').read_text()) == management_admission(metrics)
    assert json.loads((out / 'summary.json').read_text())['existing_model_and_thresholds_exact']
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    assert not list(out.rglob('trades.parquet'))
    frame.loc[0, MinuteInventoryModels.features[0]] += 1
    with pytest.raises(AssertionError):
        run_management_diagnostics(reference, labels, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'models.json').exists()
