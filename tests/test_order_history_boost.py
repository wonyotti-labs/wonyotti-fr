import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_calibration_diagnostics import source
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_inventory import MinuteInventoryModels
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES, OrderHistoryModels
from wonyotti_fr.order_history_boost import (
    OrderHistoryBoostModels,
    history_boost_admission,
    run_order_history_boost_diagnostics,
)


def history_rows():
    frame = source()
    rng = np.random.default_rng(72)
    for name in HISTORY_FEATURES:
        frame[name] = rng.integers(0, 2, len(frame)).astype(float) if name.endswith('exists') else rng.uniform(0, 4, len(frame))
    return (purged_window(frame, *period).reset_index(drop=True) for period in [TRAINING_PERIODS[1], CALIBRATION_PERIODS[1]])


@pytest.fixture(scope='module')
def learned():
    train, validation = history_rows()
    x, vx = (f[OrderHistoryModels.features].to_numpy() for f in [train, validation])
    y = train[[f'y_{a}' for a in ACTIONS]].to_numpy()
    model, support = OrderHistoryBoostModels.fit(x, y, vx)
    return model, support, x, y, vx


def test_fifty_feature_export_matches_independent_fit_and_future_inputs_do_not_train(learned):
    model, support, x, y, vx = learned
    assert len(model.features) == 50 and support['validation_export_max_error'] < 1e-12
    for i in range(3):
        with threadpool_limits(limits=1):
            estimator = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y[:, i])
            expected = estimator.predict_proba(vx)[:, 1]
        np.testing.assert_allclose(model.probabilities(vx)[:, i], expected, atol=1e-12, rtol=0)
    changed, _ = OrderHistoryBoostModels.fit(x, y, vx + 100)
    assert changed.to_dict() == model.to_dict()
    np.testing.assert_array_equal(model.probabilities(vx[:3]), np.vstack([model.probabilities(row[None, :]) for row in vx[:3]]))


@pytest.mark.parametrize('damage', ['format', 'features', 'tree', 'settings'])
def test_history_boost_rejects_other_dimensions_or_mutated_trees(learned, damage):
    data = learned[0].to_dict()
    if damage == 'format':
        data['format'] = 'histogram_management_v1'
    elif damage == 'features':
        data['features'] = MinuteInventoryModels.features
    elif damage == 'tree':
        data['models'][0]['trees'][0]['left'][0] = 0
    else:
        data['settings']['early_stopping'] = True
    with pytest.raises(ValueError):
        OrderHistoryBoostModels.from_dict(data)


def test_history_boost_must_improve_every_action_against_both_baselines():
    metrics = {a: {kind: {'log_loss': loss, 'average_precision': ap} for kind, loss, ap in
        [('original', .6, .4), ('history', .55, .5), ('boosted', .5, .6), ('constant', .69, .1)]} for a in ACTIONS}
    assert history_boost_admission(metrics)['history_boost_admitted']
    for name, loss, ap in [('boosted', .5445, .6), ('boosted', .5, .49), ('constant', .49, .1)]:
        bad = copy.deepcopy(metrics)
        bad['exit'][name] = {'log_loss': loss, 'average_precision': ap}
        assert not history_boost_admission(bad)['history_boost_admitted']


def test_full_history_boost_pipeline_reproduces_parent_and_preserves_failure(tmp_path, monkeypatch):
    history, selection, labels, audit = [tmp_path / name for name in ['history', 'selection', 'labels', 'audit']]
    for path in [history, selection, labels, audit]:
        path.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    (labels / 'files.json').write_text('{}')
    (audit / 'actions.parquet').write_text('synthetic fingerprint')
    save_json(history / 'manifest.json', {'settings': {'selection': str(selection), 'labels': str(labels), 'audit': str(audit),
        'selection_sha256': sha256(selection / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json'),
        'audit_sha256': {'actions.parquet': sha256(audit / 'actions.parquet')}, 'training_period': TRAINING_PERIODS[1],
        'diagnosis_period': CALIBRATION_PERIODS[1], 'new_features': HISTORY_FEATURES}})
    save_json(history / 'summary.json', {'complete': True})
    train, validation = history_rows()
    train.to_parquet(history / 'training_used.parquet', index=False)
    validation.to_parquet(history / 'diagnosis_used.parquet', index=False)
    old_train, old_validation = (f.drop(columns=HISTORY_FEATURES) for f in [train, validation])
    old_train.to_parquet(selection / 'training_used.parquet', index=False)
    old_validation.to_parquet(selection / 'calibration_used.parquet', index=False)
    old, _, _ = MinuteInventoryModels.fit(old_train, old_validation, 'logistic')
    previous, thresholds, _ = OrderHistoryModels.fit(train, validation, 'logistic')
    save_json(history / 'model.json', previous.to_dict())
    save_json(history / 'thresholds_unused.json', thresholds)
    scores = {'original': old.probabilities(validation[old.features].to_numpy()),
        'history': previous.probabilities(validation[previous.features].to_numpy()),
        'constant': np.tile(train[[f'y_{a}' for a in ACTIONS]].mean().to_numpy(), (len(validation), 1))}
    predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
    for i, a in enumerate(ACTIONS):
        for name, values in scores.items():
            predictions[f'{a}_{name}'] = values[:, i]
    predictions.to_parquet(history / 'predictions.parquet', index=False)
    save_json(history / 'files.json', {p.name: sha256(p) for p in history.iterdir() if p.is_file()})
    monkeypatch.setattr('wonyotti_fr.order_history_boost.load_selection',
        lambda _: ({'protocol': 'minute_inventory_micro_v21'}, SimpleNamespace(manager=old)))
    out = run_order_history_boost_diagnostics(history, tmp_path / 'runs')
    metrics = json.loads((out / 'metrics.json').read_text())
    output_scores = pd.read_parquet(out / 'predictions.parquet')
    for a in ACTIONS:
        for name in metrics[a]:
            assert metrics[a][name] == management_metrics(output_scores['y_' + a], output_scores[a + '_' + name])
    assert json.loads((out / 'decision.json').read_text()) == history_boost_admission(metrics)
    assert json.loads((out / 'summary.json').read_text())['history_model_and_thresholds_refitted_exact']
    assert not list(out.rglob('trades.parquet'))
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    train.loc[0, 'ret_5m'] += 1
    train.to_parquet(history / 'training_used.parquet', index=False)
    save_json(history / 'files.json', {p.name: sha256(p) for p in history.iterdir() if p.name != 'files.json'})
    with pytest.raises(AssertionError):
        run_order_history_boost_diagnostics(history, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()
