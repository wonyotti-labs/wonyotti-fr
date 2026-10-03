import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from wonyotti_fr.calibration_diagnostics import (
    PERIODS,
    calibration_admission,
    calibration_split,
    run_calibration_diagnostics,
)
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.management_calibration import ManagementOffset, probability_logits, sigmoid
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_inventory import MinuteInventoryModels
from wonyotti_fr.minute_management import ACTIONS, purged_window


def source():
    times = pd.date_range('2020-01-03', periods=1100, freq='6h', tz='UTC').append(
        pd.date_range('2021-01-03', periods=300, freq='8h', tz='UTC')).append(
        pd.date_range('2021-07-03', periods=300, freq='8h', tz='UTC'))
    rng = np.random.default_rng(38)
    frame = pd.DataFrame(rng.normal(size=(len(times), 44)), columns=MinuteInventoryModels.features)
    for i, action in enumerate(ACTIONS):
        frame[f'y_{action}'] = frame.iloc[:, i].gt(.5).astype(int)
        frame[f'{action}_count'] = frame[f'y_{action}']
    return frame.assign(end=times, entry_time=times-pd.Timedelta(minutes=5),
        episode_id=np.arange(len(times))+1, label_end=times+pd.Timedelta(minutes=1), usable=True)


def test_offset_matches_analytic_and_independent_roots_without_reordering():
    rng = np.random.default_rng(31)
    scores = rng.uniform(.01, .7, (600, 3))
    scores[:, 0] = .2
    y = np.zeros((600, 3), dtype=int)
    y[:300, 0], y[:90, 1], y[:160, 2] = 1, 1, 1
    model, support = ManagementOffset.fit(scores, y)
    assert abs(model.offsets[0] - np.log(4)) < 1e-12
    logits = probability_logits(scores)
    for i in range(3):
        expected = brentq(lambda b, i=i: sigmoid(logits[:, i] + b).mean()-y[:, i].mean(), -64, 64, xtol=1e-14)
        assert abs(model.offsets[i]-expected) < 1e-12
        np.testing.assert_array_equal(np.argsort(scores[:, i], kind='stable'), np.argsort(model.predict(scores)[:, i], kind='stable'))
    assert support['max_mean_residual'] < 1e-12
    restored = ManagementOffset.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict(scores), restored.predict(scores))
    np.testing.assert_array_equal(model.predict(scores[:5]), np.vstack([model.predict(row[None, :]) for row in scores[:5]]))
    assert np.isfinite(model.predict(np.array([[0., 1., .5]]))).all()


@pytest.mark.parametrize('damage', ['shape', 'nan', 'range', 'few', 'class', 'labels'])
def test_offset_rejects_invalid_scores_or_support(damage):
    scores = np.full((120, 3), .1)
    y = np.tile(np.r_[np.ones(60), np.zeros(60)][:, None], (1, 3))
    if damage == 'shape':
        scores = scores[:, :2]
    elif damage == 'nan':
        scores[0, 0] = np.nan
    elif damage == 'range':
        scores[0, 0] = 1.1
    elif damage == 'few':
        scores, y = scores[:99], y[:99]
    elif damage == 'class':
        y[:, 0] = 0
    else:
        y[0, 0] = .5
    with pytest.raises(ValueError):
        ManagementOffset.fit(scores, y)


@pytest.mark.parametrize('key,value', [('slope', 2.), ('offsets', [0., 65., 0.]),
    ('offsets', [0., float('nan'), 0.]), ('offsets', [False, 0., 0.]), ('iterations', 50), ('epsilon', .001)])
def test_numeric_offset_loader_rejects_settings_changes(key, value):
    data = {'format': ManagementOffset.format, 'actions': ACTIONS, 'slope': 1.,
        'epsilon': float(np.finfo(float).eps), 'bounds': [-64., 64.], 'iterations': 100, 'offsets': [0.]*3}
    data[key] = value
    with pytest.raises(ValueError):
        ManagementOffset.from_dict(data)


def test_three_way_split_excludes_crossing_positions_and_changes(tmp_path):
    frame = source()
    crossing = frame.iloc[:1].copy()
    crossing['end'] = pd.Timestamp('2021-01-01', tz='UTC')
    crossing['label_end'] = crossing.end + pd.Timedelta(minutes=1)
    frame = pd.concat([frame, crossing]).sort_values('end').reset_index(drop=True)
    purged_window(frame, *PERIODS['diagnosis']).to_parquet(tmp_path / 'diagnosis_used.parquet', index=False)
    parts, support = calibration_split(frame, tmp_path)
    assert [len(v) for v in parts.values()] == [1099, 300, 300]
    assert 1 not in set(parts['training'].episode_id)
    assert support['calibration']['independent_orders']['exit'] == parts['calibration'].y_exit.sum()
    frame.loc[frame.end.dt.year.eq(2021) & frame.end.dt.month.ge(7), 'ret_5m'] += .1
    with pytest.raises(AssertionError):
        calibration_split(frame, tmp_path)


def test_all_actions_must_improve_both_baselines_and_preserve_raw_model():
    metrics = {a: {kind: {'log_loss': loss, 'average_precision': ap} for kind, loss, ap in
        [('original_v21', .6, .4), ('logistic_calibrated', .59, .4), ('histogram_raw', .52, .5),
         ('histogram_calibrated', .5, .5), ('constant', .69, .1)]} for a in ACTIONS}
    assert calibration_admission(metrics)['calibrated_histogram_admitted']
    for key, loss, ap in [('histogram_calibrated', .5841, .5), ('histogram_calibrated', .5, .39),
        ('histogram_raw', .49, .5), ('constant', .49, .1)]:
        bad = copy.deepcopy(metrics)
        bad['reduce'][key] = {'log_loss': loss, 'average_precision': ap}
        assert not calibration_admission(bad)['calibrated_histogram_admitted']


def test_full_diagnosis_never_fits_future_labels_and_preserves_failure(tmp_path, monkeypatch):
    selection, labels, previous = [tmp_path / name for name in ['selection', 'labels', 'previous']]
    for path in [selection, labels, previous]:
        path.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    (labels / 'files.json').write_text('{}')
    save_json(previous / 'manifest.json', {'settings': {'reference_sha256': sha256(selection / 'frozen_selection.json'),
        'labels_files_sha256': sha256(labels / 'files.json')}})
    save_json(previous / 'summary.json', {'complete': True})
    frame = source()
    old_train = purged_window(frame, '2020-01-01', '2021-07-01')
    validation = purged_window(frame, *PERIODS['diagnosis']).reset_index(drop=True)
    old, _, _ = MinuteInventoryModels.fit(old_train, validation, 'logistic')
    monkeypatch.setattr('wonyotti_fr.calibration_diagnostics.load_selection',
        lambda _: ({'protocol': 'minute_inventory_micro_v21'}, SimpleNamespace(manager=old)))
    monkeypatch.setattr('wonyotti_fr.calibration_diagnostics.load_minute_inventory_labels', lambda _: (frame, None))

    def previous_outputs():
        val = purged_window(frame, *PERIODS['diagnosis']).reset_index(drop=True)
        val.to_parquet(previous / 'diagnosis_used.parquet', index=False)
        predictions = val[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        scores = old.probabilities(val[old.features].to_numpy())
        for i, a in enumerate(ACTIONS):
            predictions[a + '_logistic'] = scores[:, i]
        predictions.to_parquet(previous / 'predictions.parquet', index=False)
        save_json(previous / 'files.json', {p.name: sha256(p) for p in previous.iterdir() if p.name != 'files.json'})

    previous_outputs()
    out = run_calibration_diagnostics(selection, labels, previous, tmp_path / 'runs')
    predictions = pd.read_parquet(out / 'predictions.parquet')
    metrics = json.loads((out / 'metrics.json').read_text())
    for a in ACTIONS:
        for name in metrics[a]:
            assert metrics[a][name] == management_metrics(predictions['y_' + a], predictions[a + '_' + name])
    assert not list(out.rglob('trades.parquet'))
    assert json.loads((out / 'decision.json').read_text()) == calibration_admission(metrics)
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    future = frame.end.ge(pd.Timestamp('2021-07-01', tz='UTC'))
    for a in ACTIONS:
        frame.loc[future, 'y_' + a] = 1-frame.loc[future, 'y_' + a]
        frame.loc[future, a + '_count'] = frame.loc[future, 'y_' + a]
    previous_outputs()
    changed = run_calibration_diagnostics(selection, labels, previous, tmp_path / 'changed')
    for name in ['models.json', 'offsets.json', 'training_support.json']:
        assert (out / name).read_bytes() == (changed / name).read_bytes()
    frame.loc[frame.end.dt.year.eq(2021) & frame.end.dt.month.lt(7), 'y_exit'] = 0
    with pytest.raises(ValueError, match='지원'):
        run_calibration_diagnostics(selection, labels, previous, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'models.json').exists()
