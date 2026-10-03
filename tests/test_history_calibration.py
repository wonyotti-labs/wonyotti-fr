import copy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_calibration_diagnostics import source

from wonyotti_fr.calibration_diagnostics import PERIODS, run_calibration_diagnostics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.history_calibration_diagnostics import (
    history_calibration_admission,
    history_calibration_context,
    run_history_calibration_diagnostics,
)
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_inventory import MinuteInventoryModels
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


def refresh_files(root):
    save_json(root / 'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'files.json'})


def setup(tmp_path, monkeypatch):
    selection, labels, previous, history, audit = [tmp_path / name for name in ['selection', 'labels', 'previous', 'history', 'audit']]
    for path in [selection, labels, previous, history, audit]:
        path.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    (labels / 'files.json').write_text('{}')
    (audit / 'actions.parquet').write_text('synthetic fingerprint')
    settings = {'reference_sha256': sha256(selection / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json')}
    save_json(previous / 'manifest.json', {'settings': settings})
    save_json(previous / 'summary.json', {'complete': True})
    frame = source()
    old_train = purged_window(frame, *TRAINING_PERIODS[1])
    validation = purged_window(frame, *CALIBRATION_PERIODS[1])
    old, _, _ = MinuteInventoryModels.fit(old_train, validation, 'logistic')
    monkeypatch.setattr('wonyotti_fr.calibration_diagnostics.load_selection',
        lambda _: ({'protocol': 'minute_inventory_micro_v21'}, SimpleNamespace(manager=old)))
    monkeypatch.setattr('wonyotti_fr.calibration_diagnostics.load_minute_inventory_labels', lambda _: (frame, None))

    def baseline():
        val = purged_window(frame, *PERIODS['diagnosis']).reset_index(drop=True)
        val.to_parquet(previous / 'diagnosis_used.parquet', index=False)
        predictions = val[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        scores = old.probabilities(val[old.features].to_numpy())
        for i, a in enumerate(ACTIONS):
            predictions[a + '_logistic'] = scores[:, i]
        predictions.to_parquet(previous / 'predictions.parquet', index=False)
        refresh_files(previous)
        return run_calibration_diagnostics(selection, labels, previous, tmp_path / 'baseline_runs')

    reference = baseline()
    rng = np.random.default_rng(86)
    features = frame[['end']].copy()
    for name in HISTORY_FEATURES:
        features[name] = rng.integers(0, 2, len(frame)).astype(float) if name.endswith('exists') else rng.uniform(0, 4, len(frame))
    features.to_parquet(history / 'history_features.parquet', index=False)
    save_json(history / 'summary.json', {'complete': True})
    save_json(history / 'manifest.json', {'settings': {'selection_sha256': settings['reference_sha256'],
        'labels_files_sha256': settings['labels_files_sha256'], 'audit': str(audit),
        'audit_sha256': {'actions.parquet': sha256(audit / 'actions.parquet')}, 'training_period': TRAINING_PERIODS[1],
        'diagnosis_period': CALIBRATION_PERIODS[1], 'new_features': HISTORY_FEATURES}})
    refresh_files(history)
    return history, reference, frame, baseline


def test_combined_diagnosis_preserves_old_splits_and_never_fits_future_labels(tmp_path, monkeypatch):
    history, reference, frame, baseline = setup(tmp_path, monkeypatch)
    out = run_history_calibration_diagnostics(history, reference, tmp_path / 'runs')
    for name in PERIODS:
        actual = pd.read_parquet(out / f'{name}_used.parquet')
        original = pd.read_parquet(reference / f'{name}_used.parquet')
        pd.testing.assert_frame_equal(actual.drop(columns=HISTORY_FEATURES), original, check_exact=True)
    models = json.loads((out / 'models.json').read_text())
    assert models['histogram']['format'] == OrderHistoryBoostModels.format and len(models['histogram']['features']) == 50
    predictions = pd.read_parquet(out / 'predictions.parquet')
    previous = pd.read_parquet(reference / 'predictions.parquet')
    metrics = json.loads((out / 'metrics.json').read_text())
    for a in ACTIONS:
        np.testing.assert_array_equal(predictions[a + '_previous_calibrated'], previous[a + '_histogram_calibrated'])
        for name in metrics[a]:
            assert metrics[a][name] == management_metrics(predictions['y_' + a], predictions[a + '_' + name])
    assert json.loads((out / 'decision.json').read_text()) == history_calibration_admission(metrics)
    assert not list(out.rglob('trades.parquet'))
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    future = frame.end.ge(pd.Timestamp('2021-07-01', tz='UTC'))
    for a in ACTIONS:
        frame.loc[future, 'y_' + a] = 1-frame.loc[future, 'y_' + a]
        frame.loc[future, a + '_count'] = frame.loc[future, 'y_' + a]
    new_reference = baseline()
    changed = run_history_calibration_diagnostics(history, new_reference, tmp_path / 'changed')
    for name in ['models.json', 'offsets.json', 'training_support.json']:
        assert (out / name).read_bytes() == (changed / name).read_bytes()


def test_history_context_rejects_changed_period_fingerprint_or_past_inputs(tmp_path, monkeypatch):
    history, reference, _, _ = setup(tmp_path, monkeypatch)
    for damage in ['period', 'fingerprint', 'nan', 'duplicate', 'columns']:
        changed = tmp_path / damage
        shutil.copytree(history, changed)
        if damage in {'period', 'fingerprint'}:
            manifest = json.loads((changed / 'manifest.json').read_text())
            if damage == 'period':
                manifest['settings']['training_period'][0] = '2019-01-01'
            else:
                manifest['settings']['selection_sha256'] = 'wrong'
            save_json(changed / 'manifest.json', manifest)
        else:
            features = pd.read_parquet(changed / 'history_features.parquet')
            if damage == 'nan':
                features.loc[0, HISTORY_FEATURES[0]] = np.nan
            elif damage == 'duplicate':
                features.loc[1, 'end'] = features.loc[0, 'end']
            else:
                features = features.drop(columns=HISTORY_FEATURES[0])
            features.to_parquet(changed / 'history_features.parquet', index=False)
        refresh_files(changed)
        with pytest.raises(ValueError):
            history_calibration_context(changed, reference)


def test_third_baseline_cannot_be_skipped_for_any_action():
    metrics = {a: {kind: {'log_loss': loss, 'average_precision': ap} for kind, loss, ap in
        [('original_v21', .6, .4), ('logistic_calibrated', .59, .4), ('histogram_raw', .52, .5),
         ('histogram_calibrated', .5, .5), ('previous_calibrated', .55, .5), ('constant', .69, .1)]} for a in ACTIONS}
    assert history_calibration_admission(metrics)['calibrated_histogram_admitted']
    for loss, ap in [(.49, .4), (.505, .4), (.6, .51)]:
        changed = copy.deepcopy(metrics)
        changed['reduce']['previous_calibrated'] = {'log_loss': loss, 'average_precision': ap}
        assert not history_calibration_admission(changed)['calibrated_histogram_admitted']
