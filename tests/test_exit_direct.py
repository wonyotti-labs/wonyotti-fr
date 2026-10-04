import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from test_first_management import SyntheticScores, rows

from wonyotti_fr.action_model import select_threshold
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.exit_direct import (
    MODEL_FILES,
    exit_direct_admission,
    exit_direct_diagnosis,
    run_exit_direct_diagnosis,
)
from wonyotti_fr.minute_management import ACTIONS


def settings(cal):
    manager = SyntheticScores()
    threshold, support = select_threshold(cal.y_exit, manager.probabilities(cal[manager.features].to_numpy())[:, 0], 2.)
    return {'exit': threshold, 'reduce': .04, 'increase': .04}, {'reduce': .02, 'increase': .02}, support


def run(cal, val):
    thresholds, first, _ = settings(cal)
    return exit_direct_diagnosis(cal, val, SyntheticScores(), thresholds, first, 1.5)


def test_direct_exit_preserves_other_actions_and_future_label_independence():
    cal, val = rows()
    support, predictions, metrics, decision = run(cal, val)
    assert decision['exit_direct_admitted'] and all(decision['unchanged_masks'].values())
    assert support == settings(cal)[2]
    assert metrics['exit']['direct']['recall'] > metrics['exit']['original']['recall']
    for action, beta in zip(ACTIONS, [2., 1., .5], strict=True):
        for kind in ['original', 'direct']:
            expected = fbeta_score(val[f'y_{action}'], predictions[f'{action}_{kind}'], beta=beta, zero_division=0)
            assert metrics[action][kind]['f_beta'] == pytest.approx(expected, abs=1e-14)
        if action != 'exit':
            boundary = np.where(val[f'past_{action}_exists'].eq(0), .02, .06)
            np.testing.assert_array_equal(predictions[f'{action}_direct'], predictions[f'{action}_score'] >= boundary)
            assert metrics[action]['direct'] == metrics[action]['original']
    changed = val.copy()
    changed[['y_' + a for a in ACTIONS]] = 1 - changed[['y_' + a for a in ACTIONS]]
    new = run(cal, changed)
    assert support == new[0]
    pd.testing.assert_frame_equal(new[1].drop(columns=['y_' + a for a in ACTIONS]),
                                  predictions.drop(columns=['y_' + a for a in ACTIONS]), check_exact=True)
    assert new[2] != metrics


@pytest.mark.parametrize('damage', ['support', 'flags', 'time', 'duplicate', 'episode', 'features', 'threshold', 'multiplier', 'probability'])
def test_direct_exit_rejects_unsupported_inputs_and_changed_threshold(damage):
    cal, val = rows()
    thresholds, first, _ = settings(cal)
    manager, multiplier = SyntheticScores(), 1.5
    if damage == 'support':
        val['y_exit'] = 0
    elif damage == 'flags':
        val['past_reduce_exists'] = .5
    elif damage == 'time':
        cal.loc[0, 'label_end'] = pd.Timestamp('2021-07-01', tz='UTC')
    elif damage == 'duplicate':
        val.loc[1, 'end'] = val.loc[0, 'end']
    elif damage == 'episode':
        val.loc[0, 'episode_id'] = cal.episode_id.iloc[0]
    elif damage == 'features':
        val.loc[0, 'ret_5m'] = np.nan
    elif damage == 'threshold':
        thresholds['exit'] *= .9
    elif damage == 'multiplier':
        multiplier = 1.
    else:
        manager.probabilities = lambda x: np.full((len(x), 3), 1.1)
    with pytest.raises(ValueError):
        exit_direct_diagnosis(cal, val, manager, thresholds, first, multiplier)


def test_admission_requires_both_exit_improvements_and_unchanged_management():
    _, _, metrics, decision = run(*rows())
    for key in ['recall', 'f_beta']:
        changed = copy.deepcopy(metrics)
        changed['exit']['direct'][key] = changed['exit']['original'][key]
        assert not exit_direct_admission(changed, decision['unchanged_masks'])['exit_direct_admitted']
    for a in ['reduce', 'increase']:
        unchanged = {**decision['unchanged_masks'], a: False}
        assert not exit_direct_admission(metrics, unchanged)['exit_direct_admitted']


def reference(tmp_path, monkeypatch):
    selection, diagnosis = tmp_path / 'selection', tmp_path / 'diagnosis'
    selection.mkdir()
    diagnosis.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    for name in MODEL_FILES:
        (selection / name).write_text(json.dumps({'synthetic': name}))
    cal, val = rows()
    thresholds, first, support = settings(cal)
    for frame, name in [(cal, 'calibration'), (val, 'diagnosis')]:
        frame.to_parquet(diagnosis / f'{name}_used.parquet', index=False)
    save_json(diagnosis / 'summary.json', {'complete': True})
    save_json(diagnosis / 'manifest.json', {'synthetic': True})
    save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir() if p.is_file()})
    save_json(selection / 'history_admission.json', {'files_sha256': sha256(diagnosis / 'files.json')})
    save_json(selection / 'history_thresholds.json', {'thresholds': thresholds, 'support': {'exit': support},
        'rows': len(cal), 'calibration_period': ['2021-01-01', '2021-07-01'], 'minimum_predicted_positive': 20,
        'calibration_sha256': sha256(diagnosis / 'calibration_used.parquet')})
    monkeypatch.setattr('wonyotti_fr.exit_direct.load_selection',
        lambda _: ({'protocol': 'activity_ablation_v46'}, SimpleNamespace(manager=SyntheticScores(),
            thresholds=thresholds, first_thresholds=first, multiplier=1.5)))
    return selection, diagnosis


def test_pipeline_preserves_models_threshold_and_future_label_independence(tmp_path, monkeypatch):
    selection, diagnosis = reference(tmp_path, monkeypatch)
    out = run_exit_direct_diagnosis(selection, diagnosis, tmp_path / 'runs')
    assert json.loads((out / 'summary.json').read_text())['exit_direct_admitted']
    for name in MODEL_FILES:
        assert (out / name).read_bytes() == (selection / name).read_bytes()
    assert not list(out.rglob('trades.parquet'))
    future = pd.read_parquet(diagnosis / 'diagnosis_used.parquet')
    future[['y_' + a for a in ACTIONS]] = 1 - future[['y_' + a for a in ACTIONS]]
    future.to_parquet(diagnosis / 'diagnosis_used.parquet', index=False)
    files = json.loads((diagnosis / 'files.json').read_text())
    files['diagnosis_used.parquet'] = sha256(diagnosis / 'diagnosis_used.parquet')
    save_json(diagnosis / 'files.json', files)
    save_json(selection / 'history_admission.json', {'files_sha256': sha256(diagnosis / 'files.json')})
    changed = run_exit_direct_diagnosis(selection, diagnosis, tmp_path / 'future')
    for name in [*MODEL_FILES, 'exit_threshold.json']:
        assert (out / name).read_bytes() == (changed / name).read_bytes()
    assert json.loads((out / 'metrics.json').read_text()) != json.loads((changed / 'metrics.json').read_text())


@pytest.mark.parametrize('damage', ['support', 'source', 'path', 'hash'])
def test_pipeline_rejects_changed_support_source_or_file_map(tmp_path, monkeypatch, damage):
    selection, diagnosis = reference(tmp_path, monkeypatch)
    if damage == 'support':
        old = json.loads((selection / 'history_thresholds.json').read_text())
        old['support']['exit']['recall'] = 0.
        save_json(selection / 'history_thresholds.json', old)
    elif damage == 'source':
        (diagnosis / 'diagnosis_used.parquet').write_bytes(b'changed')
    elif damage == 'path':
        save_json(diagnosis / 'files.json', {'../escape': '0'*64})
    else:
        save_json(selection / 'history_admission.json', {'files_sha256': '0'*64})
    with pytest.raises(ValueError):
        run_exit_direct_diagnosis(selection, diagnosis, tmp_path / 'failures')
    if damage == 'support':
        assert len(list((tmp_path / 'failures').glob('*/failure.json'))) == 1
