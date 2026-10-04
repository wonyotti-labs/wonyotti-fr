import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_first_management import SyntheticScores, rows

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_management import first_management_diagnosis, run_first_management_diagnosis
from wonyotti_fr.first_management_direct import run_first_management_direct_diagnosis
from wonyotti_fr.minute_management import ACTIONS


def test_direct_first_thresholds_preserve_selection_repeats_and_future_label_independence():
    cal, val = rows()
    old = first_management_diagnosis(cal, val, SyntheticScores(), dict.fromkeys(ACTIONS, .5), 1.5)
    direct = first_management_diagnosis(cal, val, SyntheticScores(), dict.fromkeys(ACTIONS, .5), 1.5, first_multiplier=1.)
    assert old[:2] == direct[:2]
    assert all(direct[4]['unchanged_masks'].values())
    for a in ['reduce', 'increase']:
        first = val['past_' + a + '_exists'].eq(0)
        assert direct[2].loc[first, a + '_separate'].sum() > old[2].loc[first, a + '_separate'].sum()
        np.testing.assert_array_equal(direct[2].loc[~first, a + '_separate'], old[2].loc[~first, a + '_separate'])
    changed = val.copy()
    changed[['y_' + a for a in ACTIONS]] = 1 - changed[['y_' + a for a in ACTIONS]]
    new = first_management_diagnosis(cal, changed, SyntheticScores(), dict.fromkeys(ACTIONS, .5), 1.5, first_multiplier=1.)
    assert new[:2] == direct[:2]
    pd.testing.assert_frame_equal(new[2].drop(columns=['y_' + a for a in ACTIONS]),
                                  direct[2].drop(columns=['y_' + a for a in ACTIONS]), check_exact=True)


def reference(tmp_path, monkeypatch):
    selection, diagnosis = tmp_path / 'selection', tmp_path / 'diagnosis'
    selection.mkdir()
    diagnosis.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        (selection / name).write_text(json.dumps({'synthetic': name}))
    for frame, name in zip(rows(), ['calibration', 'diagnosis'], strict=True):
        frame.to_parquet(diagnosis / f'{name}_used.parquet', index=False)
    save_json(diagnosis / 'summary.json', {'complete': True})
    save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir() if p.is_file()})
    save_json(selection / 'history_admission.json', {'files_sha256': sha256(diagnosis / 'files.json')})
    monkeypatch.setattr('wonyotti_fr.first_management.load_selection',
        lambda _: ({'protocol': 'history_state_v36'}, SimpleNamespace(manager=SyntheticScores(),
            thresholds=dict.fromkeys(ACTIONS, .5), multiplier=1.5)))
    return run_first_management_diagnosis(selection, diagnosis, tmp_path / 'runs'), selection


def test_direct_pipeline_copies_models_and_requires_identical_first_thresholds(tmp_path, monkeypatch):
    previous, selection = reference(tmp_path, monkeypatch)
    out = run_first_management_direct_diagnosis(previous, tmp_path / 'direct')
    old = json.loads((previous / 'first_thresholds.json').read_text())
    new = json.loads((out / 'first_thresholds.json').read_text())
    assert {k: v for k, v in new.items() if k != 'first_multiplier'} == old
    assert new['first_multiplier'] == 1.
    assert json.loads((out / 'summary.json').read_text())['complete']
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        assert (out / name).read_bytes() == (selection / name).read_bytes()
    assert not list(out.rglob('trades.parquet'))
    old['thresholds']['reduce'] *= .8
    save_json(previous / 'first_thresholds.json', old)
    files = json.loads((previous / 'files.json').read_text())
    files['first_thresholds.json'] = sha256(previous / 'first_thresholds.json')
    save_json(previous / 'files.json', files)
    with pytest.raises(ValueError, match='재현'):
        run_first_management_direct_diagnosis(previous, tmp_path / 'failure')
    assert len(list((tmp_path / 'failure').glob('*/failure.json'))) == 1


@pytest.mark.parametrize('damage', ['model', 'source', 'protocol'])
def test_direct_pipeline_rejects_changed_parent_model_source_or_protocol(tmp_path, monkeypatch, damage):
    previous, selection = reference(tmp_path, monkeypatch)
    if damage == 'model':
        (selection / 'history_manager.json').write_text('changed')
    elif damage == 'source':
        settings = json.loads((previous / 'manifest.json').read_text())['settings']
        from pathlib import Path
        (Path(settings['diagnosis']) / 'files.json').write_text('{}')
    else:
        manifest = json.loads((previous / 'manifest.json').read_text())
        manifest['settings']['protocol_sha256'] = 'wrong'
        save_json(previous / 'manifest.json', manifest)
        files = json.loads((previous / 'files.json').read_text())
        files['manifest.json'] = sha256(previous / 'manifest.json')
        save_json(previous / 'files.json', files)
    with pytest.raises(ValueError):
        run_first_management_direct_diagnosis(previous, tmp_path / 'direct')
