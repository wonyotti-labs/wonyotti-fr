import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from test_calibration_diagnostics import source

from wonyotti_fr.calibration_diagnostics import PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_management import (
    BETAS,
    decision_metrics,
    first_management_admission,
    first_management_diagnosis,
    run_first_management_diagnosis,
)
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


class SyntheticScores:
    features = OrderHistoryBoostModels.features

    def probabilities(self, x):
        return .02 + .02 * (x[:, :3] > .5) + .2 * (x[:, :3] > 1.)


def rows():
    f = source()
    f[HISTORY_FEATURES] = 0.
    f['past_increase_exists'] = np.arange(len(f)) % 2
    f['past_reduce_exists'] = np.arange(len(f)) % 2
    return [purged_window(f, *PERIODS[n]).reset_index(drop=True) for n in ['calibration', 'diagnosis']]


def run(cal, val):
    return first_management_diagnosis(cal, val, SyntheticScores(), dict.fromkeys(ACTIONS, .5), 1.5)


def test_first_thresholds_cannot_train_on_last_labels_or_change_repeat_and_exit():
    cal, val = rows()
    thresholds, support, predictions, metrics, decision = run(cal, val)
    assert decision['first_thresholds_admitted'] and all(decision['unchanged_masks'].values())
    assert all(v['predicted_positive'] >= 20 for v in support.values())
    changed = val.copy()
    for a in ACTIONS:
        changed['y_' + a] = 1 - changed['y_' + a]
        for phase in ['original', 'separate']:
            p = predictions[a + '_' + phase].to_numpy()
            expected = fbeta_score(val['y_' + a], p, beta=BETAS[a], zero_division=0)
            assert metrics[a]['all'][phase]['f_beta'] == pytest.approx(expected, abs=1e-14)
    new = run(cal, changed)
    assert new[:2] == (thresholds, support)
    pd.testing.assert_frame_equal(new[2].drop(columns=['y_' + a for a in ACTIONS]),
                                  predictions.drop(columns=['y_' + a for a in ACTIONS]), check_exact=True)
    assert new[3] != metrics
    assert thresholds == {'reduce': .04, 'increase': .04}


@pytest.mark.parametrize('damage', ['first_support', 'repeat_support', 'flags', 'time', 'episode', 'features', 'multiplier'])
def test_first_thresholds_reject_unsupported_or_overlapping_inputs(damage):
    cal, val = rows()
    multiplier = 1.5
    if damage == 'first_support':
        cal.loc[cal.past_reduce_exists.eq(0), 'y_reduce'] = 0
    elif damage == 'repeat_support':
        val.loc[val.past_increase_exists.eq(1), 'y_increase'] = 0
    elif damage == 'flags':
        cal['past_reduce_exists'] = cal.past_reduce_exists.astype(float)
        cal.loc[0, 'past_reduce_exists'] = .5
    elif damage == 'time':
        cal.loc[0, 'end'] = pd.Timestamp('2021-07-01', tz='UTC')
    elif damage == 'episode':
        val.loc[0, 'episode_id'] = cal.episode_id.iloc[0]
    elif damage == 'features':
        val.loc[0, 'ret_5m'] = np.nan
    else:
        multiplier = 1.
    with pytest.raises(ValueError):
        first_management_diagnosis(cal, val, SyntheticScores(), dict.fromkeys(ACTIONS, .5), multiplier)


def test_admission_requires_both_first_actions_and_preserves_overall_f_and_masks():
    _, _, _, metrics, decision = run(*rows())
    for action in ['increase', 'reduce']:
        for reason in ['recall', 'f_beta', 'overall', 'mask']:
            values = copy.deepcopy(metrics)
            unchanged = dict(decision['unchanged_masks'])
            if reason in ['recall', 'f_beta']:
                values[action]['first']['separate'][reason] = 0.
            elif reason == 'overall':
                values[action]['all']['original']['f_beta'] = 1.
            else:
                unchanged['repeat_' + action] = False
            assert not first_management_admission(values, unchanged)['first_thresholds_admitted']
    assert decision_metrics(np.zeros(20), np.zeros(20, dtype=bool), 1.)['f_beta'] == 0.


def test_pipeline_preserves_models_and_records_preflight_failure(tmp_path, monkeypatch):
    selection, diagnosis = tmp_path / 'selection', tmp_path / 'diagnosis'
    selection.mkdir()
    diagnosis.mkdir()
    (selection / 'frozen_selection.json').write_text('{}')
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        (selection / name).write_text(json.dumps({'synthetic': name}))
    cal, val = rows()
    cal.to_parquet(diagnosis / 'calibration_used.parquet', index=False)
    val.to_parquet(diagnosis / 'diagnosis_used.parquet', index=False)
    save_json(diagnosis / 'summary.json', {'complete': True})
    def refresh():
        save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir() if p.is_file() and p.name != 'files.json'})
        save_json(selection / 'history_admission.json', {'files_sha256': sha256(diagnosis / 'files.json')})
    refresh()
    monkeypatch.setattr('wonyotti_fr.first_management.load_selection',
        lambda _: ({'protocol': 'history_state_v36'}, SimpleNamespace(manager=SyntheticScores(),
            thresholds=dict.fromkeys(ACTIONS, .5), multiplier=1.5)))
    out = run_first_management_diagnosis(selection, diagnosis, tmp_path / 'runs')
    assert json.loads((out / 'decision.json').read_text())['first_thresholds_admitted']
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        assert (out / name).read_bytes() == (selection / name).read_bytes()
    assert not list(out.rglob('trades.parquet'))
    cal.loc[cal.past_reduce_exists.eq(0), 'y_reduce'] = 0
    cal.to_parquet(diagnosis / 'calibration_used.parquet', index=False)
    refresh()
    with pytest.raises(ValueError):
        run_first_management_diagnosis(selection, diagnosis, tmp_path / 'failures')
    assert len(list((tmp_path / 'failures').glob('*/failure.json'))) == 1
