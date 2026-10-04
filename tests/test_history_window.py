import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_calibration_diagnostics import source
from test_history_calibration import refresh_files, setup
from threadpoolctl import threadpool_limits

from wonyotti_fr import calibration_diagnostics
from wonyotti_fr.calibration_diagnostics import PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.history_calibration_diagnostics import run_history_calibration_diagnostics
from wonyotti_fr.history_window import (
    HistoryWindowModels,
    expanded_splits,
    fit_history_window,
    history_window_admission,
    run_history_window_diagnosis,
)
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


def extend(frame):
    old = frame[frame.end.dt.year.eq(2020)]
    parts = []
    for year in [2018, 2019]:
        part = old.copy()
        part['end'] = pd.date_range(f'{year}-01-03', periods=len(part), freq='6h', tz='UTC')
        part['entry_time'] = part.end-pd.Timedelta(minutes=5)
        part['label_end'] = part.end+pd.Timedelta(minutes=1)
        part['episode_id'] += (2020-year)*10000
        parts.append(part)
    return pd.concat([*parts, frame], ignore_index=True)


def inputs():
    base = source()
    history = base[['end']].assign(**dict.fromkeys(HISTORY_FEATURES, 0.))
    combined = base.copy()
    combined[HISTORY_FEATURES] = history[HISTORY_FEATURES].to_numpy()
    reference = {n: purged_window(combined, *period).reset_index(drop=True) for n, period in PERIODS.items()}
    frame = extend(base)
    history = frame[['end']].assign(**dict.fromkeys(HISTORY_FEATURES, 0.))
    return frame, history, reference


def test_expansion_preserves_original_rows_and_excludes_whole_crossing_episode():
    frame, history, reference = inputs()
    selected, support, ledger = expanded_splits(frame, history, reference)
    assert len(selected['training']) == sum(support[str(y)]['rows'] for y in [2018, 2019, 2020])
    for name in ['calibration', 'diagnosis']:
        pd.testing.assert_frame_equal(selected[name], reference[name], check_exact=True)
    old = selected['training'][selected['training'].end.dt.year.eq(2020)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, reference['training'], check_exact=True)
    assert set(ledger.loc[ledger.included, 'end']) == set(selected['training'].end)
    episode = frame[frame.end.dt.year.eq(2018)].episode_id.iloc[-1]
    crossing = frame[frame.episode_id.eq(episode)].copy()
    crossing['end'] = pd.Timestamp('2019-01-03', tz='UTC')
    crossing['label_end'] = crossing.end+pd.Timedelta(minutes=1)
    # 기존 2019년 시각을 덮지 않고 경계 이후에 같은 포지션을 추가한다.
    crossing['end'] += pd.Timedelta(minutes=1)
    crossing['label_end'] += pd.Timedelta(minutes=1)
    frame = pd.concat([frame, crossing]).sort_values('end').reset_index(drop=True)
    history = frame[['end']].assign(**dict.fromkeys(HISTORY_FEATURES, 0.))
    selected, _, ledger = expanded_splits(frame, history, reference)
    assert episode not in set(selected['training'].episode_id)
    assert set(ledger.loc[ledger.episode_id.eq(episode), 'reason']) == {'crossing_episode', 'entry_before_period'}


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'negative', 'nan', 'old_row', 'support', 'overlap', 'time', 'id'])
def test_expansion_rejects_missing_history_changed_rows_and_invalid_splits(damage):
    frame, history, reference = inputs()
    if damage == 'missing':
        history = history.iloc[1:]
    elif damage == 'duplicate':
        history.loc[1, 'end'] = history.loc[0, 'end']
    elif damage == 'negative':
        history.loc[0, HISTORY_FEATURES[0]] = -1.
    elif damage == 'nan':
        frame.loc[0, 'ret_5m'] = np.nan
    elif damage == 'old_row':
        reference['training'].loc[0, 'ret_5m'] += 1.
    elif damage == 'support':
        frame.loc[frame.end.dt.year.eq(2018), 'y_exit'] = 0
    elif damage == 'overlap':
        frame.loc[frame.end.dt.year.eq(2018), 'episode_id'] = reference['diagnosis'].episode_id.iloc[0]
    elif damage == 'id':
        frame.loc[0, 'episode_id'] = 0
    else:
        frame.loc[0, 'entry_time'] = frame.end.iloc[0]
    with pytest.raises((ValueError, AssertionError)):
        expanded_splits(frame, history, reference)


def test_same_fixed_models_match_independent_refit_and_future_cannot_change_training():
    selected, _, _ = expanded_splits(*inputs())
    model, offset, support, _, scores = fit_history_window(selected)
    x = selected['training'][model.features].to_numpy()
    cx = selected['calibration'][model.features].to_numpy()
    vx = selected['diagnosis'][model.features].to_numpy()
    for i, a in enumerate(ACTIONS):
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, selected['training']['y_'+a])
            expected = learner.predict_proba(np.vstack([x, cx, vx]))[:, 1]
        np.testing.assert_allclose(model.probabilities(np.vstack([x, cx, vx]))[:, i], expected, rtol=0, atol=1e-12)
    changed = copy.deepcopy(selected)
    for a in ACTIONS:
        changed['diagnosis']['y_'+a] = 1-changed['diagnosis']['y_'+a]
    changed['diagnosis'][model.features] *= 2.
    future_model, future_offset, _, _, _ = fit_history_window(changed)
    assert future_model.to_dict() == model.to_dict() and future_offset.to_dict() == offset.to_dict()
    assert support['offset']['max_mean_residual'] < 1e-12
    np.testing.assert_array_equal(scores, offset.predict(HistoryWindowModels.from_dict(model.to_dict()).probabilities(vx)))
    with pytest.raises(ValueError):
        OrderHistoryBoostModels.from_dict(model.to_dict())


def test_all_actions_require_gain_precision_constant_and_raw_preservation():
    metrics = {a: {k: {'log_loss': loss, 'average_precision': ap} for k, loss, ap in [
        ('history_calibrated', .56, .45), ('expanded_calibrated', .5, .5),
        ('expanded_raw', .52, .5), ('constant', .69, .1)]} for a in ACTIONS}
    assert history_window_admission(metrics)['history_window_admitted']
    for action in ACTIONS:
        for baseline, key, value in [('history_calibrated', 'log_loss', .505),
            ('history_calibrated', 'average_precision', .51), ('expanded_raw', 'log_loss', .49),
            ('constant', 'log_loss', .49)]:
            changed = copy.deepcopy(metrics)
            changed[action][baseline][key] = value
            assert not history_window_admission(changed)['history_window_admitted']


def test_full_input_links_reference_scores_models_metrics_and_failures(tmp_path, monkeypatch):
    history, reference, frame, _ = setup(tmp_path, monkeypatch)
    source_run = run_history_calibration_diagnostics(history, reference, tmp_path / 'sources')
    expanded = extend(frame)
    old_features = pd.read_parquet(history / 'history_features.parquet')
    old_features = pd.concat([expanded.loc[expanded.end.dt.year.lt(2020), ['end']].assign(**dict.fromkeys(HISTORY_FEATURES, 0.)), old_features], ignore_index=True)
    old_features.to_parquet(history / 'history_features.parquet', index=False)
    refresh_files(history)
    manifest = json.loads((source_run / 'manifest.json').read_text())
    manifest['settings']['history_files_sha256'] = sha256(history / 'files.json')
    save_json(source_run / 'manifest.json', manifest)
    refresh_files(source_run)
    monkeypatch.setattr('wonyotti_fr.episode_balance.load_selection', calibration_diagnostics.load_selection)
    monkeypatch.setattr('wonyotti_fr.history_window.load_minute_inventory_labels', lambda _: (expanded, None))
    out = run_history_window_diagnosis(source_run, tmp_path / 'runs')
    p = pd.read_parquet(out / 'predictions.parquet')
    old = pd.read_parquet(source_run / 'predictions.parquet')
    for a in ACTIONS:
        np.testing.assert_array_equal(p[a+'_history_calibrated'], old[a+'_histogram_calibrated'])
    metrics = json.loads((out / 'metrics.json').read_text())
    for a, pairs in metrics.items():
        for kind, m in pairs.items():
            assert m == management_metrics(p['y_'+a], p[a+'_'+kind])
    assert json.loads((out / 'decision.json').read_text()) == history_window_admission(metrics)
    assert not list(out.rglob('trades.parquet'))
    assert json.loads((out / 'summary.json').read_text())['training_rows'] > len(frame)
    old_features.loc[0, HISTORY_FEATURES[0]] = np.nan
    old_features.to_parquet(history / 'history_features.parquet', index=False)
    refresh_files(history)
    with pytest.raises(ValueError):
        run_history_window_diagnosis(source_run, tmp_path / 'failures')
    failed, = (tmp_path / 'failures').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()
