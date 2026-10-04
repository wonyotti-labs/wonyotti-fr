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
from wonyotti_fr.episode_balance import (
    EpisodeBalanceModels,
    episode_balance_admission,
    episode_log_loss,
    episode_weights,
    fit_episode_candidate,
    load_episode_reference,
    run_episode_balance_diagnosis,
)
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.history_calibration_diagnostics import run_history_calibration_diagnostics
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


def test_weights_equal_episode_totals_and_preserve_original_order():
    ids = np.r_[np.ones(1000, dtype=int), np.arange(2, 2002)]
    w, support = episode_weights(ids)
    sums = pd.Series(w).groupby(ids).sum()
    np.testing.assert_allclose(sums, 3000/2001, rtol=1e-12, atol=1e-12)
    assert np.isclose(w.mean(), 1.) and support['effective_rows'] >= 1000
    assert np.isclose(support['effective_rows'], w.sum()**2/np.square(w).sum())
    shuffled = np.random.default_rng(42).permutation(len(ids))
    permuted, _ = episode_weights(ids[shuffled])
    np.testing.assert_array_equal(permuted, w[shuffled])


@pytest.mark.parametrize('kind', ['float', 'negative', 'missing', 'shape', 'empty', 'few_effective'])
def test_weights_reject_invalid_identifiers_and_weak_effective_support(kind):
    ids = np.arange(1, 1201)
    if kind == 'float':
        ids = ids.astype(float)
    elif kind == 'negative':
        ids[0] = -1
    elif kind == 'missing':
        ids = np.array([None]*1200)
    elif kind == 'shape':
        ids = ids[:, None]
    elif kind == 'empty':
        ids = np.array([], dtype=int)
    else:
        ids[:1190] = 1
    with pytest.raises(ValueError):
        episode_weights(ids)


def test_weighted_numeric_models_match_independent_library_and_validation_cannot_fit():
    rng = np.random.default_rng(94)
    x = rng.normal(size=(3000, 50))
    vx = rng.normal(size=(100, 50))
    y = np.column_stack([x[:, i]+rng.normal(size=len(x)) > .5 for i in range(3)]).astype(int)
    ids = np.r_[np.ones(1000, dtype=int), np.arange(2, 2002)]
    model, support, weights = EpisodeBalanceModels.fit(x, y, vx, ids)
    for i in range(3):
        with threadpool_limits(limits=1):
            expected = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y[:, i], sample_weight=weights)
            scores = expected.predict_proba(np.vstack([x, vx]))[:, 1]
        np.testing.assert_allclose(model.probabilities(np.vstack([x, vx]))[:, i], scores, atol=1e-12, rtol=0)
    changed, _, changed_weights = EpisodeBalanceModels.fit(x, y, -vx, ids)
    assert model.to_dict() == changed.to_dict()
    np.testing.assert_array_equal(weights, changed_weights)
    np.testing.assert_array_equal(model.probabilities(vx), EpisodeBalanceModels.from_dict(model.to_dict()).probabilities(vx))
    assert all(v > 0 for v in support['weighting']['effective_positive'].values())
    with pytest.raises(ValueError):
        OrderHistoryBoostModels.from_dict(model.to_dict())


@pytest.mark.parametrize('kind', ['shape', 'negative', 'mean', 'nan'])
def test_histogram_weight_interface_rejects_invalid_values(kind):
    x, y = np.zeros((1200, 50)), np.tile(np.r_[np.ones(600), np.zeros(600)][:, None], (1, 3))
    weights = np.ones(len(x))
    if kind == 'shape':
        weights = weights[:, None]
    elif kind == 'negative':
        weights[0] = -1
    elif kind == 'mean':
        weights *= 2
    else:
        weights[0] = np.nan
    with pytest.raises(ValueError):
        OrderHistoryBoostModels.fit(x, y, x[:100], sample_weight=weights)


def splits():
    frame = source()
    frame[HISTORY_FEATURES] = 0.
    return {n: purged_window(frame, *period).reset_index(drop=True) for n, period in PERIODS.items()}


def test_future_labels_do_not_change_weights_models_or_calibration():
    rows = splits()
    model, offset, weights, support, _, scores = fit_episode_candidate(rows)
    for a in ACTIONS:
        rows['diagnosis']['y_'+a] = 1-rows['diagnosis']['y_'+a]
    changed, changed_offset, changed_weights, _, _, changed_scores = fit_episode_candidate(rows)
    assert model.to_dict() == changed.to_dict() and offset.to_dict() == changed_offset.to_dict()
    np.testing.assert_array_equal(weights, changed_weights)
    np.testing.assert_array_equal(scores, changed_scores)
    assert support['offset']['max_mean_residual'] < 1e-12
    for damage in ['overlap', 'time', 'feature']:
        bad = copy.deepcopy(rows)
        if damage == 'overlap':
            bad['calibration'].loc[0, 'episode_id'] = bad['training'].episode_id.iloc[0]
        elif damage == 'time':
            bad['calibration'].loc[0, 'label_end'] = pd.Timestamp('2021-07-01', tz='UTC')
        else:
            bad['diagnosis'].loc[0, 'ret_5m'] = np.nan
        with pytest.raises((ValueError, AssertionError)):
            fit_episode_candidate(bad)


def test_all_four_baselines_and_raw_loss_remain_required():
    metrics = {a: {k: {'log_loss': loss, 'average_precision': ap} for k, loss, ap in [
        ('original_v21', .6, .4), ('logistic_calibrated', .59, .4), ('previous_calibrated', .58, .4),
        ('history_calibrated', .56, .45), ('histogram_calibrated', .5, .5),
        ('histogram_raw', .52, .5), ('constant', .69, .1)]} for a in ACTIONS}
    assert episode_balance_admission(metrics)['episode_balanced_admitted']
    for k in ['original_v21', 'logistic_calibrated', 'previous_calibrated', 'history_calibrated', 'histogram_raw']:
        changed = copy.deepcopy(metrics)
        changed['reduce'][k]['log_loss'] = .49
        assert not episode_balance_admission(changed)['episode_balanced_admitted']
    changed = copy.deepcopy(metrics)
    changed['increase']['history_calibrated']['average_precision'] = .51
    assert not episode_balance_admission(changed)['episode_balanced_admitted']


def test_episode_metric_matches_manual_per_position_mean():
    frame = pd.DataFrame({'episode_id': [1, 1, 2], **{'y_'+a: [0, 1, 1] for a in ACTIONS}})
    scores = np.tile(np.array([.1, .8, .7])[:, None], (1, 3))
    expected = ((-np.log(.9)-np.log(.8))/2-np.log(.7))/2
    assert all(abs(v-expected) < 1e-15 for v in episode_log_loss(frame, scores).values())


def test_full_reference_scores_and_failure_preservation(tmp_path, monkeypatch):
    history, reference, _, _ = setup(tmp_path, monkeypatch)
    source_run = run_history_calibration_diagnostics(history, reference, tmp_path / 'source_runs')
    monkeypatch.setattr('wonyotti_fr.episode_balance.load_selection', calibration_diagnostics.load_selection)
    out = run_episode_balance_diagnosis(source_run, tmp_path / 'runs')
    predictions = pd.read_parquet(out / 'predictions.parquet')
    previous = pd.read_parquet(source_run / 'predictions.parquet')
    metrics = json.loads((out / 'metrics.json').read_text())
    for a in ACTIONS:
        np.testing.assert_array_equal(predictions[a+'_history_calibrated'], previous[a+'_histogram_calibrated'])
        for kind in metrics[a]:
            assert metrics[a][kind] == management_metrics(predictions['y_'+a], predictions[a+'_'+kind])
    assert json.loads((out / 'decision.json').read_text()) == episode_balance_admission(metrics)
    assert not list(out.rglob('trades.parquet'))
    rows = pd.read_parquet(source_run / 'training_used.parquet')
    rows.loc[0, 'ret_5m'] = np.nan
    rows.to_parquet(source_run / 'training_used.parquet', index=False)
    refresh_files(source_run)
    with pytest.raises(ValueError):
        run_episode_balance_diagnosis(source_run, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()
    files = json.loads((source_run / 'files.json').read_text())
    files['../outside'] = '0'*64
    (source_run / 'files.json').write_text(json.dumps(files))
    with pytest.raises(ValueError):
        load_episode_reference(source_run)
