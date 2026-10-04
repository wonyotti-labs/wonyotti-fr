import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits

from wonyotti_fr import exit_horizon as eh
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.management_calibration import ManagementOffset
from wonyotti_fr.minute_management import purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels


def source(monkeypatch):
    periods = {'training': ['2020-01-01', '2020-01-05'], 'calibration': ['2020-01-05', '2020-01-09'],
               'diagnosis': ['2020-01-09', '2020-01-13']}
    monkeypatch.setattr(eh, 'PERIODS', periods)
    n = 12*1440
    rng = np.random.default_rng(51)
    frame = pd.DataFrame(rng.normal(size=(n, 50)), columns=eh.ExitHorizonModel.features)
    frame['end'] = pd.date_range('2020-01-01', periods=n, freq='min', tz='UTC')
    frame['entry_time'] = frame.end.dt.floor('h')-pd.Timedelta(seconds=1)
    frame['episode_id'] = np.arange(n)//60+1
    frame['label_end'] = frame.end+pd.Timedelta(minutes=1)
    frame['usable'] = True
    frame['y_exit'] = (np.arange(n)%60 == 49).astype(int)
    frame['y_reduce'] = (np.arange(n)%60 == 9).astype(int)
    frame['y_increase'] = (np.arange(n)%60 == 29).astype(int)
    frame[HISTORY_FEATURES] = 0.
    frame = frame[[c for c in frame.columns if c not in HISTORY_FEATURES] + HISTORY_FEATURES]
    reference = {name: purged_window(frame, *period).reset_index(drop=True) for name, period in periods.items()}
    return frame, reference


def test_half_open_horizon_keeps_episode_and_original_features(monkeypatch):
    frame, _ = source(monkeypatch)
    frame = frame.iloc[:120].copy()
    frame.loc[:, 'y_exit'] = 0
    frame.loc[[15, 59, 60, 119], 'y_exit'] = 1
    result = eh.attach_exit_horizon(frame, frame.end.iloc[-1]+pd.Timedelta(minutes=1))
    for i, row in frame.iterrows():
        matches = frame.loc[frame.episode_id.eq(row.episode_id) & frame.y_exit.eq(1)
            & frame.end.ge(row.end) & frame.end.lt(row.end+eh.HORIZON), 'end']
        assert result.y_exit.iloc[i] == int(len(matches) > 0)
        if len(matches):
            assert result.horizon_event_end.iloc[i] == matches.min()
    assert result.y_exit.iloc[0] == 0 and result.y_exit.iloc[1] == 1
    assert result.horizon_event_end.iloc[59] == frame.end.iloc[59]
    assert result.horizon_event_end.iloc[60] == frame.end.iloc[60]
    assert result.usable.iloc[-15] and not result.usable.iloc[-14:].any()
    pd.testing.assert_frame_equal(result[eh.ExitHorizonModel.features], frame[eh.ExitHorizonModel.features], check_exact=True)
    pd.testing.assert_series_equal(result.original_y_exit, frame.y_exit, check_names=False)


@pytest.mark.parametrize('damage', ['gap', 'duplicate', 'label', 'target', 'episode', 'timezone'])
def test_invalid_horizon_input_is_rejected(monkeypatch, damage):
    frame, _ = source(monkeypatch)
    if damage == 'gap':
        frame = frame.drop(index=12)
    elif damage == 'duplicate':
        frame.loc[12, 'end'] = frame.end.iloc[11]
    elif damage == 'label':
        frame.loc[12, 'label_end'] += pd.Timedelta(minutes=1)
    elif damage == 'target':
        frame.loc[12, 'y_exit'] = 2
    elif damage == 'episode':
        frame.loc[49, 'episode_id'] = -1
    else:
        frame['end'] = frame.end.dt.tz_localize(None)
    with pytest.raises(ValueError):
        eh.attach_exit_horizon(frame, pd.Timestamp('2020-01-13', tz='UTC'))


def prepared(monkeypatch):
    frame, reference = source(monkeypatch)
    targets = eh.attach_exit_horizon(frame, pd.Timestamp('2020-01-13', tz='UTC'))
    rows, support, ledger = eh.horizon_splits(targets, reference)
    train = reference['training']
    model, _ = OrderHistoryBoostModels.fit(train[eh.ExitHorizonModel.features],
        train[['y_exit', 'y_reduce', 'y_increase']], reference['calibration'][eh.ExitHorizonModel.features])
    return frame, reference, targets, rows, support, ledger, model


def test_extended_embargo_and_unchanged_feature_rows(monkeypatch):
    frame, reference = source(monkeypatch)
    targets = eh.attach_exit_horizon(frame, pd.Timestamp('2020-01-13', tz='UTC'))
    rows, support, ledger = eh.horizon_splits(targets, reference)
    assert all(v['removed_original_rows'] == 14 for v in support.values())
    assert set(ledger.loc[~ledger.included, 'reason']) == {'extended_end_embargo'}
    for name, part in rows.items():
        assert set(part.end) == set(ledger.loc[ledger.split.eq(name) & ledger.included, 'end'])
        assert not set(part.episode_id) & set(rows['diagnosis' if name == 'training' else 'training'].episode_id)
    damaged = copy.deepcopy(reference)
    damaged['training'].loc[0, 'ret_5m'] += 1.
    with pytest.raises(AssertionError):
        eh.horizon_splits(targets, damaged)


def test_fit_matches_independent_library_offsets_thresholds_and_future_isolation(monkeypatch):
    _, _, _, rows, _, _, baseline = prepared(monkeypatch)
    model, offsets, scores, thresholds, support = eh.fit_exit_horizon(rows, baseline)
    train, calibration, diagnosis = (rows[n] for n in eh.PERIODS)
    x, cx, vx = (p[model.features].to_numpy() for p in [train, calibration, diagnosis])
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, train.y_exit)
        expected = learner.predict_proba(np.vstack([x, cx, vx]))[:, 1]
    np.testing.assert_allclose(model.probabilities(np.vstack([x, cx, vx]))[:, 0], expected, atol=1e-12, rtol=0)
    for name, raw in [('candidate', model.probabilities(cx)[:, 0]), ('original', baseline.probabilities(cx)[:, 0])]:
        clipped = np.clip(raw, np.finfo(float).eps, 1-np.finfo(float).eps)
        logits = np.log(clipped)-np.log1p(-clipped)
        delta = brentq(lambda v, logits=logits: (1/(1+np.exp(-(logits+v)))).mean()-calibration.y_exit.mean(), -64, 64)
        assert offsets[name].offsets[0] == pytest.approx(delta, abs=1e-12)
        probabilities = 1/(1+np.exp(-(logits+delta)))
        precision, recall, ts = precision_recall_curve(calibration.y_exit, probabilities)
        f2 = np.divide(5*precision[:-1]*recall[:-1], 4*precision[:-1]+recall[:-1],
            out=np.zeros_like(ts), where=(4*precision[:-1]+recall[:-1]) > 0)
        counts = np.array([(probabilities >= t).sum() for t in ts])
        f2[counts < 20] = -1
        best = np.flatnonzero(np.isclose(f2, f2.max(), atol=1e-15, rtol=0))[-1]
        assert thresholds[name] == pytest.approx(ts[best], abs=1e-12)
    assert support['offset']['max_mean_residual'] < 1e-12
    changed = copy.deepcopy(rows)
    changed['diagnosis']['y_exit'] = 1-changed['diagnosis']['y_exit']
    changed['diagnosis'][model.features] *= 2
    future, future_offsets, _, future_thresholds, _ = eh.fit_exit_horizon(changed, baseline)
    assert model.to_dict() == future.to_dict() and thresholds == future_thresholds
    assert {k: v.to_dict() for k, v in offsets.items()} == {k: v.to_dict() for k, v in future_offsets.items()}
    with pytest.raises(ValueError):
        OrderHistoryBoostModels.from_dict(model.to_dict())
    np.testing.assert_array_equal(scores['candidate'], eh.exit_offset_predict(offsets['candidate'], model.probabilities(vx)[:, 0]))


def test_event_detection_uses_strict_fifteen_minutes_and_same_episode(monkeypatch):
    frame, _ = source(monkeypatch)
    frame = eh.attach_exit_horizon(frame.iloc[:120], pd.Timestamp('2020-01-02', tz='UTC'))
    scores = np.zeros(len(frame))
    scores[34] = scores[95] = 1.
    metrics, events = eh.horizon_requests(frame, scores, .5)
    assert events.detected.tolist() == [False, True]
    assert metrics['event_recall'] == .5 and metrics['original_events'] == 2
    assert metrics['true_positive_minutes'] == 1 and metrics['f2'] == pytest.approx(5/(4*30+2))


def test_all_probability_and_event_gates_are_required():
    metrics = {k: {'log_loss': loss, 'average_precision': ap} for k, loss, ap in [
        ('original', .4, .4), ('candidate', .3, .5), ('candidate_raw', .35, .5), ('constant', .5, .1)]}
    requests = {'original': {'f2': .4, 'event_recall': .5}, 'candidate': {'f2': .5, 'event_recall': .6}}
    assert eh.exit_horizon_admission(metrics, requests)['exit_horizon_admitted']
    for key, value in [('log_loss', .399), ('average_precision', .3)]:
        changed = copy.deepcopy(metrics)
        changed['candidate'][key] = value
        assert not eh.exit_horizon_admission(changed, requests)['exit_horizon_admitted']
    for key in ['f2', 'event_recall']:
        changed = copy.deepcopy(requests)
        changed['candidate'][key] = .1
        assert not eh.exit_horizon_admission(metrics, changed)['exit_horizon_admitted']


def test_complete_runner_preserves_inputs_hashes_and_failed_runs(tmp_path, monkeypatch):
    frame, reference, _, rows, _, _, baseline = prepared(monkeypatch)
    source, history, labels, audit = [tmp_path/n for n in ['source', 'history', 'labels', 'audit']]
    for path in [source, history, labels, audit]:
        path.mkdir()
    pd.DataFrame({'time': [pd.Timestamp('2020-01-13', tz='UTC')]}).to_parquet(audit/'actions.parquet', index=False)
    save_json(labels/'files.json', {})
    settings = {'selection_sha256': 'fixed', 'labels_files_sha256': sha256(labels/'files.json'),
        'audit': str(audit), 'audit_sha256': {'actions.parquet': sha256(audit/'actions.parquet')}, 'new_features': HISTORY_FEATURES}
    save_json(history/'manifest.json', {'settings': settings})
    save_json(history/'summary.json', {'complete': True})
    frame[['end', *HISTORY_FEATURES]].to_parquet(history/'history_features.parquet', index=False)
    save_json(history/'files.json', {p.name: sha256(p) for p in history.iterdir()})
    save_json(source/'manifest.json', {'settings': {**settings, 'history': str(history), 'labels': str(labels),
        'history_files_sha256': sha256(history/'files.json')}})
    save_json(source/'models.json', {'histogram': baseline.to_dict()})
    cy = reference['calibration'][['y_exit', 'y_reduce', 'y_increase']].to_numpy()
    offset, _ = ManagementOffset.fit(baseline.probabilities(reference['calibration'][baseline.features]), cy)
    save_json(source/'offsets.json', {'histogram': offset.to_dict()})
    save_json(source/'files.json', {p.name: sha256(p) for p in source.iterdir()})
    monkeypatch.setattr(eh, 'load_episode_reference', lambda _: (reference, {}))
    monkeypatch.setattr(eh, 'load_minute_inventory_labels', lambda _: (frame.drop(columns=HISTORY_FEATURES), None))
    loaded, _, _, _, _ = eh.load_exit_horizon_inputs(source)
    for name in rows:
        pd.testing.assert_frame_equal(loaded[name][rows[name].columns], rows[name], check_exact=True)
    out = eh.run_exit_horizon_diagnosis(source, tmp_path/'runs')
    assert json.loads((out/'summary.json').read_text())['complete']
    assert not list(out.rglob('trades.parquet'))
    assert all(sha256(out/n) == digest for n, digest in json.loads((out/'files.json').read_text()).items())
    (audit/'actions.parquet').write_text('changed')
    with pytest.raises(ValueError, match='원본'):
        eh.run_exit_horizon_diagnosis(source, tmp_path/'failed')
    assert len(list((tmp_path/'failed').glob('*/failure.json'))) == 1
