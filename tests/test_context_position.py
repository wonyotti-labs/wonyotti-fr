import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_position_target import reference as position_fixture
from test_position_target import source
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.context_position import (
    CONTEXT_FEATURES,
    CONTEXT_WINDOW,
    ContextPositionModel,
    attach_context,
    context_admission,
    context_features,
    context_reference,
    run_context_diagnostics,
)
from wonyotti_fr.event_features import MARKET_FEATURES, event_features
from wonyotti_fr.joint_entry import JOINT_SETTINGS
from wonyotti_fr.position_target import PositionTargetModel, run_position_diagnostics


def bars(count=21000, start='2019-09-01'):
    time = pd.date_range(start, periods=count, freq='5min', tz='UTC')
    steps = np.arange(count)
    close = 100*np.exp(.000003*steps+.03*np.sin(steps/100))
    return pd.DataFrame({'time': time, 'end': time+pd.Timedelta(minutes=5),
        'open': close, 'high': close*1.01, 'low': close*.99, 'close': close,
        'volume': 1+steps % 17})


def test_context_features_match_explicit_past_calculations_and_future_invariance():
    frame = bars()
    result = context_features(frame)
    assert result[CONTEXT_FEATURES].iloc[:CONTEXT_WINDOW].isna().all().all()
    assert np.isfinite(result[CONTEXT_FEATURES].iloc[CONTEXT_WINDOW:]).all().all()
    i = 20000
    for day in [4, 16, 64]:
        assert abs(result.loc[i, f'ret_{day}d']-(frame.close.iloc[i]/frame.close.iloc[i-day*288]-1)) < 1e-14
    means = {}
    for day in [1, 4, 16, 64]:
        value = frame.close.iloc[0]
        alpha = 2/(day*288+1)
        for p in frame.close.iloc[1:i+1]:
            value = (1-alpha)*value+alpha*p
        means[day] = value
    for first, last in [(1, 4), (4, 16), (16, 64)]:
        assert abs(result.loc[i, f'trend_{first}d_{last}d']-(means[first]/means[last]-1)) < 1e-12
    ret = np.diff(np.log(frame.close.iloc[:i+1]))
    assert abs(result.loc[i, 'vol_16d']-np.std(ret[-16*288:], ddof=1)*np.sqrt(16*288)) < 1e-12
    assert result.loc[i, 'volume_ratio_16d'] == frame.volume.iloc[i]/frame.volume.iloc[i-16*288+1:i+1].mean()
    changed = frame.copy()
    changed.loc[20001:, ['open', 'high', 'low', 'close']] *= 2
    changed.loc[20001:, 'volume'] *= 3
    pd.testing.assert_frame_equal(result.iloc[:20001], context_features(changed).iloc[:20001], check_exact=True)


def test_time_gap_restarts_context_without_dropping_rows_or_filling_missing_values():
    frame = bars(40000)
    frame.loc[20000:, ['time', 'end']] += pd.Timedelta(minutes=5)
    result = context_features(frame)
    assert len(result) == len(frame)
    assert np.isfinite(result[CONTEXT_FEATURES].iloc[19999]).all()
    assert result[CONTEXT_FEATURES].iloc[20000:20000+CONTEXT_WINDOW].isna().all().all()
    fresh = context_features(frame.iloc[20000:].reset_index(drop=True))
    pd.testing.assert_frame_equal(result.iloc[20000:].reset_index(drop=True), fresh, check_exact=True)


@pytest.mark.parametrize('damage', ['order', 'duplicate', 'price', 'volume', 'time', 'infinite'])
def test_context_invalid_market_input_rejected(damage):
    frame = bars(300)
    if damage == 'order':
        frame = frame.iloc[::-1]
    elif damage == 'duplicate':
        frame.loc[2, 'time'] = frame.time.iloc[1]
    elif damage == 'price':
        frame.loc[0, 'close'] = 0
    elif damage == 'volume':
        frame.loc[0, 'volume'] = -1
    elif damage == 'time':
        frame.loc[0, 'end'] += pd.Timedelta(seconds=1)
    else:
        frame.loc[0, 'high'] = np.inf
    with pytest.raises(ValueError):
        context_features(frame)


def test_context_attachment_preserves_original_features_and_rejects_unsupported_rows():
    frame = bars()
    features = event_features(frame).iloc[19000:].reset_index(drop=True)
    result = attach_context({'training': features}, frame)['training']
    pd.testing.assert_frame_equal(result[features.columns], features, check_exact=True)
    unsupported = event_features(frame).iloc[300:1300].reset_index(drop=True)
    with pytest.raises(ValueError):
        attach_context({'training': unsupported}, frame)
    changed = features.copy()
    changed.loc[0, 'ret_1d'] += .1
    with pytest.raises(AssertionError):
        attach_context({'training': changed}, frame)


def test_extended_model_matches_independent_library_and_rejects_legacy_schema():
    rng = np.random.default_rng(445)
    x = rng.normal(size=(1800, 22))
    y = np.argmax(np.c_[x[:, 14], x[:, 15], -x[:, 14]], axis=1)
    model, _ = ContextPositionModel.fit(x, y)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**JOINT_SETTINGS).fit(scaler.transform(x), y)
        expected = learner.predict_proba(scaler.transform(x))
    np.testing.assert_allclose(model.probabilities(x), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.probabilities(x), softmax(scaler.transform(x)@learner.coef_.T+learner.intercept_, axis=1), atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        PositionTargetModel.from_dict(model.to_dict())


def test_context_admission_requires_previous_constant_and_monthly_improvement():
    def metric(loss, precision):
        return {'log_loss': loss, 'short': {'average_precision': precision}, 'long': {'average_precision': precision}}
    metrics = {'context': metric(.55, .6), 'previous': metric(.6, .5), 'constant': metric(.69, .4)}
    months = [{'month': f'2021-{m:02}', 'context_log_loss': .55 if m <= 8 else .7,
        'previous_log_loss': .6, 'constant_log_loss': .69} for m in range(1, 13)]
    assert context_admission(metrics, months)['context_position_admitted']
    for damage in ['previous_gain', 'constant_gain', 'short', 'long', 'months']:
        changed, changed_months = copy.deepcopy(metrics), copy.deepcopy(months)
        if damage == 'previous_gain':
            changed['previous']['log_loss'] = .55/.99
        elif damage == 'constant_gain':
            changed['constant']['log_loss'] = .55/.99
        elif damage == 'months':
            changed_months[0]['context_log_loss'] = .7
        else:
            changed['context'][damage]['average_precision'] = .49
        assert not context_admission(changed, changed_months)['context_position_admitted']


def fixture(tmp_path, monkeypatch, future=False):
    activity = position_fixture(tmp_path, monkeypatch, future_flip=future)
    frame = bars(250000)
    if future:
        frame.loc[frame.time.ge(pd.Timestamp('2021-01-01', tz='UTC')), ['open', 'high', 'low', 'close']] *= 1.5
    history = tmp_path / 'history'
    history.mkdir(exist_ok=True)
    frame.to_parquet(history / 'bars.parquet', index=False)
    digest = sha256(history / 'bars.parquet')
    save_json(history / 'manifest.json', {'file': 'bars.parquet', 'sha256': digest})
    features = event_features(frame).set_index('end')
    for n in ['training', 'diagnosis']:
        rows = pd.read_parquet(activity / f'{n}_used.parquet')
        rows[MARKET_FEATURES] = features.loc[rows.end, MARKET_FEATURES].to_numpy()
        rows.to_parquet(activity / f'{n}_used.parquet', index=False)
    manifest = json.loads((activity / 'manifest.json').read_text())
    manifest['settings'].update(history=str(history), history_sha256=digest)
    save_json(activity / 'manifest.json', manifest)
    save_json(activity / 'files.json', {p.name: sha256(p) for p in activity.iterdir() if p.name != 'files.json'})
    _, actions, episodes = source(future_flip=future)
    monkeypatch.setattr('wonyotti_fr.position_target.source_inputs',
        lambda *_: ({'actions': actions, 'episodes': episodes}, {'synthetic': True, 'history_sha256': digest}))
    return run_position_diagnostics(activity, tmp_path / 'previous')


def test_full_context_pipeline_preserves_reference_and_future_cannot_fit(tmp_path, monkeypatch):
    reference = fixture(tmp_path, monkeypatch)
    frames, scores, frequencies, _ = context_reference(reference)
    assert len(frames['training']) == 1200 and scores.shape == (360, 3)
    out = run_context_diagnostics(reference, tmp_path / 'runs')
    metrics = json.loads((out / 'metrics.json').read_text())
    old = json.loads((reference / 'metrics.json').read_text())
    assert metrics['previous'] == old['position'] and metrics['constant'] == old['constant']
    changed_ref = fixture(tmp_path, monkeypatch, future=True)
    changed = run_context_diagnostics(changed_ref, tmp_path / 'changed')
    assert (out / 'model.json').read_bytes() == (changed / 'model.json').read_bytes()
    np.testing.assert_array_equal(frequencies, json.loads((changed / 'training_support.json').read_text())['training_frequencies'])
    assert not list(out.rglob('trades.parquet'))
    (changed_ref / 'training_used.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError):
        run_context_diagnostics(changed_ref, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()

