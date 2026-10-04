import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_boosted_direction import admission_fixture
from threadpoolctl import threadpool_limits

from wonyotti_fr.activity_diagnostics import ACTIVITY_PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.expansion_model import BinaryModel
from wonyotti_fr.joint_entry import (
    JOINT_SETTINGS,
    JointEntryModel,
    factorized_scores,
    joint_targets,
    validate_joint_scores,
)
from wonyotti_fr.joint_research import (
    joint_admission,
    joint_metrics,
    joint_reference,
    run_joint_diagnostics,
)


def rows():
    result = {}
    rng = np.random.default_rng(103)
    for name, year, count, offset in [('training', 2020, 1200, 0), ('diagnosis', 2021, 300, 2000)]:
        times = pd.date_range(f'{year}-01-03', periods=count, freq='h', tz='UTC')
        y = np.arange(count) % 3
        frame = pd.DataFrame(rng.normal(size=(count, 14)), columns=MARKET_FEATURES)
        frame['end'], frame['label_end'] = times, times+pd.Timedelta(minutes=5)
        frame['active'], frame['buy'] = (y != 0).astype(int), np.where(y == 0, np.nan, y == 2)
        frame['episode_id'] = 0
        frame['target_episode_id'] = np.where(y == 0, 0, offset+np.arange(count)+1)
        frame['target_time'] = pd.Series(times+pd.Timedelta(minutes=1)).where(y != 0)
        frame['usable'] = True
        result[name] = frame
    return result


def refresh(root):
    save_json(root / 'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'files.json'})


def references(tmp_path):
    frames = rows()
    activity, direction = tmp_path / 'activity', tmp_path / 'direction'
    activity.mkdir()
    direction.mkdir()
    train, val = frames.values()
    a, _ = BinaryModel.fit(train[MARKET_FEATURES].to_numpy(), train.active.to_numpy(), 'logistic')
    d = BinaryModel.from_dict({'format': 'expansion_binary_v1', 'kind': 'boosted', 'features': MARKET_FEATURES,
        'learning_rate': .05, 'trees': [{'left': [-1], 'right': [-1], 'feature': [-2],
            'threshold': [-2.], 'value': [1.]} for _ in range(64)]})
    hashes = {'audit_sha256': {}, 'history_sha256': 'synthetic', 'events_sha256': 'synthetic'}
    save_json(activity / 'manifest.json', {'settings': {**hashes, 'periods': ACTIVITY_PERIODS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V42.md')), 'training_quantile': .975}})
    save_json(activity / 'summary.json', {'complete': True})
    save_json(activity / 'models.json', {'logistic': a.to_dict()})
    for n, frame in frames.items():
        frame.to_parquet(activity / f'{n}_used.parquet', index=False)
    pred = val[['end', 'active']].copy()
    pred['logistic'] = a.probabilities(val[MARKET_FEATURES].to_numpy())
    pred.to_parquet(activity / 'predictions.parquet', index=False)
    refresh(activity)
    evidence = admission_fixture()
    evidence['settings'].update(hashes, protocol_sha256=sha256(Path('docs/EXPERIMENT_V28.md')))
    active = val[val.active.eq(1)]
    evidence['summary']['diagnosis_rows'] = len(active)
    for metric in evidence['metrics'].values():
        metric['rows'] = len(active)
    for key in ['summary', 'metrics', 'decision', 'training_support']:
        save_json(direction / f'{key}.json', evidence[key])
    save_json(direction / 'manifest.json', {'settings': evidence['settings']})
    save_json(direction / 'models.json', {'boosted': d.to_dict()})
    active.to_parquet(direction / 'diagnosis_used.parquet', index=False)
    train.to_parquet(direction / 'training_used.parquet', index=False)
    pred = active[['end', 'target_episode_id', 'buy']].copy()
    pred['boosted'] = d.probabilities(active[MARKET_FEATURES].to_numpy())
    pred.to_parquet(direction / 'predictions.parquet', index=False)
    refresh(direction)
    return activity, direction


def test_multinomial_export_matches_independent_library_and_manual_softmax():
    rng = np.random.default_rng(104)
    x, vx = rng.normal(size=(1800, 14)), rng.normal(size=(100, 14))
    y = np.argmax(np.column_stack([x[:, 0], -x[:, 0], x[:, 1]]), axis=1)
    model, _ = JointEntryModel.fit(x, y)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**JOINT_SETTINGS).fit(scaler.transform(x), y)
        expected = learner.predict_proba(scaler.transform(vx))
    np.testing.assert_allclose(model.probabilities(vx), expected, atol=1e-12, rtol=0)
    manual = softmax(scaler.transform(vx) @ learner.coef_.T+learner.intercept_, axis=1)
    np.testing.assert_allclose(model.probabilities(vx), manual, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.probabilities(vx[:3]), np.vstack([model.probabilities(v[None, :]) for v in vx[:3]]), atol=1e-14, rtol=0)
    assert np.isnan(model.probabilities(np.full((1, 14), np.nan))).all()
    np.testing.assert_array_equal(model.probabilities(vx), JointEntryModel.from_dict(model.to_dict()).probabilities(vx))


@pytest.mark.parametrize('damage', ['classes', 'coefficients', 'scale', 'nan', 'settings'])
def test_joint_loader_rejects_changed_class_order_and_numeric_parameters(damage):
    frame = rows()['training']
    model, _ = JointEntryModel.fit(frame[MARKET_FEATURES], joint_targets(frame))
    data = model.to_dict()
    if damage == 'classes':
        data['classes'] = ['hold', 'long', 'short']
    elif damage == 'coefficients':
        data['coefficients'] = data['coefficients'][:2]
    elif damage == 'scale':
        data['scale'][0] = 0
    elif damage == 'nan':
        data['intercepts'][0] = np.nan
    else:
        data['settings']['C'] = 1.
    with pytest.raises(ValueError):
        JointEntryModel.from_dict(data)


def test_factorized_mass_and_joint_targets_have_explicit_meaning():
    result = factorized_scores([.2, .4, 0.], [.75, .25, .9])
    np.testing.assert_allclose(result, [[.8, .05, .15], [.6, .3, .1], [1., 0., 0.]], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(joint_targets(rows()['training'])[:3], [0, 1, 2])
    for bad in [np.zeros((3, 3)), np.ones((3, 2)), np.full((3, 3), np.nan), [[1.1, -.1, 0.]]]:
        with pytest.raises(ValueError):
            validate_joint_scores(bad)
    with pytest.raises(ValueError):
        factorized_scores([.1], [1.1])


def test_joint_metrics_match_manual_loss_and_require_all_admission_criteria():
    y = np.tile(np.arange(3), 20)
    p = np.full((60, 3), .1)
    p[np.arange(60), y] = .8
    values = joint_metrics(y, p, .5)
    assert abs(values['log_loss']+np.log(.8)) < 1e-14
    assert values['long']['correct_requests'] == values['short']['correct_requests'] == 20
    baseline = {'log_loss': .6, 'activity': {'log_loss': .4},
        'short': {'average_precision': .4}, 'long': {'average_precision': .4}}
    new = {'log_loss': .58, 'activity': {'log_loss': .39},
        'short': {'average_precision': .45}, 'long': {'average_precision': .45}}
    metrics = {'joint': new, 'factorized': baseline, 'constant': {'log_loss': .7}}
    assert joint_admission(metrics)['joint_entry_admitted']
    for key in ['gain', 'constant', 'activity', 'short', 'long']:
        changed = copy.deepcopy(metrics)
        if key == 'gain':
            changed['joint']['log_loss'] = .594
        elif key == 'constant':
            changed['constant']['log_loss'] = .58
        elif key == 'activity':
            changed['joint'][key]['log_loss'] = .41
        else:
            changed['joint'][key]['average_precision'] = .39
        assert not joint_admission(changed)['joint_entry_admitted']


def test_full_joint_pipeline_preserves_baselines_and_future_cannot_fit(tmp_path):
    activity, direction = references(tmp_path)
    frames, _ = joint_reference(activity, direction)
    out = run_joint_diagnostics(activity, direction, tmp_path / 'runs')
    p = pd.read_parquet(out / 'predictions.parquet')
    thresholds = json.loads((out / 'thresholds.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    for name in metrics:
        assert joint_metrics(p.target.to_numpy(), p[[name+'_'+s for s in ['hold', 'short', 'long']]].to_numpy(), thresholds[name]) == metrics[name]
    validation = frames['diagnosis']
    validation['buy'] = 1-validation.buy
    validation[MARKET_FEATURES] *= -4
    validation.to_parquet(activity / 'diagnosis_used.parquet', index=False)
    a_model = BinaryModel.from_dict(json.loads((activity / 'models.json').read_text())['logistic'])
    pred = validation[['end', 'active']].copy()
    pred['logistic'] = a_model.probabilities(validation[MARKET_FEATURES].to_numpy())
    pred.to_parquet(activity / 'predictions.parquet', index=False)
    refresh(activity)
    active = validation[validation.active.eq(1)]
    active.to_parquet(direction / 'diagnosis_used.parquet', index=False)
    d_model = BinaryModel.from_dict(json.loads((direction / 'models.json').read_text())['boosted'])
    pred = active[['end', 'target_episode_id', 'buy']].copy()
    pred['boosted'] = d_model.probabilities(active[MARKET_FEATURES].to_numpy())
    pred.to_parquet(direction / 'predictions.parquet', index=False)
    refresh(direction)
    changed = run_joint_diagnostics(activity, direction, tmp_path / 'changed')
    for name in ['model.json', 'thresholds.json', 'training_support.json']:
        assert (out / name).read_bytes() == (changed / name).read_bytes()
    assert not list(out.rglob('trades.parquet'))
    old = json.loads((direction / 'manifest.json').read_text())
    old['settings']['training_period'][1] = '2022-01-01'
    save_json(direction / 'manifest.json', old)
    refresh(direction)
    with pytest.raises(ValueError):
        run_joint_diagnostics(activity, direction, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()


@pytest.mark.parametrize('kind', ['few_class', 'missing_class', 'fractional', 'features'])
def test_joint_fit_rejects_weak_class_support_and_invalid_inputs(kind):
    frame = rows()['training']
    x, y = frame[MARKET_FEATURES].to_numpy(copy=True), joint_targets(frame)
    if kind == 'few_class':
        y[y == 1] = 0
        y[:19] = 1
    elif kind == 'missing_class':
        y[y == 1] = 0
    elif kind == 'fractional':
        y = y.astype(float)+.1
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        JointEntryModel.fit(x, y)
