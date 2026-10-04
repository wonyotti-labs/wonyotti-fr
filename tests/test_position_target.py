import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.activity_diagnostics import ACTIVITY_PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.joint_entry import JOINT_SETTINGS, JointEntryModel
from wonyotti_fr.position_target import (
    PositionTargetModel,
    position_admission,
    position_metrics,
    position_reference,
    position_targets,
    run_position_diagnostics,
)


def source(future_flip=False):
    rng = np.random.default_rng(444)
    rows, actions, episodes = {}, [], []
    old_qty, episode_id = 0, 0

    def action(time, quantity):
        nonlocal old_qty, episode_id
        if np.sign(quantity) != np.sign(old_qty):
            if old_qty:
                episodes[-1]['exit_time'] = time
            if quantity:
                episode_id += 1
                episodes.append({'episode_id': episode_id, 'entry_time': time, 'exit_time': pd.NaT})
        actions.append({'time': time, 'before_qty': old_qty, 'after_qty': quantity, 'episode_id': episode_id})
        old_qty = quantity

    for name, year, per_month in [('training', 2020, 100), ('diagnosis', 2021, 30)]:
        parts = []
        for month in range(1, 13):
            times = pd.date_range(f'{year}-{month:02}-03', periods=per_month, freq='10min', tz='UTC')
            for i, t in enumerate(times):
                sign = -1 if future_flip and name == 'diagnosis' else 1
                current, target = (i % 3-1)*sign, ((i+1) % 3-1)*sign
                action(t-pd.Timedelta(minutes=1), current)
                current_id = episode_id if current else 0
                action(t, -current if current else 1)
                action(t+pd.Timedelta(minutes=1), -target if target else 1)
                action(t+pd.Timedelta(minutes=1), target)
                action(t+pd.Timedelta(minutes=5), 0)
                row = dict(zip(MARKET_FEATURES, rng.normal(size=14)*sign, strict=True))
                row.update(end=t, label_end=t+pd.Timedelta(minutes=5), direction=current,
                    episode_id=current_id, usable=True)
                parts.append(row)
        rows[name] = pd.DataFrame(parts)
    return rows, pd.DataFrame(actions), pd.DataFrame(episodes)


def reference(tmp_path, monkeypatch, future_flip=False):
    frames, actions, episodes = source(future_flip)
    root = tmp_path / 'activity'
    root.mkdir(exist_ok=True)
    save_json(root / 'manifest.json', {'settings': {'periods': ACTIVITY_PERIODS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V42.md')), 'synthetic': True,
        'audit': 'synthetic', 'study': 'synthetic', 'history': 'synthetic'}})
    save_json(root / 'summary.json', {'complete': True})
    for n, frame in frames.items():
        frame.to_parquet(root / f'{n}_used.parquet', index=False)
    save_json(root / 'files.json', {p.name: sha256(p) for p in root.iterdir() if p.name != 'files.json'})
    monkeypatch.setattr('wonyotti_fr.position_target.source_inputs',
        lambda *_: ({'actions': actions, 'episodes': episodes}, {'synthetic': True}))
    return root


def test_next_boundary_uses_last_strictly_prior_quantity_and_keeps_current_state():
    rows, actions, episodes = source()
    train, ledger = position_targets(rows['training'], actions, episodes, 'training')
    np.testing.assert_array_equal(train.position_target[:3], [0, 2, 1])
    np.testing.assert_array_equal(train.direction[:3], [-1, 0, 1])
    assert len(train) == 1200 and ledger.reason.eq('included').all()
    assert train.position_target_time.sub(train.end).eq(pd.Timedelta(minutes=1)).all()
    assert train.position_target_episode_id.eq(0).equals(train.position_target.eq(0))
    # 경계와 같은 시각의 평탄 청산은 다음 창의 정답이다.
    assert actions.loc[actions.time.eq(train.label_end.iloc[1]), 'after_qty'].iloc[-1] == 0
    assert train.position_target.iloc[1] == 2


def test_future_target_position_crossing_period_is_excluded_with_ledger():
    rows, actions, episodes = source()
    target_id = actions.loc[actions.time.eq(rows['training'].end.iloc[1]+pd.Timedelta(minutes=1)), 'episode_id'].iloc[-1]
    episodes.loc[episodes.episode_id.eq(target_id), 'exit_time'] = pd.Timestamp('2021-01-02', tz='UTC')
    train, ledger = position_targets(rows['training'], actions, episodes, 'training')
    assert len(train) == 1199
    assert ledger.reason.iloc[1] == 'right_episode_boundary'
    assert target_id not in train.position_target_episode_id.to_numpy()


@pytest.mark.parametrize('damage', ['continuity', 'current', 'unknown_position', 'missing', 'few', 'window'])
def test_invalid_position_source_fails_before_fitting(damage):
    rows, actions, episodes = source()
    frame = rows['training']
    if damage == 'continuity':
        actions.loc[1, 'before_qty'] = 100
    elif damage == 'current':
        frame.loc[0, 'direction'] = 1
    elif damage == 'unknown_position':
        episodes = episodes.iloc[1:]
    elif damage == 'missing':
        frame.loc[0, MARKET_FEATURES[0]] = np.nan
    elif damage == 'few':
        frame = frame.iloc[:999]
    else:
        frame.loc[0, 'label_end'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        position_targets(frame, actions, episodes, 'training')


def test_position_model_separates_meaning_and_matches_independent_multinomial():
    rows, actions, episodes = source()
    train, _ = position_targets(rows['training'], actions, episodes, 'training')
    x, y = train[MARKET_FEATURES].to_numpy(), train.position_target.to_numpy()
    model, _ = PositionTargetModel.fit(x, y)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**JOINT_SETTINGS).fit(scaler.transform(x), y)
        expected = learner.predict_proba(scaler.transform(x))
    np.testing.assert_allclose(model.probabilities(x), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.probabilities(x), softmax(scaler.transform(x)@learner.coef_.T+learner.intercept_, axis=1), atol=1e-12, rtol=0)
    assert model.to_dict()['classes'] == ['flat', 'short', 'long']
    with pytest.raises(ValueError):
        JointEntryModel.from_dict(model.to_dict())
    data = model.to_dict()
    data['classes'][0] = 'hold'
    with pytest.raises(ValueError):
        PositionTargetModel.from_dict(data)


def test_position_metrics_and_all_admission_conditions_match_manual_values():
    y = np.tile(np.arange(3), 20)
    p = np.full((60, 3), .1)
    p[np.arange(60), y] = .8
    metrics = position_metrics(y, p)
    assert abs(metrics['log_loss']+np.log(.8)) < 1e-14
    assert metrics['request_confusion'] == [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
    baseline = {'log_loss': .6, 'short': {'average_precision': .4}, 'long': {'average_precision': .4}}
    better = {'log_loss': .58, 'short': {'average_precision': .5}, 'long': {'average_precision': .5}}
    pair = {'position': better, 'constant': baseline}
    months = [{'month': f'2021-{m:02}', 'position_log_loss': .58 if m <= 8 else .61, 'constant_log_loss': .6} for m in range(1, 13)]
    assert position_admission(pair, months)['position_target_admitted']
    for damage in ['gain', 'short', 'long', 'months']:
        copy_pair, copy_months = copy.deepcopy(pair), copy.deepcopy(months)
        if damage == 'gain':
            copy_pair['position']['log_loss'] = .594
        elif damage == 'months':
            copy_months[0]['position_log_loss'] = .61
        else:
            copy_pair['position'][damage]['average_precision'] = .39
        assert not position_admission(copy_pair, copy_months)['position_target_admitted']
    with pytest.raises(ValueError):
        position_admission(pair, months[:-1])


def test_full_position_pipeline_future_invariance_and_failure_preservation(tmp_path, monkeypatch):
    root = reference(tmp_path, monkeypatch)
    frames, _, _ = position_reference(root)
    out = run_position_diagnostics(root, tmp_path / 'runs')
    metrics = json.loads((out / 'metrics.json').read_text())
    pred = pd.read_parquet(out / 'predictions.parquet')
    for n in ['position', 'constant']:
        assert position_metrics(pred.position_target, pred[[n+'_'+c for c in ['flat', 'short', 'long']]]) == metrics[n]
    reference(tmp_path, monkeypatch, future_flip=True)
    changed = run_position_diagnostics(root, tmp_path / 'changed')
    assert (out / 'model.json').read_bytes() == (changed / 'model.json').read_bytes()
    old_support = json.loads((out / 'training_support.json').read_text())
    new_support = json.loads((changed / 'training_support.json').read_text())
    for key in ['rows', 'class_counts', 'training_frequencies', 'training_positions']:
        assert old_support[key] == new_support[key]
    assert not list(out.rglob('trades.parquet'))
    (root / 'training_used.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError):
        run_position_diagnostics(root, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()
    assert len(frames['training']) == 1200
