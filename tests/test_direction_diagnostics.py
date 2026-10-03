import json

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import sha256
from wonyotti_fr.direction_diagnostics import (
    DIAGNOSIS_PERIOD,
    TRAIN_PERIOD,
    direction_admission,
    direction_metrics,
    direction_window,
    run_direction_diagnostics,
)
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.expansion_model import BinaryModel
from wonyotti_fr.new_position import new_position_targets


def source_fixture():
    times = pd.date_range('2020-06-01', periods=1200, freq='5min', tz='UTC').append(
        pd.date_range('2021-06-01', periods=120, freq='5min', tz='UTC')).as_unit('ns')
    ids = np.arange(1, len(times) + 1)
    data = pd.DataFrame({'end': times, 'label_end': times + pd.Timedelta(minutes=5),
        'usable': True, 'episode_id': 0, 'target_time': times, 'target_episode_id': ids,
        'active': 1, 'buy': ids % 2, 'expansion_episode_id': ids,
        'expansion_count': 1, 'both_directions': False})
    values = np.random.default_rng(8).normal(size=(len(data), len(MARKET_FEATURES)))
    for j, name in enumerate(MARKET_FEATURES):
        data[name] = values[:, j]
    actions = []
    for i, t in enumerate(times):
        side = 1 if i % 2 else -1
        actions.extend([(t + pd.Timedelta(minutes=1), 'open', 0, side, i + 1),
                        (t + pd.Timedelta(minutes=2), 'close', side, 0, i + 1)])
    actions = pd.DataFrame(actions, columns=['time', 'action', 'before_qty', 'after_qty', 'episode_id'])
    episodes = pd.DataFrame({'episode_id': ids, 'entry_time': times + pd.Timedelta(minutes=1),
                             'exit_time': times + pd.Timedelta(minutes=2)})
    return data, actions, episodes


def test_direction_windows_exclude_current_and_target_boundary_episodes():
    data, actions, episodes = source_fixture()
    data, _ = new_position_targets(data, actions)
    episodes.loc[0, 'entry_time'] = pd.Timestamp('2017-12-31', tz='UTC')
    episodes.loc[1, 'exit_time'] = pd.Timestamp('2021-01-02', tz='UTC')
    data.loc[2, 'episode_id'] = 1
    data.loc[3, 'episode_id'] = 2
    data.loc[4, 'usable'] = False
    data.loc[5, 'active'] = 0
    train, ledger = direction_window(data, episodes, TRAIN_PERIOD)
    validation, _ = direction_window(data, episodes, DIAGNOSIS_PERIOD)
    assert ledger.reason.iloc[:6].tolist() == ['left_episode_boundary', 'right_episode_boundary',
        'left_episode_boundary', 'right_episode_boundary', 'unusable_original_event', 'no_new_position']
    assert len(train) == 1194 and len(validation) == 120
    assert not set(train.target_episode_id) & set(validation.target_episode_id)
    pd.testing.assert_frame_equal(train, data.loc[ledger.reason.eq('included')], check_exact=True)


@pytest.mark.parametrize('damage', ['left', 'right', 'missing', 'duplicate', 'late_target', 'support'])
def test_invalid_direction_windows_and_embargo(damage):
    data, actions, episodes = source_fixture()
    data, _ = new_position_targets(data, actions)
    if damage == 'left':
        data.loc[0, 'end'] = pd.Timestamp('2018-01-01T23:59Z')
    elif damage == 'right':
        data.loc[0, 'label_end'] = pd.Timestamp('2020-12-31T00:00Z')
    elif damage == 'missing':
        episodes = episodes.iloc[1:]
    elif damage == 'duplicate':
        episodes = pd.concat([episodes, episodes.iloc[:1]])
    elif damage == 'late_target':
        data.loc[0, 'target_time'] = data.label_end.iloc[0]
    else:
        data = data.iloc[:999]
    if damage in {'left', 'right'}:
        rows, ledger = direction_window(data, episodes, TRAIN_PERIOD)
        assert len(rows) == 1199 and 'embargo' in ledger.reason.iloc[0]
    else:
        with pytest.raises(ValueError):
            direction_window(data, episodes, TRAIN_PERIOD)


def test_direction_metrics_and_conservative_admission_boundary():
    values = direction_metrics([0, 1, 0, 1], [.35, .65, .5, .5])
    assert values['long_decisions'] == values['long_correct'] == 1
    assert values['short_decisions'] == values['short_correct'] == 1
    assert values['abstentions'] == 2
    def decision(logistic, boosted, constant):
        return direction_admission({k: {'log_loss': v} for k, v in
            [('logistic', logistic), ('boosted', boosted), ('constant', constant)]})
    assert not decision(.61, .6, .7)['boosted_admitted']
    assert not decision(.63, .6, .6)['boosted_admitted']
    accepted = decision(.62, .6, .7)
    assert accepted['boosted_admitted']
    assert not accepted['profitability_accepted'] and not accepted['selection_uses_trading_returns']
    with pytest.raises(ValueError):
        decision(.7, float('nan'), .6)


@pytest.mark.parametrize('labels,scores', [([0, 0], [.5, .6]), ([0, 1], [0., 1.1]),
    ([0, 1], [float('nan'), .5]), ([0, 1], [[.5], [.5]])])
def test_invalid_direction_metrics_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        direction_metrics(labels, scores)


def test_temporal_direction_pipeline_replays_both_numeric_models(tmp_path, monkeypatch):
    data, actions, episodes = source_fixture()
    monkeypatch.setattr('wonyotti_fr.direction_diagnostics.source_inputs',
        lambda *_: ({'actions': actions, 'episodes': episodes}, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.direction_diagnostics.make_expansion_data', lambda _: data)
    out = run_direction_diagnostics(tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    train = pd.read_parquet(out / 'training_used.parquet')
    validation = pd.read_parquet(out / 'diagnosis_used.parquet')
    predictions = pd.read_parquet(out / 'predictions.parquet')
    models = json.loads((out / 'models.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    assert len(train) == 1200 and len(validation) == 120
    for kind in ['logistic', 'boosted']:
        model, _ = BinaryModel.fit(train[MARKET_FEATURES].to_numpy(), train.buy.to_numpy(dtype=int), kind)
        assert model.to_dict() == models[kind]
        np.testing.assert_array_equal(model.probabilities(validation[MARKET_FEATURES].to_numpy()), predictions[kind])
        assert direction_metrics(predictions.buy, predictions[kind]) == metrics[kind]
    assert predictions.constant.eq(train.buy.mean()).all()
    assert json.loads((out / 'decision.json').read_text()) == direction_admission(metrics)
    assert json.loads((out / 'summary.json').read_text())['episode_intersection'] == 0
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    assert not list(out.rglob('trades.parquet'))


def test_insufficient_diagnostic_support_preserves_failure_before_fitting(tmp_path, monkeypatch):
    data, actions, episodes = source_fixture()
    monkeypatch.setattr('wonyotti_fr.direction_diagnostics.source_inputs',
        lambda *_: ({'actions': actions, 'episodes': episodes}, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.direction_diagnostics.make_expansion_data', lambda _: data.iloc[:-21])
    with pytest.raises(ValueError, match='지원 부족'):
        run_direction_diagnostics(tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    out, = (tmp_path / 'runs').iterdir()
    assert (out / 'failure.json').exists()
    assert not (out / 'models.json').exists() and not (out / 'summary.json').exists()
