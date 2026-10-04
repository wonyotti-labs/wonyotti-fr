import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_direction_diagnostics import source_fixture
from threadpoolctl import threadpool_limits

from wonyotti_fr.activity_diagnostics import (
    ActivityHistogramModels,
    activity_admission,
    activity_metrics,
    activity_window,
    fit_activity_models,
    run_activity_diagnostics,
)
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS, HistogramManagementModels
from wonyotti_fr.new_position import new_position_targets


def source():
    data, actions, episodes = source_fixture()
    actions = actions[actions.episode_id.mod(3).eq(0)].reset_index(drop=True)
    return data, actions, episodes


def test_activity_windows_keep_negative_rows_and_exclude_crossing_positions():
    data, actions, episodes = source()
    data, _ = new_position_targets(data, actions)
    episodes.loc[2, 'entry_time'] = pd.Timestamp('2019-12-31', tz='UTC')
    episodes.loc[5, 'exit_time'] = pd.Timestamp('2021-01-02', tz='UTC')
    data.loc[8, 'episode_id'] = 3
    data.loc[11, 'episode_id'] = 6
    train, ledger = activity_window(data, episodes, 'training')
    validation, _ = activity_window(data, episodes, 'diagnosis')
    assert len(train) == 1196 and len(validation) == 120
    assert train.active.eq(0).sum() == 800
    assert ledger.reason.iloc[[2, 5, 8, 11]].tolist() == ['left_episode_boundary', 'right_episode_boundary']*2
    assert not (set(train.target_episode_id) & set(validation.target_episode_id)) - {0}


@pytest.mark.parametrize('kind', ['missing_episode', 'duplicate_episode', 'late_target', 'feature', 'negative_target', 'few'])
def test_invalid_activity_source_fails_before_learning(kind):
    data, actions, episodes = source()
    data, _ = new_position_targets(data, actions)
    if kind == 'missing_episode':
        episodes = episodes[episodes.episode_id.ne(3)]
    elif kind == 'duplicate_episode':
        episodes = pd.concat([episodes, episodes.iloc[:1]])
    elif kind == 'late_target':
        data.loc[2, 'target_time'] = data.loc[2, 'label_end']
    elif kind == 'feature':
        data.loc[0, 'ret_5m'] = np.nan
    elif kind == 'negative_target':
        data.loc[0, 'target_time'] = data.loc[0, 'end']
    else:
        data = data.iloc[:999]
    with pytest.raises(ValueError):
        activity_window(data, episodes, 'training')


def test_one_action_numeric_export_matches_library_and_rejects_management_format():
    rng = np.random.default_rng(43)
    x, vx = rng.normal(size=(1400, 14)), rng.normal(size=(100, 14))
    y = (x[:, 0]+rng.normal(size=len(x)) > .8).astype(int)
    model, _ = ActivityHistogramModels.fit(x, y[:, None], vx)
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y)
        expected = learner.predict_proba(np.vstack([x, vx]))[:, 1]
    actual = model.probabilities(np.vstack([x, vx]))
    assert actual.shape == (1500, 1)
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual[-3:], np.vstack([model.probabilities(v[None, :]) for v in vx[-3:]]))
    for bad in [model.to_dict(), {**model.to_dict(), 'models': model.to_dict()['models']*3}]:
        with pytest.raises(ValueError):
            HistogramManagementModels.from_dict(bad)
    changed = copy.deepcopy(model.to_dict())
    changed['models'][0]['action'] = 'exit'
    with pytest.raises(ValueError):
        ActivityHistogramModels.from_dict(changed)


def test_future_features_and_labels_cannot_change_models_or_training_thresholds():
    data, actions, episodes = source()
    data, _ = new_position_targets(data, actions)
    train, _ = activity_window(data, episodes, 'training')
    validation, _ = activity_window(data, episodes, 'diagnosis')
    models, thresholds, _ = fit_activity_models(train, validation)
    validation['active'] = 1-validation.active
    validation[MARKET_FEATURES] *= -2
    changed, changed_thresholds, _ = fit_activity_models(train, validation)
    assert thresholds == changed_thresholds
    assert {k: v.to_dict() for k, v in models.items()} == {k: v.to_dict() for k, v in changed.items()}
    for name, model in models.items():
        scores = model.probabilities(train[MARKET_FEATURES].to_numpy())
        assert thresholds[name] == np.quantile(scores, .975)


def test_activity_threshold_metrics_and_all_three_admission_conditions():
    values = activity_metrics([0]*20+[1]*20, [.1]*10+[.8]*10+[.2]*10+[.9]*10, .8)
    assert [values[k] for k in ['true_positive', 'false_positive', 'false_negative', 'true_negative']] == [10]*4
    assert values['precision'] == values['recall'] == .5
    metrics = {'logistic': {'log_loss': .6, 'average_precision': .4},
        'histogram': {'log_loss': .58, 'average_precision': .4}, 'constant': {'log_loss': .69}}
    assert activity_admission(metrics)['activity_histogram_admitted']
    for kind in ['gain', 'constant', 'precision', 'nonfinite']:
        changed = copy.deepcopy(metrics)
        if kind == 'gain':
            changed['histogram']['log_loss'] = .594
        elif kind == 'constant':
            changed['constant']['log_loss'] = .58
        elif kind == 'precision':
            changed['histogram']['average_precision'] = .39
        else:
            changed['histogram']['log_loss'] = np.nan
        if kind == 'nonfinite':
            with pytest.raises(ValueError):
                activity_admission(changed)
        else:
            assert not activity_admission(changed)['activity_histogram_admitted']


def test_activity_pipeline_preserves_source_and_failure(tmp_path, monkeypatch):
    data, actions, episodes = source()
    monkeypatch.setattr('wonyotti_fr.activity_diagnostics.source_inputs',
        lambda *_: ({'actions': actions, 'episodes': episodes}, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.activity_diagnostics.make_expansion_data', lambda _: data)
    out = run_activity_diagnostics(tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    train = pd.read_parquet(out / 'training_used.parquet')
    validation = pd.read_parquet(out / 'diagnosis_used.parquet')
    pred = pd.read_parquet(out / 'predictions.parquet')
    metrics = json.loads((out / 'metrics.json').read_text())
    thresholds = json.loads((out / 'thresholds.json').read_text())['thresholds']
    assert len(train) == 1200 and len(validation) == 120
    assert train.active.sum() == 400 and validation.active.sum() == 40
    assert len(pd.read_parquet(out / 'new_position_ledger.parquet')) == len(actions[actions.action.eq('open')])
    for name in metrics:
        assert metrics[name] == activity_metrics(pred.active, pred[name], thresholds[name])
    assert activity_admission(metrics) == json.loads((out / 'decision.json').read_text())
    assert not list(out.rglob('trades.parquet'))
    monkeypatch.setattr('wonyotti_fr.activity_diagnostics.make_expansion_data', lambda _: data.iloc[:-21])
    with pytest.raises(ValueError):
        run_activity_diagnostics(tmp_path, tmp_path, tmp_path, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'models.json').exists()
