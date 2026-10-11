import copy

import numpy as np
import pandas as pd
import pytest
from test_exposure_backtest import data
from test_exposure_engine import bar, config
from test_sparse_execution import model as state_model

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.market_change_backtest import backtest, verify_path
from wonyotti_fr.market_change_model import MarketChangeModel, action_labels, linear_baseline
from wonyotti_fr.market_change_policy import MarketChangePolicy
from wonyotti_fr.sparse_execution_reference import ReferenceSparsePolicy
from wonyotti_fr.sparse_exposure_reference import independent_refit, raw_forest


def model(timing='window_end', fixed_action=None):
    raw = state_model(fixed_action).to_dict()
    raw['format'], raw['timing'] = 'market_only_execution_policy_v1', timing
    for key, fmt in [('classifier', 'market_only_execution_classifier_v1'),
        ('negative', 'market_only_execution_magnitude_v1'), ('positive', 'market_only_execution_magnitude_v1')]:
        raw[key]['format'], raw[key]['features'] = fmt, MARKET_FEATURES
    return MarketChangeModel.from_dict(raw)


@pytest.mark.parametrize('timing', ['window_start', 'window_end'])
def test_all_trees_independently_refitted_and_scalar_predictions_match(timing):
    rng = np.random.default_rng(104)
    x, vx = rng.normal(size=(1500, 14)), rng.normal(size=(180, 14))
    y = np.where(x[:, 0] < -.3, -.3-.1*np.abs(x[:, 1]), np.where(x[:, 0] > .3, .2+.1*np.abs(x[:, 2]), 0.))
    learner, support = MarketChangeModel.fit(x, y, vx, timing=timing)
    linear, logistic = linear_baseline(x, action_labels(y), vx)
    expected = independent_refit(learner.to_dict(), linear, x, y, vx)
    actual = learner.predict(vx)
    for key, values in actual.items():
        np.testing.assert_allclose(values, expected[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(logistic, expected['linear'], rtol=0, atol=1e-12)
    for side in ['negative', 'positive']:
        np.testing.assert_allclose(actual[side+'_size'], np.maximum(raw_forest(learner.to_dict()[side], vx), 0), rtol=0, atol=1e-12)
    for i, row in enumerate(vx):
        single = learner.predict_one(row)
        for key in ['probabilities', 'negative_size', 'positive_size', 'decision_change', 'expected_change']:
            np.testing.assert_allclose(single[key], actual[key][i], rtol=0, atol=1e-12)
        assert single['action'] == actual['actions'][i]
    assert support['classifier']['class_counts'] == np.bincount(action_labels(y), minlength=3).tolist()
    assert MarketChangeModel.from_dict(learner.to_dict()).to_dict() == learner.to_dict()
    changed, _ = MarketChangeModel.fit(x, y, vx+100, timing=timing)
    assert changed.to_dict() == learner.to_dict()


@pytest.mark.parametrize('bad', ['current_exposure_column', 'nonfinite', 'class_support', 'too_few', 'bad_timing'])
def test_training_rejects_state_or_unsupported_inputs(bad):
    x = np.zeros((1200, 14))
    y = np.tile([-.2, 0., .2], 400)
    timing = 'window_end'
    if bad == 'current_exposure_column':
        x = np.c_[x, np.zeros(len(x))]
    elif bad == 'nonfinite':
        x[0, 0] = np.nan
    elif bad == 'class_support':
        y[:1100] = 0
    elif bad == 'too_few':
        x, y = x[:999], y[:999]
    else:
        timing = 'wrong'
    with pytest.raises(ValueError):
        MarketChangeModel.fit(x, y, x, timing=timing)


@pytest.mark.parametrize('tamper', ['features', 'state_branch', 'timing', 'format', 'cycle'])
def test_serialized_model_rejects_hidden_state_and_invalid_trees(tamper):
    raw = model().to_dict()
    if tamper == 'features':
        raw['classifier']['features'] = [*MARKET_FEATURES, 'current_net_exposure']
    elif tamper == 'state_branch':
        raw['classifier']['models'][0]['trees'][0]['feature'][0] = 14
    elif tamper == 'timing':
        raw['timing'] = 'future'
    elif tamper == 'format':
        raw['format'] = 'sparse_execution_change_policy_v1'
    else:
        raw['classifier']['models'][0]['trees'][0]['left'][0] = 0
    with pytest.raises(ValueError):
        MarketChangeModel.from_dict(raw)


@pytest.mark.parametrize('timing', ['window_start', 'window_end'])
@pytest.mark.parametrize('stop', [0., .04])
@pytest.mark.parametrize('delay', [0, 1])
def test_all_rows_decisions_and_checkpoints_against_independent_reference(tmp_path, timing, stop, delay):
    learner, frame = model(timing), data()
    settings = config(stop_fraction=stop, max_hold_bars=37, cooldown_bars=15, signal_delay_bars=delay)
    output = tmp_path/'run'
    metrics = backtest(frame, learner, settings, output, batch_size=41)
    proof = verify_path(frame, learner.to_dict(), settings, output, tmp_path/'proof.json')
    assert proof['rows']['equity'] == len(frame) and metrics['closed_trades'] > 0
    assert proof['all_checkpoint_states_compared'] == [500, 1000]
    assert proof['max_equity_difference'] < 1e-7
    assert read_json(output/'policy.json')['kind'] == 'market_only_execution_'+timing


def test_probabilities_do_not_read_original_or_own_account_state():
    learner = model()
    policy = MarketChangePolicy(learner)
    event = bar(4)
    event.update(feature_end=event['end'], features=[1.]+[0.]*13)
    journal = []
    policy.audit_sink = journal.append
    expected = learner.predict_one(event['features'])
    for state in [-1e9, 0., 1e9]:
        event.update(original_quantity=state, current_net_exposure=state, original_target=state)
        policy(event, {'quantity': state/1e7, 'equity': 10000.})
        recorded = journal[-1]
        assert recorded['probability_increase'] == expected['probabilities'][2]
        assert recorded['decision_change'] == expected['decision_change']
        target, reference = ReferenceSparsePolicy(learner.to_dict())(event, 10000.-state/1e7*100, state/1e7)
        assert target == recorded['target']
        assert reference['decision_change'] == expected['decision_change']
    assert len({row['target'] for row in journal}) > 1
    with pytest.raises(ValueError):
        learner.predict_one([*event['features'], 0.])


@pytest.mark.parametrize('mode', ['no_change', 'zero_size', 'missing_features'])
def test_quantity_hold_cases_and_missing_feature_journal(tmp_path, mode):
    raw = model(fixed_action=0 if mode == 'no_change' else 1).to_dict()
    if mode == 'zero_size':
        raw['positive']['baseline'] = 0.
    frame = data()
    if mode == 'missing_features':
        frame[MARKET_FEATURES] = np.nan
    learner, settings, output = MarketChangeModel.from_dict(raw), config(), tmp_path/'run'
    result = backtest(frame, learner, settings, output)
    assert result['closed_trades'] == 0 and result['total_return'] == 0
    verify_path(frame, raw, settings, output, tmp_path/'proof.json')
    assert set(pd.read_parquet(output/'decisions.parquet').reason) == {mode}


@pytest.mark.parametrize('tamper', ['checkpoint', 'decision', 'policy', 'extra_decision'])
def test_resealed_output_tampering_rejected(tmp_path, tamper):
    frame, learner, settings = data(), model(), config(max_hold_bars=21)
    run = tmp_path/'run'
    backtest(frame, learner, settings, run)
    if tamper in ['checkpoint', 'policy']:
        path = run/('checkpoint-500.json' if tamper == 'checkpoint' else 'policy.json')
        value = read_json(path)
        if tamper == 'checkpoint':
            value['state']['cash'] += 1.
        else:
            value['model']['timing'] = 'window_start'
        save_json(path, value)
    else:
        path = run/'decisions.parquet'
        table = pd.read_parquet(path)
        if tamper == 'extra_decision':
            table = pd.concat([table, table.tail(1)], ignore_index=True)
        else:
            table.loc[table.index[-1], 'probability_increase'] += .01
        table.to_parquet(path, index=False)
    save_json(run/'files.json', {p.name: sha256(p) for p in run.iterdir() if p.is_file() and p.name != 'files.json'})
    with pytest.raises(ValueError):
        verify_path(frame, learner.to_dict(), settings, run, tmp_path/'proof.json')
    assert not (tmp_path/'proof.json').exists()


def test_future_market_change_cannot_change_previous_policy_decisions():
    learner, left, right = model(), [], []
    policies = [MarketChangePolicy(learner), MarketChangePolicy(copy.deepcopy(learner))]
    for policy, rows, future in zip(policies, [left, right], [1., -1.], strict=True):
        policy.audit_sink = rows.append
        for i in range(40):
            event = bar(i)
            event.update(feature_end=pd.Timestamp(event['end']).floor('5min').isoformat(), features=[future if i > 19 else .1]+[0.]*13)
            policy(event, {'equity': 10000., 'quantity': 0.})
    assert left[:4] == right[:4] and left[4:] != right[4:]
