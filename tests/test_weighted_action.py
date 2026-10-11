import copy

import numpy as np
import pandas as pd
import pytest
from test_exposure_backtest import data
from test_exposure_engine import bar, config
from test_market_change import model as market_model

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.market_change_backtest import backtest as market_backtest
from wonyotti_fr.market_change_policy import MarketChangePolicy
from wonyotti_fr.sparse_exposure import action_labels, choose_actions
from wonyotti_fr.sparse_exposure_reference import verify_metrics
from wonyotti_fr.weighted_action_backtest import backtest, verify_path
from wonyotti_fr.weighted_action_evaluation import calibration, evaluate
from wonyotti_fr.weighted_action_model import (
    WeightedActionModel,
    WeightedMarketClassifier,
    class_weights,
    transform_scores,
)
from wonyotti_fr.weighted_action_reference import (
    independent_weighted_fit,
    reference_evaluate,
    reference_prediction,
)


def model(condition='C', fixed_action=None):
    raw = market_model(fixed_action=fixed_action).to_dict()
    raw.update(format=WeightedActionModel.format, condition=condition, class_counts=[200, 1100, 200])
    if condition in ('C', 'D'):
        raw['classifier']['format'] = WeightedMarketClassifier.format
    return WeightedActionModel.from_dict(raw)


def test_full_weighted_refit_scores_and_future_validation_invariance():
    rng = np.random.default_rng(105)
    x, vx = rng.normal(size=(1800, 14)), rng.normal(size=(200, 14))
    y = np.where(x[:, 0] < -1.1, -.2, np.where(x[:, 0] > 1.1, .3, 0.))
    labels = action_labels(y)
    classifier, support = WeightedMarketClassifier.fit(x, labels, vx)
    counts = np.bincount(labels, minlength=3).tolist()
    assert support['class_counts'] == counts
    np.testing.assert_array_equal(support['class_weights'], class_weights(counts))
    expected = independent_weighted_fit(classifier.to_dict(), x, y, vx)
    np.testing.assert_allclose(classifier.probabilities(vx), expected, rtol=0, atol=1e-12)
    later, _ = WeightedMarketClassifier.fit(x, labels, vx+100)
    assert later.to_dict() == classifier.to_dict()
    for condition in ('C', 'D'):
        base = market_model()
        combined = WeightedActionModel(classifier, base.negative, base.positive, counts, condition)
        prediction, reference = combined.predict(vx), reference_prediction(combined.to_dict(), vx)
        for key in prediction:
            np.testing.assert_allclose(prediction[key], reference[key], rtol=0, atol=1e-12)
        for i, row in enumerate(vx):
            single = combined.predict_one(row)
            assert single['action'] == prediction['actions'][i]
            for key in ('probabilities', 'negative_size', 'positive_size', 'decision_change', 'expected_change'):
                np.testing.assert_allclose(single[key], prediction[key][i], rtol=0, atol=1e-12)


@pytest.mark.parametrize('condition', ['A', 'B', 'C', 'D'])
def test_model_roundtrip_full_predictions_and_fixed_magnitudes(condition):
    learner = model(condition)
    raw = learner.to_dict()
    assert WeightedActionModel.from_dict(raw).to_dict() == raw
    assert raw['negative'] == market_model().to_dict()['negative']
    assert raw['positive'] == market_model().to_dict()['positive']
    values = np.random.default_rng(1).normal(size=(150, 14))
    actual, expected = learner.predict(values), reference_prediction(raw, values)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    raw['class_counts'][0] += 1
    assert learner.to_dict()['class_counts'] != raw['class_counts']


def test_decoder_only_transform_and_inverse_restore():
    p = np.array([[.1, .8, .1], [0., 1., 0.], [.5, 0., .5]])
    counts = [200, 1600, 200]
    assert np.array_equal(transform_scores(p, counts, 'A'), p)
    q = transform_scores(p, counts, 'B')
    np.testing.assert_allclose(q, [[1/3]*3, [0., 1., 0.], [.5, 0., .5]], rtol=0, atol=1e-15)
    assert np.array_equal(choose_actions(q), [0, 0, 0])
    np.testing.assert_allclose(transform_scores(q, counts, 'D'), p, rtol=0, atol=1e-15)


@pytest.mark.parametrize('counts', [[0, 500, 500], [127, 800, 200], [200, 200, 200], [True, 1100, 200],
    [200., 1100, 200], [200, 1100], [200, 1100, 200, 200], (200, 1100, 200)])
def test_unsupported_training_counts_rejected(counts):
    with pytest.raises(ValueError):
        class_weights(counts)


@pytest.mark.parametrize('bad', ['missing_class', 'rare_class', 'float_label', 'extra_state', 'nonfinite', 'short', 'empty_validation'])
def test_weighted_fit_rejects_unsupported_input(bad):
    x, labels, vx = np.zeros((1200, 14)), np.tile([0, 1, 2], 400), np.zeros((12, 14))
    if bad == 'missing_class':
        labels[:] = 1
    elif bad == 'rare_class':
        labels[:1000] = 1
    elif bad == 'float_label':
        labels = labels.astype(float)
    elif bad == 'extra_state':
        x = np.c_[x, np.zeros(len(x))]
    elif bad == 'nonfinite':
        x[0, 0] = np.nan
    elif bad == 'short':
        x, labels = x[:999], labels[:999]
    else:
        vx = vx[:0]
    with pytest.raises(ValueError):
        WeightedMarketClassifier.fit(x, labels, vx)


@pytest.mark.parametrize('bad', ['format', 'timing', 'condition', 'classifier', 'counts', 'state_branch', 'cycle', 'extra'])
def test_invalid_serialized_object_rejected(bad):
    raw = model().to_dict()
    if bad in ('format', 'timing', 'condition'):
        raw[bad] = 'wrong'
    elif bad == 'classifier':
        raw['classifier']['format'] = market_model().classifier.format
    elif bad == 'counts':
        raw['class_counts'] = [0, 1000, 0]
    elif bad == 'state_branch':
        raw['classifier']['models'][0]['trees'][0]['feature'][0] = 14
    elif bad == 'cycle':
        raw['classifier']['models'][0]['trees'][0]['left'][0] = 0
    else:
        raw['extra'] = 1
    with pytest.raises(ValueError):
        WeightedActionModel.from_dict(raw)


@pytest.mark.parametrize('condition', ['A', 'B', 'C', 'D'])
@pytest.mark.parametrize('delay', [0, 1])
@pytest.mark.parametrize('stop', [0., .04])
def test_all_minute_rows_fills_funding_checkpoints_and_delay(tmp_path, condition, delay, stop):
    learner, frame = model(condition), data()
    settings = config(signal_delay_bars=delay, stop_fraction=stop, max_hold_bars=37, cooldown_bars=15)
    output = tmp_path/'run'
    metrics = backtest(frame, learner, settings, output, batch_size=41)
    proof = verify_path(frame, learner.to_dict(), settings, output, tmp_path/'proof.json')
    assert proof['rows']['equity'] == len(frame) and metrics['closed_trades'] > 0
    assert proof['all_checkpoint_states_compared'] == [500, 1000]
    assert metrics['fees'] > 0 and metrics['funding_cost'] != 0
    if condition == 'A':
        control = tmp_path/'control'
        expected = market_backtest(frame, market_model(), settings, control, batch_size=41)
        assert metrics == expected
        for filename in ('equity', 'fills', 'trades', 'decisions', 'rejections'):
            pd.testing.assert_frame_equal(pd.read_parquet(output/f'{filename}.parquet'), pd.read_parquet(control/f'{filename}.parquet'), check_exact=True)


@pytest.mark.parametrize('tamper', ['checkpoint', 'condition', 'scores', 'extra_decision'])
def test_resealed_account_output_tampering_rejected(tmp_path, tamper):
    learner, frame, settings = model(), data(), config(max_hold_bars=21)
    output = tmp_path/'run'
    backtest(frame, learner, settings, output)
    if tamper in ('checkpoint', 'condition'):
        path = output/('checkpoint-500.json' if tamper == 'checkpoint' else 'policy.json')
        value = read_json(path)
        if tamper == 'checkpoint':
            value['state']['cash'] += 1
        else:
            value['model']['condition'] = 'D'
        save_json(path, value)
    else:
        path = output/'decisions.parquet'
        frame = pd.read_parquet(path)
        if tamper == 'scores':
            frame.loc[0, 'probability_decrease'] += .01
        else:
            frame = pd.concat([frame, frame.tail(1)])
        frame.to_parquet(path, index=False)
    save_json(output/'files.json', {p.name: sha256(p) for p in output.iterdir() if p.name != 'files.json'})
    with pytest.raises(ValueError):
        verify_path(data(), learner.to_dict(), settings, output, tmp_path/'proof.json')
    assert not (tmp_path/'proof.json').exists()


def test_all_classification_regression_calibration_against_independent_formulas():
    x = np.random.default_rng(2).normal(size=(150, 14))
    y = np.tile([-.2, 0., .3], 50)
    predictions = {k: model(k).predict(x) for k in ('A', 'B', 'C', 'D')}
    expected = {k: reference_prediction(model(k).to_dict(), x) for k in predictions}
    frequency = np.array([.1, .8, .1])
    verify_metrics(evaluate(y, predictions, frequency), reference_evaluate(y, expected, frequency))
    actual = calibration(y, np.eye(3)[action_labels(y)])
    assert actual['multiclass_brier'] == 0
    assert actual['reliability']['-1'][9]['rows'] == 50
    assert sum(row['rows'] for row in actual['reliability']['1']) == len(y)


def test_own_state_only_changes_target_and_future_cannot_change_past():
    learner, rows = model('B'), []
    policy = MarketChangePolicy(learner)
    policy.audit_sink = rows.append
    event = bar(4)
    event.update(features=[1.]+[0.]*13, feature_end=event['end'])
    for q in [-30., 0., 30.]:
        event.update(original_target=q, original_quantity=q)
        policy(event, {'quantity': q, 'equity': 10000.})
    assert len({r['probability_increase'] for r in rows}) == 1
    assert len({r['decision_change'] for r in rows}) == 1
    assert len({r['target'] for r in rows}) > 1
    prior = copy.deepcopy(rows)
    event['features'] = [-1.]+[0.]*13
    policy(event, {'quantity': 0., 'equity': 10000.})
    assert prior == rows[:-1]
