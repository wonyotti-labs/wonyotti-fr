import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_first_state import fixed_budget  # noqa: F401
from test_policy_entry import fixture as policy_fixture
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.entry_regression_diagnostics import run_entry_regression_diagnosis
from wonyotti_fr.entry_stop_risk import (
    EntryStopModel,
    reproduce_regression,
    run_entry_stop_diagnosis,
    stop_admission,
    stop_expected_bps,
    stop_inputs,
    stop_probability_metrics,
)
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.net_edge_model import NET_FEATURES
from wonyotti_fr.outcome_journal import OutcomeJournal


def stop_training():
    rng = np.random.default_rng(58)
    frame = pd.DataFrame(rng.normal(size=(1402, len(MARKET_FEATURES))), columns=MARKET_FEATURES)
    frame['decision_time'] = pd.to_datetime([
        *pd.date_range('2021-01-02', periods=1200, freq='5h', tz='UTC'),
        pd.Timestamp('2021-09-29T23:45Z'), pd.Timestamp('2021-10-01T00:10Z'),
        *pd.date_range('2021-10-02', periods=200, freq='6h', tz='UTC')])
    frame['label_end'] = frame.decision_time+pd.Timedelta(hours=1)
    frame['order_direction'] = np.tile([-1, 1], 701)
    frame['net_bps'] = np.where(frame.ret_5m > .3, -700., 150.)
    frame['favorable_bps'], frame['wait_minutes'] = 18., 2.
    return frame


def model_data():
    rng = np.random.default_rng(158)
    values = rng.normal(size=(1200, len(NET_FEATURES)))
    stop = (values[:, 0] > .3).astype(int)
    weights = rng.uniform(.1, 2., len(values))
    return values, stop, weights/weights.mean()


def test_probability_export_matches_reference_and_future_inputs_do_not_change_model():
    x, y, w = model_data()
    model, support = EntryStopModel.fit(x, y[:, None], x[:100], sample_weight=w)
    altered, _ = EntryStopModel.fit(x, y[:, None], x[:100]*-100, sample_weight=w)
    assert model.to_dict() == altered.to_dict()
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y, sample_weight=w)
        expected = learner.predict_proba(x)[:, 1]
    np.testing.assert_allclose(model.probabilities(x)[:, 0], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose([model.probabilities(row[None, :])[0, 0] for row in x[:10]], expected[:10], rtol=0, atol=1e-12)
    assert support['rows'] == len(y) and support['positive']['intrabar_stop'] == y.sum()
    bad = x[:2].copy()
    bad[0, 0] = np.nan
    assert np.isnan(model.probabilities(bad)[0, 0])
    exported = model.to_dict()
    exported['models'][0]['trees'][0]['left'][0] = 0
    with pytest.raises(ValueError):
        EntryStopModel.from_dict(exported)


def test_mixture_preserves_loss_severity_and_weighted_total_identity():
    means = {'intrabar_stop': -700., 'non_stop': 150.}
    np.testing.assert_allclose(stop_expected_bps([0., .25, 1.], means), [150., -62.5, -700.])
    y, w, stop = np.array([-600., -800., 120., 180.]), np.array([1., 3., 2., 2.]), np.array([1, 1, 0, 0])
    means = {'intrabar_stop': float(np.average(y[stop == 1], weights=w[stop == 1])),
             'non_stop': float(np.average(y[stop == 0], weights=w[stop == 0]))}
    assert stop_expected_bps([np.average(stop, weights=w)], means)[0] == pytest.approx(np.average(y, weights=w))
    with pytest.raises(ValueError):
        stop_expected_bps([1.1], means)
    with pytest.raises(ValueError):
        stop_expected_bps([.1], {**means, 'non_stop': float('nan')})


def test_probability_metrics_and_every_admission_condition():
    actual, predicted, weights = np.array([0, 1, 0, 1]), np.array([.2, .7, .1, .9]), np.array([1., 2., 3., 4.])
    result = stop_probability_metrics(actual, predicted, weights)
    assert result['weighted_log_loss'] == pytest.approx(-np.dot(weights, np.log([.8, .7, .9, .9]))/10)
    assert result['weighted_brier'] == pytest.approx(np.dot(weights, [.04, .09, .01, .01])/10)
    candidate = {'rows': 100, 'weighted_mse': 50., 'mse': 50., 'selected': 30,
                 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    metrics = {name: {**candidate, 'weighted_mse': 100., 'mse': 100.} for name in ['ridge', 'boosted', 'constant']}
    metrics['stop_mixture'] = candidate
    probability = {'stop_mixture': {'rows': 100, 'weighted_log_loss': .5, 'weighted_brier': .1},
                   'constant': {'rows': 100, 'weighted_log_loss': 1., 'weighted_brier': .2}}
    assert stop_admission(metrics, probability)['stop_risk_admitted']
    for key, value in [('weighted_mse', 99.), ('mse', 101.), ('selected', 29), ('selected_mean_bps', 0.), ('selected_weighted_mean_bps', 0.)]:
        changed = copy.deepcopy(metrics)
        changed['stop_mixture'][key] = value
        assert not stop_admission(changed, probability)['stop_risk_admitted']
    for key, value in [('weighted_log_loss', .99), ('weighted_brier', .3)]:
        changed = copy.deepcopy(probability)
        changed['stop_mixture'][key] = value
        assert not stop_admission(metrics, changed)['stop_risk_admitted']
    for name in ['ridge', 'boosted', 'constant']:
        changed = copy.deepcopy(metrics)
        changed[name]['weighted_mse'] = 49.
        assert not stop_admission(changed, probability)['stop_risk_admitted']


@pytest.mark.parametrize('damage', ['reason', 'censored', 'support', 'weight', 'target', 'features'])
def test_stop_input_rejects_invalid_sources_and_loss_removal(damage):
    frame = stop_training().iloc[:1200].assign(label_status='closed')
    frame['exit_reason'] = np.where(frame.net_bps < 0, 'intrabar_stop', 'signal_exit')
    weights = np.ones(len(frame))
    if damage == 'reason':
        frame.loc[0, 'exit_reason'] = 'unknown'
    elif damage == 'censored':
        frame.loc[0, 'label_status'] = 'right_censored'
    elif damage == 'support':
        frame['exit_reason'] = 'signal_exit'
    elif damage == 'weight':
        weights[0] = 0
    elif damage == 'target':
        frame.loc[0, 'net_bps'] = np.nan
    else:
        frame.loc[0, 'ret_5m'] = np.nan
    with pytest.raises(ValueError):
        stop_inputs(frame, weights, training=True)


def test_full_source_reproduction_future_outcome_invariance_and_tamper_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr('test_policy_entry.training', stop_training)
    append = OutcomeJournal.append
    def with_stop(self, sequence, opportunity, record):
        if record['outcome']['label_status'] == 'closed' and record['outcome']['net_bps'] < 0:
            record['outcome']['exit_reason'] = 'intrabar_stop'
            record['fills'][-1]['reason'] = 'intrabar_stop'
        return append(self, sequence, opportunity, record)
    monkeypatch.setattr(OutcomeJournal, 'append', with_stop)
    root, parent, labels, _, _ = policy_fixture(tmp_path, monkeypatch)
    save_json(root/'manifest.json', {'settings': {'labels': str(labels), 'reference': str(parent),
        'labels_files_sha256': sha256(labels/'files.json')}})
    diagnosis = run_entry_regression_diagnosis(root, tmp_path/'regression')
    out = run_entry_stop_diagnosis(diagnosis, tmp_path/'risk')
    result = json.loads((out/'summary.json').read_text())
    assert result['complete'] and result['training_rows'] == 1200 and result['diagnosis_rows'] == 200
    original = stop_inputs
    def changed_future(frame, weights, *, training):
        x, y, w, stop = original(frame, weights, training=training)
        return (x, y, w, stop) if training else (x*-2, -y*10, w, 1-stop)
    monkeypatch.setattr('wonyotti_fr.entry_stop_risk.stop_inputs', changed_future)
    altered = run_entry_stop_diagnosis(diagnosis, tmp_path/'future')
    for name in ['model.json', 'conditional_means.json', 'training_weights.parquet', 'training_used.parquet']:
        assert (out/name).read_bytes() == (altered/name).read_bytes()
    for name in ['training_used.parquet', 'diagnosis_used.parquet', 'exclusion_ledger.parquet']:
        assert (out/name).read_bytes() == (diagnosis/name).read_bytes()
    damaged = pd.read_parquet(diagnosis/'predictions.parquet')
    damaged.loc[0, 'predicted_ridge'] += 100
    damaged.to_parquet(diagnosis/'predictions.parquet', index=False)
    with pytest.raises(ValueError, match='지문'):
        reproduce_regression(diagnosis, tmp_path/'bad_hash')
    hashes = json.loads((diagnosis/'files.json').read_text())
    hashes['predictions.parquet'] = sha256(diagnosis/'predictions.parquet')
    save_json(diagnosis/'files.json', hashes)
    with pytest.raises(AssertionError):
        reproduce_regression(diagnosis, tmp_path/'bad_refingerprint')
