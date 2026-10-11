import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_exposure_backtest import data
from test_exposure_engine import bar, config

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.exposure_change_model import EXPOSURE_FEATURES, SETTINGS
from wonyotti_fr.exposure_engine import ExposureEngine
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.sparse_execution_reference import ReferenceSparsePolicy
from wonyotti_fr.sparse_exposure import SparseExposureModel
from wonyotti_fr.sparse_exposure_backtest import backtest, verify_path
from wonyotti_fr.sparse_exposure_policy import SparseExposurePolicy


def model(fixed_action=None):
    leaf = {'left': [-1], 'right': [-1], 'feature': [-2], 'threshold': [0.], 'value': [0.]}
    classifiers = []
    for k, action in enumerate(['decrease', 'no_change', 'increase']):
        trees = [copy.deepcopy(leaf) for _ in range(64)]
        if fixed_action is None:
            trees[0] = {'left': [1, -1, 3, -1, -1], 'right': [2, -1, 4, -1, -1],
                'feature': [0, -2, 0, -2, -2], 'threshold': [-.3, 0., .3, 0., 0.],
                'value': [0., 4. if k == 0 else 0., 0., 4. if k == 1 else 0., 4. if k == 2 else 0.]}
        classifiers.append({'action': action, 'baseline': 4. if k-1 == fixed_action else 0., 'trees': trees})
    regressor = {'format': 'sparse_execution_magnitude_histogram_v1', 'features': EXPOSURE_FEATURES,
        'settings': SETTINGS, 'baseline': .25, 'trees': [copy.deepcopy(leaf) for _ in range(64)]}
    raw = {'format': 'sparse_execution_change_policy_v1', 'classifier': {
        'format': 'sparse_execution_change_classifier_v1', 'features': EXPOSURE_FEATURES,
        'settings': HISTOGRAM_SETTINGS, 'models': classifiers}, 'negative': regressor, 'positive': copy.deepcopy(regressor)}
    return SparseExposureModel.from_dict(raw)


def event(i, feature=0., **kwargs):
    row = bar(i, **kwargs)
    row.update(feature_end=pd.Timestamp(row['end']).floor('5min').isoformat(), features=[feature]+[0.]*13)
    return row


@pytest.mark.parametrize('delay', [0, 1])
def test_all_minute_probability_size_target_and_ledger_replay(tmp_path, delay):
    frame, learner = data(), model()
    settings = config(signal_delay_bars=delay, stop_fraction=.04, cooldown_bars=15, max_hold_bars=333)
    output = tmp_path/'run'
    metrics = backtest(frame, SparseExposurePolicy(learner), settings, output, batch_size=37)
    proof = verify_path(frame, 'sparse_execution', learner.to_dict(), settings, output, tmp_path/'review.json')
    assert proof['rows']['equity'] == len(frame) and proof['rows']['decisions'] > 0
    assert proof['metrics']['final_equity'] == pytest.approx(metrics['final_equity'])
    decisions = pd.read_parquet(output/'decisions.parquet')
    assert set(decisions.action) == {-1, 0, 1}
    assert decisions.loc[decisions.action == 0, 'target'].isna().all()
    assert metrics['closed_trades'] > 0
    assert read_json(output/'storage.json')['json_restored_checkpoints'] == [500, 1000]


def test_no_change_preserves_quantity_when_price_drifts_and_stop_still_exits():
    learner = model()
    policy = SparseExposurePolicy(learner)
    engine = ExposureEngine(config(stop_fraction=.04))
    for i in range(6):
        result = engine.step(event(i, feature=1.), policy)
    held = result['quantity']
    assert held > 0
    for i in range(6, 15):
        result = engine.step(event(i, feature=0., price=103.), policy)
        assert result['quantity'] == held and result['fills'] == [] and result['decision'] is None
    result = engine.step(event(15, feature=0., price=103., low=90.), policy)
    assert result['quantity'] == 0 and result['closed_trades'][0]['exit_reason'] == 'intrabar_stop'


def test_only_own_state_controls_target_and_missing_features_are_recorded():
    learner, row = model(1), event(4)
    policy, journal = SparseExposurePolicy(learner), []
    policy.audit_sink = journal.append
    row['original_current_exposure'] = -1e9
    assert policy(row, {'equity': 10000., 'quantity': 20.}) == pytest.approx(.45)
    row['original_current_exposure'] = 1e9
    assert policy(row, {'equity': 10000., 'quantity': 20.}) == pytest.approx(.45)
    assert policy(row, {'equity': 10000., 'quantity': -20.}) == pytest.approx(.05)
    expected, recorded = ReferenceSparsePolicy(learner.to_dict())(row, 12000., -20.)
    assert expected == pytest.approx(.05) and recorded['current_exposure'] == -.2
    row['features'][0] = np.nan
    assert policy(row, {'equity': 10000., 'quantity': 20.}) is None
    assert journal[-1]['reason'] == 'missing_features' and journal[-1]['action'] is None
    row['feature_end'] = event(3)['end']
    with pytest.raises(ValueError, match='시각'):
        policy(row, {'equity': 10000., 'quantity': 20.})


@pytest.mark.parametrize('fixed_action', [0, 1])
def test_no_change_and_zero_size_create_no_order(fixed_action):
    raw = model(fixed_action).to_dict()
    if fixed_action == 1:
        raw['positive']['baseline'] = -1.
    policy = SparseExposurePolicy(SparseExposureModel.from_dict(raw))
    journal = []
    policy.audit_sink = journal.append
    assert policy(event(4), {'equity': 10000., 'quantity': 20.}) is None
    assert journal[-1]['reason'] == ('no_change' if fixed_action == 0 else 'zero_size')


@pytest.mark.parametrize('field', ['probability_decrease', 'positive_size', 'current_exposure', 'action', 'target', 'reason', 'extra'])
def test_resealed_decision_journal_tamper_rejected(tmp_path, field):
    frame, learner, settings, output = data(), model(), config(), tmp_path/'run'
    backtest(frame, SparseExposurePolicy(learner), settings, output)
    path = output/'decisions.parquet'
    journal = pd.read_parquet(path)
    i = journal.target.first_valid_index()
    if field == 'extra':
        journal = pd.concat([journal, journal.tail(1)], ignore_index=True)
    elif field == 'reason':
        journal.loc[i, field] = 'no_change'
    else:
        journal.loc[i, field] += 1.
    journal.to_parquet(path, index=False)
    sealed = read_json(output/'files.json')
    sealed[path.name] = sha256(path)
    save_json(output/'files.json', sealed)
    with pytest.raises(ValueError):
        verify_path(frame, 'sparse_execution', learner.to_dict(), settings, output, tmp_path/'review.json')


def test_zero_liquidity_and_future_market_do_not_alter_previous_path(tmp_path):
    frame, learner, settings = data(), model(), config(signal_delay_bars=1)
    frame.loc[5:10, ['count', 'volume']] = 0
    frame.loc[:4, MARKET_FEATURES] = np.nan
    full = tmp_path/'original'
    backtest(frame, SparseExposurePolicy(learner), settings, full)
    verify_path(frame, 'sparse_execution', learner.to_dict(), settings, full, tmp_path/'review.json')
    changed = frame.copy()
    changed.loc[800:, ['open', 'high', 'low', 'close']] *= 1.02
    changed.loc[800:, MARKET_FEATURES] *= -1
    second = tmp_path/'future'
    backtest(changed, SparseExposurePolicy(learner), settings, second)
    for name in ['equity', 'decisions', 'fills', 'trades']:
        a, b = pd.read_parquet(full/f'{name}.parquet'), pd.read_parquet(second/f'{name}.parquet')
        time = 'exit_time' if name == 'trades' else 'time'
        cutoff = frame.end.iloc[799].isoformat()
        pd.testing.assert_frame_equal(a[a[time] <= cutoff].reset_index(drop=True), b[b[time] <= cutoff].reset_index(drop=True), check_exact=True)


def test_all_no_change_and_missing_features_empty_trade_paths(tmp_path):
    for case in ['no_change', 'missing']:
        frame, learner, settings = data(), model(0), config()
        if case == 'missing':
            frame[MARKET_FEATURES] = np.nan
        out = tmp_path/case
        metrics = backtest(frame, SparseExposurePolicy(learner), settings, out)
        assert metrics['closed_trades'] == 0 and metrics['total_return'] == 0
        verify_path(frame, 'sparse_execution', learner.to_dict(), settings, out, tmp_path/f'{case}.json')
        with pytest.raises(ValueError, match='설정'):
            verify_path(frame, 'sparse_execution', learner.to_dict(), replace(settings, fee_bps=9), out, tmp_path/'wrong.json')
