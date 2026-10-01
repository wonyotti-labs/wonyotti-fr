import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.expansion_model import BinaryModel, ExpansionPolicy


def constant(probability):
    return BinaryModel.from_dict({'format': 'expansion_binary_v1', 'kind': 'logistic', 'features': MARKET_FEATURES,
                                  'mean': [0.] * 14, 'scale': [1.] * 14, 'coefficients': [0.] * 14,
                                  'intercept': float(np.log(probability / (1 - probability)))})


@pytest.mark.parametrize('kind', ['logistic', 'boosted'])
def test_export_matches_library_and_nonfinite_features_abstain(kind):
    rng = np.random.default_rng(4)
    values = rng.normal(size=(1200, 14))
    labels = (values[:, 0] * values[:, 1] + values[:, 2] > 0).astype(int)
    model, diagnostics = BinaryModel.fit(values, labels, kind)
    assert diagnostics['export_max_error'] < 1e-12
    restored = BinaryModel.from_dict(model.to_dict())
    assert np.array_equal(model.probabilities(values), restored.probabilities(values))
    values[1, 3] = np.inf
    assert np.isnan(restored.probabilities(values)[1])


@pytest.mark.parametrize('mutation', ['cycle', 'shared', 'index', 'nan', 'orphan'])
def test_untrusted_tree_structure_is_rejected(mutation):
    tree = {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [0, -2, -2],
            'threshold': [0., -2., -2.], 'value': [0., -1., 1.]}
    if mutation == 'cycle':
        tree['left'][0] = 0
    elif mutation == 'shared':
        tree['right'][0] = 1
    elif mutation == 'index':
        tree['feature'][0] = 14
    elif mutation == 'nan':
        tree['value'][1] = float('nan')
    elif mutation == 'orphan':
        tree['left'][0] = tree['right'][0] = -1
        tree['feature'][0] = -2
    with pytest.raises(ValueError):
        BinaryModel.from_dict({'format': 'expansion_binary_v1', 'kind': 'boosted', 'features': MARKET_FEATURES,
                              'learning_rate': 0.05, 'trees': [tree]})


def test_policy_gates_holds_and_cache_checks_actual_features():
    policy = ExpansionPolicy(constant(.2), constant(.9), .1, .65)
    state = {'halted': False, 'direction': -1, 'hold_bars': 11}
    event = {'end': '2020-01-01T00:05:00+00:00', 'features': [0.] * 14}
    assert policy(event, state) == 'hold'
    state['hold_bars'] = 12
    assert policy(event, state) == 'enter_long'
    frame = pd.DataFrame([dict(zip(MARKET_FEATURES, event['features'], strict=True))])
    frame['end'] = pd.to_datetime([event['end']], utc=True)
    policy.prepare(frame)
    assert policy(event, state) == 'enter_long'
    # 캐시는 시각만 같고 특징이 바뀐 입력을 재사용하지 않는다.
    policy._scores[0] = [.2, .1]
    changed = copy.deepcopy(event)
    changed['features'][0] = 1.
    assert policy(changed, state) == 'enter_long'
    assert policy(event, state) == 'hold'
    state['halted'] = True
    assert policy(changed, state) == 'hold'
    state['halted'] = False
    assert ExpansionPolicy(constant(.01), constant(.9), .1, .65)(event, state) == 'hold'


def test_risk_stop_overrides_minimum_hold_and_signal_executes_next_bar(tmp_path):
    times = pd.date_range('2020-01-01', periods=3, freq='5min', tz='UTC')
    frame = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5),
                          'open': 100., 'high': 101., 'low': [99., 95., 99.], 'close': 100., 'funding_rate': 0.})
    frame[MARKET_FEATURES] = .01
    policy = ExpansionPolicy(constant(.2), constant(.9), .1, .65)
    backtest(frame, policy, EngineConfig(stop_fraction=.04, max_hold_bars=72, cooldown_bars=3), tmp_path / 'run')
    trades = pd.read_parquet(tmp_path / 'run' / 'trades.parquet')
    assert len(trades) == 1
    assert pd.Timestamp(trades.iloc[0].entry_time) == times[1]
    assert trades.iloc[0]['exit_reason'] == 'intrabar_stop'
