import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.edge_model import EDGE_FEATURES, EdgeModel, EdgePolicy, edge_targets, edge_values
from wonyotti_fr.event_features import MARKET_FEATURES, purged_train
from wonyotti_fr.expansion_model import BinaryModel, ExpansionPolicy


def constant(probability):
    return BinaryModel.from_dict({'format': 'expansion_binary_v1', 'kind': 'logistic', 'features': MARKET_FEATURES,
                                  'mean': [0.] * 14, 'scale': [1.] * 14, 'coefficients': [0.] * 14,
                                  'intercept': float(np.log(probability / (1 - probability)))})


def test_future_price_changes_labels_without_changing_features_and_purge_uses_label_end():
    times = pd.date_range('2019-12-30T23:00Z', periods=24, freq='5min').as_unit('ns')
    events = pd.DataFrame({'end': times, 'buy': 1, 'active': 1, 'usable': True, 'episode_id': 1, 'target_episode_id': 1})
    events[MARKET_FEATURES] = .01
    bars = pd.DataFrame({'end': times, 'close': np.arange(100., 124.)})
    labeled = edge_targets(events, bars, 6)
    assert labeled.outcome_bps.iloc[0] == pytest.approx(np.log(106 / 100) * 10000)
    changed = bars.copy()
    changed.loc[6:, 'close'] *= 2
    altered = edge_targets(events, changed, 6)
    pd.testing.assert_frame_equal(labeled[MARKET_FEATURES], altered[MARKET_FEATURES])
    assert labeled.outcome_bps.iloc[0] != altered.outcome_bps.iloc[0]
    episodes = pd.DataFrame({'episode_id': [1], 'entry_time': [times[0]], 'exit_time': [times[-1]]})
    train = purged_train(labeled, episodes, '2020-01-01')
    assert train.outcome_time.max() < pd.Timestamp('2019-12-31', tz='UTC')
    assert train.end.max() == pd.Timestamp('2019-12-30T23:25Z')
    assert not labeled.usable.iloc[-1]


def test_ridge_export_matches_library_and_invalid_values_abstain():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(1200, 14)), columns=MARKET_FEATURES)
    data['order_direction'] = rng.choice([-1, 1], len(data))
    data['outcome_bps'] = data.order_direction * data.ret_1h * 4 + rng.normal(size=len(data))
    data['outcome_time'] = pd.date_range('2019-01-01', periods=len(data), freq='5min', tz='UTC')
    model, diagnostics = EdgeModel.fit(data, 10)
    assert diagnostics['export_max_error'] < 1e-10
    values = edge_values(data[MARKET_FEATURES].to_numpy(), data.order_direction.to_numpy())
    values[0, 0] = np.nan
    assert np.isnan(model.predict(values)[0])
    payload = model.to_dict()
    payload['scale'][2] = 0
    with pytest.raises(ValueError, match='계수'):
        EdgeModel.from_dict(payload)


def test_expected_cost_gate_and_fixed_holding():
    base = ExpansionPolicy(constant(.5), constant(.9), .1, .65)
    payload = {'format': 'edge_ridge_v1', 'features': EDGE_FEATURES, 'alpha': 10,
               'mean': [0.] * 29, 'scale': [1.] * 29, 'coefficients': [0.] * 29, 'intercept': 20.}
    policy = EdgePolicy(base, EdgeModel.from_dict(payload), 0)
    bar = {'features': [0.] * 14, 'end': '2020-01-01T00:05:00+00:00'}
    state = {'direction': 0, 'halted': False, 'hold_bars': 0}
    assert policy(bar, state) == 'enter_long'
    assert EdgePolicy(base, policy.model, 8)(bar, state) == 'hold'
    assert EdgePolicy(base, policy.model, 8, False)(bar, state) == 'enter_long'
    state['direction'] = -1
    assert policy(bar, state) == 'hold'
