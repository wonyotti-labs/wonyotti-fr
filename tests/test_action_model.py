import copy

import numpy as np
import pandas as pd
import pytest
from test_pullback_evaluation import ConstantBase, bars

from wonyotti_fr.action_model import ActionModels, MinuteActionPolicy, select_threshold
from wonyotti_fr.action_research import action_diagnostics
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.minute_management import ACTIONS, FEATURES
from wonyotti_fr.pullback_policy import PullbackPolicy


def data():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(1800, 32)), columns=FEATURES)
    for i, action in enumerate(ACTIONS):
        frame[f'y_{action}'] = frame[FEATURES[i]].gt(.5).astype(int)
    frame['end'] = pd.date_range('2018-01-01', periods=len(frame), freq='D', tz='UTC')
    frame['label_end'] = frame.end + pd.Timedelta(minutes=1)
    return frame.iloc[:1200], frame.iloc[1200:]


@pytest.mark.parametrize('kind', ['logistic', 'tree'])
def test_export_threshold_time_separation_and_scalar_vector_parity(kind):
    train, calibration = data()
    model, thresholds, report = ActionModels.fit(train, calibration, kind)
    assert report['export_max_error'] < 1e-12 and set(thresholds) == set(ACTIONS)
    values = calibration[FEATURES].to_numpy(copy=True)
    expected = model.probabilities(values[:20])
    np.testing.assert_allclose(np.vstack([model.probabilities(v.reshape(1, -1))[0] for v in values[:20]]), expected, rtol=0, atol=1e-15)
    if kind == 'logistic':
        np.testing.assert_allclose(model.data['mean'], train[FEATURES].mean())
        broken = model.to_dict()
        broken['scale'][0] = 0
    else:
        broken = model.to_dict()
        broken['trees'][0]['left'][0] = 0
    with pytest.raises(ValueError):
        ActionModels.from_dict(broken)
    with pytest.raises(ValueError, match='시간'):
        ActionModels.fit(train, train, kind)
    values[0, 0] = np.nan
    assert np.isnan(model.probabilities(values)[0]).all()


def test_threshold_uses_scores_and_labels_with_ties_and_minimum_support():
    labels = np.r_[np.zeros(80), np.ones(20)]
    scores = np.r_[np.full(80, .01), np.full(20, .2)]
    threshold, report = select_threshold(labels, scores, 2.)
    assert threshold == .2 and report['recall'] == report['precision'] == 1
    with pytest.raises(ValueError, match='지원'):
        select_threshold(labels[:99], scores[:99], 2.)


def test_scalar_tree_keeps_double_precision_split_between_adjacent_float32_values():
    low = np.nextafter(np.float32(1), np.float32(2))
    high = np.nextafter(low, np.float32(2))
    threshold = (float(low) + float(high)) / 2
    tree = {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [0, -2, -2],
            'threshold': [threshold, -2., -2.], 'probability': [.5, .1, .9]}
    model = ActionModels.from_dict({'format': 'minute_action_v1', 'features': FEATURES, 'actions': ACTIONS,
                                   'kind': 'tree', 'trees': [tree, tree, tree]})
    values = np.zeros((2, 32))
    values[:, 0] = [low, high]
    np.testing.assert_array_equal(model.probabilities(values), [[.1]*3, [.9]*3])
    np.testing.assert_array_equal(model.probabilities(values[1:]), [[.9]*3])


class FixedScores:
    def __init__(self, values):
        self.values = values

    def probabilities(self, values):
        return np.tile(self.values, (len(values), 1))


def state():
    return {'direction': 1, 'halted': False, 'policy_state': {}, 'pending': 'hold',
            'bar_seconds': 60, 'hold_bars': 31, 'adds': 0, 'favorable_move': .01}


def test_exit_without_activity_gate_at_minute_boundary_and_cooldown():
    policy = MinuteActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([.12, .2, .4]),
                                {'exit': .1, 'reduce': .1, 'increase': .1})
    bar = {'features': [0.]*14, 'end': '2021-01-01T01:01:00+00:00', 'close': 100.}
    first = policy(bar, state())
    assert first.intent == 'exit'
    held = {**state(), 'policy_state': first.state}
    assert policy({**bar, 'end': '2021-01-01T01:02:00+00:00'}, held).event == 'action_cooldown'
    assert policy({**bar, 'end': '2021-01-01T01:04:00+00:00'}, held).intent == 'exit'
    flat = {**held, 'direction': 0}
    assert policy({**bar, 'end': '2021-01-01T01:02:00+00:00'}, flat).state == {}
    corrupt = copy.deepcopy(held)
    corrupt['policy_state']['management_after'] = '2021-01-02T01:01:00+00:00'
    with pytest.raises(ValueError, match='시각'):
        policy(bar, corrupt)


def test_partial_reductions_are_spaced_and_accounting_matches(tmp_path):
    policy = MinuteActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([0., .3, .2]),
                                {'exit': .1, 'reduce': .1, 'increase': .1})
    backtest(bars(), policy, EngineConfig(bar_seconds=60, max_hold_bars=0), tmp_path/'run')
    fills = pd.read_parquet(tmp_path/'run/fills.parquet')
    reduced = fills[fills.reason.eq('signal_reduce')]
    assert len(reduced) >= 2
    assert pd.to_datetime(reduced.time, utc=True).diff().dropna().ge(pd.Timedelta(minutes=3)).all()
    curve = pd.read_parquet(tmp_path/'run/equity.parquet')
    assert curve.accounting_residual.abs().max() < 1e-7
    diagnostics = action_diagnostics(tmp_path/'run', bars(), EngineConfig(bar_seconds=60, max_hold_bars=0))
    assert diagnostics['waiting']['entry_fills'] > 0


def test_delayed_entry_cancels_a_new_wait_before_managing_position(tmp_path):
    policy = MinuteActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([0., 0., .2]),
                                {'exit': .1, 'reduce': .1, 'increase': .1})
    frame = bars()
    # 9분 경계에서 충족된 진입을 1분 지연하면 10분 경계의 새 대기가 남을 수 있다.
    frame.loc[:7, ['open', 'high', 'low', 'close']] = [100., 100.1, 99.9, 100.]
    frame.loc[8:, ['open', 'high', 'low', 'close']] = [99., 99.1, 98.9, 99.]
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, signal_delay_bars=1)
    backtest(frame, policy, config, tmp_path/'delayed')
    report = action_diagnostics(tmp_path/'delayed', frame, config)
    curve = pd.read_parquet(tmp_path/'delayed/equity.parquet')
    assert report['waiting']['matched_entry_fills'] > 0
    assert ((curve.policy_event == 'cleared') & curve.quantity.ne(0)).any()
