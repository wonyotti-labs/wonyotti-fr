import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_addition_effect import event
from test_close_effect import CLOSE_FEATURES, CollectionPolicy, collection_fixture
from test_engine import config
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot

from wonyotti_fr.engine import PolicyDecision, TradingEngine
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.exit_move_state import ExitMovePolicy
from wonyotti_fr.first_exit_control import FirstExitControl
from wonyotti_fr.first_exit_research import first_exit_breakdown, verify_parent_replay


class Parent:
    manager = SimpleNamespace(features=CLOSE_FEATURES)

    def __init__(self, intent='hold', *, calls=1, finite=True, failure=False):
        self.intent, self.calls, self.finite, self.failure = intent, calls, finite, failure

    def feature_values(self, *_):
        return np.zeros(len(CLOSE_FEATURES)) if self.finite else np.full(len(CLOSE_FEATURES), np.nan)

    def __call__(self, bar, state):
        for _ in range(self.calls):
            self.feature_values(bar, state)
        if self.failure:
            raise RuntimeError('부모 오류')
        return PolicyDecision(self.intent, {'preserved': 3}, 'parent_event', .3 if self.intent == 'reduce' else None)


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('intent', ['hold', 'increase', 'reduce', 'exit'])
def test_original_exit_and_state_preserved_with_full_close_size(direction, intent):
    parent, records = Parent(intent), []
    policy = FirstExitControl(parent, record=records.append)
    state = {'direction': direction, 'halted': False}
    before = parent.feature_values
    result = policy(event(0), state)
    assert result.intent == 'exit' and result.state == {'preserved': 3} and result.reduction_fraction is None
    assert result.event == ('parent_event' if intent == 'exit' else 'first_exit_control')
    assert parent.feature_values == before and records[0]['changed'] == (intent != 'exit')
    assert records[0]['original_intent'] == intent and records[0]['features_supported']


@pytest.mark.parametrize('state,calls,finite,reason', [({'direction': 0, 'halted': False}, 0, True, 'flat'),
    ({'direction': 1, 'halted': True}, 0, True, 'halted'),
    ({'direction': 1, 'halted': False}, 0, True, 'no_management_call'),
    ({'direction': 1, 'halted': False}, 1, False, 'nonfinite_features')])
def test_unsupported_current_decisions_remain_unchanged(state, calls, finite, reason):
    parent, records = Parent(calls=calls, finite=finite), []
    expected = parent(event(0), state)
    assert FirstExitControl(parent, record=records.append)(event(0), state) == expected
    assert records[0]['reason'] == reason and not records[0]['changed']


def test_invalid_parent_and_failed_capture_restore_original_method():
    state = {'direction': 1, 'halted': False}
    for parent in [Parent(calls=2), Parent(failure=True)]:
        before = parent.feature_values
        with pytest.raises((RuntimeError, ValueError)):
            FirstExitControl(parent)(event(0), state)
        assert parent.feature_values == before
    for enabled in [1, 'true', None]:
        with pytest.raises(ValueError):
            FirstExitControl(Parent(), enabled=enabled)
    with pytest.raises(ValueError):
        FirstExitControl(SimpleNamespace(manager=SimpleNamespace(features=['wrong'])))


@pytest.mark.parametrize('direction,delay', [(1, 0), (-1, 0), (1, 1), (-1, 1)])
def test_delayed_exit_no_trade_wait_funding_and_restart(direction, delay):
    cfg = config(bar_seconds=60, fee_bps=5, slippage_bps=3, signal_delay_bars=delay)
    engine = TradingEngine(cfg)
    engine.state['pending'] = 'enter_long' if direction == 1 else 'enter_short'
    records, outputs = [], []
    for minute in range(7):
        liquid = minute not in (1, 2, 3)
        bar = event(minute, 100.+minute/10, count=int(liquid), volume=float(liquid), funding_rate=.001)
        parent = Parent(calls=1 if engine.state['quantity'] or minute == 0 else 0)
        control = FirstExitControl(parent, record=records.append)
        result = engine.step(bar, control, final=minute == 6)
        assert abs(result['accounting_residual']) < 1e-8
        if not liquid:
            assert not result['fills']
        outputs.append(result)
        engine = TradingEngine(cfg, engine.snapshot())
    fills = [f for r in outputs for f in r['fills']]
    assert [v['reason'] for v in fills] == ['entry', 'signal_exit']
    assert fills[-1]['time'] == event(4)['time']
    assert engine.state['closed_trades'] == 1 and engine.state['total_funding']*direction > 0
    assert outputs[4]['closed_trades'][0]['fees'] == pytest.approx(sum(f['fee'] for f in fills))


@pytest.mark.parametrize('direction', [1, -1])
def test_gap_stop_keeps_priority_over_control_exit(direction):
    cfg = config(bar_seconds=60, stop_fraction=.04)
    engine = TradingEngine(cfg)
    engine.state['pending'] = 'enter_long' if direction == 1 else 'enter_short'
    engine.step(event(0), FirstExitControl(Parent()))
    result = engine.step(event(1, 90 if direction == 1 else 110), FirstExitControl(Parent(calls=0)))
    assert result['fills'][0]['reason'] == 'gap_stop'


def test_disabled_real_parent_matches_all_outputs_and_future_changes_do_not_affect_prefix(tmp_path):
    bars = ready_bars().assign(count=1, volume=1.)
    cfg = config(bar_seconds=60, max_hold_bars=20, fee_bps=5, slippage_bps=3)
    def parent():
        policy = ExitMovePolicy(bot((0., .05, .8)), .02)
        policy.manager.features = CLOSE_FEATURES
        return policy
    backtest(bars, parent(), cfg, tmp_path/'parent')
    backtest(bars, FirstExitControl(parent(), enabled=False), cfg, tmp_path/'disabled')
    verify_parent_replay(tmp_path/'disabled', tmp_path/'parent')
    backtest(bars, FirstExitControl(parent()), cfg, tmp_path/'control')
    altered = bars.copy()
    altered.loc[altered.index >= 40, ['open', 'high', 'low', 'close']] *= 1.2
    backtest(altered, FirstExitControl(parent()), cfg, tmp_path/'future')
    old, future = [pd.read_parquet(tmp_path/name/'equity.parquet') for name in ['control', 'future']]
    pd.testing.assert_frame_equal(old.iloc[:40], future.iloc[:40], check_exact=True)
    records = list(iter_events(bars))
    def run(resume):
        policy, engine, results = FirstExitControl(parent()), TradingEngine(cfg), []
        policy.prepare(bars)
        for i, bar in enumerate(records):
            results.append(engine.step(bar, policy, final=i == len(records)-1))
            if resume and i % 3 == 0:
                engine, policy = TradingEngine(cfg, engine.snapshot()), FirstExitControl(parent())
                policy.prepare(bars)
        return results, engine.snapshot()
    assert run(False) == run(True)


def test_control_reentry_changes_full_account_and_loss_breakdown_is_preserved(tmp_path):
    bars, cfg, root = collection_fixture(tmp_path)
    records = []
    backtest(bars, FirstExitControl(CollectionPolicy(), record=records.append), cfg, tmp_path/'control')
    parent = json.loads((root/'candidate-00/metrics.json').read_text())
    current = json.loads((tmp_path/'control/metrics.json').read_text())
    assert current['closed_trades'] > parent['closed_trades'] and current['fees'] > parent['fees']
    parts = first_exit_breakdown(tmp_path/'control')
    assert sum(p['closed_trades'] for p in parts if p['kind'] == 'direction') == current['closed_trades']
    assert any(row['changed'] for row in records)
    damaged = copy.deepcopy(current)
    damaged['fees'] += 1
    (tmp_path/'control/metrics.json').write_text(json.dumps(damaged))
    with pytest.raises((ValueError, AssertionError)):
        verify_parent_replay(tmp_path/'control', root/'candidate-00')
