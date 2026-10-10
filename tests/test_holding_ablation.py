import copy
from dataclasses import replace

import numpy as np
import pytest
from test_activity_ablation import bots
from test_history_state import history_inputs
from test_minute_inventory_research import ready_bars

from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.exit_state import ExitStatePolicy
from wonyotti_fr.holding_ablation import DisabledManagementScores, HoldOnlyControl


@pytest.mark.parametrize('scores', [(.8,.9,.9),(0.,.9,.9),(0.,0.,.9)])
def test_original_management_requests_blocked_and_parent_restored(scores):
    _, old = bots(manager=scores)
    parent = ExitStatePolicy(old)
    bar, state = history_inputs(1)
    before = copy.deepcopy(state)
    original_manager = parent.manager
    assert parent(bar,state).intent in {'exit','reduce','increase'}
    decision = HoldOnlyControl(parent)(bar,state)
    assert decision.intent=='hold' and decision.reduction_fraction is None
    assert parent.manager is original_manager and state==before
    assert 'management_after' not in decision.state
    assert {'path_end','path_entry_time'} <= set(decision.state)


def test_disabled_ablation_matches_original_and_rejects_zero_threshold():
    _, old = bots(manager=(.8,.9,.9))
    parent=ExitStatePolicy(old)
    inputs=history_inputs(1)
    assert HoldOnlyControl(parent,enabled=False)(*inputs)==parent(*inputs)
    parent.thresholds['exit']=0.
    with pytest.raises(ValueError):
        HoldOnlyControl(parent)


def test_original_manager_restored_on_parent_exception():
    _, old=bots(manager=(.8,.9,.9))
    parent=ExitStatePolicy(old)
    original=parent.manager
    bar,state=history_inputs(1)
    state['bar_seconds']=300
    with pytest.raises(ValueError):
        HoldOnlyControl(parent)(bar,state)
    assert parent.manager is original


def test_missing_inputs_preserved_as_unavailable():
    scores=DisabledManagementScores()
    values=np.zeros((2,len(scores.features)))
    values[1,0]=np.nan
    actual=scores.probabilities(values)
    np.testing.assert_array_equal(actual[0],np.zeros(3))
    assert np.isnan(actual[1]).all()
    with pytest.raises(ValueError):
        scores.probabilities(values[:,:-1])


@pytest.mark.parametrize('reason', ['time_limit', 'gap_stop', 'intrabar_stop', 'risk_halt', 'end_of_test'])
def test_hold_only_keeps_partial_entry_and_engine_risk_priority(reason):
    _, old = bots(manager=(.8,.9,.9))
    policy = HoldOnlyControl(ExitStatePolicy(old))
    risk = replace(EngineConfig(), bar_seconds=60, entry_fraction=.125, addition_fraction=.25,
        max_hold_bars=5 if reason == 'time_limit' else 100, stop_fraction=.04 if 'stop' in reason else 0.,
        max_drawdown=.001 if reason == 'risk_halt' else .25, cooldown_bars=15)
    engine = TradingEngine(risk)
    fills = []
    events = list(iter_events(ready_bars()))[:18]
    for i, bar in enumerate(events):
        close = 99.8 if i >= 5 else 100.
        current = {**bar, 'open': 100., 'high': 100., 'low': close, 'close': close}
        if i == 8 and reason in {'gap_stop', 'risk_halt'}:
            current.update(open=90., high=90., low=90., close=90.)
        if i == 8 and reason == 'intrabar_stop':
            current['low'] = 90.
        result = engine.step(current, policy, final=i == len(events)-1)
        fills.extend(result['fills'])
    assert fills[0]['reason'] == 'entry' and fills[1]['reason'] == reason
    assert abs(fills[0]['delta_quantity'])*fills[0]['price']+fills[0]['fee'] == pytest.approx(
        risk.initial_equity*risk.allocation*risk.entry_fraction)
    assert not {'increase','signal_reduce','signal_exit','signal_reverse'} & {f['reason'] for f in fills}
    assert engine.state['quantity'] == 0 and engine.state['completed']


def test_actual_partial_fill_history_and_future_observation_invariance():
    _, old = bots(manager=(.8,.9,.9))
    policy = HoldOnlyControl(ExitStatePolicy(old))
    stored = policy(*history_inputs(1)).state
    stored = policy(*history_inputs(2, stored, adds=1)).state
    stored = policy(*history_inputs(3, stored, adds=1, fraction=.7)).state
    assert stored['fill_adds'] == 1 and stored['fill_fraction'] == .7
    assert stored['fill_increase_mask'] == 2 and stored['fill_reduce_mask'] == 1
    results = []
    for changed in [False, True]:
        _, old = bots(manager=(.8,.9,.9))
        policy = HoldOnlyControl(ExitStatePolicy(old))
        engine = TradingEngine(replace(EngineConfig(), bar_seconds=60, max_hold_bars=100, stop_fraction=0.))
        frame = ready_bars().copy()
        frame[['open','high','low','close']] = 100.
        frame.loc[frame.index >= 5, ['low','close']] = 99.8
        if changed:
            frame.loc[frame.index >= 40, ['open','high','low','close']] *= 1.2
        policy.prepare(frame)
        actual = [engine.step(bar, policy) for bar in list(iter_events(frame))[:40]]
        results.append((actual, engine.snapshot()))
    assert results[0] == results[1]


def test_no_adds_keeps_entry_and_blocks_an_otherwise_executed_addition():
    risk = replace(EngineConfig(), bar_seconds=60, max_hold_bars=20, stop_fraction=0.,
        allow_adverse_add=True, max_adds=1)
    outputs = []
    for limit in [1, 0]:
        _, old = bots(manager=(0.,0.,.9))
        policy = ExitStatePolicy(old)
        engine = TradingEngine(replace(risk, max_adds=limit))
        fills = []
        for i, bar in enumerate(list(iter_events(ready_bars()))[:18]):
            close = 99.8 if i >= 5 else 100.
            result = engine.step({**bar, 'open': 100., 'high': 100., 'low': close, 'close': close}, policy, final=i == 17)
            fills.extend(result['fills'])
        outputs.append(fills)
    assert outputs[0][0] == outputs[1][0]
    assert any(f['reason'] == 'increase' for f in outputs[0])
    assert not any(f['reason'] == 'increase' for f in outputs[1])


def test_disabled_control_matches_all_engine_decisions_and_final_state():
    outputs = []
    for enabled in [None, False]:
        _, old = bots(manager=(0.,.9,.9))
        parent = ExitStatePolicy(old)
        policy = parent if enabled is None else HoldOnlyControl(parent, enabled=False)
        engine = TradingEngine(replace(EngineConfig(), bar_seconds=60, max_hold_bars=20, stop_fraction=0.))
        rows = []
        for i, bar in enumerate(list(iter_events(ready_bars()))[:30]):
            close = 99.8 if i >= 5 else 100.
            rows.append(engine.step({**bar, 'open': 100., 'high': 100., 'low': close, 'close': close}, policy, final=i == 29))
        outputs.append((rows, engine.snapshot()))
    assert outputs[0] == outputs[1]
