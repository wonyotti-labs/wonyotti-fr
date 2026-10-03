import json
import subprocess
import sys
from dataclasses import replace

import pytest
from test_market_liquidity import event, hold

from wonyotti_fr.engine import EngineConfig, PolicyDecision, TradingEngine
from wonyotti_fr.journal import EventJournal, canonical


def settings(delay=0):
    return EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0, signal_delay_bars=delay, allow_adverse_add=True)


def test_fractional_reduction_and_inventory_follow_actual_fills():
    engine = TradingEngine(settings())
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), lambda *_: PolicyDecision('reduce', {}, 'sized', .75))
    initial = engine.state['quantity']
    assert engine.view(100)['remaining_fraction'] == 1
    first = engine.step(event(1), lambda *_: 'increase')
    assert first['fills'][0]['delta_quantity'] == pytest.approx(-initial*.75)
    assert engine.view(100)['remaining_fraction'] == pytest.approx(.25)
    engine.step(event(2), lambda *_: PolicyDecision('reduce', {}, 'sized', 1.))
    assert engine.view(100)['remaining_fraction'] == 1
    result = engine.step(event(3), hold)
    assert result['quantity'] == 0 and result['closed_trades'][0]['exit_reason'] == 'signal_reduce'
    assert engine.view(100)['remaining_fraction'] == 0
    assert abs(result['accounting_residual']) < 1e-7


def test_delayed_fraction_remains_attached_to_order_across_liquidity_and_restore():
    config = settings(1)
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), lambda *_: PolicyDecision('reduce', {}, 'first', .2))
    original = engine.state['quantity']
    engine.step(event(1), lambda *_: PolicyDecision('reduce', {}, 'second', .8))
    assert engine.state['pending_reduction_fraction'] == .2
    assert engine.state['deferred_reduction_fractions'] == [.8]
    fees = engine.state['total_fees']
    for n in [2, 3]:
        result = engine.step(event(n, active=False), lambda *_: PolicyDecision('reduce', {}, 'changed_score', .6))
        assert result['fills'] == [] and engine.state['total_fees'] == fees
        assert engine.state['pending_reduction_fraction'] == .2
        assert engine.state['deferred_reduction_fractions'] == [.8]
        engine = TradingEngine(config, engine.snapshot())
    first = engine.step(event(4), hold)
    second = engine.step(event(5), hold)
    assert first['fills'][0]['delta_quantity'] == pytest.approx(-original*.2)
    assert second['fills'][0]['delta_quantity'] == pytest.approx(-original*.8*.8)
    assert second['quantity'] == pytest.approx(original*.8*.2)
    assert 'pending_reduction_fraction' not in engine.state and 'deferred_reduction_fractions' not in engine.state


@pytest.mark.parametrize('intent,fraction', [('hold', .5), ('reduce', True), ('reduce', float('nan')),
                                           ('reduce', -.1), ('reduce', 0), ('reduce', 1.1), ('reduce', '.5')])
def test_invalid_size_is_rejected_transactionally(intent, fraction):
    engine = TradingEngine(settings())
    before = engine.snapshot()
    with pytest.raises(ValueError, match='비율'):
        engine.step(event(0), lambda *_: PolicyDecision(intent, {}, 'invalid', fraction))
    assert engine.snapshot() == before


@pytest.mark.parametrize('change', [{'pending_reduction_fraction': .5}, {'pending_reduction_fraction': None},
                                    {'deferred_reduction_fractions': [.5]},
                                    {'deferred_intents': ['exit'], 'deferred_reduction_fractions': [.5]}])
def test_corrupted_order_size_alignment_is_rejected(change):
    config = settings(1)
    state = TradingEngine(config).snapshot()
    state.update(change)
    with pytest.raises(ValueError):
        TradingEngine(config, state)


@pytest.mark.parametrize('reason', ['manual', 'funding_risk', 'final'])
def test_halt_or_completion_clears_all_fraction_metadata(reason):
    engine = TradingEngine(settings(1))
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), hold)
    engine.state.update(pending='reduce', pending_reduction_fraction=.25,
                        deferred_intents=['reduce'], deferred_reduction_fractions=[.75])
    if reason == 'manual':
        engine.halt()
    result = engine.step(event(1, funding=.2 if reason == 'funding_risk' else 0), hold, final=reason == 'final')
    assert result['quantity'] == 0
    assert 'pending_reduction_fraction' not in engine.state and 'deferred_reduction_fractions' not in engine.state


CHILD = '''import os,sys
import pandas as pd
from wonyotti_fr.engine import EngineConfig,PolicyDecision
from wonyotti_fr.journal import EventJournal

def policy(bar,state):
    minute=pd.Timestamp(bar['time']).minute
    if minute==0:return 'enter_long'
    if minute in (1,2):return PolicyDecision('reduce',{},'sized',.25 if minute==1 else .75)
    return 'hold'

def bar(i):
    time=pd.Timestamp('2021-01-01T00:00Z')+pd.Timedelta(minutes=i)
    return {'time':time.isoformat(),'end':(time+pd.Timedelta(minutes=1)).isoformat(),'open':100.,'high':100.,'low':100.,'close':100.}

if __name__=='__main__':
    config=EngineConfig(bar_seconds=60,max_hold_bars=0,slippage_bps=0,signal_delay_bars=1)
    with EventJournal(__import__('pathlib').Path(sys.argv[1]),config,{}) as journal:
        for i in range(3):
            journal.process(bar(i),policy,before_commit=(lambda:os._exit(73)) if i==2 and sys.argv[2]=='before' else None)
            if i==2:os._exit(73)
'''


@pytest.mark.parametrize('when', ['before', 'after'])
def test_actual_process_exit_preserves_sized_order_queue_without_duplicate_fills(tmp_path, when):
    script = tmp_path / 'child.py'
    script.write_text(CHILD)
    namespace = {'__name__': 'size_test'}
    exec(CHILD, namespace)
    config = settings(1)
    config = replace(config, allow_adverse_add=False)
    memory = TradingEngine(config)
    expected = [memory.step(namespace['bar'](i), namespace['policy'], final=i == 6) for i in range(7)]
    path = tmp_path / 'journal.sqlite'
    child = subprocess.run([sys.executable, str(script), str(path), when], capture_output=True, text=True)
    assert child.returncode == 73, child.stderr
    with EventJournal(path, config, {}) as journal:
        if when == 'after':
            assert journal.snapshot()['pending_reduction_fraction'] == .25
            assert journal.snapshot()['deferred_reduction_fractions'] == [.75]
        for i in range(7):
            journal.process(namespace['bar'](i), namespace['policy'], final=i == 6)
        assert canonical(journal.results()) == canonical(expected)
        assert journal.snapshot() == memory.snapshot()


def test_default_policy_outputs_and_state_have_no_size_metadata():
    engine = TradingEngine(settings(1))
    engine.state['pending'] = 'enter_long'
    for minute in range(5):
        engine.step(event(minute), lambda *_: PolicyDecision('reduce', {}, 'fixed'))
        assert 'pending_reduction_fraction' not in engine.state
        assert 'deferred_reduction_fractions' not in engine.state
    json.dumps(engine.snapshot(), allow_nan=False)
