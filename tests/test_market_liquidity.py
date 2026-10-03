from dataclasses import replace

import pandas as pd
import pytest
from test_lifecycle_edge import edge_selection
from test_pullback_evaluation import bars

from wonyotti_fr.engine import EngineConfig, TradingEngine, validate_bar
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.journal import EventJournal, canonical
from wonyotti_fr.lifecycle_edge_research import lifecycle_edge_diagnostics


def event(minute, price=100., active=True, funding=0.):
    time = pd.Timestamp('2021-01-01T23:58Z') + pd.Timedelta(minutes=minute)
    return {'time': time.isoformat(), 'end': (time+pd.Timedelta(minutes=1)).isoformat(),
            'open': price, 'high': price, 'low': price, 'close': price, 'count': 2 if active else 0,
            'volume': 1. if active else 0., 'funding_rate': funding}


def hold(_bar, _state):
    return 'hold'


def settings():
    return EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0, cooldown_bars=0)


def test_pending_exit_waits_without_fills_or_fees_and_keeps_funding():
    engine = TradingEngine(settings())
    engine.state['pending'] = 'enter_long'
    entry = engine.step(event(0), hold)['fills'][0]
    quantity, fees = engine.state['quantity'], engine.state['total_fees']
    engine.state['pending'] = 'exit'
    for minute in [1, 2]:
        result = engine.step(event(minute, active=False, funding=.001), hold)
        assert result['fills'] == [] and result['next_intent'] == 'exit'
        assert engine.state['quantity'] == quantity and engine.state['total_fees'] == fees
        assert result['rejected'] == ['market_no_trades']
    result = engine.step(event(3, 101), hold)
    trade = result['closed_trades'][0]
    assert result['fills'][0]['time'] == event(3)['time'] and result['fills'][0]['reason'] == 'signal_exit'
    assert trade['funding_cost'] == pytest.approx(quantity*100*.002)
    assert trade['net_pnl'] == pytest.approx(quantity-entry['fee']-quantity*101*.0005-trade['funding_cost'])
    assert abs(result['accounting_residual']) < 1e-8


def test_pending_entry_and_deferred_queue_resume_in_order():
    engine = TradingEngine(replace(settings(), signal_delay_bars=1))
    engine.state['pending'], engine.state['deferred_intents'] = 'enter_long', ['exit']
    for minute in [0, 1]:
        result = engine.step(event(minute, active=False), hold)
        assert not result['fills'] and result['next_intent'] == 'enter_long'
        assert engine.state['deferred_intents'] == ['exit']
    entry = engine.step(event(2), hold)
    assert entry['fills'][0]['reason'] == 'entry' and entry['next_intent'] == 'exit'
    exit_result = engine.step(event(3), hold)
    assert exit_result['fills'][0]['reason'] == 'signal_exit' and not exit_result['quantity']


def test_risk_exit_survives_closed_market_midnight_and_blocks_old_entry():
    engine = TradingEngine(settings())
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), hold)
    engine.state['pending'] = 'enter_short'
    first = engine.step(event(1, active=False, funding=.2), hold)
    assert first['daily_halted'] and first['quantity'] > 0 and not first['fills']
    assert engine.state['liquidity_exit_reason'] == 'risk_halt' and engine.state['pending'] == 'hold'
    second = engine.step(event(2, active=False), hold)
    assert not second['daily_halted'] and second['quantity'] > 0 and not second['fills']
    active = engine.step(event(3), hold)
    assert active['quantity'] == 0 and [f['reason'] for f in active['fills']] == ['liquidity_risk_halt']
    assert 'liquidity_exit_reason' not in engine.state


@pytest.mark.parametrize('reason', ['manual_halt', 'time_limit', 'gap_stop'])
def test_forced_exits_wait_for_real_market_activity(reason):
    config = replace(settings(), max_hold_bars=1) if reason == 'time_limit' else settings()
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), hold)
    if reason == 'manual_halt':
        engine.halt()
    price = 90. if reason == 'gap_stop' else 100.
    closed = engine.step(event(1, price, active=False), hold)
    assert not closed['fills'] and closed['quantity'] > 0
    assert engine.state['liquidity_exit_reason'] == reason
    opened = engine.step(event(2, price), hold)
    assert opened['fills'][0]['reason'] == f'liquidity_{reason}' and opened['quantity'] == 0


def test_untradable_final_bar_fails_without_changing_held_state():
    engine = TradingEngine(settings())
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), hold)
    state = engine.snapshot()
    with pytest.raises(ValueError, match='종료 봉'):
        engine.step(event(1, active=False, funding=.01), hold, final=True)
    assert engine.snapshot() == state and not engine.state['completed']
    empty = TradingEngine(settings())
    assert empty.step(event(0, active=False), hold, final=True)['completed']


@pytest.mark.parametrize('change', [{'count': -1}, {'count': .1}, {'count': True}, {'count': 0},
                                   {'volume': 0}, {'volume': float('nan')}, {'count': 0, 'volume': 0, 'high': 101}])
def test_invalid_trade_activity_is_rejected(change):
    with pytest.raises(ValueError, match='거래 활동'):
        validate_bar({**event(0), **change}, 60)


@pytest.mark.parametrize('delay', [0, 1])
def test_waiting_diagnostics_and_journal_link_deferred_liquidity_entry(tmp_path, delay):
    root = tmp_path / 'selection'
    frozen = edge_selection(root, weighted=True, direction_only=True)
    _, policy = load_selection(root)
    frame = bars().assign(count=2, volume=1.)
    frame.loc[6:8, ['count', 'volume']] = 0
    frame.loc[6:8, ['open', 'high', 'low', 'close']] = frame.close.iloc[5]
    config = replace(EngineConfig(**frozen['risk']), signal_delay_bars=delay)
    out = tmp_path / 'run'
    backtest(frame, policy, config, out)
    report = lifecycle_edge_diagnostics(out, frame, policy, config)
    waits = pd.read_parquet(out / 'waiting_episodes.parquet')
    assert waits[waits.executed].liquidity_wait_minutes.max() >= 2
    assert report['waiting']['matched_entry_fills'] == report['waiting']['entry_fills'] > 0
    events = list(iter_events(frame))
    assert events[6]['count'] == events[6]['volume'] == 0
    memory = TradingEngine(config)
    expected = [memory.step(e, policy, final=i == len(events)-1) for i, e in enumerate(events)]
    path = tmp_path / 'journal.sqlite'
    with EventJournal(path, config, {}) as journal:
        for e in events[:9]:
            journal.process(e, policy)
        assert journal.snapshot()['pending'] == 'enter_long'
    with EventJournal(path, config, {}) as journal:
        for i in range(9, len(events)):
            journal.process(events[i], policy, final=i == len(events)-1)
        assert canonical(journal.results()) == canonical(expected)
        assert canonical(journal.snapshot()) == canonical(memory.snapshot())


def test_actual_process_crash_during_unfilled_market_entry(tmp_path):
    root = tmp_path / 'selection'
    edge_selection(root, weighted=True, direction_only=True)
    frame = bars().assign(count=2, volume=1.)
    frame.loc[6:8, ['count', 'volume']] = 0
    frame.loc[6:8, ['open', 'high', 'low', 'close']] = frame.close.iloc[5]
    report = verify_stress(list(iter_events(frame)), root, tmp_path, {}, 'liquidity', True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73
    assert report['interruption_after_bars'] == 7 and not report['position_open_at_interruption']
