import copy

import pandas as pd
import pytest

from wonyotti_fr.engine import EngineConfig, PolicyDecision, TradingEngine
from wonyotti_fr.journal import EventJournal
from wonyotti_fr.pullback_policy import PullbackPolicy


def bar(minute, close=100., **extra):
    stamp = pd.Timestamp('2020-01-01T00:00Z') + pd.Timedelta(minutes=minute)
    return {'time': stamp.isoformat(), 'end': (stamp + pd.Timedelta(minutes=1)).isoformat(),
            'open': close, 'high': close, 'low': close, 'close': close, **extra}


def config():
    return EngineConfig(bar_seconds=60, max_hold_bars=30, max_adds=0, cooldown_bars=15)


def policy(side='enter_long'):
    return PullbackPolicy(lambda *_: side, 8, 5)


def test_wait_requires_completed_close_and_executes_at_next_open_with_cost():
    engine, rule = TradingEngine(config()), policy()
    engine.step(bar(4), rule)
    watched = engine.snapshot()
    assert watched['policy_state']['reference_price'] == 100
    touched = engine.step(bar(5, low=90), rule)
    assert touched['policy_event'] == 'waiting' and not touched['fills']
    ready = engine.step(bar(6, 99.9), rule)
    assert ready['next_intent'] == 'enter_long' and not ready['fills']
    executed = engine.step(bar(7, 102.), rule)
    assert executed['fills'][0]['price'] == pytest.approx(102 * 1.0003)
    assert executed['fills'][0]['fee'] > 0
    assert not engine.state['policy_state']


@pytest.mark.parametrize('side,close', [('enter_long', 99.9), ('enter_short', 100.1)])
def test_expiry_boundary_is_inclusive_and_expiry_does_not_rearm(side, close):
    engine, rule = TradingEngine(config()), policy(side)
    engine.step(bar(4), rule)
    for minute in range(5, 9):
        engine.step(bar(minute), rule)
    snapshot = engine.snapshot()
    assert engine.step(bar(9, close), rule)['next_intent'] == side
    expired = TradingEngine(config(), snapshot)
    output = expired.step(bar(9), rule)
    assert output['policy_event'] == 'expired' and not expired.state['policy_state']
    assert expired.step(bar(10, close), rule)['next_intent'] == 'hold'


def test_policy_state_is_atomic_bounded_and_cleared_on_halt_or_final():
    engine, rule = TradingEngine(config()), policy()
    engine.step(bar(4), rule)
    snapshot = engine.snapshot()
    def invalid(_, state):
        state['policy_state']['reference_price'] = 1
        return PolicyDecision('invalid', state['policy_state'], 'invalid')
    with pytest.raises(ValueError, match='주문 의도'):
        engine.step(bar(5), invalid)
    assert engine.snapshot() == snapshot
    with pytest.raises(ValueError, match='정책 상태'):
        engine.step(bar(5), lambda *_: PolicyDecision('hold', {'nested': []}, 'bad'))
    assert engine.snapshot() == snapshot
    engine.halt()
    assert not engine.state['policy_state']
    ending = TradingEngine(config(), snapshot)
    ending.step(bar(5), rule, final=True)
    assert not ending.state['policy_state']


def test_waiting_journal_restart_duplicate_and_failed_transaction_match_memory(tmp_path):
    events = [bar(4), bar(5), bar(6, 99.9), bar(7, 101.), bar(8, 100.)]
    expected = TradingEngine(config())
    outputs = [expected.step(event, policy(), final=i == 4) for i, event in enumerate(events)]
    path = tmp_path / 'journal.sqlite'
    with EventJournal(path, config(), {'test': 'pullback'}) as journal:
        journal.process(events[0], policy())
        assert journal.snapshot()['policy_state']
        def failed_commit():
            raise RuntimeError('중단')
        before = copy.deepcopy(journal.snapshot())
        with pytest.raises(RuntimeError, match='중단'):
            journal.process(events[1], policy(), before_commit=failed_commit)
        assert journal.snapshot() == before
    with EventJournal(path, config(), {'test': 'pullback'}) as journal:
        assert journal.process(events[0], policy())['duplicate']
        for i, event in enumerate(events[1:], start=1):
            journal.process(event, policy(), final=i == 4)
        assert journal.results() == outputs
        assert journal.snapshot() == expected.snapshot()


def test_malformed_watch_and_wrong_interval_fail_without_advancing():
    engine, rule = TradingEngine(config()), policy()
    engine.step(bar(4), rule)
    bad = engine.snapshot()
    bad['policy_state']['expires_at'] = '2020-01-01T00:20:00+00:00'
    broken = TradingEngine(config(), bad)
    with pytest.raises(ValueError, match='진입 대기 상태'):
        broken.step(bar(5), rule)
    assert broken.snapshot() == bad
    with pytest.raises(ValueError, match='1분'):
        rule(bar(4), {**engine.view(100), 'bar_seconds': 300})
