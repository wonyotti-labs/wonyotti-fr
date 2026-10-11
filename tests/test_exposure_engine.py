import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.exposure_engine import ExposureConfig, ExposureEngine, target_quantity


def bar(i, price=100., high=None, low=None, closing=None, funding=0., count=1):
    end = pd.Timestamp('2021-01-01', tz='UTC')+pd.Timedelta(minutes=i+1)
    close = price if closing is None else closing
    return {'time': (end-pd.Timedelta(minutes=1)).isoformat(), 'end': end.isoformat(), 'open': price,
        'high': max(price, close) if high is None else high, 'low': min(price, close) if low is None else low,
        'close': close, 'funding_rate': funding, 'count': count, 'volume': count}


def config(**overrides):
    values = {'allocation': .5, 'stop_fraction': 0., 'max_hold_bars': 0, 'cooldown_bars': 0,
        'daily_loss_limit': 1., 'max_drawdown': 1.}
    values.update(overrides)
    return ExposureConfig(**values)


@pytest.mark.parametrize('quantity,target', [(0., .5), (0., -.5), (10., .3), (30., .1), (30., -.4), (-30., .4), (-30., -.1), (30., 0.)])
@pytest.mark.parametrize('fee,slip', [(0., 0.), (5., 3.), (15., 9.)])
def test_post_cost_target_quantity_against_independent_bisection(quantity, target, fee, slip):
    price, cash = 123., 10000.-quantity*123.
    solved = target_quantity(cash, quantity, price, target, fee, slip)

    def difference(candidate):
        delta = candidate-quantity
        execution = price*(1+np.sign(delta)*slip/10000)
        new_cash = cash-delta*execution-abs(delta)*execution*fee/10000
        return candidate*price/(new_cash+candidate*price)-target

    left, right = -80., 80.
    for _ in range(100):
        middle = (left+right)/2
        if difference(middle) > 0:
            right = middle
        else:
            left = middle
    assert solved == pytest.approx((left+right)/2, abs=1e-10)
    assert difference(solved) == pytest.approx(0., abs=1e-14)


def test_next_open_funding_partial_close_reversal_and_direct_cash_ledger():
    engine = ExposureEngine(config())
    targets = [.4, .2, -.3, 0., None]
    cash, quantity, closed, fees, funding = 10000., 0., [], 0., 0.
    for i, target in enumerate(targets):
        event = bar(i, price=100.+i, closing=100.5+i, funding=.0001)
        payment = quantity*event['open']*event['funding_rate']
        cash -= payment
        funding += payment
        result = engine.step(event, lambda b, s, target=target: target, final=i == 4)
        if i == 0:
            assert result['fills'] == []
        for fill in result['fills']:
            cash -= fill['delta_quantity']*fill['price']+fill['fee']
            quantity += fill['delta_quantity']
            fees += fill['fee']
        assert result['equity'] == pytest.approx(cash+quantity*event['close'], abs=1e-9)
        assert result['quantity'] == pytest.approx(quantity, abs=1e-12)
        assert abs(result['accounting_residual']) < 1e-8
        closed.extend(result['closed_trades'])
    assert len(closed) == engine.state['closed_trades'] == 2
    assert sum(t['net_pnl'] for t in closed) == pytest.approx(cash-10000., abs=1e-8)
    assert engine.state['total_fees'] == pytest.approx(fees)
    assert engine.state['total_funding'] == pytest.approx(funding)


@pytest.mark.parametrize('delay', [0, 1])
def test_snapshot_resume_and_signal_delay_exact(delay):
    settings = config(signal_delay_bars=delay)
    full, first = ExposureEngine(settings), ExposureEngine(settings)
    choices = [.4, -.2, .1, None, -.1, 0., None]
    expected = []
    for i, target in enumerate(choices):
        expected.append(full.step(bar(i, price=100.+i), lambda b, s, target=target: target, final=i == 6))
    prefix = []
    for i, target in enumerate(choices[:3]):
        prefix.append(first.step(bar(i, price=100.+i), lambda b, s, target=target: target))
    resumed = ExposureEngine(settings, json.loads(json.dumps(first.snapshot())))
    for i, target in enumerate(choices[3:], start=3):
        prefix.append(resumed.step(bar(i, price=100.+i), lambda b, s, target=target: target, final=i == 6))
    assert prefix == expected and resumed.snapshot() == full.snapshot()
    assert next(i for i, value in enumerate(expected) if value['fills']) == 1+delay


def test_no_trade_bar_keeps_original_pending_target():
    engine = ExposureEngine(config())
    engine.step(bar(0), lambda b, s: .4)
    waiting = engine.step(bar(1, count=0), lambda b, s: -.4)
    assert waiting['fills'] == [] and waiting['next_target'] == .4
    filled = engine.step(bar(2), lambda b, s: None)
    assert filled['quantity'] > 0 and filled['net_exposure'] == pytest.approx(.4)


def test_intrabar_stop_and_cooldown_cannot_be_blocked_by_target():
    engine = ExposureEngine(config(stop_fraction=.04, cooldown_bars=2))
    engine.step(bar(0), lambda b, s: .4)
    result = engine.step(bar(1, low=95.), lambda b, s: .4)
    assert result['quantity'] == 0 and result['closed_trades'][0]['exit_reason'] == 'intrabar_stop'
    assert engine.step(bar(2), lambda b, s: .4)['fills'] == []
    assert engine.step(bar(3), lambda b, s: .4)['fills'] == []
    assert engine.step(bar(4), lambda b, s: .4)['quantity'] > 0


def test_gap_stop_waits_for_liquidity_without_reopening():
    engine = ExposureEngine(config(stop_fraction=.04, cooldown_bars=2))
    engine.step(bar(0), lambda b, s: .4)
    engine.step(bar(1), lambda b, s: .4)
    deferred = engine.step(bar(2, price=90., count=0), lambda b, s: .4)
    assert deferred['quantity'] > 0 and deferred['next_target'] is None
    closed = engine.step(bar(3, price=89.), lambda b, s: .4)
    assert closed['quantity'] == 0 and closed['closed_trades'][0]['exit_reason'] == 'liquidity_gap_stop'


def test_maximum_hold_and_drawdown_halt():
    engine = ExposureEngine(config(max_hold_bars=1, cooldown_bars=2))
    engine.step(bar(0), lambda b, s: .4)
    engine.step(bar(1), lambda b, s: .4)
    assert engine.step(bar(2), lambda b, s: .4)['closed_trades'][0]['exit_reason'] == 'time_limit'
    risk = ExposureEngine(config(max_drawdown=.02))
    risk.step(bar(0), lambda b, s: .5)
    risk.step(bar(1), lambda b, s: .5)
    stopped = risk.step(bar(2, price=90.), lambda b, s: .5)
    assert stopped['halted'] and stopped['quantity'] == 0
    assert risk.step(bar(3, price=110.), lambda b, s: .5)['fills'] == []


@pytest.mark.parametrize('mode', ['nan', 'limit', 'bool', 'bad_bar', 'callback', 'future_time'])
def test_invalid_step_leaves_state_unchanged(mode):
    engine = ExposureEngine(config())
    engine.step(bar(0), lambda b, s: .4)
    saved = engine.snapshot()
    event = bar(1)
    value = {'nan': float('nan'), 'limit': .5001, 'bool': True}.get(mode, .4)
    if mode == 'bad_bar':
        event['low'] = 101.
    if mode == 'future_time':
        event = bar(2)
    def decide(b, s):
        if mode == 'callback':
            raise ValueError('검사 예외')
        return value
    with pytest.raises(ValueError):
        engine.step(event, decide)
    assert engine.snapshot() == saved


def test_state_tampering_and_final_no_trade_rejected():
    engine = ExposureEngine(config())
    engine.step(bar(0), lambda b, s: .4)
    engine.step(bar(1), lambda b, s: .4)
    source = engine.snapshot()
    bad = copy.deepcopy(source)
    bad['state']['active_trade']['fees'] = float('nan')
    with pytest.raises(ValueError):
        ExposureEngine(config(), bad)
    with pytest.raises(ValueError):
        ExposureEngine(replace(config(), fee_bps=7.), source)
    with pytest.raises(ValueError):
        engine.step(bar(2, count=0), lambda b, s: None, final=True)
    assert engine.snapshot() == source


@pytest.mark.parametrize('key,value', [('peak', 0.), ('closed_trades', -1), ('wins', 1),
    ('last_end', '2021-01-01T00:01:01+00:00'), ('day', '2021-01-02'), ('liquidity_exit_reason', 'unknown')])
def test_invalid_snapshot_ranges_and_time_boundaries(key, value):
    engine = ExposureEngine(config())
    engine.step(bar(0), lambda b, s: None)
    snapshot = engine.snapshot()
    snapshot['state'][key] = value
    with pytest.raises(ValueError):
        ExposureEngine(config(), snapshot)
