import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_engine import config
from test_minute_inventory_research import ready_bars, selection

from wonyotti_fr.addition_effect import (
    collect_addition_states,
    paired_addition_outcome,
    position_weights,
)
from wonyotti_fr.engine import TradingEngine
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_inventory import MinuteInventoryModels


def event(i, price=100., **kwargs):
    t = pd.Timestamp('2021-01-01', tz='UTC') + pd.Timedelta(minutes=i)
    return {'time': t.isoformat(), 'end': (t + pd.Timedelta(minutes=1)).isoformat(),
            'open': price, 'high': price, 'low': price, 'close': price,
            'funding_rate': 0., 'count': 1, 'volume': 1., 'step': i, **kwargs}


def pending(cfg):
    engine = TradingEngine(cfg)
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), lambda *_: 'increase')
    return engine.snapshot()


def exit_next(_bar, _state):
    return 'exit'


def test_paired_add_has_exact_marginal_fee_funding_and_cash():
    cfg = config(bar_seconds=60, fee_bps=5, slippage_bps=3)
    state = pending(cfg)
    untouched = copy.deepcopy(state)
    events = [event(1, 100.), event(2, 110., funding_rate=.001), event(3, 100.)]
    outcome, trace = paired_addition_outcome(events, 0, state, cfg, exit_next, pd.Timestamp('2021-01-02', tz='UTC'))
    add = trace['allow']['fills'][0]
    # 두 경로는 같은 가격에 종료하므로 추가 수량의 가격·수수료·펀딩 차이만 남는다.
    qty, purchase, sale = add['delta_quantity'], add['price'], 110*(1-.0003)
    expected = qty*(sale-purchase) - add['fee'] - qty*sale*.0005 - qty*110*.001
    assert outcome['incremental_pnl'] == pytest.approx(expected, abs=1e-9)
    assert outcome['incremental_bps'] == pytest.approx(expected/outcome['decision_equity']*10000)
    assert state == untouched
    assert all(len(t['closed_trades']) == 1 and t['max_accounting_residual'] < 1e-8 for t in trace.values())
    assert all(t['final_state']['quantity'] == 0 for t in trace.values())


@pytest.mark.parametrize('price,sign', [(110., 1), (90., -1), (100., 0)])
def test_positive_negative_and_zero_labels_are_retained(price, sign):
    cfg = config(bar_seconds=60)
    outcome, _ = paired_addition_outcome([event(1), event(2, price)], 0, pending(cfg), cfg, exit_next,
                                         pd.Timestamp('2021-01-02', tz='UTC'))
    assert outcome['label_status'] == 'closed'
    assert np.sign(outcome['incremental_pnl']) == sign


def test_branches_can_close_at_different_times_without_new_positions():
    cfg = config(bar_seconds=60)
    def manage(bar, view):
        assert view['direction'] != 0
        return 'exit' if view['adds'] or bar['step'] >= 3 else 'hold'
    outcome, trace = paired_addition_outcome([event(i, 100+i) for i in range(1, 7)], 0, pending(cfg), cfg, manage,
                                             pd.Timestamp('2021-01-02', tz='UTC'))
    assert trace['allow']['label_end'] < trace['skip']['label_end'] == outcome['label_end']
    assert all(len(t['closed_trades']) == 1 for t in trace.values())


def test_boundary_censoring_does_not_force_close_or_make_zero_target():
    cfg = config(bar_seconds=60)
    cutoff = pd.Timestamp('2021-01-01T00:03Z')
    outcome, trace = paired_addition_outcome([event(i) for i in range(1, 5)], 0, pending(cfg), cfg,
                                             lambda *_: 'hold', cutoff)
    assert outcome['label_status'] == 'right_censored' and outcome['incremental_bps'] is None
    assert outcome['label_end'] == cutoff
    assert all(t['final_state']['quantity'] > 0 and not t['closed_trades'] for t in trace.values())
    assert all(pd.Timestamp(t['final_state']['last_end']) < cutoff for t in trace.values())


def test_zero_trade_delay_and_risk_exit_use_the_same_engine():
    cfg = config(bar_seconds=60, stop_fraction=.04)
    events = [event(1, count=0, volume=0), event(2), event(3, 90.)]
    outcome, trace = paired_addition_outcome(events, 0, pending(cfg), cfg, lambda *_: 'hold',
                                             pd.Timestamp('2021-01-02', tz='UTC'))
    assert outcome['label_status'] == 'closed'
    assert trace['allow']['fills'][0]['time'] == events[1]['time']
    assert all(t['closed_trades'][0]['exit_reason'] == 'gap_stop' for t in trace.values())


def test_engine_rejected_addition_remains_a_zero_effect_label():
    cfg = config(bar_seconds=60, max_adds=0)
    result, trace = paired_addition_outcome([event(1), event(2, 110)], 0, pending(cfg), cfg, exit_next,
                                           pd.Timestamp('2021-01-02', tz='UTC'))
    assert result['incremental_pnl'] == 0
    assert trace['allow']['fills'] == trace['skip']['fills']


def test_wrong_start_and_deferred_configuration_are_rejected():
    cfg = config(bar_seconds=60)
    for first, altered in [(2, cfg), (1, replace(cfg, signal_delay_bars=1))]:
        with pytest.raises(ValueError, match='시작 상태'):
            paired_addition_outcome([event(first)], 0, pending(cfg), altered, exit_next,
                                    pd.Timestamp('2021-01-02', tz='UTC'))


def test_collection_preserves_full_baseline_and_position_weights(tmp_path):
    root = tmp_path/'selection'
    frozen = selection(root, tmp_path/'previous')
    cfg = config(**{k: v for k, v in frozen['risk'].items() if k not in {'stop_fraction', 'fee_bps', 'slippage_bps', 'max_hold_bars', 'daily_loss_limit', 'max_drawdown', 'allow_adverse_add'}})
    bars = ready_bars().assign(volume=1., count=1)
    _, bot = load_selection(root)
    model = bot.manager.to_dict()
    model['coef'] = [[0.]*44 for _ in range(3)]
    model['intercept'] = [-100., -100., 0.]
    bot.manager = MinuteInventoryModels.from_dict(model)
    bot.scales = {'exit': 1., 'reduce': 1., 'increase': 1.}
    backtest(bars, bot, cfg, tmp_path/'old')
    opportunities, state = collect_addition_states(bars, bot, cfg, tmp_path/'new')
    for name in ['equity', 'fills', 'trades']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/f'old/{name}.parquet'),
                                      pd.read_parquet(tmp_path/f'new/{name}.parquet'), check_exact=True)
    import json
    assert state == json.loads((tmp_path/'old/final_state.json').read_text())
    assert len(opportunities) >= 2
    for row in opportunities:
        assert row['state']['pending'] == 'increase' and len(row['features']) == 44
        assert row['decision_time'] == row['state']['last_end']
    frame = pd.DataFrame({'position_entry_time': ['a', 'a', 'b']})
    weights = position_weights(frame)
    assert weights.mean() == 1 and weights[0]+weights[1] == weights[2]
