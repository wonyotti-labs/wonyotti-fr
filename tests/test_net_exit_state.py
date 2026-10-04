import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_activity_ablation import bots
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import history_inputs
from test_horizon_exit_state import fixture as horizon_fixture
from test_market_liquidity import event, hold
from test_minute_inventory_research import ready_bars

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.horizon_exit_state import HORIZON_EXIT_FILES, HorizonExitPolicy
from wonyotti_fr.net_exit_research import run_net_exit_selection
from wonyotti_fr.net_exit_state import NetExitPolicy, copy_horizon_parent
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def bot(scores=(.03, 0., 0.)):
    _, parent = bots(manager=scores)
    return NetExitPolicy(HorizonExitPolicy(parent, .02))


def fixture(tmp_path, monkeypatch):
    parent, previous, _, _, frozen = horizon_fixture(tmp_path, monkeypatch)
    name = HORIZON_EXIT_FILES[0]
    value = json.loads((parent/name).read_text())
    value['threshold'] = .02
    save_json(parent/name, value)
    frozen['horizon_exit_files_sha256'][name] = sha256(parent/name)
    save_json(parent/'frozen_selection.json', frozen)
    save_json(parent/'frozen_integrity.json', {'frozen_selection_sha256': sha256(parent/'frozen_selection.json')})
    root = tmp_path/'net-exit'
    root.mkdir()
    copy_horizon_parent(parent, root)
    frozen = {**frozen, 'protocol': 'net_exit_v53',
        'horizon_exit_selection_sha256': sha256(root/'horizon_exit_selection.json'),
        'net_exit_rule_sha256': sha256(root/'net_exit_rule.json')}
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    return root, parent, previous, frozen


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('cost', [1, 2, 3])
def test_estimate_equals_independent_cashflows_and_actual_close(direction, cost):
    risk = EngineConfig(bar_seconds=60, max_hold_bars=0, stop_fraction=0., max_drawdown=1., daily_loss_limit=1.,
        fee_bps=5.*cost, slippage_bps=3.*cost, entry_fraction=.125, addition_fraction=.25,
        allow_adverse_add=True, cooldown_bars=0)
    engine = TradingEngine(risk)
    cashflow = 0.
    for minute, intent, price, funding in [(0, 'enter_long' if direction == 1 else 'enter_short', 100., 0.),
        (1, 'increase', 99., .001), (2, 'reduce', 102., -.002), (3, 'hold', 101., .001)]:
        engine.state['pending'] = intent
        cashflow -= engine.state['quantity'] * price * funding
        result = engine.step(event(minute, price, funding=funding), hold)
        cashflow -= sum(f['delta_quantity']*f['price']+f['fee'] for f in result['fills'])
        assert abs(result['accounting_residual']) < 1e-8
        engine = TradingEngine(risk, engine.snapshot())
    quantity = engine.state['quantity']
    execution = 101.*(1-direction*risk.slippage_bps/10000)
    expected = cashflow+quantity*execution-abs(quantity)*execution*risk.fee_bps/10000
    estimate = engine.view(101.)['estimated_exit_net']
    assert estimate == pytest.approx(expected, abs=1e-10)
    assert engine.state['active_trade']['adds'] == 1
    engine.state['pending'] = 'exit'
    closed = engine.step(event(4, 101.), hold)
    assert closed['closed_trades'][0]['net_pnl'] == pytest.approx(estimate, abs=1e-10)
    assert engine.view(101.)['estimated_exit_net'] == 0.


@pytest.mark.parametrize('net,scores,intent', [(0., (.03, 0., 0.), 'exit'), (1., (.03, 0., 0.), 'exit'),
    (-1., (.03, 0., 0.), 'hold'), (-1., (.6, 0., 0.), 'exit'),
    (-1., (.03, .8, .9), 'reduce'), (-1., (.03, 0., .8), 'increase')])
def test_only_additional_early_exit_is_guarded(net, scores, intent):
    assert bot(scores)(*history_inputs(1, estimated_exit_net=net)).intent == intent


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf')])
def test_bad_net_estimate_is_rejected(value):
    with pytest.raises(ValueError, match='추정 순손익'):
        bot()(*history_inputs(1, estimated_exit_net=value))


@pytest.mark.parametrize('kind', ['manual_halt', 'time_limit', 'gap_stop', 'risk_halt'])
def test_forced_loss_exit_remains_active_and_waits_for_liquidity(kind):
    risk = EngineConfig(bar_seconds=60, max_hold_bars=int(kind == 'time_limit'), cooldown_bars=0,
        daily_loss_limit=.001 if kind == 'risk_halt' else 1., max_drawdown=1.)
    engine = TradingEngine(risk)
    engine.state['pending'] = 'enter_long'
    engine.step(event(0), hold)
    if kind == 'manual_halt':
        engine.halt()
    price = 90. if kind == 'gap_stop' else 100.
    result = engine.step(event(1, price, active=False, funding=.01 if kind == 'risk_halt' else 0.), hold)
    assert not result['fills'] and engine.view(price)['estimated_exit_net'] < 0
    assert engine.state['liquidity_exit_reason'] == kind
    restored = TradingEngine(risk, engine.snapshot())
    closed = restored.step(event(2, price), hold)
    assert closed['fills'][0]['reason'] == f'liquidity_{kind}'
    assert closed['closed_trades'][0]['net_pnl'] < 0


@pytest.mark.parametrize('delay', [0, 1])
def test_future_gap_can_turn_positive_estimate_into_real_loss(delay):
    risk = EngineConfig(bar_seconds=60, signal_delay_bars=delay, max_hold_bars=0, stop_fraction=0.,
        max_drawdown=1., daily_loss_limit=1., cooldown_bars=0)
    engine = TradingEngine(risk)
    engine.state['pending'] = 'enter_long'
    first = history_inputs(1)[0]
    first['time'] = (pd.Timestamp(first['end'])-pd.Timedelta(minutes=1)).isoformat()
    first['minute_features'] = first['minute_features'].tolist()
    first.update(open=100., low=100., high=101., close=101., count=1, volume=100., funding_rate=0.)
    engine.step(first, bot())
    first = {**first, 'time': first['end'], 'end': (pd.Timestamp(first['end'])+pd.Timedelta(minutes=1)).isoformat()}
    result = engine.step(first, bot())
    assert result.get('policy_event') == 'action_exit'
    assert engine.view(101.)['estimated_exit_net'] > 0
    snapshot = engine.snapshot()
    outcomes = []
    for future_price in [101., 90.]:
        restored = TradingEngine(risk, copy.deepcopy(snapshot))
        assert restored.view(101.) == engine.view(101.)
        for minute in range(1, delay+2):
            later = {**first, 'time': (pd.Timestamp(first['time'])+pd.Timedelta(minutes=minute)).isoformat(),
                'end': (pd.Timestamp(first['end'])+pd.Timedelta(minutes=minute)).isoformat(),
                'open': future_price, 'high': future_price, 'low': future_price, 'close': future_price}
            result = restored.step(later, hold)
        outcomes.append(result['closed_trades'][0]['net_pnl'])
    assert outcomes[0] > 0 > outcomes[1]


@pytest.mark.parametrize('damage', ['parent', 'risk', 'model', 'rule', 'plan'])
def test_loader_rejects_changed_frozen_inputs(tmp_path, monkeypatch, damage):
    root, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'parent':
        (root/'horizon_exit_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'model':
        (root/'history_manager.json').write_text('{}')
    else:
        rule = json.loads((root/'net_exit_rule.json').read_text())
        rule['minimum_estimated_exit_net' if damage == 'rule' else 'protocol_sha256'] = -1
        save_json(root/'net_exit_rule.json', rule)
        frozen['net_exit_rule_sha256'] = sha256(root/'net_exit_rule.json')
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_and_period_guard(tmp_path, monkeypatch, kind):
    root, _, _, frozen = fixture(tmp_path, monkeypatch)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path/'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_eleven_variants_preserve_previous_outputs_and_risk(tmp_path, monkeypatch):
    root, parent, previous, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path/'runs', 'seen_2026', ['BTCUSDT'])
    names = {r['strategy'] for r in json.loads((out/'results.json').read_text())}
    assert len(names) == 11 and {'previous_v48', 'previous_v52', 'unfiltered_v14'} <= names
    assert 'previous_v46' not in names and isinstance(load_selection(out)[1], NetExitPolicy)
    for label, reference in [('previous_v52', parent), ('previous_v48', previous)]:
        frozen, policy = load_selection(reference)
        expected = tmp_path/label
        backtest(ready_bars(), policy, EngineConfig(**frozen['risk']), expected)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(expected/name), pd.read_parquet(out/f'BTCUSDT/{label}'/name), check_exact=True)
        assert json.loads((expected/'final_state.json').read_text()) == json.loads((out/f'BTCUSDT/{label}/final_state.json').read_text())
        assert json.loads((out/f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']


def test_future_prices_cannot_change_rule_or_model(tmp_path, monkeypatch):
    _, parent, _, _ = fixture(tmp_path, monkeypatch)
    old, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent/'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.net_exit_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_net_exit_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'runs')
    frozen, policy = load_selection(out)
    assert frozen['risk'] == old['risk']
    assert policy.manager.model.to_dict() == original.manager.model.to_dict()
    assert policy.manager.offset.to_dict() == original.manager.offset.to_dict()
    assert json.loads((out/'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    for flags in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]:
        for net in [-1., 0., 1.]:
            state = {'_fill_features': np.array([flags[0], 0., 0., flags[1], 0., 0.]), 'estimated_exit_net': net}
            for action in ['increase', 'reduce']:
                assert policy.action_threshold(action, state) == original.action_threshold(action, state)
    def changed_market(*args, **kwargs):
        bars = ready_bars()
        if args[3] == '2022-01-01':
            bars[['open', 'high', 'low', 'close']] *= 2.
        return bars, {}
    monkeypatch.setattr('wonyotti_fr.net_exit_research.prepare_minute_period', changed_market)
    future = run_net_exit_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'future')
    for name in ['net_exit_rule.json', *HORIZON_EXIT_FILES, 'history_manager.json', 'history_offset.json',
                 'history_thresholds.json', 'first_policy_thresholds.json', 'horizon_exit_selection.json']:
        assert (out/name).read_bytes() == (future/name).read_bytes()
