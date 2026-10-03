import json

import numpy as np
import pandas as pd
import pytest
from test_market_liquidity import event
from test_pullback_evaluation import bars
from test_rate_policy import inputs, policy, rate_selection

from wonyotti_fr.action_research import action_diagnostics, load_action_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.rate_policy import ReversalRatePolicy
from wonyotti_fr.reversal_research import run_reversal_selection


def reverse(scores=(.25, 0., 0.)):
    base = policy(scores)
    return ReversalRatePolicy(base, base.manager, base.thresholds, base.multiplier, base.scales)


def source_selection(root):
    frozen = rate_selection(root)
    model = json.loads((root / 'action_model.json').read_text())
    model['intercept'] = [float(np.log(.25/.75)), -10., -10.]
    save_json(root / 'action_model.json', model)
    fingerprint = {'action_model.json': sha256(root / 'action_model.json')}
    path = json.loads((root / 'path_selection.json').read_text())
    path['model_sha256'] = fingerprint
    save_json(root / 'path_selection.json', path)
    frozen.update(model_sha256=fingerprint, path_selection_sha256=sha256(root / 'path_selection.json'))
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def selection(root):
    frozen = source_selection(root)
    (root / 'rate_selection.json').write_bytes((root / 'frozen_selection.json').read_bytes())
    frozen.update(protocol='minute_rate_reverse_v18', candidate=0,
                  rate_selection_sha256=sha256(root / 'rate_selection.json'))
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_accumulation_matches_original_until_exit_and_keeps_pending_position_identity():
    one, two = policy([.25, .1, .05]), reverse([.25, .1, .05])
    first, second = {}, {}
    for minute in range(1, 5):
        a, b = one(*inputs(minute, first)), two(*inputs(minute, second))
        first, second = a.state, b.state
        if minute < 4:
            assert a == b and b.intent == 'hold'
        else:
            assert a.intent == 'exit' and b.intent == 'enter_short'
            assert first['rate_exit'] == second['rate_exit'] == 0
            assert second['management_direction'] == -1
            assert second['rate_entry_time'] == first['rate_entry_time']
    pending = two(*inputs(5, second, 'enter_short'))
    assert pending.intent == 'hold' and pending.state['rate_exit'] == .25
    bar, state = inputs(6, pending.state)
    state.update(direction=-1, hold_bars=1, position_entry_time='2021-01-01T00:05:00+00:00')
    fresh = two(bar, state)
    assert fresh.state['rate_exit'] == .25 and fresh.state['rate_reduce'] == .1
    assert fresh.state['path_entry_time'] == fresh.state['rate_entry_time'] == state['position_entry_time']
    assert fresh.state['path_direction'] == -1 and fresh.event == 'action_rate_cooldown'


@pytest.mark.parametrize('delay', [0, 1])
def test_actual_reversal_two_legs_costs_and_new_position_counters(tmp_path, delay):
    frame, bot = bars(), reverse()
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, signal_delay_bars=delay)
    result = backtest(frame, bot, config, tmp_path / 'run')
    report = action_diagnostics(tmp_path / 'run', frame, config)
    fills = pd.read_parquet(tmp_path / 'run/fills.parquet')
    equity = pd.read_parquet(tmp_path / 'run/equity.parquet')
    reversals = fills[fills.reason.eq('signal_reverse')]
    assert len(reversals) >= 3 and result['max_accounting_residual'] < 1e-7
    assert pd.to_datetime(reversals.time, utc=True).diff().dropna().ge(pd.Timedelta(minutes=4)).all()
    for leg in reversals.itertuples():
        pair = fills[fills.time.eq(leg.time)]
        assert pair.reason.tolist() == ['signal_reverse', 'entry']
        assert pair.fee.gt(0).all() and pair.delta_quantity.prod() > 0
        row = equity[pd.to_datetime(equity.time, utc=True).eq(pd.Timestamp(leg.time)+pd.Timedelta(minutes=1))].iloc[0]
        if row.completed:
            continue
        stored = json.loads(row.policy_state)
        assert stored['rate_entry_time'] == stored['path_entry_time'] == leg.time
        assert stored['rate_exit'] == .25 and stored['rate_reduce'] == stored['rate_increase'] == 0
        assert stored['path_low'] == stored['path_high']
        assert pd.Timestamp(stored['path_end']) - pd.Timestamp(stored['path_entry_time']) == pd.Timedelta(minutes=1)
    assert report['waiting']['reversal_entries'] == len(reversals)
    assert report['waiting']['matched_entry_fills'] == result['closed_trades']


@pytest.mark.parametrize('risk_halt', [False, True])
def test_pending_reversal_waits_for_liquidity_and_risk_halt_closes_flat(risk_halt):
    bot = reverse([.5, 0, 0])
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0)
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long'
    def bar(n, active=True, funding=0.):
        return {**event(n, active=active, funding=funding), 'features': np.zeros(14)}
    engine.step(bar(0), bot)
    engine.step(bar(1), bot)
    assert engine.state['pending'] == 'enter_short'
    wait = engine.step(bar(2, active=False, funding=.2 if risk_halt else 0), bot)
    assert not wait['fills'] and wait['quantity'] > 0
    snapshot = engine.snapshot()
    engine = TradingEngine(config, snapshot)
    result = engine.step(bar(3), bot)
    if risk_halt:
        assert result['quantity'] == 0 and [f['reason'] for f in result['fills']] == ['liquidity_risk_halt']
    else:
        assert result['quantity'] < 0 and [f['reason'] for f in result['fills']] == ['signal_reverse', 'entry']
        assert engine.state['policy_state']['rate_exit'] == .5
    assert abs(result['accounting_residual']) < 1e-7


@pytest.mark.parametrize('damage', ['parent', 'scale', 'risk'])
def test_frozen_rate_model_chain_rejects_changes(tmp_path, damage):
    root = tmp_path / 'selection'
    frozen = selection(root)
    assert isinstance(load_selection(root)[1], ReversalRatePolicy)
    if damage == 'parent':
        (root / 'rate_selection.json').write_text('{}')
    elif damage == 'scale':
        frozen['rate_scales']['exit'] *= 2
    else:
        frozen['risk']['fee_bps'] = 0
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_crash_recovery_and_period_guard(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = selection(root)
    guard_replay_period(root, frozen, '2021-01-01', '2022-01-01')
    with pytest.raises(ValueError, match='관찰한 기간'):
        guard_replay_period(root, frozen, '2026-09-01', '2026-11-01')
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_selection_and_evaluation_preserve_original_rate_control(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    frozen = source_selection(root)
    metrics = {'total_return': -.1, 'max_drawdown': -.15, 'closed_trades': 30, 'permanent_halt': False}
    frozen['development_metrics'] = metrics
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    save_json(root / 'confirmation-2022/metrics.json', metrics)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.reversal_research.prepare_minute_period', lambda *_: (bars(), {}))
    output = run_reversal_selection(root, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs', rate_based=True)
    assert isinstance(load_selection(output)[1], ReversalRatePolicy)
    assert [r['policy'] for r in json.loads((output / 'comparison.json').read_text())] == ['v14_flat', 'v18_reverse']*2
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_: (bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic_scope': True})
    evaluated = run_pullback_evaluation(output, tmp_path, tmp_path, tmp_path / 'eval', 'seen_2026', ['BTCUSDT'])
    assert isinstance(load_selection(evaluated)[1], ReversalRatePolicy)
    assert len(json.loads((evaluated / 'results.json').read_text())) == 9
    _, original = load_action_selection(root, frozen)
    backtest(bars(), original, EngineConfig(**frozen['risk']), tmp_path / 'original')
    for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'original' / name),
            pd.read_parquet(evaluated / 'BTCUSDT/unfiltered_v14' / name), check_exact=True)
    assert json.loads((tmp_path / 'original/final_state.json').read_text()) == json.loads((evaluated / 'BTCUSDT/unfiltered_v14/final_state.json').read_text())
