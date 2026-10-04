import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import history_inputs
from test_history_state import policy as history_policy
from test_holding_support import fixture as holding_fixture
from test_minute_inventory_research import ready_bars

from wonyotti_fr.activity_ablation import copy_holding_parent, without_activity_gate
from wonyotti_fr.activity_ablation_research import run_activity_ablation_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.expansion_model import ExpansionPolicy
from wonyotti_fr.first_state import FirstStatePolicy
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


class ConstantBinary:
    def __init__(self, value):
        self.value = value

    def probabilities(self, values):
        return np.full(len(values), self.value)


def bots(buy=.8, activity=.001, manager=(0., 0., 0.)):
    parent = FirstStatePolicy(history_policy(manager), {'reduce': .2, 'increase': .2})
    parent.base = ExpansionPolicy(ConstantBinary(activity), ConstantBinary(buy), .5, .65)
    return parent, without_activity_gate(parent)


def selection(tmp_path):
    parent, first, _, _, old = holding_fixture(tmp_path)
    root = tmp_path / 'ablation'
    root.mkdir()
    copy_holding_parent(parent, root)
    frozen = {**old, 'protocol': 'activity_ablation_v46',
        'holding_selection_sha256': sha256(root / 'holding_selection.json'),
        'activity_gate_disabled': True, 'effective_activity_threshold': 0.,
        'ablation_protocol_sha256': sha256(Path('docs/EXPERIMENT_V46.md'))}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, first, frozen


@pytest.mark.parametrize('buy,intent,close', [(.8, 'enter_long', 99.8), (.2, 'enter_short', 100.2)])
def test_only_activity_gate_changes_and_price_wait_is_preserved(buy, intent, close):
    parent, candidate = bots(buy)
    bar, state = history_inputs(5, direction=0, hold_bars=0, position_entry_time=None)
    assert parent(bar, state).event == 'no_signal'
    result = candidate(bar, state)
    assert result.event == 'armed' and result.intent == 'hold'
    later, state = history_inputs(6, result.state, direction=0, hold_bars=0, position_entry_time=None)
    later['close'] = close
    triggered = candidate(later, state)
    assert triggered.intent == intent and triggered.event == 'triggered'
    assert parent.base.activity_threshold == .5 and candidate.base.activity_threshold == 0.
    assert candidate.base.direction is parent.base.direction and candidate.base.activity is parent.base.activity


@pytest.mark.parametrize('buy,activity,bad_features,halted', [
    (.5, .001, False, False), (np.nan, .001, False, False), (.8, np.nan, False, False),
    (.8, .001, True, False), (.8, .001, False, True)])
def test_direction_abstention_bad_scores_and_halt_still_block_entry(buy, activity, bad_features, halted):
    _, bot = bots(buy, activity)
    bar, state = history_inputs(5, direction=0, hold_bars=0, position_entry_time=None, halted=halted)
    if bad_features:
        bar['features'][:] = np.nan
    result = bot(bar, state)
    assert result.intent == 'hold' and not result.state


@pytest.mark.parametrize('scores', [(0., 0., 0.), (.8, .9, .9), (0., .8, .9), (0., 0., .8)])
def test_held_management_and_executed_history_are_exact(scores):
    old, new = bots(.2, manager=scores)
    old_memory, new_memory = {}, {}
    for minute in range(1, 30):
        changes = {'adds': int(minute >= 3), 'fraction': .7 if minute >= 5 else 1.}
        before = old(*history_inputs(minute, old_memory, **changes))
        after = new(*history_inputs(minute, new_memory, **changes))
        assert before == after and after.intent not in ('enter_long', 'enter_short')
        old_memory, new_memory = before.state, after.state


@pytest.mark.parametrize('delay,liquid', [(0, True), (1, True), (0, False), (1, False)])
def test_actual_execution_keeps_risk_quantity_and_nontradable_pending(delay, liquid):
    _, bot = bots()
    risk = replace(EngineConfig(), bar_seconds=60, signal_delay_bars=delay,
        max_hold_bars=5, stop_fraction=0., entry_fraction=.125, addition_fraction=.25,
        max_adds=5, allow_adverse_add=True, cooldown_bars=15)
    engine = TradingEngine(risk)
    events = list(iter_events(ready_bars()))[:18]
    fills, decisions = [], []
    for i, bar in enumerate(events):
        # 첫 5분 경계 이후에만 가격 대기를 통과시킨다.
        close = 99.8 if i >= 5 else 100.
        event = {**bar, 'open': 100., 'close': close, 'high': 100., 'low': close}
        if not liquid and i in [6+delay, 7+delay]:
            event.update(count=0, volume=0., open=close, high=close, low=close)
        result = engine.step(event, bot)
        decisions.append(result['policy_event'])
        fills.extend(result['fills'])
        if not liquid and i in [6+delay, 7+delay]:
            assert not result['fills']
        snapshot = engine.snapshot()
        engine = TradingEngine(risk, copy.deepcopy(snapshot))
        assert engine.snapshot() == snapshot
    assert 'armed' in decisions and 'triggered' in decisions
    assert fills[0]['reason'] == 'entry'
    assert abs(fills[0]['delta_quantity'])*fills[0]['price']+fills[0]['fee'] == pytest.approx(
        risk.initial_equity*risk.allocation*risk.entry_fraction)
    assert fills[1]['reason'] == 'time_limit'
    assert pd.Timestamp(fills[1]['time'])-pd.Timestamp(fills[0]['time']) == pd.Timedelta(minutes=5)


@pytest.mark.parametrize('damage', ['risk', 'threshold', 'disabled', 'parent', 'plan', 'model'])
def test_ablation_loader_rejects_any_unplanned_change(tmp_path, damage):
    root, _, _, frozen = selection(tmp_path)
    if damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'threshold':
        frozen['effective_activity_threshold'] = .1
    elif damage == 'disabled':
        frozen['activity_gate_disabled'] = False
    elif damage == 'parent':
        (root / 'holding_selection.json').write_text('{}')
    elif damage == 'plan':
        frozen['ablation_protocol_sha256'] = '0'*64
    else:
        (root / 'boosted_direction.json').write_text('{}')
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_restores_ungated_entry_and_management(tmp_path, kind):
    root, _, _, frozen = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_eleven_variants_preserve_original_risks_and_both_direct_controls(tmp_path, monkeypatch):
    root, parent, first, _ = selection(tmp_path)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v39', 'previous_v40', 'unfiltered_v14'} <= strategies
    assert 'previous_v36' not in strategies
    assert load_selection(out)[1].base.activity_threshold == 0.
    for label, reference in [('previous_v39', first), ('previous_v40', parent)]:
        frozen, bot = load_selection(reference)
        expected = tmp_path / label
        backtest(ready_bars(), bot, EngineConfig(**frozen['risk']), expected)
        for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / f'BTCUSDT/{label}' / n), pd.read_parquet(expected / n), check_exact=True)
        assert json.loads((out / f'BTCUSDT/{label}/final_state.json').read_text()) == json.loads((expected / 'final_state.json').read_text())
        assert json.loads((out / f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']
    assert json.loads((out / 'BTCUSDT/unfiltered_v14/config.json').read_text())['max_hold_bars'] == 0


def test_selection_preserves_all_models_risk_thresholds_and_full_original_control(tmp_path, monkeypatch):
    parent, _, _, _, old = holding_fixture(tmp_path)
    _, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.activity_ablation_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_activity_ablation_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, bot = load_selection(out)
    assert frozen['risk'] == old['risk'] and bot.base.activity_threshold == 0.
    assert bot.multiplier == original.multiplier and bot.scales == original.scales
    assert bot.first_thresholds == original.first_thresholds and bot.thresholds == original.thresholds
    assert bot.manager.model.to_dict() == original.manager.model.to_dict()
    assert bot.manager.offset.to_dict() == original.manager.offset.to_dict()
    assert bot.size_model.to_dict() == original.size_model.to_dict()
    assert bot.base.direction.to_dict() == original.base.direction.to_dict()
    assert bot.base.activity.to_dict() == original.base.activity.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    assert json.loads((out / 'manifest.json').read_text())['settings']['new_models_fitted'] is False
