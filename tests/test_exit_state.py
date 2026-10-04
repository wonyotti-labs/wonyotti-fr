import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from test_activity_ablation import bots
from test_activity_ablation import selection as activity_selection
from test_exit_direct import run as diagnosis_values
from test_first_management import rows
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import history_inputs
from test_history_state import selection as history_selection
from test_minute_inventory_research import ready_bars

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.exit_direct import MODEL_FILES
from wonyotti_fr.exit_state import (
    EXIT_POLICY_FILES,
    ExitStatePolicy,
    copy_activity_parent,
    prepare_exit_evidence,
)
from wonyotti_fr.exit_state_research import run_exit_state_selection
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def fixture(tmp_path, monkeypatch):
    support, predictions, metrics, decision = diagnosis_values(*rows())
    def with_support(path):
        root, previous, frozen = history_selection(path)
        value = json.loads((root / 'history_thresholds.json').read_text())
        value['thresholds']['exit'] = .04
        value['support'] = {'exit': support}
        save_json(root / 'history_thresholds.json', value)
        frozen['history_files_sha256']['history_thresholds.json'] = sha256(root / 'history_thresholds.json')
        save_json(root / 'frozen_selection.json', frozen)
        save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
        return root, previous, frozen
    monkeypatch.setattr('test_first_state.history_selection', with_support)
    parent, holding, _, old = activity_selection(tmp_path)
    diagnosis = tmp_path / 'exit-diagnosis'
    diagnosis.mkdir()
    settings = {'selection_sha256': sha256(parent / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V47.md')), 'exit_multiplier': 1.,
        'original_multiplier': 1.5, 'new_models_fitted': False,
        'calibration_period': ['2021-01-01', '2021-07-01'], 'diagnosis_period': ['2021-07-01', '2022-01-01']}
    for name in MODEL_FILES:
        (diagnosis / name).write_bytes((parent / name).read_bytes())
    save_json(diagnosis / 'exit_threshold.json', {'threshold': .04, 'support': support,
        'exit_multiplier': 1., 'original_multiplier': 1.5, 'minimum_predicted_positive': 20,
        'calibration_period': settings['calibration_period'], 'beta': 2.})
    save_json(diagnosis / 'manifest.json', {'settings': settings})
    save_json(diagnosis / 'metrics.json', metrics)
    save_json(diagnosis / 'decision.json', decision)
    save_json(diagnosis / 'summary.json', {'complete': True, 'diagnosis_rows': len(predictions),
        'episode_intersection': 0, 'new_models_fitted': False, **decision})
    save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir()})
    root = tmp_path / 'exit-state'
    root.mkdir()
    copy_activity_parent(parent, root)
    prepare_exit_evidence(parent, diagnosis, root)
    frozen = {**old, 'protocol': 'exit_state_v48',
        'activity_selection_sha256': sha256(root / 'activity_selection.json'),
        'exit_files_sha256': {n: sha256(root / n) for n in EXIT_POLICY_FILES},
        'exit_protocol_sha256': sha256(Path('docs/EXPERIMENT_V48.md'))}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, holding, diagnosis, frozen


@pytest.mark.parametrize('scores,old_intent,new_intent', [
    ((.6, .3, .3), 'reduce', 'exit'), ((.5, 0., 0.), 'hold', 'exit'),
    ((.49, 0., 0.), 'hold', 'hold'), ((.8, .9, .9), 'exit', 'exit'),
    ((0., .3, .3), 'reduce', 'reduce'), ((0., 0., .3), 'increase', 'increase')])
def test_only_exit_boundary_changes_with_original_priority(scores, old_intent, new_intent):
    _, parent = bots(manager=scores)
    candidate = ExitStatePolicy(parent)
    old, new = (bot(*history_inputs(1)) for bot in [parent, candidate])
    assert old.intent == old_intent and new.intent == new_intent
    if old_intent == new_intent:
        assert old == new
    assert parent.action_threshold('exit', {}) == .75
    assert candidate.action_threshold('exit', {}) == .5
    assert candidate.base is parent.base and candidate.manager is parent.manager
    assert candidate(*history_inputs(1, pending='increase')).intent == 'hold'


def test_first_repeat_history_and_new_position_reset_are_unchanged():
    _, parent = bots(manager=(0., .3, .3))
    candidate = ExitStatePolicy(parent)
    memory = {}
    for minute in range(1, 25):
        changes = {'adds': int(minute >= 3), 'fraction': .7 if minute >= 5 else 1.}
        old = parent(*history_inputs(minute, memory, **changes))
        new = candidate(*history_inputs(minute, memory, **changes))
        assert old == new
        memory = new.state
    new = candidate(*history_inputs(25, memory, direction=-1, hold_bars=1,
        position_entry_time='2021-01-01T00:24:00+00:00'))
    assert new.state['fill_increase_last'] == new.state['fill_reduce_last'] == 0


@pytest.mark.parametrize('delay,liquid', [(0, True), (1, True), (0, False), (1, False)])
def test_actual_exit_waits_for_liquidity_preserves_accounting_and_allows_reentry(delay, liquid):
    _, parent = bots(manager=(.6, 0., 0.))
    bot = ExitStatePolicy(parent)
    risk = replace(EngineConfig(), bar_seconds=60, signal_delay_bars=delay, max_hold_bars=50,
        stop_fraction=0., entry_fraction=.125, addition_fraction=.25, max_adds=5, cooldown_bars=0)
    engine = TradingEngine(risk)
    fills = []
    for i, bar in enumerate(list(iter_events(ready_bars()))[:40]):
        close = 100. - .2*i
        event = {**bar, 'open': 100., 'close': close, 'high': 100., 'low': close, 'count': 1, 'volume': 100.}
        if not liquid and i in [7+2*delay, 8+2*delay]:
            event.update(count=0, volume=0., open=close, high=close, low=close)
        result = engine.step(event, bot)
        fills.extend(result['fills'])
        assert abs(result['accounting_residual']) < 1e-7
        if event['count'] == 0:
            assert not result['fills']
        snapshot = engine.snapshot()
        engine = TradingEngine(risk, copy.deepcopy(snapshot))
        assert engine.snapshot() == snapshot
    reasons = [f['reason'] for f in fills]
    assert reasons.count('entry') >= 2 and reasons.count('signal_exit') >= 2
    assert not {'increase', 'signal_reduce', 'time_limit', 'intrabar_stop'} & set(reasons)
    first, exit_fill = fills[:2]
    assert abs(first['delta_quantity'])*first['price']+first['fee'] == pytest.approx(
        risk.initial_equity*risk.allocation*risk.entry_fraction)
    assert exit_fill['delta_quantity'] == -first['delta_quantity']
    assert pd.Timestamp(exit_fill['time']) >= pd.Timestamp(first['time'])+pd.Timedelta(minutes=1+delay)
    if not liquid:
        assert pd.Timestamp(exit_fill['time']) == pd.Timestamp(first['time'])+pd.Timedelta(minutes=3+delay)


@pytest.mark.parametrize('damage', ['parent', 'risk', 'model', 'plan', 'threshold', 'multiplier', 'admission', 'support', 'link'])
def test_loader_rejects_unplanned_changes_and_failed_diagnosis(tmp_path, monkeypatch, damage):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'parent':
        (root / 'activity_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'model':
        (root / 'history_manager.json').write_text('{}')
    elif damage == 'plan':
        frozen['exit_protocol_sha256'] = '0'*64
    else:
        name = 'exit_admission.json' if damage in ['admission', 'link'] else 'exit_threshold.json'
        value = json.loads((root / name).read_text())
        if damage == 'admission':
            value['metrics']['exit']['direct']['recall'] = 0.
        elif damage == 'link':
            value['settings']['selection_sha256'] = '0'*64
        elif damage == 'threshold':
            value['threshold'] *= .5
        elif damage == 'support':
            value['support']['predicted_positive'] = 19
        else:
            value['exit_multiplier'] = 1.5
        save_json(root / name, value)
        frozen['exit_files_sha256'][name] = sha256(root / name)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_and_training_period_guard(tmp_path, monkeypatch, kind):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_eleven_variants_preserve_both_original_controls_and_risks(tmp_path, monkeypatch):
    root, parent, holding, _, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v40', 'previous_v46', 'unfiltered_v14'} <= strategies
    assert 'previous_v39' not in strategies and isinstance(load_selection(out)[1], ExitStatePolicy)
    for label, reference in [('previous_v46', parent), ('previous_v40', holding)]:
        frozen, bot = load_selection(reference)
        expected = tmp_path / label
        backtest(ready_bars(), bot, EngineConfig(**frozen['risk']), expected)
        for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / f'BTCUSDT/{label}' / n), pd.read_parquet(expected / n), check_exact=True)
        assert json.loads((out / f'BTCUSDT/{label}/final_state.json').read_text()) == json.loads((expected / 'final_state.json').read_text())
        assert json.loads((out / f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']
    assert json.loads((out / 'BTCUSDT/unfiltered_v14/config.json').read_text())['max_hold_bars'] == 0


def test_selection_uses_frozen_diagnosis_preserves_models_and_exact_control(tmp_path, monkeypatch):
    _, parent, _, diagnosis, old = fixture(tmp_path, monkeypatch)
    _, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.exit_state_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_exit_state_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
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
