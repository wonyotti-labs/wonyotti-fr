import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_action_model import FixedScores
from test_current_state import current_inputs, policies
from test_current_state import selection as current_selection
from test_lifecycle import frozen_selection
from test_minute_inventory_research import ready_bars

from wonyotti_fr.calibration_diagnostics import PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, PolicyDecision, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.history_calibration_diagnostics import history_calibration_admission
from wonyotti_fr.history_state import (
    FILL_STATE,
    HISTORY_POLICY_FILES,
    CalibratedHistoryModels,
    HistoryStatePolicy,
    copy_current_parent,
    fill_context,
)
from wonyotti_fr.management_calibration import ManagementOffset
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.order_history_boost import OrderHistoryBoostModels
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


@pytest.fixture(autouse=True)
def fixed_budget(monkeypatch):
    monkeypatch.setattr('test_action_replay.frozen_selection',
                        lambda root: frozen_selection(root, addition_fraction=.25))


def manager():
    tree = {'left': [-1], 'right': [-1], 'feature': [-2], 'threshold': [0.], 'value': [0.]}
    model = OrderHistoryBoostModels.from_dict({'format': OrderHistoryBoostModels.format,
        'features': OrderHistoryBoostModels.features, 'settings': HISTOGRAM_SETTINGS,
        'models': [{'action': a, 'baseline': -3., 'trees': [tree]*64} for a in ACTIONS]})
    offset = ManagementOffset.from_dict({'format': ManagementOffset.format, 'actions': ACTIONS,
        'slope': 1., 'epsilon': float(np.finfo(float).eps), 'bounds': [-64., 64.], 'iterations': 100,
        'offsets': [-.1, -.2, -.3]})
    return CalibratedHistoryModels(model, offset)


def admission():
    metrics = {a: {k: {'log_loss': loss, 'average_precision': ap, 'rows': 300, 'positive': 50, 'negative': 250} for k, loss, ap in [
        ('original_v21', .6, .4), ('logistic_calibrated', .55, .5), ('previous_calibrated', .56, .5),
        ('histogram_calibrated', .5, .6), ('histogram_raw', .51, .6), ('constant', .69, .1)]} for a in ACTIONS}
    decision = history_calibration_admission(metrics)
    return {'metrics': metrics, 'decision': decision,
        'summary': {'complete': True, 'episode_intersection': 0, **decision},
        'settings': {'periods': {k: list(v) for k, v in PERIODS.items()},
            'protocol_sha256': sha256(Path('docs/EXPERIMENT_V35.md')), 'history_files_sha256': 'synthetic'}}


def selection(tmp_path):
    parent, _, old = current_selection(tmp_path)
    root = tmp_path / 'history'
    root.mkdir()
    copy_current_parent(parent, root)
    model = manager()
    save_json(root / 'history_manager.json', model.model.to_dict())
    save_json(root / 'history_offset.json', model.offset.to_dict())
    save_json(root / 'history_admission.json', admission())
    save_json(root / 'history_thresholds.json', {'calibration_period': list(PERIODS['calibration']),
        'betas': [2., 1., .5], 'model_refitted': False, 'profit_selected': False,
        'minimum_predicted_positive': 20, 'thresholds': dict.fromkeys(ACTIONS, .5)})
    frozen = {**old, 'protocol': 'history_state_v36',
        'current_selection_sha256': sha256(root / 'current_selection.json'),
        'history_files_sha256': {n: sha256(root / n) for n in HISTORY_POLICY_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, frozen


def history_inputs(minute, stored=None, adds=0, fraction=1., **changes):
    bar, state = current_inputs(minute, stored)
    state.update(adds=adds, remaining_fraction=fraction, **changes)
    return bar, state


def policy(scores=(0., 0., 0.)):
    parent, _ = policies(scores)
    return HistoryStatePolicy(parent, FixedScores(scores), dict.fromkeys(ACTIONS, .5))


def test_history_uses_executed_quantity_and_includes_fifteen_minute_boundary():
    bot = policy()
    stored = bot(*history_inputs(1)).state
    stored = bot(*history_inputs(2, stored, adds=1)).state
    stored = bot(*history_inputs(3, stored, adds=1, fraction=.7)).state
    for minute in range(4, 18):
        bar, state = history_inputs(minute, stored, adds=1, fraction=.7)
        _, _, features = fill_context(bar, state)
        np.testing.assert_allclose(features, [1, np.log1p(minute-1), np.log1p(int(minute<=16)),
            1, np.log1p(minute-2), np.log1p(int(minute<=17))], atol=1e-14, rtol=0)
        stored = bot(bar, state).state
    assert stored['fill_increase_mask'] == 0 and stored['fill_reduce_mask'] == 2**14


@pytest.mark.parametrize('scores,action,fraction', [([.8, .9, .9], 'exit', None),
    ([0, .8, .9], 'reduce', .3), ([0, 0, .8], 'increase', None)])
def test_history_policy_preserves_priority_multiplier_size_and_cooldown(scores, action, fraction):
    bot = policy(scores)
    first = bot(*history_inputs(1))
    assert first.intent == action and first.reduction_fraction == fraction
    second = bot(*history_inputs(2, first.state))
    assert second.event == 'action_cooldown' and second.state['fill_increase_last'] == 0
    assert len(second.state) == 13


def test_history_resets_on_reverse_and_cleared_position():
    bot = policy()
    stored = bot(*history_inputs(1)).state
    stored = bot(*history_inputs(2, stored, adds=1)).state
    reversed_state = bot(*history_inputs(3, stored, direction=-1, hold_bars=1,
        position_entry_time='2021-01-01T00:02:00+00:00')).state
    assert reversed_state['fill_adds'] == reversed_state['fill_increase_last'] == 0
    flat = bot(*history_inputs(4, reversed_state, direction=0, hold_bars=0,
                              fraction=0., position_entry_time=None))
    assert not FILL_STATE & set(flat.state)


@pytest.mark.parametrize('damage', ['missing', 'old_path', 'future', 'mask', 'overlap', 'adds', 'quantity', 'duplicate'])
def test_history_rejects_corrupt_or_incompatible_journal(damage):
    bot = policy()
    stored = bot(*history_inputs(1)).state
    stored = bot(*history_inputs(2, stored, adds=1)).state
    bar, state = history_inputs(3, stored, adds=1)
    if damage == 'missing':
        del stored['fill_reduce_last']
    elif damage == 'old_path':
        for k in FILL_STATE:
            del stored[k]
    elif damage == 'future':
        stored['fill_increase_last'] += 10
    elif damage == 'mask':
        stored['fill_increase_mask'] = 2
    elif damage == 'overlap':
        stored['fill_reduce_mask'] = stored['fill_increase_mask']
        stored['fill_reduce_last'] = stored['fill_increase_last']
    elif damage == 'adds':
        state['adds'] = 3
    elif damage == 'quantity':
        stored['fill_fraction'] = .5
    else:
        bar['end'] = stored['path_end']
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        bot(bar, state)
    assert state == before


@pytest.mark.parametrize('delay', [0, 1])
def test_engine_fills_rejections_no_trade_delay_and_recovery_match_independent_history(delay):
    bot = policy()
    config = replace(EngineConfig(), bar_seconds=60, signal_delay_bars=delay, max_adds=1,
        max_hold_bars=0, stop_fraction=0, allow_adverse_add=True, entry_fraction=.125, addition_fraction=.25)
    engine = TradingEngine(config)
    events = list(iter_events(ready_bars()))[:40]
    requests = {0: 'enter_long', 4: 'increase', 9: 'increase', 12: 'reduce', 20: 'enter_short', 24: 'increase'}
    fills = []
    rejected = []
    for i, event in enumerate(events):
        event = {**event, 'open': 100., 'close': 100., 'high': 100., 'low': 100.}
        if i in [5, 6, 13, 14]:
            event.update(count=0, volume=0.)
        def choose(bar, state, i=i):
            decision = bot(bar, state)
            return PolicyDecision(requests.get(i, 'hold'), decision.state, 'synthetic', .3 if i == 12 else None)
        result = engine.step(event, choose)
        fills.extend(result['fills'])
        rejected.extend(result['rejected'])
        memory = engine.snapshot()['policy_state']
        if FILL_STATE <= set(memory):
            entry, end = pd.Timestamp(memory['path_entry_time']), pd.Timestamp(event['end'])
            for action, reason in [('increase', 'increase'), ('reduce', 'signal_reduce')]:
                past = [pd.Timestamp(f['time']) for f in fills if f['reason'] == reason and entry < pd.Timestamp(f['time']) < end]
                expected_last = int(past[-1].value // pd.Timedelta(minutes=1).value) if past else 0
                assert memory[f'fill_{action}_last'] == expected_last
                recent = sum(end-pd.Timedelta(minutes=15) <= t for t in past)
                assert memory[f'fill_{action}_mask'].bit_count() == recent
        resumed = TradingEngine(config, engine.snapshot())
        assert resumed.snapshot() == engine.snapshot()
        engine = resumed
    assert 'max_adds' in rejected and 'market_no_trades' in rejected
    assert {'increase', 'signal_reduce', 'signal_reverse'} <= {f['reason'] for f in fills}
    snapshot = engine.snapshot()
    with pytest.raises(ValueError):
        engine.step(events[-1], bot)
    assert engine.snapshot() == snapshot
    event = {**events[-1], 'time': snapshot['last_end'],
        'end': (pd.Timestamp(snapshot['last_end'])+pd.Timedelta(minutes=1)).isoformat()}
    def broken(bar, state):
        bot(bar, state)
        raise ValueError('synthetic rollback')
    with pytest.raises(ValueError, match='rollback'):
        engine.step(event, broken)
    assert engine.snapshot() == snapshot


def test_calibrated_single_row_and_batch_prediction_match_and_nonfinite_stays_unavailable():
    model = manager()
    x = np.zeros((3, 50))
    np.testing.assert_array_equal(model.probabilities(x), np.vstack([model.probabilities(v[None, :]) for v in x]))
    x[0, 0] = np.nan
    assert np.isnan(model.probabilities(x)[0]).all()


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_history_actual_process_exit_recovery(tmp_path, kind):
    root, _, _ = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


@pytest.mark.parametrize('damage', ['file', 'risk', 'parent', 'admission', 'threshold'])
def test_history_selection_rejects_changed_frozen_inputs(tmp_path, damage):
    root, _, frozen = selection(tmp_path)
    if damage == 'file':
        (root / 'history_manager.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['allocation'] = .25
    elif damage == 'parent':
        (root / 'current_selection.json').write_text('{}')
    elif damage == 'admission':
        value = admission()
        value['metrics']['reduce']['histogram_calibrated']['log_loss'] = .56
        save_json(root / 'history_admission.json', value)
        frozen['history_files_sha256']['history_admission.json'] = sha256(root / 'history_admission.json')
    else:
        path = root / 'history_thresholds.json'
        value = json.loads(path.read_text())
        value['calibration_period'] = list(PERIODS['diagnosis'])
        save_json(path, value)
        frozen['history_files_sha256'][path.name] = sha256(path)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


def test_eleven_conditions_preserve_v31_v29_v14_and_original_risks(tmp_path, monkeypatch):
    root, parent, frozen = selection(tmp_path)
    previous, v31 = load_selection(parent)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v31', 'previous_v29', 'unfiltered_v14'} <= strategies
    assert isinstance(load_selection(out)[1], HistoryStatePolicy)
    backtest(ready_bars(), v31, EngineConfig(**previous['risk']), tmp_path / 'original')
    for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT/previous_v31' / name),
                                      pd.read_parquet(tmp_path / 'original' / name), check_exact=True)
    assert json.loads((out / 'BTCUSDT/previous_v31/final_state.json').read_text()) == json.loads((tmp_path / 'original/final_state.json').read_text())
    guard_replay_period(root, frozen, '2021-01-01', '2022-01-01')
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2026-10-01', '2026-11-01')

def diagnosis_fixture(root, parent):
    from test_calibration_diagnostics import source

    from wonyotti_fr.minute_management import purged_window
    from wonyotti_fr.order_history import HISTORY_FEATURES
    root.mkdir()
    evidence = admission()
    evidence['settings']['selection_sha256'] = parent['minute_selection_sha256']
    save_json(root / 'manifest.json', {'settings': evidence['settings']})
    for name in ['summary', 'decision', 'metrics']:
        save_json(root / f'{name}.json', evidence[name])
    model = manager()
    save_json(root / 'models.json', {'histogram': model.model.to_dict()})
    save_json(root / 'offsets.json', {'histogram': model.offset.to_dict()})
    frame = source()
    frame[HISTORY_FEATURES] = 0.
    for name, period in PERIODS.items():
        purged_window(frame, *period).to_parquet(root / f'{name}_used.parquet', index=False)
    save_json(root / 'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file()})
    return root


def test_thresholds_use_only_calibration_and_preserve_frozen_models(tmp_path):
    from wonyotti_fr.history_state_research import prepare_history_manager
    parent = {'minute_selection_sha256': 'synthetic'}
    diagnosis = diagnosis_fixture(tmp_path / 'diagnosis', parent)
    out, changed = tmp_path / 'out', tmp_path / 'changed'
    out.mkdir()
    changed.mkdir()
    prepare_history_manager(diagnosis, parent, out)
    validation = pd.read_parquet(diagnosis / 'diagnosis_used.parquet')
    for a in ACTIONS:
        validation['y_' + a] = 1-validation['y_' + a]
    validation.to_parquet(diagnosis / 'diagnosis_used.parquet', index=False)
    hashes = json.loads((diagnosis / 'files.json').read_text())
    hashes['diagnosis_used.parquet'] = sha256(diagnosis / 'diagnosis_used.parquet')
    save_json(diagnosis / 'files.json', hashes)
    prepare_history_manager(diagnosis, parent, changed)
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        assert (out / name).read_bytes() == (changed / name).read_bytes()
    thresholds = json.loads((out / 'history_thresholds.json').read_text())
    assert all(v['predicted_positive'] >= 20 for v in thresholds['support'].values())
    assert thresholds['thresholds'] == dict(zip(ACTIONS, manager().probabilities(np.zeros((1, 50)))[0], strict=True))


def test_selection_pipeline_preserves_original_full_outputs_and_all_entry_risk(tmp_path, monkeypatch):
    from wonyotti_fr.history_state_research import run_history_state_selection
    parent, _, old = current_selection(tmp_path)
    _, original = load_selection(parent)
    diagnosis = diagnosis_fixture(tmp_path / 'diagnosis', old)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.history_state_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_history_state_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, bot = load_selection(out)
    assert len(json.loads((out / 'comparison.json').read_text())) == 3
    assert frozen['risk'] == old['risk'] and bot.multiplier == original.multiplier
    assert bot.size_model.to_dict() == original.size_model.to_dict()
    assert bot.base.activity.to_dict() == original.base.activity.to_dict()
    assert bot.base.direction.to_dict() == original.base.direction.to_dict()
    assert bot.manager.model.to_dict() == manager().model.to_dict()
    assert bot.manager.offset.to_dict() == manager().offset.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
