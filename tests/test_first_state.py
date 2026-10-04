import json
from pathlib import Path

import pandas as pd
import pytest
from test_first_management import SyntheticScores, rows
from test_history_state import history_inputs
from test_history_state import policy as history_policy
from test_history_state import selection as history_selection
from test_lifecycle import frozen_selection
from test_minute_inventory_research import ready_bars

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.first_management import first_management_diagnosis
from wonyotti_fr.first_state import FIRST_POLICY_FILES, FirstStatePolicy, copy_history_parent
from wonyotti_fr.first_state_research import run_first_state_selection
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


@pytest.fixture(autouse=True)
def fixed_budget(monkeypatch):
    monkeypatch.setattr('test_action_replay.frozen_selection',
                        lambda root: frozen_selection(root, addition_fraction=.25))
    from test_action_replay import selection as action_selection
    def selected(*args, **kwargs):
        value = action_selection(*args, **kwargs)
        value['multiplier'] = 1.5
        return value
    monkeypatch.setattr('test_path_management.selection', selected)


def admission(parent):
    *_, metrics, decision = first_management_diagnosis(*rows(), SyntheticScores(),
        dict.fromkeys(ACTIONS, .5), 1.5, first_multiplier=1.)
    return {'metrics': metrics, 'decision': decision,
        'summary': {'complete': True, **decision},
        'settings': {'protocol_sha256': sha256(Path('docs/EXPERIMENT_V38.md')),
            'first_multiplier': 1., 'new_models_fitted': False,
            'selection_sha256': sha256(parent / 'frozen_selection.json')}}


def thresholds():
    return {'thresholds': {'reduce': .02, 'increase': .02}, 'multiplier': 1.5, 'first_multiplier': 1.,
        'betas': {'reduce': 1., 'increase': .5}, 'calibration_period': ['2021-01-01', '2021-07-01'],
        'minimum_predicted_positive': 20}


def selection(tmp_path):
    parent, _, old = history_selection(tmp_path)
    root = tmp_path / 'first'
    root.mkdir()
    copy_history_parent(parent, root)
    save_json(root / 'first_policy_thresholds.json', thresholds())
    save_json(root / 'first_admission.json', admission(parent))
    frozen = {**old, 'protocol': 'first_state_v39',
        'history_selection_sha256': sha256(root / 'history_selection.json'),
        'first_files_sha256': {n: sha256(root / n) for n in FIRST_POLICY_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, frozen


@pytest.mark.parametrize('action,scores', [('increase', [0, 0, .3]), ('reduce', [0, .3, 0])])
def test_only_actual_first_fill_switches_threshold_and_recent_window_does_not_reset(action, scores):
    parent = history_policy(scores)
    bot = FirstStatePolicy(parent, {'increase': .2, 'reduce': .2})
    stored = bot(*history_inputs(1)).state
    assert parent(*history_inputs(1)).intent == 'hold'
    assert bot(*history_inputs(1)).intent == action
    for minute in range(2, 5):
        result = bot(*history_inputs(minute, stored))
        stored = result.state
    assert result.intent == action
    adds, fraction = (1, 1.) if action == 'increase' else (0, .7)
    for minute in range(5, 24):
        result = bot(*history_inputs(minute, stored, adds=adds, fraction=fraction))
        assert result.intent == 'hold'
        stored = result.state
    assert stored['fill_' + action + '_mask'] == 0
    assert stored['fill_' + action + '_last'] != 0
    assert bot(*history_inputs(24, stored, direction=-1, hold_bars=1,
        position_entry_time='2021-01-01T00:23:00+00:00')).intent == action


def test_first_policy_preserves_exit_priority_size_and_pending_requests():
    parent = history_policy([.8, .3, .3])
    bot = FirstStatePolicy(parent, {'increase': .2, 'reduce': .2})
    assert bot(*history_inputs(1)).intent == 'exit'
    waiting = bot(*history_inputs(1, pending='increase'))
    assert waiting.intent == 'hold' and waiting.state['fill_increase_last'] == 0
    bot = FirstStatePolicy(history_policy([0, .3, .3]), {'increase': .2, 'reduce': .2})
    decision = bot(*history_inputs(1))
    assert decision.intent == 'reduce' and decision.reduction_fraction == .3


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_first_policy_actual_process_exit_recovery(tmp_path, kind):
    root, _, _ = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


@pytest.mark.parametrize('damage', ['file', 'risk', 'admission', 'threshold', 'parent'])
def test_first_policy_rejects_tampered_models_risk_and_diagnosis(tmp_path, damage):
    root, _, frozen = selection(tmp_path)
    if damage == 'file':
        (root / 'first_policy_thresholds.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] = .5
    elif damage == 'parent':
        (root / 'history_selection.json').write_text('{}')
    else:
        name = 'first_admission.json' if damage == 'admission' else 'first_policy_thresholds.json'
        value = json.loads((root / name).read_text())
        if damage == 'admission':
            value['metrics']['increase']['first']['separate']['recall'] = 0.
        else:
            value['first_multiplier'] = 1.5
        save_json(root / name, value)
        frozen['first_files_sha256'][name] = sha256(root / name)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


def test_eleven_conditions_preserve_previous_history_policy_exact(tmp_path, monkeypatch):
    root, parent, _ = selection(tmp_path)
    old, original = load_selection(parent)
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v31', 'previous_v36', 'unfiltered_v14'} <= strategies
    assert isinstance(load_selection(out)[1], FirstStatePolicy)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), tmp_path / 'original')
    for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT/previous_v36' / n),
                                      pd.read_parquet(tmp_path / 'original' / n), check_exact=True)
    assert json.loads((out / 'BTCUSDT/previous_v36/final_state.json').read_text()) == json.loads((tmp_path / 'original/final_state.json').read_text())


def test_selection_pipeline_copies_every_model_and_preserves_original_control(tmp_path, monkeypatch):
    _, parent, _ = selection(tmp_path)
    old, original = load_selection(parent)
    diagnosis = tmp_path / 'diagnosis'
    diagnosis.mkdir()
    evidence = admission(parent)
    for key in ['summary', 'decision', 'metrics']:
        save_json(diagnosis / f'{key}.json', evidence[key])
    save_json(diagnosis / 'manifest.json', {'settings': evidence['settings']})
    save_json(diagnosis / 'first_thresholds.json', thresholds())
    for n in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
        (diagnosis / n).write_bytes((parent / n).read_bytes())
    save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir() if p.is_file()})
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.first_state_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_first_state_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, policy = load_selection(out)
    assert len(json.loads((out / 'comparison.json').read_text())) == 3
    assert frozen['risk'] == old['risk'] and policy.multiplier == original.multiplier
    assert policy.thresholds == original.thresholds and policy.scales == original.scales
    assert policy.manager.model.to_dict() == original.manager.model.to_dict()
    assert policy.manager.offset.to_dict() == original.manager.offset.to_dict()
    assert policy.size_model.to_dict() == original.size_model.to_dict()
    assert policy.base.activity.to_dict() == original.base.activity.to_dict()
    assert policy.base.direction.to_dict() == original.base.direction.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']


@pytest.mark.parametrize('delay,max_adds', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_first_phase_changes_only_after_engine_fill_with_rejections_and_delay(delay, max_adds):
    from dataclasses import replace

    from wonyotti_fr.engine import PolicyDecision, TradingEngine
    bot = FirstStatePolicy(history_policy([0, 0, .3]), {'reduce': .2, 'increase': .2})
    config = replace(EngineConfig(), bar_seconds=60, max_hold_bars=0, stop_fraction=0.,
        signal_delay_bars=delay, max_adds=max_adds, entry_fraction=.125, addition_fraction=.25,
        allow_adverse_add=True)
    engine = TradingEngine(config)
    fills, rejected, requests = [], [], []
    for i, event in enumerate(list(iter_events(ready_bars()))[:20]):
        event = {**event, 'open': 100., 'close': 100., 'high': 100., 'low': 100.}
        if i in [2, 3, 6, 7]:
            event.update(count=0, volume=0.)
        def decide(bar, state, i=i):
            decision = bot(bar, state)
            if decision.intent == 'increase':
                requests.append(decision.state['fill_increase_last'])
            return PolicyDecision('enter_long', decision.state, 'synthetic_entry') if i == 0 else decision
        result = engine.step(event, decide)
        fills.extend(result['fills'])
        rejected.extend(result['rejected'])
        memory = engine.snapshot()['policy_state']
        if 'fill_increase_last' in memory:
            actual = [r for r in fills if r['reason'] == 'increase']
            assert bool(memory['fill_increase_last']) == bool(actual)
    assert 'market_no_trades' in rejected
    assert requests and set(requests) == {0}
    assert sum(r['reason'] == 'increase' for r in fills) == max_adds
    if not max_adds:
        assert 'max_adds' in rejected and len(requests) > 1
