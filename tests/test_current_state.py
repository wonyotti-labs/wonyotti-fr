import json

import numpy as np
import pandas as pd
import pytest
from test_action_model import FixedScores
from test_lifecycle import frozen_selection
from test_minute_inventory_research import ready_bars
from test_probe_entry import selection as probe_selection
from test_pullback_evaluation import ConstantBase
from test_rate_policy import inputs

from wonyotti_fr.action_research import load_action_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.current_state import CurrentStatePolicy, copy_probe_parent
from wonyotti_fr.current_state_research import run_current_state_selection
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_inventory import MinuteInventoryPolicy, MinuteReductionModel
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.pullback_policy import PullbackPolicy
from wonyotti_fr.rate_policy import RATE_STATE


@pytest.fixture(autouse=True)
def fixed_budget(monkeypatch):
    monkeypatch.setattr('test_action_replay.frozen_selection',
                        lambda root: frozen_selection(root, addition_fraction=.25))


def selection(tmp_path):
    parent, _, old = probe_selection(tmp_path)
    root = tmp_path / 'current'
    root.mkdir()
    copy_probe_parent(parent, root)
    frozen = {**old, 'protocol': 'current_state_v31',
              'probe_selection_sha256': sha256(root / 'probe_selection.json')}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, frozen


def policies(scores):
    n = len(MinuteReductionModel.features)
    size = MinuteReductionModel.from_dict({'format': MinuteReductionModel.format,
        'features': MinuteReductionModel.features, 'alpha': 100., 'mean': [0.]*n,
        'scale': [1.]*n, 'coef': [0.]*n, 'intercept': .3})
    old = MinuteInventoryPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores(scores),
        dict.fromkeys(ACTIONS, .5), 1.5, dict.fromkeys(ACTIONS, 1.), size)
    return CurrentStatePolicy(old), old


def current_inputs(minute, stored=None, pending='hold'):
    bar, state = inputs(minute, stored, pending)
    bar['minute_features'] = np.zeros(4)
    state['remaining_fraction'] = .8
    return bar, state


def test_low_scores_do_not_accumulate_and_high_current_score_acts_immediately():
    policy, rate = policies([.2, 0, 0])
    current_memory, rate_memory = {}, {}
    for minute in range(1, 6):
        current = policy(*current_inputs(minute, current_memory))
        accumulated = rate(*current_inputs(minute, rate_memory))
        current_memory, rate_memory = current.state, accumulated.state
        assert current.intent == 'hold' and not RATE_STATE & set(current.state)
    assert accumulated.intent == 'exit'
    policy, rate = policies([.75, 0, 0])
    assert policy(*current_inputs(1)).intent == 'exit'
    assert rate(*current_inputs(1)).intent == 'hold'


@pytest.mark.parametrize('scores,intent,fraction', [([.75, .9, .9], 'exit', None),
    ([0, .75, .9], 'reduce', .3), ([0, 0, .75], 'increase', None)])
def test_priority_existing_multiplier_and_learned_reduction(scores, intent, fraction):
    policy, _ = policies(scores)
    decision = policy(*current_inputs(1))
    assert decision.intent == intent and decision.reduction_fraction == fraction
    assert decision.state['path_low'] == decision.state['path_high'] == 100
    assert policy(*current_inputs(2, decision.state)).event == 'action_cooldown'


def test_pending_requests_cooldown_and_nonfinite_scores_preserve_path():
    policy, _ = policies([.9, 0, 0])
    pending = policy(*current_inputs(1, pending='increase'))
    assert pending.event == 'action_pending'
    fired = policy(*current_inputs(2, pending.state))
    assert fired.intent == 'exit'
    memory = fired.state
    for minute in [3, 4]:
        result = policy(*current_inputs(minute, memory))
        assert result.event == 'action_cooldown'
        memory = result.state
    assert policy(*current_inputs(5, memory)).intent == 'exit'
    unavailable, _ = policies([float('nan'), 0, 0])
    decision = unavailable(*current_inputs(1))
    assert decision.intent == 'hold' and decision.event == 'action_unavailable'
    assert decision.state['path_end'] == current_inputs(1)[0]['end']


def test_rate_journal_cannot_silently_migrate_to_current_policy():
    policy, rate = policies([.2, 0, 0])
    stored = rate(*current_inputs(1)).state
    with pytest.raises(ValueError, match='누적량'):
        policy(*current_inputs(2, stored))


@pytest.mark.parametrize('damage', ['parent', 'risk', 'multiplier', 'threshold', 'scale'])
def test_frozen_parent_and_all_settings_are_immutable(tmp_path, damage):
    root, _, frozen = selection(tmp_path)
    if damage == 'parent':
        (root / 'probe_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['allocation'] *= .5
    elif damage == 'multiplier':
        frozen['multiplier'] = 1. if frozen['multiplier'] == 1.5 else 1.5
    elif damage == 'threshold':
        frozen['thresholds']['exit'] *= .5
    else:
        frozen['inventory_scales']['exit'] *= .5
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_current_policy_actual_process_recovery(tmp_path, kind):
    root, _, _ = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_conditions_have_exact_parent_policies_and_original_risks(tmp_path, monkeypatch):
    root, parent, frozen = selection(tmp_path)
    _, v29 = load_selection(parent)
    old, v28 = load_action_selection(root, json.loads((root / 'boost_selection.json').read_text()))
    _, v14 = load_action_selection(root, json.loads((root / 'rate_selection.json').read_text()))
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v28', 'previous_v29', 'unfiltered_v14'} <= strategies
    assert 'matched_initial_risk' not in strategies
    assert isinstance(load_selection(out)[1], CurrentStatePolicy)
    for name, policy, risk in [('previous_v28', v28, old['risk']),
        ('previous_v29', v29, frozen['risk']), ('unfiltered_v14', v14, old['risk'])]:
        backtest(ready_bars(), policy, EngineConfig(**risk), tmp_path / name)
        for filename in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT' / name / filename),
                                          pd.read_parquet(tmp_path / name / filename), check_exact=True)
        assert json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text()) == json.loads((tmp_path / name / 'final_state.json').read_text())


def test_single_candidate_comparison_preserves_models_and_risk(tmp_path, monkeypatch):
    _, parent, _ = selection(tmp_path)
    old, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.current_state_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_current_state_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, policy = load_selection(out)
    assert len(json.loads((out / 'comparison.json').read_text())) == 3
    assert frozen['risk'] == old['risk']
    assert policy.base.direction.to_dict() == original.base.direction.to_dict()
    assert policy.base.activity.to_dict() == original.base.activity.to_dict()
    assert policy.manager.to_dict() == original.manager.to_dict()
    assert policy.size_model.to_dict() == original.size_model.to_dict()
    assert (policy.thresholds, policy.multiplier) == (original.thresholds, original.multiplier)
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
