import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_action_model import FixedScores
from test_path_management import path_selection
from test_pullback_evaluation import ConstantBase, bars

from wonyotti_fr.action_research import action_diagnostics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.path_management import PathActionModels
from wonyotti_fr.pullback_policy import PullbackPolicy
from wonyotti_fr.rate_policy import RateActionPolicy, calibrate_rates


def policy(scores):
    return RateActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores(scores),
                            dict.fromkeys(ACTIONS, .9), 1.5, dict.fromkeys(ACTIONS, 1.))


def inputs(minute, stored=None, pending='hold'):
    end = pd.Timestamp('2021-01-01', tz='UTC') + pd.Timedelta(minutes=minute)
    bar = {'end': end.isoformat(), 'close': 100., 'features': np.zeros(14)}
    state = {'direction': 1, 'halted': False, 'policy_state': stored or {}, 'pending': pending,
             'bar_seconds': 60, 'hold_bars': minute, 'adds': 0, 'favorable_move': 0., 'average_entry': 100.,
             'position_entry_time': '2021-01-01T00:00:00+00:00'}
    return bar, state


def rate_selection(root):
    frozen = path_selection(root)
    (root / 'path_selection.json').write_bytes((root / 'frozen_selection.json').read_bytes())
    scales = dict.fromkeys(ACTIONS, 1.)
    save_json(root / 'rate_calibration.json', {'scales': scales})
    frozen.update(protocol='minute_rate_v14', candidate=0, rate_scales=scales,
                  rate_calibration_sha256=sha256(root / 'rate_calibration.json'),
                  path_selection_sha256=sha256(root / 'path_selection.json'))
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_rate_accumulation_priority_and_cooldown_keeps_other_actions():
    bot = policy([0., .25, .25])
    stored = {}
    for minute in range(1, 8):
        decision = bot(*inputs(minute, stored))
        stored = decision.state
        if minute < 4:
            assert decision.intent == 'hold'
            assert stored['rate_reduce'] == stored['rate_increase'] == minute * .25
        elif minute == 4:
            assert decision.intent == 'reduce' and stored['rate_reduce'] == 0 and stored['rate_increase'] == 1
        elif minute < 7:
            assert decision.intent == 'hold' and decision.event == 'action_rate_cooldown'
            assert stored['rate_increase'] == 1
        else:
            assert decision.intent == 'increase' and stored['rate_reduce'] == .75 and stored['rate_increase'] == 0
    assert len(stored) <= 16


def test_pending_intent_accumulates_without_duplicates_and_flat_resets():
    bot = policy([.2, 0, 0])
    stored = {}
    for minute in range(1, 6):
        decision = bot(*inputs(minute, stored, 'increase' if minute < 5 else 'hold'))
        stored = decision.state
        assert decision.intent == ('exit' if minute == 5 else 'hold')
    bar, state = inputs(6, stored)
    flat = bot(bar, {**state, 'direction': 0})
    assert flat.state == {}
    first_bar, first_state = inputs(7)
    first_state.update(hold_bars=1, position_entry_time='2021-01-01T00:06:00+00:00')
    first = bot(first_bar, first_state)
    assert first.state['rate_exit'] == .2 and first.intent == 'hold'


@pytest.mark.parametrize('damage', ['missing_field', 'invalid_number', 'position_mismatch', 'all_rates_missing'])
def test_rate_state_corruption_is_rejected(damage):
    bot = policy([.2, .1, .1])
    stored = copy.deepcopy(bot(*inputs(1)).state)
    if damage == 'missing_field':
        del stored['rate_exit']
    elif damage == 'invalid_number':
        stored['rate_exit'] = 1.1
    elif damage == 'position_mismatch':
        stored['rate_entry_time'] = '2021-01-01T00:00:01+00:00'
    else:
        stored = {k: v for k, v in stored.items() if not k.startswith('rate_')}
    with pytest.raises(ValueError):
        bot(*inputs(2, stored))


def test_rate_calibration_matches_event_mass_and_rejects_future_labels():
    frame = pd.DataFrame(np.zeros((100, 35)), columns=PathActionModels.features)
    frame['end'] = pd.date_range('2020-07-04', periods=100, freq='min', tz='UTC')
    frame['label_end'] = frame.end + pd.Timedelta(minutes=1)
    for a in ACTIONS:
        frame[f'y_{a}'] = np.tile([1, 0, 0, 0, 0], 20)
    model = FixedScores([.1, .2, .4])
    model.features = PathActionModels.features
    report = calibrate_rates(model, frame)
    np.testing.assert_allclose([report['scales'][a] for a in ACTIONS], [2, 1, .5])
    frame.loc[99, 'label_end'] = pd.Timestamp('2021-01-01', tz='UTC')
    with pytest.raises(ValueError, match='시간'):
        calibrate_rates(model, frame)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_rate_state_actual_crash_recovery_and_frozen_scale_integrity(tmp_path, kind):
    root = tmp_path / 'selection'
    rate_selection(root)
    assert isinstance(load_selection(root)[1], RateActionPolicy)
    report = verify_stress(list(iter_events(bars())), root, tmp_path, {}, kind, True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73
    (root / 'rate_calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        load_selection(root)


@pytest.mark.parametrize('delay', [0, 1])
def test_rate_engine_partial_management_accounting_and_final_reset(tmp_path, delay):
    frame = bars()
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, signal_delay_bars=delay)
    backtest(frame, policy([.01, .2, .1]), config, tmp_path / 'run')
    report = action_diagnostics(tmp_path / 'run', frame, config)
    assert report['waiting']['entry_fills'] > 0
    state = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert state['quantity'] == 0 and state['policy_state'] == {}
