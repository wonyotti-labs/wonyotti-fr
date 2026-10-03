import json

import pytest
from test_lifecycle import frozen_selection
from test_pullback_evaluation import bars

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.event_replay import run_event_replay
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_management import ACTIONS, FEATURES
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import evaluation_period


def selection(root):
    frozen = frozen_selection(root)
    payload = {'format': 'minute_action_v1', 'features': FEATURES, 'actions': ACTIONS, 'kind': 'logistic',
               'mean': [0.]*32, 'scale': [1.]*32, 'coef': [[0.]*32 for _ in ACTIONS], 'intercept': [-10., -10., 10.]}
    save_json(root/'action_model.json', payload)
    frozen.update(protocol='minute_action_v10', kind='logistic', multiplier=1.,
                  thresholds={'exit': .9, 'reduce': .9, 'increase': .1},
                  model_sha256={'action_model.json': sha256(root/'action_model.json')},
                  training_period=['2018-03-01', '2020-01-01'], calibration_period=['2020-01-01', '2021-01-01'])
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_action_policy_actual_crash_and_recovery(tmp_path, kind):
    root = tmp_path/'selection'
    selection(root)
    report = verify_stress(list(iter_events(bars())), root, tmp_path, {'kind': kind}, kind, True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73


def test_action_policy_journal_management_state_and_tamper(monkeypatch, tmp_path):
    root = tmp_path/'selection'
    frozen = selection(root)
    assert evaluation_period(frozen, 'observed') == ('2023-01-01', '2026-01-01')
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2026-10-01', '2026-11-01')
    feature = tmp_path/'features'
    feature.mkdir()
    (feature/'manifest-5m.json').write_text('{}')
    (tmp_path/'manifest-1m.json').write_text('{}')
    monkeypatch.setattr('wonyotti_fr.event_replay.prepare_minute_period', lambda *_: (bars(), {}))
    args = (root, tmp_path, 'BTCUSDT', '2021-01-01', '2021-01-02', tmp_path/'journal.sqlite', tmp_path/'runs')
    first = run_event_replay(*args, max_bars=8, feature_market=feature)
    state = json.loads((first/'replay.json').read_text())['final_state']
    assert state['quantity'] != 0 and 'management_after' in state['policy_state']
    final = run_event_replay(*args, feature_market=feature, verify_memory=True)
    assert json.loads((final/'replay.json').read_text())['memory_parity']
    (root/'action_model.json').write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        load_selection(root)
