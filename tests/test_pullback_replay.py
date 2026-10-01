import json

import pytest
from test_pullback_evaluation import bars
from test_pullback_research import selection

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.event_replay import run_event_replay
from wonyotti_fr.period_guard import guard_replay_period


def model(root):
    frozen = selection(root)
    frozen.update(evaluation_end_exclusive='2026-10-01', unseen_evaluation_available=False)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_restores_waiting_and_open_position(tmp_path, kind):
    root = tmp_path / 'selection'
    model(root)
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {'source': 'synthetic', 'kind': kind}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    assert result['waiting_at_interruption'] == (kind == 'waiting')
    assert result['position_open_at_interruption'] == (kind == 'position')


def test_v7_command_resumes_wait_and_checks_both_market_manifests(monkeypatch, tmp_path):
    root = tmp_path / 'selection'
    model(root)
    features = tmp_path / 'features'
    features.mkdir()
    (features / 'manifest-5m.json').write_text('{}')
    (tmp_path / 'manifest-1m.json').write_text('{}')
    monkeypatch.setattr('wonyotti_fr.event_replay.prepare_minute_period', lambda *_: (bars(), {}))
    args = (root, tmp_path, 'BTCUSDT', '2020-01-01', '2020-01-02', tmp_path / 'journal.sqlite', tmp_path / 'runs')
    first = run_event_replay(*args, max_bars=5, feature_market=features)
    report = json.loads((first / 'replay.json').read_text())
    assert report['final_state']['policy_state'] and not report['completed']
    (features / 'manifest-5m.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='지문'):
        run_event_replay(*args, feature_market=features)
    (features / 'manifest-5m.json').write_text('{}')
    complete = run_event_replay(*args, verify_memory=True, feature_market=features)
    report = json.loads((complete / 'replay.json').read_text())
    assert report['completed'] and report['memory_parity'] and not report['final_state']['policy_state']
    with pytest.raises(ValueError, match='별도'):
        run_event_replay(*args)


def test_v7_replay_refuses_unobserved_period_before_market_reads(tmp_path):
    root = tmp_path / 'selection'
    frozen = model(root)
    guard_replay_period(root, frozen, '2026-09-01', '2026-10-01')
    with pytest.raises(ValueError, match='관찰한'):
        guard_replay_period(root, frozen, '2026-09-01', '2026-10-02')


def test_requested_crash_state_cannot_pass_without_occurring(tmp_path):
    root = tmp_path / 'selection'
    model(root)
    short = bars().iloc[:4]
    with pytest.raises(ValueError, match='상태가 없어'):
        verify_stress(list(iter_events(short)), root, tmp_path, {}, 'waiting', True)
