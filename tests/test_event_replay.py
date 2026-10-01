import json
from dataclasses import asdict

import pandas as pd

from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_replay import run_event_replay


def test_command_resume_does_not_liquidate_at_chunk_boundary(monkeypatch, tmp_path):
    times = pd.date_range('2020-01-01', periods=10, freq='5min', tz='UTC')
    bars = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5),
                         'open': 100, 'high': 101, 'low': 99, 'close': 100, 'funding_rate': 0.0001})
    bars[MARKET_FEATURES] = 0.0
    config = EngineConfig(stop_fraction=0, max_hold_bars=0, allow_adverse_add=True)

    def policy(event, state):
        if not state['direction']:
            return 'enter_long'
        return {1: 'increase', 2: 'reduce', 3: 'exit'}.get(state['hold_bars'], 'hold')

    monkeypatch.setattr('wonyotti_fr.event_replay.load_selection', lambda _: ({'risk': asdict(config)}, policy))
    monkeypatch.setattr('wonyotti_fr.event_replay.prepare_period', lambda *_: bars)
    (tmp_path / 'frozen_selection.json').write_text('{}')
    (tmp_path / 'manifest-5m.json').write_text('{}')
    args = (tmp_path, tmp_path, 'BTCUSDT', '2020-01-01', '2020-01-02', tmp_path / 'journal.sqlite', tmp_path / 'runs')
    first = run_event_replay(*args, max_bars=3)
    saved = json.loads((first / 'replay.json').read_text())
    assert not saved['completed'] and saved['final_state']['quantity'] > 0
    resumed = run_event_replay(*args, verify_memory=True)
    report = json.loads((resumed / 'replay.json').read_text())
    assert report['memory_parity'] and report['completed']
    assert report['processed_this_run'] == 7
    duplicate = run_event_replay(*args, verify_memory=True)
    report = json.loads((duplicate / 'replay.json').read_text())
    assert report['processed_this_run'] == 0 and report['memory_parity']
