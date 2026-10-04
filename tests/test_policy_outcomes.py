import copy
import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_exit_move_state import fixture
from test_first_state import fixed_budget  # noqa: F401
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot

from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.exit_move_state import ExitMovePolicy
from wonyotti_fr.lifecycle_edge import lifecycle_outcome
from wonyotti_fr.outcome_journal import OutcomeJournal
from wonyotti_fr.policy_outcomes import run_policy_outcomes, validate_outcome


@pytest.mark.parametrize('before', [True, False])
def test_real_process_exit_and_resume_keep_atomic_prefix(tmp_path, before):
    path = tmp_path/'outcomes.sqlite'
    with OutcomeJournal(path, {'source': 'fixed'}) as journal:
        journal.append(0, {'time': 0}, {'net': -1.})
    code = '''import os,sys
from pathlib import Path
from wonyotti_fr.outcome_journal import OutcomeJournal
with OutcomeJournal(Path(sys.argv[1]), {'source':'fixed'}) as journal:
 journal.append(1, {'time':1}, {'net':2.}, before_commit=(lambda:os._exit(73)) if sys.argv[2]=='True' else None)
 os._exit(73)
'''
    assert subprocess.run([sys.executable, '-c', code, str(path), str(before)], check=False).returncode == 73
    with OutcomeJournal(path, {'source': 'fixed'}) as journal:
        assert journal.count() == (1 if before else 2)
        assert journal.append(1, {'time': 1}, {'net': 2.}) is before
        journal.append(2, {'time': 2}, {'net': -3.})
        journal.verify()
        assert [journal.read(i, {'time': i}) for i in range(3)] == [{'net': -1.}, {'net': 2.}, {'net': -3.}]


def test_journal_rejects_changed_identity_input_output_and_missing_prefix(tmp_path):
    path = tmp_path/'outcomes.sqlite'
    with OutcomeJournal(path, {'source': 'fixed'}) as journal:
        journal.append(0, {'time': 0}, {'net': -1.})
        for invoke in [lambda: journal.read(0, {'time': 1}),
            lambda: journal.append(0, {'time': 0}, {'net': 1.}),
            lambda: journal.append(2, {'time': 2}, {'net': 1.})]:
            with pytest.raises(ValueError):
                invoke()
        assert journal.count() == 1
    with pytest.raises(ValueError, match='지문'):
        OutcomeJournal(path, {'source': 'changed'})
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE outcomes SET payload=? WHERE sequence=0', ('{}',))
    with pytest.raises(ValueError, match='해시'):
        OutcomeJournal(path, {'source': 'fixed'})


@pytest.mark.parametrize('direction', [1, -1])
def test_current_policy_holding_limit_keeps_cost_and_boundary_censoring(direction):
    frame = ready_bars().assign(count=1, volume=1.)
    frame[['open', 'high', 'low', 'close']] = 100.
    frame.loc[1, 'funding_rate'] = .001
    policy = ExitMovePolicy(bot((0., 0., 0.)), .02)
    risk = EngineConfig(bar_seconds=60, max_hold_bars=4, stop_fraction=0., max_drawdown=1., daily_loss_limit=1.)
    events = list(iter_events(frame))
    cutoff = pd.Timestamp('2020-01-02', tz='UTC')
    outcome, fills = lifecycle_outcome(events, 0, direction, risk, policy, cutoff)
    opportunity = {'decision_time': frame.time.iloc[0], 'order_direction': direction}
    validate_outcome(opportunity, {'outcome': outcome, 'fills': fills}, cutoff)
    assert outcome['exit_reason'] == 'time_limit' and outcome['hold_minutes'] == 4
    assert outcome['fees'] > 0 and outcome['funding_cost']*direction > 0
    cutoff = frame.end.iloc[2]
    outcome, fills = lifecycle_outcome(events, 0, direction, replace(risk, max_hold_bars=50), policy, cutoff)
    assert outcome['label_status'] == 'right_censored' and outcome['remaining_quantity']*direction > 0
    validate_outcome(opportunity, {'outcome': outcome, 'fills': fills}, cutoff)


def test_saved_outcome_rejects_changed_net_boundary_and_quantity():
    frame = ready_bars().assign(count=1, volume=1.)
    policy = ExitMovePolicy(bot((.8, 0., 0.)), .02)
    cutoff = pd.Timestamp('2020-01-02', tz='UTC')
    outcome, fills = lifecycle_outcome(list(iter_events(frame)), 0, 1,
        EngineConfig(bar_seconds=60, max_hold_bars=10), policy, cutoff)
    opportunity = {'decision_time': frame.time.iloc[0], 'order_direction': 1}
    for field, value in [('net_pnl', 100.), ('label_end', cutoff), ('fees', 0.), ('fills', 99)]:
        damaged = copy.deepcopy(outcome)
        damaged[field] = value
        with pytest.raises(ValueError):
            validate_outcome(opportunity, {'outcome': damaged, 'fills': fills}, cutoff)


@pytest.mark.parametrize('direction', [1, -1])
def test_current_fill_history_keeps_wait_add_reduce_funding_and_natural_exit(direction):
    frame = ready_bars().assign(count=1, volume=1.)
    frame[['open', 'high', 'low', 'close']] = 100.
    frame.loc[:1, ['count', 'volume']] = 0
    frame.loc[5, 'funding_rate'] = .001
    policy = ExitMovePolicy(bot(), .02)
    def scores(values):
        return np.array([[0., 0., .8] if row[-6] == 0 else
                         ([0., .8, 0.] if row[-3] == 0 else [.8, 0., 0.]) for row in values])
    policy.manager.probabilities = scores
    risk = EngineConfig(bar_seconds=60, max_hold_bars=50, stop_fraction=0., allow_adverse_add=True,
        max_adds=5, max_drawdown=1., daily_loss_limit=1., entry_fraction=.125, addition_fraction=.25)
    cutoff = pd.Timestamp('2020-01-02', tz='UTC')
    outcome, fills = lifecycle_outcome(list(iter_events(frame)), 0, direction, risk, policy, cutoff)
    assert [f['reason'] for f in fills] == ['entry', 'increase', 'signal_reduce', 'signal_exit']
    assert pd.Timestamp(fills[0]['time']) == frame.time.iloc[2]
    assert outcome['funding_cost']*direction > 0 and outcome['adds'] == 1
    validate_outcome({'decision_time': frame.time.iloc[0], 'order_direction': direction},
        {'outcome': outcome, 'fills': fills}, cutoff)
    frame.loc[frame.time.ge(outcome['label_end']), ['open', 'high', 'low', 'close']] *= 3
    assert lifecycle_outcome(list(iter_events(frame)), 0, direction, risk, policy, cutoff) == (outcome, fills)


def test_all_labels_equal_resumed_prefix_and_changed_source_is_rejected(tmp_path, monkeypatch):
    root, _, _, _, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    frame = ready_bars().assign(count=1, volume=100.)
    frame[['time', 'end']] += pd.Timedelta(days=366)
    monkeypatch.setattr('wonyotti_fr.policy_outcomes.prepare_minute_period', lambda *_, **__: (frame.copy(), {}))
    out = run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'runs', max_opportunities=2)
    initial = json.loads((out/'summary.json').read_text())
    assert not initial['complete'] and initial['processed'] == 2
    run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'unused', resume=out)
    fresh = run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'fresh')
    for name in ['potential_entries.parquet', 'opportunity_ledger.parquet', 'training_labels.parquet', 'label_intervals.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/name), pd.read_parquet(fresh/name), check_exact=True)
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['processed'] == summary['opportunities']
    assert summary['losing_labels'] > 0 and not summary['losing_labels_removed']
    with pytest.raises(ValueError, match='완료'):
        run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'unused', resume=out)
    partial = run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'partial', max_opportunities=1)
    (tmp_path/'manifest-1m.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='지문|변경'):
        run_policy_outcomes(root, tmp_path, tmp_path, tmp_path/'unused', resume=partial)
    for name, checksum in json.loads((fresh/'files.json').read_text()).items():
        from wonyotti_fr.common import sha256
        assert sha256(Path(fresh)/name) == checksum
