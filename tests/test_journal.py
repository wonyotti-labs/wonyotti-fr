import sqlite3

import pytest
from test_engine import actions, bar, config

from wonyotti_fr.journal import EventJournal


def test_crash_resume_and_duplicate_events_preserve_exact_results(tmp_path):
    path = tmp_path / "events.sqlite"
    identity = {"strategy": "synthetic", "input": "fixed"}
    def crash():
        raise RuntimeError("중단 시험")
    with EventJournal(path, config(), identity) as journal:
        journal.process(bar(0), actions)
        snapshot = journal.snapshot()
        with pytest.raises(RuntimeError):
            journal.process(bar(1), actions, before_commit=crash)
        assert journal.snapshot() == snapshot
        assert len(journal.results()) == 1
    with EventJournal(path, config(), identity) as journal:
        assert journal.process(bar(0), actions)["duplicate"]
        for i in range(1, 6):
            journal.process(bar(i), actions, final=i == 5)
        state = journal.snapshot()
        results = journal.results()
    with EventJournal(tmp_path / "reference.sqlite", config(), identity) as reference:
        for i in range(6):
            reference.process(bar(i), actions, final=i == 5)
        assert reference.snapshot() == state
        assert reference.results() == results
    assert path.stat().st_mode & 0o777 == 0o600


def test_conflicting_duplicate_and_changed_identity_are_rejected(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventJournal(path, config(), {"model": "one"}) as journal:
        journal.process(bar(0), actions)
        with pytest.raises(ValueError, match="다른 시세"):
            journal.process(bar(0, price=101), actions)
    with pytest.raises(ValueError, match="지문"):
        EventJournal(path, config(), {"model": "two"})


def test_persisted_manual_halt_closes_at_next_valid_price(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventJournal(path, config(), {}) as journal:
        journal.process(bar(0), actions)
        journal.process(bar(1), actions)
        journal.halt()
    with EventJournal(path, config(), {}) as journal:
        result = journal.process(bar(2, price=90), actions)
        assert result["quantity"] == 0
        assert result["fills"][0]["reason"] == "manual_halt"
        assert result["equity"] == 9750
        assert result["halted"]


def test_modified_journal_fails_integrity_check(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventJournal(path, config(), {}) as journal:
        journal.process(bar(0), actions)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE events SET output='{}' WHERE sequence=1")
    with pytest.raises(ValueError, match="손상"):
        EventJournal(path, config(), {})
