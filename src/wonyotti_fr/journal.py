from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .common import json_default
from .engine import EngineConfig, TradingEngine, validate_bar


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=json_default)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class EventJournal:
    def __init__(self, path: Path, config: EngineConfig, identity: dict):
        config.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("저널 경로에 심볼릭 링크를 허용하지 않습니다.")
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            path.chmod(0o600)
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA trusted_schema=OFF")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS events (
            sequence INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL, payload_hash TEXT NOT NULL,
            output TEXT NOT NULL, state TEXT NOT NULL, previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL)""")
        self.config = config
        self.identity = {"format": "offline_event_journal_v1", "config": asdict(config), "identity": identity}
        expected = canonical(self.identity)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute("SELECT value FROM metadata WHERE id=1").fetchone()
            if row is None:
                self.connection.execute("INSERT INTO metadata VALUES (1, ?)", (expected,))
            elif row[0] != expected:
                raise ValueError("설정·모델·입력의 지문이 기존 실행과 다릅니다.")
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            self.connection.close()
            raise
        try:
            self.verify()
        except BaseException:
            self.connection.close()
            raise

    def verify(self) -> None:
        previous = digest(canonical(self.identity))
        expected_sequence = 1
        for sequence, key, payload, output, state, prior, chained in self.connection.execute(
            "SELECT sequence,event_key,payload_hash,output,state,previous_hash,chain_hash FROM events ORDER BY sequence"
        ):
            expected = digest(canonical([sequence, key, payload, output, state, previous]))
            if sequence != expected_sequence or prior != previous or chained != expected:
                raise ValueError("저널 해시 연결 또는 순서가 손상됐습니다.")
            previous = chained
            expected_sequence += 1

    def _latest(self):
        row = self.connection.execute("SELECT sequence,state,chain_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        if row is None:
            return 0, TradingEngine(self.config), digest(canonical(self.identity))
        return row[0], TradingEngine(self.config, json.loads(row[1])), row[2]

    def snapshot(self) -> dict:
        return self._latest()[1].snapshot()

    def _append(self, sequence: int, key: str, payload: str, result: dict, state: dict, previous: str):
        out, checkpoint = canonical(result), canonical(state)
        chained = digest(canonical([sequence, key, payload, out, checkpoint, previous]))
        self.connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                                (sequence, key, payload, out, checkpoint, previous, chained))

    def process(self, bar: dict, policy: Callable[[dict, dict], str], final: bool = False,
                before_commit: Callable | None = None) -> dict:
        normalized = validate_bar(bar, self.config.bar_seconds)
        event_key = "bar:" + normalized["time"]
        payload = digest(canonical({"bar": normalized, "final": final}))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            prior = self.connection.execute("SELECT payload_hash,output FROM events WHERE event_key=?", (event_key,)).fetchone()
            if prior:
                if prior[0] != payload:
                    raise ValueError("같은 사건 ID에 다른 시세 또는 종료 상태가 전달됐습니다.")
                self.connection.commit()
                return {**json.loads(prior[1]), "duplicate": True}
            sequence, engine, previous = self._latest()
            result = engine.step(normalized, policy, final)
            self._append(sequence + 1, event_key, payload, result, engine.snapshot(), previous)
            if before_commit is not None:
                before_commit()
            self.connection.commit()
            return {**result, "duplicate": False}
        except BaseException:
            self.connection.rollback()
            raise

    def halt(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            sequence, engine, previous = self._latest()
            if engine.state["manual_halt"]:
                self.connection.commit()
                return
            engine.halt()
            self._append(sequence + 1, f"control:halt:{sequence + 1}", digest("manual_halt"),
                         {"type": "manual_halt", "liquidation": "next_valid_bar_open"}, engine.snapshot(), previous)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def results(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT output FROM events WHERE event_key LIKE 'bar:%' ORDER BY sequence")]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
