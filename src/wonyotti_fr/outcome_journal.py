from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .journal import canonical, digest


class OutcomeJournal:
    def __init__(self, path: Path, identity: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError('진입 효용 저널의 심볼릭 링크 경로 오류')
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            path.chmod(0o600)
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=10)
        try:
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('PRAGMA trusted_schema=OFF')
            self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
            self.connection.execute('''CREATE TABLE IF NOT EXISTS outcomes (
                sequence INTEGER PRIMARY KEY, input_hash TEXT NOT NULL, payload TEXT NOT NULL,
                previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL)''')
            self.identity = canonical({'format': 'offline_outcome_journal_v1', 'identity': identity})
            self.connection.execute('BEGIN IMMEDIATE')
            previous = self.connection.execute('SELECT value FROM metadata WHERE id=1').fetchone()
            if previous is None:
                self.connection.execute('INSERT INTO metadata VALUES (1,?)', (self.identity,))
            elif previous[0] != self.identity:
                raise ValueError('진입 효용의 입력·정책·코드 지문 변경')
            self.connection.commit()
            self.verify()
        except BaseException:
            self.connection.rollback()
            self.connection.close()
            raise

    def verify(self):
        if self.connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
            raise ValueError('진입 효용 저널의 저장 무결성 오류')
        previous, number = digest(self.identity), 0
        for sequence, source, payload, prior, chained in self.connection.execute('SELECT * FROM outcomes ORDER BY sequence'):
            if (sequence != number or prior != previous
                or chained != digest(canonical([sequence, source, payload, previous]))):
                raise ValueError('진입 효용 저널의 순서·해시 연결 오류')
            json.loads(payload)
            previous, number = chained, number+1

    def read(self, sequence: int, opportunity: dict):
        row = self.connection.execute('SELECT input_hash,payload FROM outcomes WHERE sequence=?', (sequence,)).fetchone()
        if row is None:
            return None
        if row[0] != digest(canonical(opportunity)):
            raise ValueError('완료한 진입 기회의 입력·순서 변경')
        return json.loads(row[1])

    def append(self, sequence: int, opportunity: dict, outcome: dict, before_commit=None):
        if type(sequence) is not int or sequence < 0:
            raise ValueError('진입 효용의 기회 순서 오류')
        source, payload = digest(canonical(opportunity)), canonical(outcome)
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            saved = self.read(sequence, opportunity)
            if saved is not None:
                if canonical(saved) != payload:
                    raise ValueError('완료한 진입 기회의 정답 변경')
                self.connection.commit()
                return False
            row = self.connection.execute('SELECT sequence,chain_hash FROM outcomes ORDER BY sequence DESC LIMIT 1').fetchone()
            expected, previous = (row[0]+1, row[1]) if row else (0, digest(self.identity))
            if sequence != expected:
                raise ValueError('진입 효용의 미완료 기회 누락')
            chained = digest(canonical([sequence, source, payload, previous]))
            # 완료한 기회 하나를 단일 거래로 저장해 중단 뒤 같은 위치에서 재개한다.
            self.connection.execute('INSERT INTO outcomes VALUES (?,?,?,?,?)', (sequence, source, payload, previous, chained))
            if before_commit is not None:
                before_commit()
            self.connection.commit()
            return True
        except BaseException:
            self.connection.rollback()
            raise

    def count(self):
        return self.connection.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
