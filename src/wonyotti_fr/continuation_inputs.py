from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .close_economics import ECONOMIC_FEATURES, EconomicCloseModel
from .close_learning_inputs import CLOSE_FILES
from .common import sha256
from .entry_regression import EntryRegressionModel
from .journal import canonical, digest

PENDING_ACTIONS = ['hold', 'exit', 'reduce', 'increase']
PENDING_FEATURES = ['pending_'+action for action in PENDING_ACTIONS]


class ContinuationCloseModel(EntryRegressionModel):
    features = EconomicCloseModel.features+PENDING_FEATURES
    format = 'close_continuation_histogram_v1'

    @classmethod
    def fit(cls, values, target, weights, validation_values):
        for value in [values, validation_values]:
            matrix = np.asarray(value, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != len(cls.features):
                raise ValueError('현재 관리 의도 학습의 입력 차원 오류')
            validate_pending_matrix(matrix[:, -4:])
        return super().fit(values, target, weights, validation_values)

    def predict(self, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError('현재 관리 의도 예측의 입력 차원 오류')
        validate_pending_matrix(matrix[:, -4:])
        return super().predict(matrix)


def pending_values(intents):
    intents = np.asarray(intents, dtype=object)
    if (intents.ndim != 1 or any(not isinstance(v, str) for v in intents)
        or not np.isin(intents, PENDING_ACTIONS).all()):
        raise ValueError('현재 관리 의도의 누락·알 수 없는 값')
    return np.column_stack([(intents == action).astype(float) for action in PENDING_ACTIONS])


def validate_pending_matrix(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 4 or not np.isin(values, [0., 1.]).all()
        or not (values.sum(axis=1) == 1).all()):
        raise ValueError('현재 관리 의도의 one-hot 입력 오류')
    return values


def load_pending_inputs(labels, ledger, *, decision_seconds=300):
    if type(decision_seconds) is not int or decision_seconds not in (60, 300):
        raise ValueError('현재 관리 의도의 수집 간격 오류')
    hashes = json.loads((labels/'files.json').read_text())
    expected_files = CLOSE_FILES | ({'legacy_parity.json'} if decision_seconds == 60 else set())
    if (set(hashes) != expected_files or (labels/'files.json').is_symlink()
        or any((labels/n).is_symlink() or sha256(labels/n) != h for n, h in hashes.items())
        or any((labels/('outcomes.sqlite'+suffix)).exists() for suffix in ['-wal', '-shm'])):
        raise ValueError('현재 관리 의도의 원장 파일·지문 오류')
    pd.testing.assert_frame_equal(ledger.drop(columns=ECONOMIC_FEATURES), pd.read_parquet(labels/'opportunity_ledger.parquet'), check_exact=True)
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    if (settings.get('decision_seconds', 300) != decision_seconds
        or (decision_seconds == 60 and settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V74.md')))):
        raise ValueError('현재 관리 의도의 원래 간격·계획 오류')
    reference = Path(settings['reference'])
    curve_path = reference/'candidate-00/equity.parquet'
    if (sha256(reference/'frozen_selection.json') != settings['reference_sha256']
        or sha256(curve_path) != settings['reference_outputs_sha256']['equity.parquet']
        or json.loads((labels/'summary.json').read_text())['complete'] is not True):
        raise ValueError('현재 관리 의도의 원래 계좌 연결·완료 오류')
    curve = pd.read_parquet(curve_path, columns=['time', 'next_intent'])
    curve['time'] = pd.to_datetime(curve.time, utc=True).astype('datetime64[ns, UTC]')
    curve = curve.set_index('time')
    times = ledger.decision_time.astype('datetime64[ns, UTC]').array.asi8
    matrix = validate_pending_matrix(pending_values(ledger.original_intent))
    identity = canonical({'format': 'offline_outcome_journal_v1', 'identity': settings})
    previous, count = digest(identity), 0
    connection = sqlite3.connect((labels/'outcomes.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    try:
        connection.execute('PRAGMA trusted_schema=OFF')
        if (connection.execute('PRAGMA quick_check').fetchone() != ('ok',)
            or connection.execute('SELECT value FROM metadata').fetchall() != [(identity,)]
            or connection.execute('SELECT COUNT(*) FROM outcomes').fetchone() != (len(ledger),)
            or connection.execute('SELECT COALESCE(MAX(length(payload)),0) FROM outcomes').fetchone()[0] > 1024**2):
            raise ValueError('현재 관리 의도의 원자 크기·무결성 오류')
        for number, source, payload, prior, chained in connection.execute('SELECT * FROM outcomes ORDER BY sequence'):
            op = json.loads(payload)['opportunity']
            if (number != count or prior != previous or source != digest(canonical(op))
                or chained != digest(canonical([number, source, payload, previous]))):
                raise ValueError('현재 관리 의도의 원자 순서·입력·해시 연결 오류')
            time = pd.Timestamp(op['decision_time'])
            expected = ledger.original_intent.iloc[number]
            if (time.value != times[number] or pd.Timestamp(op['state']['last_end']).value != times[number]
                or any(v != expected for v in [op['original_intent'], op['state']['pending'], curve.loc[time, 'next_intent']])):
                raise ValueError('현재 관리 의도의 실제 요청·저장 상태 연결 오류')
            previous, count = chained, count+1
    finally:
        connection.close()
    if (count != len(ledger) or any(sha256(labels/n) != h for n, h in hashes.items())
        or sha256(curve_path) != settings['reference_outputs_sha256']['equity.parquet']):
        raise ValueError('현재 관리 의도의 전체 행·읽기 전용 지문 오류')
    result = ledger.copy()
    result[PENDING_FEATURES] = matrix
    return result, {'rows': count, 'all_current_pending_and_original_requests_exact': True,
        'journal_read_only': True, 'future_execution_or_target_used': False,
        'labels_files_sha256': sha256(labels/'files.json'), 'reference_equity_sha256': sha256(curve_path)}
