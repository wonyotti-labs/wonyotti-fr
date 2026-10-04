from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .close_effect import CLOSE_FEATURES, CLOSE_PERIOD, IndexedEvents, validate_close_record
from .common import sha256
from .engine import EngineConfig
from .event_research import load_selection
from .journal import canonical, digest
from .minute_data import prepare_minute_period

CLOSE_FILES = {'manifest.json', 'input_verification.json', 'outcomes.sqlite', 'opportunity_ledger.parquet',
               'training_labels.parquet', 'support.json', 'summary.json'}
CLOSE_SPLITS = {'training': ['2021-01-01', '2021-09-30'], 'diagnosis': ['2021-10-02', '2021-12-31']}


def snapshot_close_values(opportunity, event):
    state = opportunity['state']
    stored, trade = state['policy_state'], state['active_trade']
    market, minute = (np.asarray(event[k], dtype=float) for k in ['features', 'minute_features'])
    direction, entry = np.sign(state['quantity']), state['entry_price']
    move = direction*(state['last_close']/entry-1)
    best = direction*((stored['path_high'] if direction > 0 else stored['path_low'])/entry-1)
    worst = direction*((stored['path_low'] if direction > 0 else stored['path_high'])/entry-1)
    history, epoch = [], pd.Timestamp(opportunity['decision_time']).value//(60*10**9)
    for action in ['increase', 'reduce']:
        last, mask = stored[f'fill_{action}_last'], stored[f'fill_{action}_mask']
        history.extend([float(last != 0), np.log1p(epoch-last) if last else 0., np.log1p(mask.bit_count())])
    return np.r_[market, market*direction, direction, move, np.log1p(state['index']-state['entry_index']),
        min(trade['adds'], 5), best, worst, best-move, abs(state['quantity'])/trade['max_quantity'],
        minute, minute*direction, history]


def load_close_training(labels: Path):
    hashes = json.loads((labels/'files.json').read_text())
    if (set(hashes) != CLOSE_FILES or (labels/'files.json').is_symlink()
        or any((labels/n).is_symlink() or sha256(labels/n) != h for n, h in hashes.items())):
        raise ValueError('청산 학습 원장의 필수 파일·지문 오류')
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    summary = json.loads((labels/'summary.json').read_text())
    if (summary['complete'] is not True or summary['new_models_fitted'] is not False
        or summary['forced_boundary_closes'] is not False or summary['profitability_accepted'] is not False
        or settings['period'] != CLOSE_PERIOD or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V59.md'))
        or settings['new_models_fitted'] is not False or settings['forced_boundary_closes'] is not False):
        raise ValueError('청산 학습의 미완료·기간·사전 가정 오류')
    sources = settings['implementation_sha256']
    code = labels/'generation_source/wonyotti_fr'
    if (not {'close_effect.py', 'engine.py'} <= set(sources) or set(sources) != {p.name for p in code.glob('*.py')}
        or any(Path(n).name != n or not n.endswith('.py') or (code/n).is_symlink()
               or not (code/n).resolve().is_relative_to(labels.resolve()) or sha256(code/n) != h for n, h in sources.items())):
        raise ValueError('청산 학습의 생성 코드 사본·지문 오류')
    reference, replay = Path(settings['reference']), Path(summary['replay'])
    frozen, _ = load_selection(reference)
    if (frozen['protocol'] != 'exit_move_v54' or settings['reference_sha256'] != sha256(reference/'frozen_selection.json')
        or set(settings['reference_outputs_sha256']) != {'equity.parquet', 'trades.parquet', 'fills.parquet', 'final_state.json'}
        or any(sha256(reference/'candidate-00'/n) != h for n, h in settings['reference_outputs_sha256'].items())):
        raise ValueError('청산 학습의 고정 부모·원래 출력 연결 오류')
    risk = EngineConfig(**frozen['risk'])
    parity = json.loads((replay/'parity.json').read_text())
    if (summary['replay_parity_sha256'] != sha256(replay/'parity.json')
        or any(parity[k] is not True for k in ['complete', 'all_replayed_outputs_exact', 'final_state_exact'])):
        raise ValueError('청산 학습의 전체 재생 확인 오류')
    for name in ['equity.parquet', 'trades.parquet', 'fills.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(replay/name), pd.read_parquet(reference/'candidate-00'/name), check_exact=True)
    if json.loads((replay/'final_state.json').read_text()) != json.loads((reference/'candidate-00/final_state.json').read_text()):
        raise ValueError('청산 학습의 원래 최종 상태 오류')
    market, features = Path(settings['market']), Path(settings['features'])
    if (settings['market_manifest_sha256'] != sha256(market/'manifest-1m.json')
        or settings['feature_manifest_sha256'] != sha256(features/'manifest-5m.json')):
        raise ValueError('청산 학습의 시세 원본 지문 오류')
    bars, checks = prepare_minute_period(market, features, 'BTCUSDT', *CLOSE_PERIOD, minute_inputs=True)
    if canonical(checks) != canonical(json.loads((labels/'input_verification.json').read_text())):
        raise ValueError('청산 학습의 시세 검증 재현 오류')
    events = IndexedEvents(bars)
    ledger, training = (pd.read_parquet(labels/n) for n in ['opportunity_ledger.parquet', 'training_labels.parquet'])
    pd.testing.assert_frame_equal(training, ledger[ledger.label_status.eq('closed')].reset_index(drop=True), check_exact=True)
    if (len(ledger) != summary['opportunities'] or len(training) != summary['closed']
        or ledger.label_status.value_counts().to_dict() != summary['statuses']
        or parity['processed_bars'] != len(bars) or parity['counts'] != summary['counts']
        or summary['counts']['opportunities'] != len(ledger)
        or summary['counts']['boundaries'] != len(bars)//5):
        raise ValueError('청산 학습의 전체 기회·정답 지원 오류')
    equity = pd.read_parquet(replay/'equity.parquet', columns=['time', 'quantity', 'equity', 'policy_state', 'policy_event'])
    equity['time'] = pd.to_datetime(equity.time, utc=True).astype('datetime64[ns, UTC]')
    equity = equity.set_index('time')
    eligible = equity.index[(equity.index.asi8 % (5*60*10**9) == 0) & equity.quantity.ne(0)
        & equity.policy_event.isin(['action_hold', 'action_exit', 'action_reduce', 'action_sized_reduce',
                                   'action_zero_reduction', 'action_increase', 'action_unavailable'])]
    np.testing.assert_array_equal(ledger.decision_time.astype('datetime64[ns, UTC]').array.asi8, eligible.asi8)
    trades = pd.read_parquet(reference/'candidate-00/trades.parquet').to_dict('records')
    prior_net = np.r_[0., np.cumsum([r['net_pnl'] for r in trades])]
    cutoff, count = pd.Timestamp('2021-12-31', tz='UTC'), 0
    identity = canonical({'format': 'offline_outcome_journal_v1', 'identity': settings})
    previous = digest(identity)
    before = sha256(labels/'outcomes.sqlite')
    if any((labels/('outcomes.sqlite'+suffix)).exists() for suffix in ['-wal', '-shm']):
        raise ValueError('청산 학습 원자가 아직 열려 있습니다.')
    connection = sqlite3.connect((labels/'outcomes.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    try:
        connection.execute('PRAGMA trusted_schema=OFF')
        if (connection.execute('PRAGMA quick_check').fetchone() != ('ok',)
            or connection.execute('SELECT value FROM metadata').fetchall() != [(identity,)]
            or connection.execute('SELECT COUNT(*),COALESCE(MAX(length(payload)),0) FROM outcomes').fetchone()[0] != len(ledger)
            or connection.execute('SELECT COALESCE(MAX(length(payload)),0) FROM outcomes').fetchone()[0] > 1024**2):
            raise ValueError('청산 학습 원자의 크기·무결성 오류')
        for number, source, payload, prior, chained in connection.execute('SELECT * FROM outcomes ORDER BY sequence'):
            record = json.loads(payload)
            op, state = record['opportunity'], record['opportunity']['state']
            if (number != count or source != digest(canonical(op)) or prior != previous
                or chained != digest(canonical([number, source, payload, previous]))):
                raise ValueError('청산 학습 원자의 순서·입력·해시 연결 오류')
            previous, count = chained, count+1
            trade = trades[op['trade_index']]
            validate_close_record(op, record, trade, risk, cutoff, events)
            current = equity.loc[pd.Timestamp(op['decision_time'])]
            if (current.quantity != state['quantity'] or json.loads(current.policy_state) != state['policy_state']
                or not np.isclose(current.equity, state['cash']+state['quantity']*state['last_close'], rtol=0, atol=1e-9)
                or not np.isclose(state['closed_net'], prior_net[op['trade_index']], rtol=0, atol=1e-8)):
                raise ValueError('청산 학습의 실제 계좌 상태 연결 오류')
            np.testing.assert_allclose(np.asarray([op[k] for k in CLOSE_FEATURES], dtype=float), snapshot_close_values(op, events[op['start']-1]),
                                       rtol=0, atol=1e-12, equal_nan=True)
            expected = {k: v for k, v in op.items() if k != 'state'} | record['outcome']
            actual = ledger.iloc[number]
            if set(expected)-set(ledger):
                raise ValueError('청산 학습 원장의 열 누락')
            for key, value in actual.items():
                target = expected.get(key)
                if target is None:
                    valid = pd.isna(value)
                elif key in {'decision_time', 'position_entry_time', 'label_end', 'continue_end'}:
                    valid = pd.Timestamp(value) == pd.Timestamp(target)
                else:
                    valid = value == target
                if not valid:
                    raise ValueError('청산 학습 원장과 원자 저장 값 불일치')
    finally:
        connection.close()
    if before != sha256(labels/'outcomes.sqlite') or count != len(ledger):
        raise ValueError('청산 학습의 읽기 전용 검증·행 수 오류')
    return ledger, {'rows': len(ledger), 'closed': len(training), 'all_actual_inputs_and_cashflows_verified': True,
        'journal_read_only': True, 'generation_implementation_sha256': sources, 'labels_files_sha256': sha256(labels/'files.json')}


def close_learning_splits(ledger):
    closed = ledger.label_status.eq('closed')
    if (ledger.decision_time.isna().any() or ledger.decision_time.duplicated().any()
        or not ledger.decision_time.is_monotonic_increasing
        or ledger.loc[closed, ['position_entry_time', 'label_end']].isna().any().any()
        or ledger.loc[closed, 'position_entry_time'].ge(ledger.loc[closed, 'decision_time']).any()
        or ledger.loc[closed, 'decision_time'].ge(ledger.loc[closed, 'label_end']).any()):
        raise ValueError('청산 학습 분리의 중복·시간 순서 오류')
    assignment = ledger.copy()
    assignment['split'] = np.where(closed, 'excluded_boundary', 'excluded_not_closed')
    rows = {}
    for name, (start, end) in CLOSE_SPLITS.items():
        mask = (closed & ledger.decision_time.ge(pd.Timestamp(start, tz='UTC'))
                & ledger.position_entry_time.ge(pd.Timestamp(start, tz='UTC'))
                & ledger.label_end.lt(pd.Timestamp(end, tz='UTC')))
        frame = ledger.loc[mask].reset_index(drop=True)
        positions = frame.groupby('direction').position_entry_time.nunique()
        if (len(frame) < (1000 if name == 'training' else 100)
            or frame.position_entry_time.nunique() < (100 if name == 'training' else 30)
            or any(positions.get(side, 0) < 20 for side in [-1, 1])
            or frame.position_entry_time.ge(frame.decision_time).any() or frame.decision_time.ge(frame.label_end).any()
            or frame.groupby('position_entry_time').direction.nunique().gt(1).any()
            or not np.isfinite(frame[CLOSE_FEATURES+['close_advantage_bps']]).all().all()
            or (name == 'training' and frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=180))):
            raise ValueError('청산 학습 분리의 기간·포지션·방향·숫자 지원 부족')
        assignment.loc[mask, 'split'] = name
        rows[name] = frame
    if (set(rows['training'].position_entry_time) & set(rows['diagnosis'].position_entry_time)
        or rows['training'].label_end.max() >= rows['diagnosis'].decision_time.min()):
        raise ValueError('청산 학습과 진단의 포지션·정답 교차')
    return rows, assignment
