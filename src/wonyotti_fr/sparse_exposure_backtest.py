from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_backtest import iter_events, summarize_observations
from wonyotti_fr.exposure_engine import ExposureEngine
from wonyotti_fr.exposure_reference import ReferenceAccount, require_near
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.sparse_execution_reference import ReferenceSparsePolicy

SCHEMAS = {
    'equity': pa.schema([('time', pa.string()), *[(key, pa.float64()) for key in
        ['equity', 'quantity', 'exposure', 'net_exposure', 'drawdown', 'requested_target',
         'executed_target', 'decision', 'next_target', 'accounting_residual']],
        ('halted', pa.bool_()), ('daily_halted', pa.bool_()), ('completed', pa.bool_())]),
    'fills': pa.schema([('time', pa.string()), ('delta_quantity', pa.float64()),
        ('price', pa.float64()), ('fee', pa.float64()), ('reason', pa.string())]),
    'trades': pa.schema([('entry_time', pa.string()), ('entry_price', pa.float64()), ('direction', pa.int64()),
        *[(key, pa.float64()) for key in ['gross_realized', 'fees', 'funding_cost', 'max_quantity']],
        ('exit_time', pa.string()), ('exit_price', pa.float64()), ('exit_reason', pa.string()),
        ('net_pnl', pa.float64()), ('hold_bars', pa.int64())]),
    'decisions': pa.schema([('time', pa.string()), *[(key, pa.float64()) for key in
        ['equity', 'quantity', 'current_exposure', 'probability_decrease', 'probability_no_change', 'probability_increase',
         'negative_size', 'positive_size', 'expected_change', 'decision_change', 'target']],
         ('action', pa.int64()), ('reason', pa.string())]),
    'rejections': pa.schema([('time', pa.string()), ('reason', pa.string())]),
}


class TypedRows:
    def __init__(self, path, schema, batch_size):
        self.schema, self.limit = schema, batch_size
        self.writer = pq.ParquetWriter(path, schema, compression='zstd')
        self.buffer, self.rows = [], 0

    def append(self, row):
        if set(row) != set(self.schema.names):
            raise ValueError('실행 출력 필드 불일치')
        self.buffer.append(row)
        self.rows += 1
        if len(self.buffer) >= self.limit:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self):
        self.flush()
        self.writer.close()


def events_with_feature_time(data):
    for event, stamp in zip(iter_events(data), data.feature_end, strict=True):
        event['feature_end'] = None if pd.isna(stamp) else stamp.isoformat()
        yield event


def backtest(data, policy, config, output, batch_size=8192, progress=None):
    if type(batch_size) is not int or not 1 <= batch_size <= 100000 or len(data) < 2:
        raise ValueError('노출 실행 저장 묶음·입력 행 오류')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'config.json', asdict(config))
    save_json(output/'policy.json', {'kind': policy.kind, 'model': None if policy.model is None else policy.model.to_dict()})
    engine = ExposureEngine(config)
    writers = {name: TypedRows(output/f'{name}.parquet', schema, batch_size) for name, schema in SCHEMAS.items()}
    if policy.audit_sink is not None:
        raise ValueError('희소 실행 정책의 중복 기록기 오류')
    policy.audit_sink = writers['decisions'].append
    daily, rejected, exposures, totals = {}, {}, [], []
    maximum, count = 0., 0
    checkpoints = sorted({len(data)//3, 2*len(data)//3})
    try:
        for i, event in enumerate(events_with_feature_time(data)):
            result = engine.step(event, policy, final=i == len(data)-1)
            for name, key in [('trades', 'closed_trades'), ('fills', 'fills')]:
                for row in result.pop(key):
                    writers[name].append(row)
            for reason in result.pop('rejected'):
                rejected[reason] = rejected.get(reason, 0)+1
                writers['rejections'].append({'time': result['time'], 'reason': reason})
            writers['equity'].append(result)
            maximum = max(maximum, abs(result['accounting_residual']))
            daily[result['time'][:10]] = result['equity']
            exposures.append(result['exposure'])
            if len(exposures) == batch_size:
                totals.append(math.fsum(exposures))
                exposures.clear()
            count += 1
            if count in checkpoints:
                snapshot = engine.snapshot()
                save_json(output/f'checkpoint-{count}.json', snapshot)
                engine = ExposureEngine(config, json.loads(json.dumps(snapshot, allow_nan=False)))
                if engine.snapshot() != snapshot:
                    raise ValueError('실행 계좌 직렬화·복원 불일치')
            if progress is not None and count % 100000 == 0:
                progress(count)
        totals.append(math.fsum(exposures))
        daily_series = pd.Series(daily, dtype=float)
        daily_series.index = pd.to_datetime(daily_series.index, utc=True)
        metrics = summarize_observations(engine, count, result['equity'], math.fsum(totals)/count, maximum, daily_series, rejected)
        save_json(output/'metrics.json', metrics)
        save_json(output/'final_state.json', engine.snapshot())
    except Exception as error:
        save_json(output/'failure.json', {'type': type(error).__name__, 'message': str(error), 'processed_bars': count, 'complete': False})
        raise
    finally:
        policy.audit_sink = None
        for writer in writers.values():
            writer.close()
    save_json(output/'storage.json', {'batch_size': batch_size, 'rows': {key: writer.rows for key, writer in writers.items()},
        'json_restored_checkpoints': checkpoints})
    save_json(output/'files.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'files.json'})
    return metrics


def parquet_records(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def verify_path(data, kind, model, config, output, review, progress=None):
    sealed = read_json(output/'files.json')
    for name, digest in sealed.items():
        if '/' in name or '\\' in name or sha256(output/name) != digest:
            raise ValueError('노출 실행 검산 입력 해시 불일치')
    if read_json(output/'config.json') != asdict(config) or read_json(output/'policy.json') != {'kind': kind, 'model': model}:
        raise ValueError('노출 실행 검산 설정·정책 불일치')
    account = ReferenceAccount(config)
    streams = {name: iter(parquet_records(output/f'{name}.parquet')) for name in SCHEMAS}
    counts = dict.fromkeys(SCHEMAS, 0)
    daily, rejected, exposure = {}, {}, 0.
    max_equity_error, max_quantity_error, residual = 0., 0., 0.

    reference = ReferenceSparsePolicy(model)

    def decision(bar, cash, q):
        target, journal = reference(bar, cash, q)
        if journal is not None:
            require_near(next(streams['decisions'], None), journal, 'decision.'+bar['end'])
            counts['decisions'] += 1
        return target

    for i, event in enumerate(events_with_feature_time(data)):
        expected = account.step(event, decision, final=i == len(data)-1)
        for name, key in [('trades', 'closed_trades'), ('fills', 'fills')]:
            for row in expected.pop(key):
                saved = next(streams[name], None)
                require_near(saved, row, f'{i}.{name}')
                counts[name] += 1
        for reason in expected.pop('rejected'):
            require_near(next(streams['rejections'], None), {'time': event['end'], 'reason': reason}, f'{i}.rejections')
            counts['rejections'] += 1
            rejected[reason] = rejected.get(reason, 0)+1
        saved = next(streams['equity'], None)
        if saved is None:
            raise ValueError('노출 검산 계좌 행 누락')
        value = saved.pop('accounting_residual')
        if not math.isfinite(value) or abs(value) > max(1e-7, saved['equity']*1e-10):
            raise ValueError('노출 검산 회계 잔차 오류')
        residual = max(residual, abs(value))
        max_equity_error = max(max_equity_error, abs(saved['equity']-expected['equity']))
        max_quantity_error = max(max_quantity_error, abs(saved['quantity']-expected['quantity']))
        require_near(saved, expected, f'{i}.equity')
        counts['equity'] += 1
        daily[event['end'][:10]] = expected['equity']
        exposure += expected['exposure']
        if progress is not None and (i+1) % 100000 == 0:
            progress(i+1)
    for name, stream in streams.items():
        if next(stream, None) is not None:
            raise ValueError(f'노출 검산 예상 밖 추가 행: {name}')
    final = read_json(output/'final_state.json')
    if not final['state']['completed'] or final['state']['quantity'] or final['state']['pending'] is not None or final['state']['deferred']:
        raise ValueError('노출 검산 최종 상태 오류')
    for key, expected in {'cash': account.cash, 'quantity': account.q, 'closed_net': account.closed_net,
        'closed_trades': account.trades, 'wins': account.wins, 'total_fees': account.fees,
        'total_funding': account.funding, 'max_drawdown': account.max_dd, 'permanent_halted': account.halted}.items():
        require_near(final['state'][key], expected, 'final.'+key)
    returns = pd.Series(daily, dtype=float).pct_change().dropna()
    deviation = returns.std()
    metrics = {'total_return': account.cash/config.initial_equity-1,
        'annualized_return': (account.cash/config.initial_equity)**(365.25*1440/len(data))-1,
        'max_drawdown': account.max_dd, 'closed_trades': account.trades,
        'win_rate': account.wins/account.trades if account.trades else None,
        'profit_factor': account.gains/account.losses if account.losses else None,
        'daily_sharpe': float(np.sqrt(365.25)*returns.mean()/deviation) if deviation > 0 else None,
        'fees': account.fees, 'funding_cost': account.funding, 'average_exposure': exposure/len(data),
        'permanent_halt': account.halted, 'max_accounting_residual': residual, 'rejected': rejected,
        'final_equity': account.cash, 'bars': len(data)}
    require_near(read_json(output/'metrics.json'), metrics, 'metrics')
    if read_json(output/'storage.json')['rows'] != counts:
        raise ValueError('노출 검산 저장 행 수 불일치')
    for name, digest in sealed.items():
        if sha256(output/name) != digest:
            raise ValueError('노출 실행 검산 중 입력 변경')
    proof = {'complete': True, 'source_files_sha256': sha256(output/'files.json'), 'rows': counts,
        'full_independent_policy_and_account_replay': True, 'all_invoked_boundary_probabilities_sizes_actions_verified': True, 'bisection_quantity_sizing': True,
        'closed_trade_cash_flow_pnl': True, 'every_fill_and_trade_compared': True,
        'max_equity_difference': max_equity_error, 'max_quantity_difference': max_quantity_error,
        'metrics': metrics}
    save_json(review, proof)
    return proof
