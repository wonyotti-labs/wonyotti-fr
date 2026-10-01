from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import save_json
from .engine import TradingEngine
from .event_backtest import iter_events, summarize_observations


class ParquetRows:
    def __init__(self, path: Path | None, batch_size: int):
        self.path, self.batch_size = path, batch_size
        self.buffer, self.writer = [], None
        self.rows, self.peak_buffer = 0, 0

    def append(self, row: dict):
        self.rows += 1
        if self.path is None:
            return
        self.buffer.append(row)
        self.peak_buffer = max(self.peak_buffer, len(self.buffer))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(self.buffer), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='snappy')
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self):
        try:
            self.flush()
        finally:
            if self.writer is not None:
                self.writer.close()
            elif self.path is not None:
                pd.DataFrame().to_parquet(self.path, index=False)


def streaming_backtest(data, policy, config, output: Path | None, batch_size: int) -> dict:
    if type(batch_size) is not int or not 1 <= batch_size <= 100_000:
        raise ValueError('저장 묶음은 1~100000행의 정수여야 합니다.')
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'config.json', asdict(config))
    engine = TradingEngine(config)
    writers = {name: ParquetRows(output / f'{name}.parquet' if output else None, batch_size)
               for name in ['equity', 'trades', 'fills']}
    rejected, daily, exposure_sums, exposures = {}, {}, [], []
    max_residual, last_equity, count = 0.0, config.initial_equity, 0
    try:
        if hasattr(policy, 'prepare'):
            policy.prepare(data)
        for index, event in enumerate(iter_events(data)):
            result = engine.step(event, policy, final=index == len(data) - 1)
            for trade in result.pop('closed_trades'):
                writers['trades'].append(trade)
            for fill in result.pop('fills'):
                writers['fills'].append(fill)
            for reason in result.pop('rejected'):
                rejected[reason] = rejected.get(reason, 0) + 1
            writers['equity'].append(result)
            count += 1
            last_equity = result['equity']
            daily[result['time'][:10]] = last_equity
            max_residual = max(max_residual, abs(result['accounting_residual']))
            exposures.append(result['exposure'])
            if len(exposures) == batch_size:
                exposure_sums.append(math.fsum(exposures))
                exposures.clear()
        exposure_sums.append(math.fsum(exposures))
        daily_series = pd.Series(daily, dtype=float)
        daily_series.index = pd.to_datetime(daily_series.index, utc=True)
        metrics = summarize_observations(engine, count, last_equity, math.fsum(exposure_sums) / count,
                                          max_residual, daily_series, rejected)
    except Exception as error:
        if output is not None:
            save_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                                               'processed_bars': count, 'complete': False})
        raise
    finally:
        for writer in writers.values():
            writer.close()
    if output is not None:
        save_json(output / 'metrics.json', metrics)
        save_json(output / 'final_state.json', engine.snapshot())
        save_json(output / 'storage.json', {'mode': 'bounded_output_buffers', 'batch_size': batch_size,
                  'outputs': {name: {'rows': writer.rows, 'peak_buffer_rows': writer.peak_buffer} for name, writer in writers.items()},
                  'limit': '입력 시세와 모델 특징 캐시는 별도 메모리를 사용함'})
    return metrics
