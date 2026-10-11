from __future__ import annotations

import pandas as pd
import pyarrow as pa

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.exposure_backtest import TypedRows
from wonyotti_fr.exposure_backtest import backtest as account_backtest
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.recorded_change_policy import RecordedChangePolicy

JOURNAL_SCHEMA = pa.schema([
    *[(k, pa.string()) for k in ['time', 'source_end', 'source_label_end', 'available_at']],
    ('lookahead_seconds', pa.int64()), ('historical_lookahead', pa.bool_()),
    ('source_supported', pa.bool_()), ('source_reason', pa.string()),
    *[(k, pa.float64()) for k in ['supplied_change', 'equity', 'quantity', 'current_exposure',
        'unclipped_target', 'target']], ('clipped', pa.bool_()), ('reason', pa.string())])


def coverage(source, policy, decisions, curve):
    key = 'end' if policy['timing'] == 'window_start' else 'label_end'
    available = pd.DatetimeIndex(source[key])
    times = pd.to_datetime(curve.time, utc=True)
    linked = curve.loc[times.isin(available)]
    linked_times = pd.to_datetime(linked.time, utc=True)
    invoked = pd.DatetimeIndex(pd.to_datetime(decisions.time, utc=True))
    absent = linked.loc[~linked_times.isin(invoked)]
    risk = absent.halted | absent.daily_halted
    return {'source_rows': len(source), 'source_supported_rows': int(source.contract_supported.sum()),
        'source_nonzero_rows': int((source.contract_supported & source[policy['target']].abs().gt(1e-12)).sum()),
        'source_rows_within_market_boundaries': len(linked),
        'source_rows_outside_market_boundaries': len(source)-len(linked),
        'source_rows_invoked': int(decisions.source_end.notna().sum()),
        'source_rows_not_invoked': len(absent), 'not_invoked_risk_blocked': int(risk.sum()),
        'not_invoked_final_only': int((~risk & absent.completed).sum()),
        'not_invoked_liquidity_only': int((~risk & ~absent.completed).sum()),
        'decision_rows': len(decisions), 'decision_reasons': decisions.reason.value_counts().to_dict(),
        'clipped_targets': int(decisions.clipped.sum()),
        'historical_lookahead_rows': int(decisions.lookahead_seconds.gt(0).sum()),
        'submitted_targets': int(decisions.target.notna().sum()),
        'executed_targets': int(curve.executed_target.notna().sum())}


def backtest(data, source, config, output, *, target, timing, historical_lookahead=False,
             batch_size=8192, progress=None):
    config.validate()
    if (config.allocation != .5 or type(batch_size) is not int or not 1 <= batch_size <= 100000
        or len(data) < 2):
        raise ValueError('기록 변경량 진단 계좌·저장 설정 오류')
    policy = RecordedChangePolicy(source, target=target, timing=timing, historical_lookahead=historical_lookahead)
    output.mkdir(parents=True, exist_ok=False)
    policy.source.to_parquet(output/'source.parquet', index=False)
    writer = TypedRows(output/'decisions.parquet', JOURNAL_SCHEMA, batch_size)
    policy.audit_sink = writer.append
    try:
        metrics = account_backtest(data, policy, config, output/'account', batch_size, progress)
    except Exception as error:
        save_json(output/'failure.json', {'complete': False, 'type': type(error).__name__,
            'message': str(error), 'decisions_preserved': writer.rows})
        raise
    finally:
        writer.close()
        policy.audit_sink = None
    decisions = pd.read_parquet(output/'decisions.parquet')
    curve = pd.read_parquet(output/'account/equity.parquet')
    save_json(output/'coverage.json', coverage(policy.source, policy.to_dict(), decisions, curve))
    save_json(output/'storage.json', {'decision_rows': writer.rows,
        'source_rows': len(policy.source), 'account_files_sha256': sha256(output/'account/files.json'),
        'account_rows': read_json(output/'account/storage.json')['rows']})
    save_json(output/'files.json', {p.name: sha256(p) for p in sorted(output.iterdir())
        if p.is_file() and p.name != 'files.json'})
    return metrics
