from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_verification import selected_aggregate_trades, verify_aggregate_coverage
from .common import save_json, sha256
from .market import archive_csv, verified_archive
from .minute_data import compare_minute_bars, validate_minutes
from .minute_repair import BAR_COLUMNS, rebuilt_minutes, selected_archive_trades
from .research import load_market


def canonical_window(minute: pd.DataFrame, five: pd.DataFrame, trades: pd.DataFrame,
                     aggregates: pd.DataFrame, ends: pd.DatetimeIndex) -> tuple[dict, dict]:
    if (ends.empty or ends.has_duplicates or not ends.is_monotonic_increasing or ends.tz is None
        or (ends.as_unit('ns').asi8 % pd.Timedelta(minutes=5).value).any()):
        raise ValueError('양방향 복원 대상의 5분 경계 오류')
    rebuilt = rebuilt_minutes(trades)
    rebuilt = rebuilt[rebuilt.end.dt.ceil('5min').isin(ends)].reset_index(drop=True)
    expected = pd.DatetimeIndex(sorted(t - pd.Timedelta(minutes=k) for t in ends for k in range(1, 6)))
    if not np.array_equal(rebuilt.time.dt.as_unit('ns').array.asi8, expected.as_unit('ns').asi8):
        raise ValueError('복원 대상의 전체 1분 체결 구간 부족')
    # 두 봉 자료 모두 오류가 있을 수 있어 모든 대상 체결을 별도 집계 형식과 대조한다.
    corroboration = verify_aggregate_coverage(trades, aggregates, rebuilt.time)
    derived = rebuilt.groupby(rebuilt.end.dt.ceil('5min')).agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
        volume=('volume', 'sum'), quote_volume=('quote_volume', 'sum'), count=('count', 'sum'),
        taker_buy_volume=('taker_buy_volume', 'sum'), taker_buy_quote=('taker_buy_quote', 'sum')).reset_index()
    derived['time'] = derived.end - pd.Timedelta(minutes=5)
    validate_minutes(derived, 5)
    updates, changes = {}, {}
    for interval, original, canonical in [('1m', minute, rebuilt), ('5m', five, derived)]:
        prior = original[original.time.isin(canonical.time)].reset_index(drop=True)
        if not np.array_equal(prior.time.dt.as_unit('ns').array.asi8, canonical.time.dt.as_unit('ns').array.asi8):
            raise ValueError('양방향 복원 대상 봉의 중복·누락·순서 오류')
        changed = np.zeros(len(prior), dtype=bool)
        for name in ['open', 'high', 'low', 'close', 'volume', 'count']:
            changed |= ~np.isclose(prior[name], canonical[name], rtol=0,
                                  atol=0 if name == 'count' else 1e-8)
        updates[interval] = canonical.loc[changed, ['time', 'end', *BAR_COLUMNS]]
        changes[interval] = prior.loc[changed, ['time', *BAR_COLUMNS]].merge(
            canonical.loc[changed, ['time', *BAR_COLUMNS]], on='time',
            suffixes=('_original', '_rebuilt'), validate='one_to_one')
    return updates, {'corroboration': corroboration, 'changes': changes,
                     'changed_minutes': len(updates['1m']), 'changed_five_minutes': len(updates['5m'])}


def context_ends(ends: pd.DatetimeIndex, date: pd.Timestamp) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(sorted({t + pd.Timedelta(minutes=k) for t in ends for k in [-5, 0, 5]}))
    if ((values - pd.Timedelta(nanoseconds=1)).floor('1D') != date).any():
        raise ValueError('날짜 경계 복원에는 전후 날짜의 별도 체결 대조가 필요합니다.')
    return values


def repair_paired_market(source: Path, feature_source: Path, output: Path, feature_output: Path, cache: Path) -> dict:
    sources, outputs = {'1m': source, '5m': feature_source}, {'1m': output, '5m': feature_output}
    paths = [p.resolve() for p in [source, feature_source, output, feature_output]]
    if (any(p.exists() for p in outputs.values()) or
        any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i+1:])):
        raise ValueError('양방향 복원에는 서로 겹치지 않는 새 출력 폴더가 필요합니다.')
    manifests = {interval: copy.deepcopy(json.loads((p / f'manifest-{interval}.json').read_text()))
                 for interval, p in sources.items()}
    symbols = manifests['1m']['summary']
    if (not 1 <= len(symbols) <= 5 or not set(symbols) <= set(manifests['5m']['summary'])
        or any(not s.isalnum() or len(s) > 20 for s in symbols)
        or any(m.get('interval') != interval for interval, m in manifests.items())):
        raise ValueError('양방향 복원의 심볼·간격 오류')
    evidence = {'source_sha256': {k: sha256(v / f'manifest-{k}.json') for k, v in sources.items()},
                'rule': '대상 모든 체결의 두 형식 대조 후 두 해상도 복원; 원본 보존', 'symbols': {}}
    for interval, path in outputs.items():
        shutil.copytree(sources[interval], path)
        # 완료 전에는 입력으로 사용할 최종 매니페스트를 노출하지 않는다.
        (path / f'manifest-{interval}.json').rename(path / f'paired-prior-manifest-{interval}.json')
    evidence_dir = output / 'paired-reconciliation'
    evidence_dir.mkdir()
    try:
        for symbol in symbols:
            minute, _ = load_market(source, symbol, '1m')
            five, _ = load_market(feature_source, symbol, '5m')
            reference = five[(five.time >= minute.time.min()) & (five.end <= minute.end.max())]
            comparison, before = compare_minute_bars(minute, reference)
            if before['incomplete']:
                raise ValueError('봉 누락은 양방향 값 복원 전에 보완해야 합니다.')
            mismatches = comparison[~comparison.matched]
            dates = (mismatches.end - pd.Timedelta(nanoseconds=1)).dt.floor('1D')
            if dates.nunique() > 31:
                raise ValueError('한 심볼의 양방향 복원은 31일 이하여야 합니다.')
            details = []
            evidence['symbols'][symbol] = {'before': before, 'details': details}
            for date in sorted(dates.unique()):
                ends = pd.DatetimeIndex(mismatches.loc[dates.eq(date), 'end'])
                base = 'https://data.binance.vision/data/futures/um/daily'
                raw_path, raw_meta = verified_archive(f'{base}/trades/{symbol}/{symbol}-trades-{date:%Y-%m-%d}.zip', cache)
                trades, raw_checks = selected_archive_trades(archive_csv(raw_path), date, context_ends(ends, date))
                agg_path, agg_meta = verified_archive(f'{base}/aggTrades/{symbol}/{symbol}-aggTrades-{date:%Y-%m-%d}.zip', cache)
                aggregates, agg_checks = selected_aggregate_trades(archive_csv(agg_path), date, trades)
                label = f'{symbol}-{date:%Y-%m-%d}'
                trades.to_parquet(evidence_dir / f'{label}-trades.parquet', index=False)
                aggregates.to_parquet(evidence_dir / f'{label}-aggregates.parquet', index=False)
                detail = {'date': str(date), 'ends': list(ends), 'trades_archive': raw_meta,
                          'aggregate_archive': agg_meta, 'raw_checks': raw_checks, 'aggregate_checks': agg_checks}
                details.append(detail)
                save_json(output / 'paired-reconciliation.partial.json', evidence)
                updates, checks = canonical_window(minute, five, trades, aggregates, ends)
                for interval, frame in [('1m', minute), ('5m', five)]:
                    checks['changes'][interval].to_parquet(evidence_dir / f'{label}-{interval}-changes.parquet', index=False)
                    for row in updates[interval].itertuples(index=False):
                        mask = frame.time.eq(row.time)
                        if mask.sum() != 1:
                            raise ValueError('봉 복원 대상 중복·누락')
                        frame.loc[mask, BAR_COLUMNS] = [getattr(row, name) for name in BAR_COLUMNS]
                checks.pop('changes')
                detail['checks'] = checks
                detail['evidence_sha256'] = {p.name: sha256(p) for p in sorted(evidence_dir.glob(f'{label}-*.parquet'))}
                save_json(output / 'paired-reconciliation.partial.json', evidence)
                print(f'{label}: 1분 {checks["changed_minutes"]}개·5분 {checks["changed_five_minutes"]}개 복원', flush=True)
            reference = five[(five.time >= minute.time.min()) & (five.end <= minute.end.max())]
            _, after = compare_minute_bars(minute, reference)
            if after['incomplete'] or after['mismatched']:
                raise ValueError('양방향 복원 후 집계 불일치')
            evidence['symbols'][symbol]['after'] = after
            for interval, frame in [('1m', minute), ('5m', five)]:
                path = (outputs[interval] / manifests[interval]['summary'][symbol]['klines']['file']).resolve()
                if not path.is_relative_to(outputs[interval].resolve()):
                    raise ValueError('복원 시세 출력 경로 오류')
                frame.to_parquet(path, index=False)
                manifests[interval]['summary'][symbol]['klines']['sha256'] = sha256(path)
        for interval, path in outputs.items():
            save_json(path / 'paired-reconciliation.json', evidence)
            manifests[interval]['paired_reconciliation'] = {
                'source_sha256': evidence['source_sha256'], 'evidence_file': 'paired-reconciliation.json',
                'evidence_sha256': sha256(path / 'paired-reconciliation.json')}
            save_json(path / f'manifest-{interval}.json', manifests[interval])
        return evidence
    except Exception as error:
        for interval, path in outputs.items():
            (path / f'manifest-{interval}.json').unlink(missing_ok=True)
            save_json(path / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
