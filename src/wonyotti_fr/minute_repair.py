from __future__ import annotations

import copy
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_verification import selected_aggregate_trades, verify_aggregate_coverage
from .common import save_json, sha256
from .market import archive_csv, verified_archive
from .minute_data import compare_minute_bars, validate_minutes
from .research import load_market

TRADE_COLUMNS = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
BAR_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote']


class UnverifiedTradeGap(ValueError):
    pass


def selected_archive_trades(content: bytes, date: pd.Timestamp, ends: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    if (date.tzinfo is None or date != date.floor('1D') or len(ends) == 0 or len(ends) > 288
        or ((ends - pd.Timedelta(nanoseconds=1)).floor('1D') != date).any()
        or (ends.as_unit('ns').asi8 % pd.Timedelta(minutes=5).value).any()):
        raise ValueError('체결 대조의 날짜·구간 경계 오류')
    header = content[:min(content.find(b'\n') + 1, 4096)].split(b',', 1)[0]
    skip = int(header in (b'id', b'tradeId'))
    targets = ends.as_unit('ms').asi8
    day_start, day_end = date.value // 10**6, (date + pd.Timedelta(days=1)).value // 10**6
    selected, previous_id, previous_time, count, gaps = [], None, None, 0, 0
    for frame in pd.read_csv(io.BytesIO(content), header=None, names=TRADE_COLUMNS, skiprows=skip, chunksize=100_000):
        numeric = frame[TRADE_COLUMNS[:-1]].apply(pd.to_numeric, errors='raise')
        if (not np.isfinite(numeric).all().all() or numeric[['id', 'time']].mod(1).ne(0).any().any()
            or numeric[['price', 'qty']].le(0).any().any() or numeric.quote_qty.lt(0).any()
            or frame.is_buyer_maker.dtype != bool or numeric.id.diff().dropna().le(0).any()
            or numeric.time.diff().dropna().lt(0).any() or numeric.time.lt(day_start).any() or numeric.time.ge(day_end).any()
            or (previous_id is not None and numeric.id.iloc[0] <= previous_id)
            or (previous_time is not None and numeric.time.iloc[0] < previous_time)):
            raise ValueError('원체결의 숫자·순서·날짜·방향 오류')
        gaps += int(numeric.id.diff().dropna().ne(1).sum())
        gaps += int(previous_id is not None and numeric.id.iloc[0] != previous_id + 1)
        previous_id, previous_time = int(numeric.id.iloc[-1]), int(numeric.time.iloc[-1])
        frame[TRADE_COLUMNS[:-1]] = numeric
        bucket_end = (numeric.time.astype('int64') // 300_000 + 1) * 300_000
        selected.append(frame.loc[bucket_end.isin(targets)].copy())
        count += len(frame)
    result = pd.concat(selected, ignore_index=True)
    if result.empty:
        raise ValueError('대조 구간의 원체결 없음')
    return result, {'archive_rows': count, 'archive_id_gaps': gaps, 'selected_rows': len(result),
                    'id_gap_note': '일별 ID 공백을 기록하며 수정할 분봉의 연속성은 별도 검사'}


def rebuilt_minutes(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame['minute'] = pd.to_datetime(frame.time.astype('int64'), unit='ms', utc=True).dt.floor('1min')
    frame['quote'] = frame.price * frame.qty
    frame['taker_base'] = frame.qty.where(~frame.is_buyer_maker, 0)
    frame['taker_quote'] = frame.quote.where(~frame.is_buyer_maker, 0)
    rebuilt = frame.groupby('minute').agg(open=('price', 'first'), high=('price', 'max'), low=('price', 'min'),
        close=('price', 'last'), volume=('qty', 'sum'), quote_volume=('quote', 'sum'), count=('id', 'size'),
        taker_buy_volume=('taker_base', 'sum'), taker_buy_quote=('taker_quote', 'sum'),
        id_gaps=('id', lambda values: int(values.diff().dropna().ne(1).sum()))).reset_index().rename(columns={'minute': 'time'})
    rebuilt['end'] = rebuilt.time + pd.Timedelta(minutes=1)
    validate_minutes(rebuilt, 1)
    return rebuilt


def reconcile_window(original: pd.DataFrame, reference: pd.DataFrame, trades: pd.DataFrame,
                     *, aggregates: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rebuilt = rebuilt_minutes(trades)
    _, check = compare_minute_bars(rebuilt, reference)
    if check['incomplete'] or check['mismatched']:
        raise ValueError('원체결 집계가 공식 5분 가격·거래량·건수와 다릅니다.')
    prior = original[original.time.isin(rebuilt.time)].reset_index(drop=True)
    if not np.array_equal(prior.time.astype('datetime64[ns, UTC]').array.asi8,
                          rebuilt.time.astype('datetime64[ns, UTC]').array.asi8):
        raise ValueError('복원 대상 분봉의 시각·순서 불일치')
    core = ['open', 'high', 'low', 'close', 'volume', 'count']
    changed = np.zeros(len(rebuilt), dtype=bool)
    for name in core:
        changed |= ~np.isclose(prior[name], rebuilt[name], rtol=0, atol=0 if name == 'count' else 1e-8)
    corroboration = None
    if rebuilt.loc[changed, 'id_gaps'].ne(0).any():
        if aggregates is None:
            raise UnverifiedTradeGap('수정할 분봉의 원체결 ID가 연속적이지 않습니다. 집계 체결 대조가 필요합니다.')
        corroboration = verify_aggregate_coverage(trades, aggregates, rebuilt.loc[changed, 'time'])
    changes = prior.loc[changed, ['time', *BAR_COLUMNS]].merge(
        rebuilt.loc[changed, ['time', *BAR_COLUMNS]], on='time', suffixes=('_original', '_rebuilt'), validate='one_to_one')
    return rebuilt.loc[changed, ['time', 'end', *BAR_COLUMNS]], changes, {
        'comparison': check, 'changed_minutes': int(changed.sum()),
        'changed_minute_id_gaps': int(rebuilt.loc[changed, 'id_gaps'].sum()), 'aggregate_corroboration': corroboration,
        'unchanged_minute_id_gaps': int(rebuilt.loc[~changed, 'id_gaps'].sum()),
        'quote_rule': '가격×수량 합계; 원체결 quote_qty의 별도 반올림 미사용'}


def repair_minute_market(source: Path, feature_market: Path, output: Path, cache: Path) -> dict:
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError('분봉 복원에는 존재하지 않는 별도 폴더가 필요합니다.')
    manifest_path = source / 'manifest-1m.json'
    manifest = copy.deepcopy(json.loads(manifest_path.read_text()))
    if manifest.get('interval') != '1m' or not 1 <= len(manifest['summary']) <= 5:
        raise ValueError('분봉 복원 매니페스트 형식 오류')
    shutil.copytree(source, output)
    (output / 'prior-manifest-1m.json').write_bytes(manifest_path.read_bytes())
    evidence = {'source_manifest_sha256': sha256(manifest_path),
                'feature_manifest_sha256': sha256(feature_market / 'manifest-5m.json'), 'symbols': {}}
    try:
        for symbol in manifest['summary']:
            if not symbol.isalnum() or len(symbol) > 20:
                raise ValueError('분봉 복원 심볼 오류')
            minute, _ = load_market(source, symbol, '1m')
            five, _ = load_market(feature_market, symbol, '5m')
            reference = five[(five.time >= minute.time.min()) & (five.end <= minute.end.max())]
            comparison, before = compare_minute_bars(minute, reference)
            if before['incomplete']:
                raise ValueError('분봉 구간 누락은 값 복원 전에 별도로 보완해야 합니다.')
            mismatches = comparison[~comparison.matched]
            dates = (mismatches.end - pd.Timedelta(nanoseconds=1)).dt.floor('1D')
            if dates.nunique() > 31:
                raise ValueError('한 번의 분봉 복원은 31일 이하여야 합니다.')
            details = []
            for date in sorted(dates.unique()):
                ends = pd.DatetimeIndex(mismatches.loc[dates.eq(date), 'end'])
                url = f'https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date:%Y-%m-%d}.zip'
                archive, archive_meta = verified_archive(url, cache)
                trades, trade_meta = selected_archive_trades(archive_csv(archive), date, ends)
                label = f'{symbol}-{date:%Y-%m-%d}'
                evidence_dir = output / 'reconciliation'
                evidence_dir.mkdir(exist_ok=True)
                trades.to_parquet(evidence_dir / f'{label}-trades.parquet', index=False)
                aggregate_source = None
                try:
                    replacements, changes, checks = reconcile_window(minute, reference[reference.end.isin(ends)], trades)
                except UnverifiedTradeGap:
                    aggregate_url = f'https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date:%Y-%m-%d}.zip'
                    aggregate_path, aggregate_meta = verified_archive(aggregate_url, cache)
                    aggregates, aggregate_checks = selected_aggregate_trades(archive_csv(aggregate_path), date, trades)
                    aggregates.to_parquet(evidence_dir / f'{label}-aggregates.parquet', index=False)
                    aggregate_source = {'archive': aggregate_meta, 'checks': aggregate_checks,
                                        'subset_sha256': sha256(evidence_dir / f'{label}-aggregates.parquet')}
                    save_json(evidence_dir / f'{label}-aggregate-source.json', aggregate_source)
                    replacements, changes, checks = reconcile_window(minute, reference[reference.end.isin(ends)], trades, aggregates=aggregates)
                changes.to_parquet(evidence_dir / f'{label}-changes.parquet', index=False)
                for row in replacements.itertuples(index=False):
                    mask = minute.time.eq(row.time)
                    if mask.sum() != 1:
                        raise ValueError('분봉 복원 대상 중복·누락')
                    minute.loc[mask, BAR_COLUMNS] = [getattr(row, name) for name in BAR_COLUMNS]
                details.append({'date': str(date), 'archive': archive_meta, 'trade_checks': trade_meta, 'reconstruction': checks,
                                'aggregate_source': aggregate_source,
                                'trade_subset_sha256': sha256(evidence_dir / f'{label}-trades.parquet'),
                                'changes_sha256': sha256(evidence_dir / f'{label}-changes.parquet')})
            _, after = compare_minute_bars(minute, reference)
            if after['incomplete'] or after['mismatched']:
                raise ValueError('복원 후에도 분봉 집계 불일치가 남았습니다.')
            path = (output / manifest['summary'][symbol]['klines']['file']).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError('분봉 복원 출력 경로 오류')
            minute.to_parquet(path, index=False)
            manifest['summary'][symbol]['klines']['sha256'] = sha256(path)
            evidence['symbols'][symbol] = {'before': before, 'after': after, 'details': details}
        manifest['minute_reconciliation'] = {'evidence_file': 'reconciliation.json', 'source_manifest_sha256': sha256(manifest_path),
            'note': '원본 보존. 공식 원체결·5분 집계와 ID 연속성 또는 집계 체결 교차 검증 후 별도 파일 생성'}
        save_json(output / 'reconciliation.json', evidence)
        manifest['minute_reconciliation']['evidence_sha256'] = sha256(output / 'reconciliation.json')
        save_json(output / 'manifest-1m.json', manifest)
        return evidence
    except Exception as error:
        save_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
