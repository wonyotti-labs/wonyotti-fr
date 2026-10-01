from __future__ import annotations

import io

import numpy as np
import pandas as pd

AGGREGATE_COLUMNS = ['aggregate_id', 'price', 'qty', 'first_id', 'last_id', 'time', 'is_buyer_maker']


def selected_aggregate_trades(content: bytes, date: pd.Timestamp, trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    header = content[:min(content.find(b'\n') + 1, 4096)].split(b',', 1)[0]
    skip = int(header in (b'agg_trade_id', b'aggregate_trade_id', b'aggregate_id'))
    first, last = int(trades.id.min()), int(trades.id.max())
    day_start, day_end = date.value // 10**6, (date + pd.Timedelta(days=1)).value // 10**6
    chunks, previous, rows = [], None, 0
    for frame in pd.read_csv(io.BytesIO(content), header=None, names=AGGREGATE_COLUMNS, skiprows=skip, chunksize=100_000):
        numeric = frame[AGGREGATE_COLUMNS[:-1]].apply(pd.to_numeric, errors='raise')
        if (not np.isfinite(numeric).all().all() or numeric[['aggregate_id', 'first_id', 'last_id', 'time']].mod(1).ne(0).any().any()
            or numeric[['price', 'qty']].le(0).any().any() or numeric.first_id.gt(numeric.last_id).any()
            or numeric.aggregate_id.diff().dropna().le(0).any() or frame.is_buyer_maker.dtype != bool
            or (previous is not None and numeric.aggregate_id.iloc[0] <= previous)
            or numeric.time.lt(day_start).any() or numeric.time.ge(day_end).any()):
            raise ValueError('집계 체결의 숫자·ID·날짜·방향 오류')
        previous = int(numeric.aggregate_id.iloc[-1])
        frame[AGGREGATE_COLUMNS[:-1]] = numeric
        chunks.append(frame[numeric.last_id.ge(first) & numeric.first_id.le(last)].copy())
        rows += len(frame)
    result = pd.concat(chunks, ignore_index=True)
    if result.empty:
        raise ValueError('대조 구간의 집계 체결 없음')
    return result, {'archive_rows': rows, 'selected_rows': len(result)}


def verify_aggregate_coverage(trades: pd.DataFrame, aggregates: pd.DataFrame, minutes: pd.Series) -> dict:
    ids, quantities, prices, sides, times = (trades[name].to_numpy() for name in ['id', 'qty', 'price', 'is_buyer_maker', 'time'])
    target = pd.to_datetime(trades.time, unit='ms', utc=True).dt.floor('1min').isin(minutes).to_numpy()
    if not target.any() or (np.diff(ids) <= 0).any() or (np.diff(times) < 0).any():
        raise ValueError('집계 교차 검증의 대상·순서 오류')
    coverage = np.zeros(len(trades), dtype=np.int16)
    verified, skipped_partial, residual = 0, 0, 0.
    for row in aggregates.itertuples(index=False):
        low, high = int(np.searchsorted(ids, row.first_id)), int(np.searchsorted(ids, row.last_id, side='right'))
        if low == high or not target[low:high].any():
            continue
        if ids[low] != row.first_id or ids[high - 1] != row.last_id:
            # 원체결 범위 밖까지 걸친 집계는 증거로 쓰지 않고 최종 연결 범위 검사에서 거부한다.
            skipped_partial += 1
            continue
        error = abs(float(quantities[low:high].sum()) - row.qty)
        if (not np.isfinite(error) or error > 1e-7 or not np.all(prices[low:high] == row.price)
            or not np.all(sides[low:high] == row.is_buyer_maker) or row.time != times[low]
            or not 0 <= times[high - 1] - times[low] <= 100):
            raise ValueError('개별 체결과 집계 체결의 가격·수량·방향·시각 불일치')
        if coverage[low:high][target[low:high]].any():
            raise ValueError('수정할 분봉의 집계 체결 연결이 중복됐습니다.')
        coverage[low:high] += 1
        residual = max(residual, error)
        verified += 1
    if not np.all(coverage[target] == 1):
        raise ValueError('수정할 분봉의 개별 체결이 집계 자료에 정확히 한 번씩 연결되지 않습니다.')
    return {'target_trades': int(target.sum()), 'verified_aggregates': verified,
            'target_exactly_once': True, 'skipped_partial_edges': skipped_partial,
            'max_quantity_residual': residual, 'quantity_absolute_tolerance': 1e-7,
            'limit': '공개 시장 체결의 두 형식 대조. ID 공백의 개별 원인이나 비공개 거래의 인증이 아님'}
