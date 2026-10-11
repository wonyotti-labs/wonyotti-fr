from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.market import safe_get

STEP = pd.Timedelta(minutes=5)
MAX_BYTES = 5*1024*1024


def retained_xbt_prices(bars, points):
    bars = bars.copy()
    if str(bars.end.dt.tz) != 'UTC' or bars.end.isna().any() or bars.end.duplicated().any():
        raise ValueError('기존 BTC 가격의 시간·중복 오류')
    bars['end'] = bars.end.astype('datetime64[ns, UTC]')
    queries = pd.DatetimeIndex(points).as_unit('ns')
    if str(queries.tz) != 'UTC' or queries.hasnans or not queries.is_unique:
        raise ValueError('기존 BTC 가격의 조회 시각 오류')
    if not queries.isin(bars.end).all():
        raise ValueError('기존 BTC 가격의 지원 경계 누락')
    selected = bars.set_index('end').loc[queries, ['close']].rename_axis('end').reset_index()
    if not np.isfinite(selected.close).all() or not selected.close.gt(0).all():
        raise ValueError('기존 BTC 가격의 양수 종가 오류')
    selected['symbol'], selected['support'] = 'XBTUSD', 'available'
    return selected[['symbol', 'end', 'close', 'support']]


def required_prices(events, points):
    if (not events.time.is_monotonic_increasing or events.time.isna().any()
        or not events.settlcurrency.eq('XBt').all()
        or not events.exectype.isin(['Trade', 'Settlement', 'Funding']).all()):
        raise ValueError('공동 계약 가격의 원본 순서·통화 오류')
    queries = pd.DatetimeIndex(points).as_unit('ns')
    if (str(queries.tz) != 'UTC' or queries.hasnans or not queries.is_unique
        or not queries.is_monotonic_increasing or np.any(queries.asi8 % STEP.value)):
        raise ValueError('공동 계약 가격의 조회 경계 오류')
    result = []
    for symbol, rows in events.groupby('symbol', sort=True):
        if not re.fullmatch('[A-Z0-9_]{3,20}', symbol):
            raise ValueError('원본 계약 심볼 형식 오류')
        traded = rows.exectype.ne('Funding')
        if not rows.loc[traded, 'side'].isin(['Buy', 'Sell']).all():
            raise ValueError('원본 계약 방향 오류')
        qty = rows.lastqty.to_numpy()
        if qty.dtype.kind not in 'iu' or (qty < 0).any() or sum(abs(int(v)) for v in qty) > np.iinfo(np.int64).max:
            raise ValueError('원본 계약 수량 범위 오류')
        qty = qty.astype(np.int64)
        change = np.where(traded, np.where(rows.side.eq('Buy'), qty, -qty), 0)
        selected = np.searchsorted(rows.time.astype('datetime64[ns, UTC]').array.asi8, queries.asi8, side='left')
        positions = np.r_[0, change.cumsum()][selected]
        if np.any(positions):
            result.append(pd.DataFrame({'symbol': symbol, 'end': queries[positions != 0]}))
    if not result:
        raise ValueError('공동 계약 가격의 지원 요청 없음')
    # 담보 환산 가격은 XBTUSD 계약을 보유하지 않은 경계에도 필요하다.
    result.append(pd.DataFrame({'symbol': 'XBTUSD', 'end': queries}))
    return pd.concat(result, ignore_index=True).drop_duplicates().sort_values(['symbol', 'end']).reset_index(drop=True)


def price_requests(needed):
    if (set(needed.columns) != {'symbol', 'end'} or needed.duplicated().any()
        or needed.end.isna().any() or str(needed.end.dt.tz) != 'UTC'):
        raise ValueError('공동 계약 가격의 요청 원장 오류')
    result = []
    for symbol, group in needed.groupby('symbol', sort=True):
        if symbol == 'XBTUSD':
            continue
        if not re.fullmatch('[A-Z0-9_]{3,20}', symbol):
            raise ValueError('공동 계약 가격의 요청 심볼 오류')
        ends = pd.DatetimeIndex(group.end).as_unit('ns')
        if not ends.is_monotonic_increasing or np.any(ends.asi8 % STEP.value):
            raise ValueError('공동 계약 가격의 요청 시간 오류')
        breaks = np.r_[0, np.flatnonzero(np.diff(ends.asi8) != STEP.value)+1, len(ends)]
        for first, stop in zip(breaks[:-1], breaks[1:], strict=True):
            for start in range(first, stop, 1000):
                chunk = ends[start:min(start+1000, stop)]
                result.append({'symbol': symbol, 'first_end': chunk[0].isoformat(),
                    'last_end': chunk[-1].isoformat(), 'expected_bars': len(chunk)})
    return result


def request_url(spec):
    if (set(spec) != {'symbol', 'first_end', 'last_end', 'expected_bars'}
        or not re.fullmatch('[A-Z0-9_]{3,20}', spec['symbol'])
        or type(spec['expected_bars']) is not int or not 1 <= spec['expected_bars'] <= 1000):
        raise ValueError('공동 계약 가격 요청 설정 오류')
    first, last = (pd.Timestamp(spec[k]) for k in ['first_end', 'last_end'])
    if (str(first.tz) != 'UTC' or str(last.tz) != 'UTC' or first > last
        or first.value % STEP.value or last.value % STEP.value
        or last-first != STEP*(spec['expected_bars']-1)):
        raise ValueError('공동 계약 가격 요청 범위 오류')
    return 'https://www.bitmex.com/api/v1/trade/bucketed?'+urlencode({
        'symbol': spec['symbol'], 'binSize': '5m', 'partial': 'false', 'count': 1000,
        'reverse': 'false', 'startTime': first.isoformat(), 'endTime': last.isoformat()})


def response_cache(url, cache, getter=safe_get):
    if cache.is_symlink():
        raise ValueError('공동 계약 가격 캐시 폴더의 심볼릭 링크 오류')
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache/(key+'.json.gz')
    if path.is_symlink():
        raise ValueError('공동 계약 가격 캐시의 심볼릭 링크 오류')
    if path.exists():
        with gzip.open(path, 'rb') as handle:
            payload = handle.read(2*MAX_BYTES+1)
        if len(payload) > 2*MAX_BYTES:
            raise ValueError('공동 계약 가격 캐시 압축 해제 상한 초과')
        envelope = json.loads(payload)
        content = envelope['response_text'].encode('utf-8')
        if (envelope['url'] != url or len(content) > MAX_BYTES
            or hashlib.sha256(content).hexdigest() != envelope['response_sha256']):
            raise ValueError('공동 계약 가격 캐시 출처·내용 불일치')
        return content, {'file': path.name, 'sha256': sha256(path), 'url': url,
            'response_sha256': envelope['response_sha256'], 'from_cache': True}
    content = getter(url, MAX_BYTES)
    if not isinstance(content, bytes) or len(content) > MAX_BYTES:
        raise ValueError('공동 계약 가격 응답 용량 오류')
    envelope = {'url': url, 'response_text': content.decode('utf-8'),
        'response_sha256': hashlib.sha256(content).hexdigest(), 'received_utc': datetime.now(UTC).isoformat(),
        'hash_type': 'local_sha256_not_provider_signature'}
    temporary = path.with_suffix('.part')
    if temporary.is_symlink():
        raise ValueError('공동 계약 가격 임시 파일의 심볼릭 링크 오류')
    payload = json.dumps(envelope, ensure_ascii=False).encode('utf-8')
    if len(payload) > 2*MAX_BYTES:
        raise ValueError('공동 계약 가격 저장 상한 초과')
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)
    return content, {'file': path.name, 'sha256': sha256(path), 'url': url,
        'response_sha256': envelope['response_sha256'], 'from_cache': False}


def parse_price_response(content, spec):
    request_url(spec)
    data = json.loads(content)
    if not isinstance(data, list) or len(data) > spec['expected_bars'] or any(not isinstance(r, dict) for r in data):
        raise ValueError('공동 계약 가격 응답 목록 오류')
    expected = pd.date_range(spec['first_end'], spec['last_end'], freq='5min').as_unit('ns')
    if not data:
        return pd.DataFrame({'symbol': spec['symbol'], 'end': expected,
            'close': np.full(len(expected), np.nan), 'support': 'missing_response'})
    for row in data:
        if 'timestamp' not in row or 'close' not in row:
            raise ValueError('공동 계약 가격 응답 필수 열 오류')
        stamp = pd.Timestamp(row['timestamp'])
        if stamp.tz is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('공동 계약 가격 응답 시간대 오류')
        if row['close'] is not None and type(row['close']) not in (int, float):
            raise ValueError('공동 계약 가격 종가 유형 오류')
    frame = pd.DataFrame(data)
    if not {'timestamp', 'symbol', 'close'} <= set(frame) or not frame.symbol.eq(spec['symbol']).all():
        raise ValueError('공동 계약 가격 응답 심볼·열 오류')
    ends = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True, errors='raise')).as_unit('ns')
    if (ends.hasnans or not ends.is_unique or not ends.is_monotonic_increasing
        or not ends.isin(expected).all()):
        raise ValueError('공동 계약 가격 응답 경계·순서 오류')
    values = pd.to_numeric(frame.close, errors='raise').to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    parsed = pd.DataFrame({'end': ends, 'close': values,
        'support': np.where(valid, 'available', 'invalid_close')}).set_index('end')
    parsed.loc[~valid, 'close'] = np.nan
    parsed = parsed.reindex(expected)
    parsed['support'] = parsed.support.fillna('missing_response')
    parsed['symbol'] = spec['symbol']
    return parsed.rename_axis('end').reset_index()[['symbol', 'end', 'close', 'support']]


def collect_prices(requests, output, getter=safe_get, sleeper=time.sleep):
    if not isinstance(requests, list) or not requests or output.is_symlink():
        raise ValueError('공동 계약 수집 요청·출력 오류')
    for request in requests:
        request_url(request)
    for name in ['requests.json', 'manifest.json', 'prices.parquet', 'progress.json', 'responses']:
        if (output/name).is_symlink():
            raise ValueError('공동 계약 수집 파일의 심볼릭 링크 오류')
    output.mkdir(parents=True, exist_ok=True)
    request_path = output/'requests.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != requests:
            raise ValueError('공동 계약 가격 재개 요청 변경')
    else:
        save_json(request_path, requests)
    if (output/'manifest.json').exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if (manifest['requests_sha256'] != sha256(request_path) or len(manifest['responses']) != len(requests)
            or sha256(output/'prices.parquet') != manifest['prices_sha256']):
            raise ValueError('공동 계약 완료 가격 변경')
        for index, row in enumerate(manifest['responses']):
            if (row['url'] != request_url(requests[index]) or Path(row['file']).name != row['file']
                or (output/'responses'/row['file']).is_symlink()
                or sha256(output/'responses'/row['file']) != row['sha256']):
                raise ValueError('공동 계약 완료 응답 변경')
        return manifest
    frames, responses = [], []
    for index, spec in enumerate(requests):
        url = request_url(spec)
        content, record = response_cache(url, output/'responses', getter)
        frame = parse_price_response(content, spec)
        frames.append(frame)
        responses.append(record)
        if not record['from_cache']:
            sleeper(2.1)
        if (index+1) % 10 == 0 or index+1 == len(requests):
            save_json(output/'progress.json', {'completed_requests': index+1, 'total_requests': len(requests),
                'latest_symbol': spec['symbol'], 'latest_end': spec['last_end']})
            print(f'공동 계약 시세 {index+1}/{len(requests)}: {spec["symbol"]} {spec["last_end"]}', flush=True)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['symbol', 'end']).any():
        raise ValueError('공동 계약 가격의 중복 수집')
    result.to_parquet(output/'prices.parquet', index=False)
    manifest = {'complete': True, 'requests_sha256': sha256(request_path),
        'prices_sha256': sha256(output/'prices.parquet'), 'responses': responses,
        'counts': {k: int(v) for k, v in result.support.value_counts().items()}, 'rows': len(result),
        'mark_prices_reconstructed': False, 'trading_returns_evaluated': False,
        'source': 'BitMEX public 5m trade closing prices', 'maximum_staleness_exclusive_minutes': 5}
    save_json(output/'manifest.json', manifest)
    return manifest


def collect_price_file(request_path, expected, output):
    if request_path.is_symlink() or sha256(request_path) != expected:
        raise ValueError('공동 계약 가격 요청 파일 지문 오류')
    requests = json.loads(request_path.read_text())
    if sha256(request_path) != expected:
        raise ValueError('공동 계약 가격 요청 읽기 중 변경')
    return collect_prices(requests, output)
