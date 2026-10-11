from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.holding_stress import seal


def read(path):
    return json.loads(path.read_text())


def independent_price_page(content, request):
    first, last = pd.Timestamp(request['first_end']), pd.Timestamp(request['last_end'])
    if str(first.tz) != 'UTC' or str(last.tz) != 'UTC':
        raise ValueError('독립 가격 요청 시간대 오류')
    start, stop = first.value, last.value
    step = 300*10**9
    if start % step or stop % step or (stop-start)//step+1 != request['expected_bars']:
        raise ValueError('독립 가격 요청 범위 오류')
    values = json.loads(content)
    if not isinstance(values, list) or len(values) > request['expected_bars']:
        raise ValueError('독립 가격 응답 개수 오류')
    present, previous = {}, start-1
    for value in values:
        if not isinstance(value, dict) or value.get('symbol') != request['symbol']:
            raise ValueError('독립 가격 응답 계약 오류')
        stamp = datetime.fromisoformat(value['timestamp'].replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('독립 가격 응답 시간대 오류')
        ns = pd.Timestamp(stamp).value
        if not start <= ns <= stop or ns % step or ns <= previous:
            raise ValueError('독립 가격 응답 경계·순서 오류')
        previous = ns
        number = value['close']
        if number is not None and type(number) not in (int, float):
            raise ValueError('독립 가격 종가 유형 오류')
        valid = number is not None and math.isfinite(number) and number > 0
        present[ns] = (float(number) if valid else float('nan'), 'available' if valid else 'invalid_close')
    return [(request['symbol'], pd.Timestamp(ns, tz='UTC'), *present.get(ns, (float('nan'), 'missing_response')))
        for ns in range(start, stop+step, step)]


def verify_price_collection(root, expected, output):
    complete = root/'collection_final_verification.json'
    if root.is_symlink() or sha256(complete) != expected:
        raise ValueError('공동 계약 수집 완료 호출자 지문 오류')
    final = read(complete)
    preparation = read(root/'price_preparation.json')
    manifest = read(root/'collected/manifest.json')
    requests = read(root/'requests.json')
    if (preparation['complete'] is not True or preparation['independent_inventory_all_queries_exact'] is not True
        or final['complete'] is not True or final['valuation_performed'] is not False
        or final['collection_manifest_sha256'] != sha256(root/'collected/manifest.json')
        or final['prices_sha256'] != sha256(root/'all_prices.parquet')
        or manifest['requests_sha256'] != sha256(root/'collected/requests.json')
        or requests != read(root/'collected/requests.json')
        or len(requests) != len(manifest['responses']) or len(requests) != preparation['api_requests']):
        raise ValueError('공동 계약 수집 완료·범위 연결 오류')
    for name, digest in preparation['prepared_sha256'].items():
        if Path(name).name != name or (root/name).is_symlink() or sha256(root/name) != digest:
            raise ValueError('공동 계약 가격 준비 근거 변경')
    reconstructed = []
    for request, record in zip(requests, manifest['responses'], strict=True):
        target = root/'collected/responses'/record['file']
        if Path(record['file']).name != record['file'] or target.is_symlink() or sha256(target) != record['sha256']:
            raise ValueError('공동 계약 가격 응답 지문 오류')
        with gzip.open(target, 'rb') as handle:
            payload = handle.read(10*1024*1024+1)
        if len(payload) > 10*1024*1024:
            raise ValueError('공동 계약 응답 독립 읽기 상한 초과')
        envelope = json.loads(payload)
        address = urlsplit(envelope['url'])
        query = parse_qs(address.query, strict_parsing=True)
        if (address.scheme != 'https' or address.netloc != 'www.bitmex.com'
            or address.path != '/api/v1/trade/bucketed' or address.fragment
            or envelope['url'] != record['url']
            or query != {'symbol': [request['symbol']], 'binSize': ['5m'], 'partial': ['false'],
                'count': ['1000'], 'reverse': ['false'], 'startTime': [request['first_end']],
                'endTime': [request['last_end']]}):
            raise ValueError('공동 계약 독립 요청 주소·범위 오류')
        content = envelope['response_text'].encode('utf-8')
        digest = hashlib.sha256(content).hexdigest()
        if len(content) > 5*1024*1024 or digest != record['response_sha256'] or digest != envelope['response_sha256']:
            raise ValueError('공동 계약 응답 원문 변경')
        reconstructed.extend(independent_price_page(content, request))
    new = pd.DataFrame(reconstructed, columns=['symbol', 'end', 'close', 'support'])
    new['end'] = new.end.astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(new, pd.read_parquet(root/'collected/prices.parquet'), check_exact=True)
    if sha256(root/'collected/prices.parquet') != manifest['prices_sha256']:
        raise ValueError('공동 계약 가격 파일 변경')
    previous = read(root/'retained_xbt_reference.json')
    if (sha256(Path(previous['manifest'])) != previous['manifest_sha256']
        or sha256(Path(previous['file'])) != previous['sha256']):
        raise ValueError('기존 BTC 원거래소 가격 변경')
    need = pd.read_parquet(root/'needed.parquet')
    raw = pd.read_parquet(previous['file'], columns=['end', 'close'])
    raw['end'] = raw.end.astype('datetime64[ns, UTC]')
    if raw.end.duplicated().any():
        raise ValueError('기존 BTC 원거래소 가격 중복')
    old = need.loc[need.symbol.eq('XBTUSD')].merge(raw, on='end', how='left', validate='one_to_one')
    if not np.isfinite(old.close).all() or not old.close.gt(0).all():
        raise ValueError('기존 BTC 원거래소 가격 누락')
    old['support'] = 'available'
    pd.testing.assert_frame_equal(old.reset_index(drop=True), pd.read_parquet(root/'retained_xbt_prices.parquet'), check_exact=True)
    all_rows = pd.concat([old, new], ignore_index=True).sort_values(['symbol', 'end']).reset_index(drop=True)
    pd.testing.assert_frame_equal(all_rows, pd.read_parquet(root/'all_prices.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(all_rows[['symbol', 'end']], need, check_exact=True)
    counts = {k: int(v) for k, v in all_rows.support.value_counts().items()}
    if counts != final['counts'] or sum(counts.values()) != preparation['required_symbol_boundaries']:
        raise ValueError('공동 계약 독립 전체 지원 분모 오류')
    out = new_run(output, 'portfolio-price-review', {'source': str(root), 'source_final_sha256': expected,
        'reviewer_sha256': sha256(Path(__file__)), 'valuation_performed': False})
    (out/'reviewer.py').write_bytes(Path(__file__).read_bytes())
    save_json(out/'verification.json', {'complete': True, 'independent_raw_responses_all_prices_exact': True,
        'independent_retained_btc_prices_exact': True, 'all_required_boundaries_preserved': True,
        'responses': len(requests), 'counts': counts,
        'by_symbol': all_rows.groupby(['symbol', 'support']).size().unstack(fill_value=0).to_dict(orient='index'),
        'mark_prices_reconstructed': False, 'valuation_performed': False, 'trading_returns_evaluated': False})
    seal(out)
    return out
