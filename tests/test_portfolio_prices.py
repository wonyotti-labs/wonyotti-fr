import gzip
import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
from test_collateral_direction import events

from wonyotti_fr.portfolio_prices import (
    collect_prices,
    parse_price_response,
    price_requests,
    request_url,
    required_prices,
    response_cache,
    retained_xbt_prices,
)


def spec(symbol='ETHUSD', start='2020-01-01T00:05:00+00:00', count=3):
    return {'symbol': symbol, 'first_end': start,
        'last_end': (pd.Timestamp(start)+pd.Timedelta(minutes=5*(count-1))).isoformat(),
        'expected_bars': count}


def response(s):
    return [{'timestamp': str(t), 'symbol': s['symbol'], 'close': 100.}
        for t in pd.date_range(s['first_end'], s['last_end'], freq='5min')]


def test_request_chunks_cover_only_needed_grid():
    times = pd.date_range('2020-01-01', periods=2021, freq='5min', tz='UTC')
    needed = pd.DataFrame({'symbol': 'ETHUSD', 'end': times.delete([1001, 2010])})
    requests = price_requests(needed)
    assert max(r['expected_bars'] for r in requests) <= 1000
    emitted = pd.DatetimeIndex([t for r in requests for t in pd.date_range(r['first_end'], r['last_end'], freq='5min')])
    assert list(emitted) == needed.end.tolist()
    assert not any('api-key' in request_url(r) for r in requests)


def test_retained_xbt_unnamed_query_preserves_named_boundary():
    points = pd.date_range('2020-01-01', periods=3, freq='5min', tz='UTC')
    bars = pd.DataFrame({'end': points, 'close': [100., 101., 102.]})
    out = retained_xbt_prices(bars, points)
    assert list(out.columns) == ['symbol', 'end', 'close', 'support']
    assert out.end.tolist() == list(points) and out.close.tolist() == [100., 101., 102.]


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'negative', 'nan'])
def test_retained_xbt_invalid_input_rejected(bad):
    points = pd.date_range('2020-01-01', periods=3, freq='5min', tz='UTC')
    bars = pd.DataFrame({'end': points, 'close': [100., 101., 102.]})
    if bad == 'duplicate':
        bars.loc[1, 'end'] = bars.end.iloc[0]
    elif bad == 'missing':
        bars = bars.iloc[1:]
    else:
        bars.loc[1, 'close'] = -1 if bad == 'negative' else np.nan
    with pytest.raises(ValueError):
        retained_xbt_prices(bars, points)


def test_actual_inventory_strict_before_boundary_and_future_invariance():
    e = events([('XBTUSD', 'Trade', 'Buy', 1, -1000, 0),
        ('ETHUSD', 'Trade', 'Buy', 1, 1000, 0), ('ETHUSD', 'Trade', 'Sell', 1, -1100, 0)])
    e['time'] = pd.date_range('2020-01-02', periods=3, freq='5min', tz='UTC')
    points = pd.date_range('2020-01-02', periods=4, freq='5min', tz='UTC')
    got = required_prices(e, points)
    assert got[got.symbol.eq('ETHUSD')].end.tolist() == [points[2]]
    assert got[got.symbol.eq('XBTUSD')].end.tolist() == list(points)
    later = e.iloc[[1]].copy()
    later['time'] = points[-1]+pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(got, required_prices(pd.concat([e, later], ignore_index=True), points))


def test_historical_underscore_symbol_and_closed_contract_do_not_block_scope():
    e = events([('XBT7D_U110', 'Trade', 'Buy', 1, 1000, 0),
        ('XBT7D_U110', 'Settlement', 'Sell', 1, -1100, 0),
        ('XBTUSD', 'Trade', 'Buy', 1, -1000, 0)])
    e['time'] = pd.date_range('2020-01-02', periods=3, freq='5min', tz='UTC')
    points = pd.date_range('2020-01-02 00:10', periods=2, freq='5min', tz='UTC')
    got = required_prices(e, points)
    assert set(got.symbol) == {'XBTUSD'}


def test_underscore_symbol_remains_literal_public_query_parameter():
    params = parse_qs(urlparse(request_url(spec('XBT7D_U110'))).query)
    assert params['symbol'] == ['XBT7D_U110'] and 'filter' not in params


@pytest.mark.parametrize('bad', ['symbol', 'count', 'grid', 'range', 'timezone', 'extra_key'])
def test_unsafe_or_inconsistent_request_rejected(bad):
    s = spec()
    if bad == 'symbol':
        s['symbol'] = '../ETH'
    elif bad == 'count':
        s['expected_bars'] = 1001
    elif bad == 'grid':
        s['first_end'] = '2020-01-01T00:05:01+00:00'
    elif bad == 'range':
        s['last_end'] = '2020-01-01T00:25:00+00:00'
    elif bad == 'timezone':
        s['first_end'] = '2020-01-01T00:05:00'
    else:
        s['endpoint'] = 'order'
    with pytest.raises(ValueError):
        request_url(s)


@pytest.mark.parametrize('bad', ['symbol', 'duplicate', 'future', 'past', 'order', 'dict', 'excess'])
def test_invalid_response_rejected(bad):
    s = spec()
    value = response(s)
    if bad == 'symbol':
        value[0]['symbol'] = 'XBTUSD'
    elif bad == 'duplicate':
        value[1] = value[0]
    elif bad == 'future':
        value[-1]['timestamp'] = '2020-01-01T00:20:00+00:00'
    elif bad == 'past':
        value[0]['timestamp'] = '2020-01-01T00:00:00+00:00'
    elif bad == 'order':
        value = value[::-1]
    elif bad == 'dict':
        value = {'error': 'unavailable'}
    else:
        value.append(value[0])
    with pytest.raises(ValueError):
        parse_price_response(json.dumps(value).encode(), s)


def test_missing_and_invalid_closes_preserved_without_future_fill():
    s = spec(count=5)
    value = response(s)
    value[0]['close'] = None
    value[1]['close'] = -1
    value[2]['close'] = float('inf')
    del value[3]
    got = parse_price_response(json.dumps(value).encode(), s)
    assert got.support.tolist() == ['invalid_close']*3+['missing_response', 'available']
    assert got.close.iloc[:4].isna().all()
    empty = parse_price_response(b'[]', s)
    assert len(empty) == 5 and empty.support.eq('missing_response').all()


def test_exact_request_cache_roundtrip_and_tamper(tmp_path):
    url = request_url(spec())
    body = json.dumps(response(spec())).encode()
    calls = []
    def getter(address, limit):
        calls.append(address)
        return body
    first, record = response_cache(url, tmp_path, getter)
    second, reused = response_cache(url, tmp_path, getter)
    assert first == second == body and len(calls) == 1 and reused['from_cache']
    target = tmp_path/record['file']
    payload = json.loads(gzip.decompress(target.read_bytes()))
    payload['response_text'] += ' '
    target.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match='내용 불일치'):
        response_cache(url, tmp_path, getter)


def test_network_interruption_resume_reuses_only_complete_responses(tmp_path):
    requests = [spec(), spec('XRPUSD')]
    calls, fail, sleeps = [], [True], []
    def getter(url, limit):
        calls.append(url)
        symbol = parse_qs(urlparse(url).query)['symbol'][0]
        if symbol == 'XRPUSD' and fail[0]:
            raise ConnectionError('합성 연결 중단')
        return json.dumps(response(spec(symbol))).encode()
    with pytest.raises(ConnectionError):
        collect_prices(requests, tmp_path, getter, sleeps.append)
    assert not (tmp_path/'manifest.json').exists()
    fail[0] = False
    result = collect_prices(requests, tmp_path, getter, sleeps.append)
    assert len(calls) == 3 and sleeps == [2.1, 2.1]
    assert result['counts'] == {'available': 6}
    assert collect_prices(requests, tmp_path, getter, sleeps.append) == result
    assert len(calls) == 3
    (tmp_path/'prices.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='완료 가격 변경'):
        collect_prices(requests, tmp_path, getter, sleeps.append)


def test_resume_rejects_changed_scope(tmp_path):
    (tmp_path/'requests.json').write_text(json.dumps([spec()]))
    with pytest.raises(ValueError, match='재개 요청 변경'):
        collect_prices([spec('XRPUSD')], tmp_path)


def test_cache_symlink_rejected(tmp_path):
    link = tmp_path/'link'
    real = tmp_path/'real'
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='심볼릭'):
        response_cache(request_url(spec()), link)


@pytest.mark.parametrize('bad', ['float_qty', 'unsigned_overflow', 'negative_qty', 'off_grid'])
def test_inventory_invalid_values_rejected(bad):
    e = events()
    points = pd.date_range('2020-01-02', periods=2, freq='5min', tz='UTC')
    if bad == 'float_qty':
        e['lastqty'] = e.lastqty.astype(float)+.1
    elif bad == 'unsigned_overflow':
        e['lastqty'] = np.full(len(e), np.iinfo(np.uint64).max, dtype=np.uint64)
    elif bad == 'negative_qty':
        e.loc[0, 'lastqty'] = -1
    else:
        points += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        required_prices(e, points)


@pytest.mark.parametrize('value', [True, '1.5', {}, []])
def test_production_parser_rejects_non_numeric_close(value):
    rows = response(spec())
    rows[0]['close'] = value
    with pytest.raises(ValueError, match='유형'):
        parse_price_response(json.dumps(rows).encode(), spec())


def test_production_parser_rejects_naive_timestamp():
    rows = response(spec())
    rows[0]['timestamp'] = '2020-01-01T00:05:00'
    with pytest.raises(ValueError, match='시간대'):
        parse_price_response(json.dumps(rows).encode(), spec())


@pytest.mark.parametrize('name', ['requests.json', 'manifest.json', 'prices.parquet', 'progress.json', 'responses'])
def test_collection_rejects_linked_output_before_network(tmp_path, name):
    outside = tmp_path/'outside'
    outside.write_bytes(b'unchanged')
    output = tmp_path/'output'
    output.mkdir()
    (output/name).symlink_to(outside)
    calls = []
    with pytest.raises(ValueError, match='심볼릭'):
        collect_prices([spec()], output, lambda *args: calls.append(args))
    assert calls == [] and outside.read_bytes() == b'unchanged'


def test_completed_collection_rejects_linked_response(tmp_path):
    output = tmp_path/'output'
    manifest = collect_prices([spec()], output, lambda *a: json.dumps(response(spec())).encode(), lambda _: None)
    record = output/'responses'/manifest['responses'][0]['file']
    external = tmp_path/'external.gz'
    external.write_bytes(record.read_bytes())
    record.unlink()
    record.symlink_to(external)
    with pytest.raises(ValueError, match='완료 응답'):
        collect_prices([spec()], output)


def test_collection_command_checks_request_hash_before_network(tmp_path, monkeypatch):
    import wonyotti_fr.portfolio_prices as module
    from wonyotti_fr.common import sha256
    request_file = tmp_path/'requested.json'
    request_file.write_text(json.dumps([spec()]))
    calls = []
    monkeypatch.setattr(module, 'collect_prices', lambda requests, output: calls.append((requests, output)))
    with pytest.raises(ValueError, match='지문'):
        module.collect_price_file(request_file, '0'*64, tmp_path/'output')
    assert not calls
    module.collect_price_file(request_file, sha256(request_file), tmp_path/'output')
    assert calls == [([spec()], tmp_path/'output')]
