from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import records
from .event_features import MARKET_FEATURES, event_features
from .minute_inputs import MINUTE_FEATURES, minute_features
from .research import check_funding_coverage, load_market


def validate_minutes(frame: pd.DataFrame, minutes: int) -> None:
    step = pd.Timedelta(minutes=minutes)
    prices = frame[['open', 'high', 'low', 'close']]
    if (frame.time.duplicated().any() or not frame.time.is_monotonic_increasing
        or not frame.end.sub(frame.time).eq(step).all()
        or (frame.time.astype('datetime64[ns, UTC]').array.asi8 % step.value).any()
        or not np.isfinite(prices).all().all() or prices.le(0).any().any()
        or frame.high.lt(prices.max(axis=1)).any() or frame.low.gt(prices.min(axis=1)).any()
        or not np.isfinite(frame[['volume', 'count']]).all().all()
        or frame[['volume', 'count']].lt(0).any().any() or frame['count'].mod(1).ne(0).any()
        or (frame['count'].eq(0) & (frame.volume.ne(0) | prices.max(axis=1).ne(prices.min(axis=1)))).any()):
        raise ValueError('실행 시세의 시간·OHLC·거래량 오류')


def aggregate_minutes(minute: pd.DataFrame) -> pd.DataFrame:
    grouped = minute.groupby(minute.end.dt.ceil('5min')).agg(
        minutes=('close', 'size'), open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'), count=('count', 'sum'))
    active = minute[minute['count'].gt(0)]
    actual = active.groupby(active.end.dt.ceil('5min')).agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
    # 무거래 분봉의 이월 가격을 이후 첫 실제 체결이나 고가·저가로 취급하지 않는다.
    grouped.loc[actual.index, actual.columns] = actual
    return grouped.reset_index()


def compare_minute_bars(minute: pd.DataFrame, five: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_minutes(minute, 1)
    validate_minutes(five, 5)
    grouped = aggregate_minutes(minute)
    names = ['open', 'high', 'low', 'close', 'volume', 'count']
    joined = grouped.merge(five[['end', *names]], on='end', how='outer', suffixes=('_minute', '_five'),
                           validate='one_to_one', indicator=True)
    joined['complete'] = joined._merge.eq('both') & joined.minutes.eq(5)
    for name in names:
        tolerance = 0 if name == 'count' else (1e-7 if name == 'volume' else 1e-8)
        joined[f'{name}_match'] = np.isclose(joined[f'{name}_minute'], joined[f'{name}_five'], rtol=0, atol=tolerance)
    joined['matched'] = joined.complete & joined[[f'{name}_match' for name in names]].all(axis=1)
    return joined, {'buckets': len(joined), 'matched': int(joined.matched.sum()),
                    'incomplete': int((~joined.complete).sum()), 'mismatched': int((joined.complete & ~joined.matched).sum()),
                    'mismatch_sample': records(joined[~joined.matched].head(20))}


def attach_confirmed_features(minute: pd.DataFrame, five: pd.DataFrame) -> pd.DataFrame:
    features = event_features(five).rename(columns={'end': 'feature_end'})
    left = minute.copy().astype({'time': 'datetime64[ns, UTC]', 'end': 'datetime64[ns, UTC]'})
    # 5분 특징은 해당 봉의 종료 이후에만 1분 실행에 제공한다.
    return pd.merge_asof(left, features[['feature_end', *MARKET_FEATURES]], left_on='end', right_on='feature_end',
                          direction='backward', tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))


def normalized_funding(frame: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    aligned = frame.time.dt.floor('1min')
    offset = (frame.time - aligned).dt.total_seconds()
    if ((offset < 0) | (offset >= 1)).any() or not np.isfinite(frame.rate).all() or frame.rate.abs().gt(1).any():
        raise ValueError('분봉 펀딩 시각·값 오류')
    selected = frame.assign(time=aligned)
    selected = selected[(selected.time >= first) & (selected.time < last)][['time', 'rate']].reset_index(drop=True)
    if selected.time.duplicated().any() or not selected.time.is_monotonic_increasing:
        raise ValueError('분봉 펀딩 중복·시간순 오류')
    return selected


def prepare_minute_period(market: Path, feature_market: Path, symbol: str, start: str, end: str,
                          minute_inputs: bool = False) -> tuple[pd.DataFrame, dict]:
    first, last = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    if (first >= last or last - first > pd.Timedelta(days=1500)
        or first.value % pd.Timedelta(minutes=5).value or last.value % pd.Timedelta(minutes=5).value):
        raise ValueError('분봉 평가 기간은 1500일 이하의 순서가 맞는 5분 경계여야 합니다.')
    minute, funding = load_market(market, symbol, '1m')
    five, five_funding = load_market(feature_market, symbol, '5m')
    check_funding_coverage(market, symbol, start, end, '1m')
    check_funding_coverage(feature_market, symbol, start, end, '5m')
    selected = minute[(minute.time >= first) & (minute.time < last)].reset_index(drop=True)
    expected = pd.date_range(first, last, freq='1min', inclusive='left').as_unit('ns')
    if not np.array_equal(selected.time.astype('datetime64[ns, UTC]').array.asi8, expected.asi8):
        raise ValueError('평가 기간의 1분 시세가 누락·중복·역순입니다.')
    reference = five[(five.time >= first) & (five.time < last)].reset_index(drop=True)
    _, comparison = compare_minute_bars(selected, reference)
    if comparison['incomplete'] or comparison['mismatched']:
        raise ValueError(f'1분 실행 시세와 5분 특징 시세 불일치: {comparison}')
    rates = normalized_funding(funding, first, last)
    previous_rates = normalized_funding(five_funding, first, last)
    if not rates.equals(previous_rates) or not rates.time.isin(selected.time).all():
        raise ValueError('두 해상도 자료의 펀딩 시각·값 불일치')
    selected['funding_rate'] = selected.time.map(rates.set_index('time').rate).fillna(0)
    result = attach_confirmed_features(selected, five)
    minute_checks = {}
    if minute_inputs:
        context = minute[(minute.time >= first - pd.Timedelta(minutes=60)) & (minute.time < last)]
        result = result.merge(minute_features(context), on='end', how='left', validate='one_to_one')
        minute_checks = {'minute_input_rule': '현재 확정 분봉과 과거 60분, 현재 거래량은 기준 평균에서 제외',
                         'minute_input_available_rows': int(np.isfinite(result[MINUTE_FEATURES]).all(axis=1).sum()),
                         'minute_input_warmup_rows': int(context.time.lt(first).sum())}
    valid = result[MARKET_FEATURES].notna().all(axis=1)
    known = result.feature_end.notna()
    if (result.loc[known, 'feature_end'] > result.loc[known, 'end']).any():
        raise ValueError('미확정 특징의 실행 입력 유입')
    return result, {'aggregate_comparison': comparison, 'minute_rows': len(result), 'funding_rows': len(rates),
                    'valid_feature_rows': int(valid.sum()), 'unavailable_feature_rows': int((~valid).sum()),
                    'feature_rule': '가장 최근 확정 5분 특징, 미래 봉 및 시간 길이 변경 없음', **minute_checks}
