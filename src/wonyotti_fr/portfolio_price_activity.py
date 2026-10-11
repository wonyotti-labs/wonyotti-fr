from __future__ import annotations

import math

import pandas as pd


def activity_record(symbol, end, trades, volume):
    stamp = pd.Timestamp(end)
    if stamp.tz is None or str(stamp.tz) != 'UTC' or stamp.value % (300*10**9):
        raise ValueError('가격 활동 봉 종료 경계 오류')
    for number in (trades, volume):
        if number is not None and (type(number) not in (int, float) or not math.isfinite(number)
            or number < 0 or number != int(number)):
            raise ValueError('가격 활동 거래 수·수량 오류')
    reason = 'missing_activity' if trades is None or volume is None else (
        'no_observed_trade' if trades == 0 or volume == 0 else 'observed_trade')
    return {'symbol': symbol, 'end': stamp, 'trades': trades, 'contract_volume': volume,
        'activity': reason, 'observed_trade': reason == 'observed_trade'}


def boundary_activity(needed, records):
    if needed[['symbol', 'end']].duplicated().any() or records[['symbol', 'end']].duplicated().any():
        raise ValueError('가격 활동 계약·경계 중복')
    joined = needed.merge(records, on=['symbol', 'end'], how='left', validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all() or len(joined) != len(records):
        raise ValueError('가격 활동 전체 필요 범위 불일치')
    keys = ['symbol', 'end', 'trades', 'contract_volume', 'activity', 'observed_trade']
    details = joined[keys].sort_values(['symbol', 'end']).reset_index(drop=True)
    counts = details.groupby('end').observed_trade.agg(['all', 'size', 'sum']).rename(
        columns={'all': 'all_prices_have_observed_trade', 'size': 'required_prices', 'sum': 'observed_prices'})
    counts['unobserved_prices'] = counts.required_prices-counts.observed_prices
    return details, counts.reset_index()
