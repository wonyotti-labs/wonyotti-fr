import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.minute_data import (
    attach_confirmed_features,
    compare_minute_bars,
    normalized_funding,
)


def source():
    time = pd.date_range('2020-01-01', periods=3000, freq='1min', tz='UTC').as_unit('ns')
    close = 100 + np.sin(np.arange(len(time)) / 100)
    minute = pd.DataFrame({'time': time, 'end': time + pd.Timedelta(minutes=1),
                           'open': close, 'high': close + 1, 'low': close - 1, 'close': close,
                           'volume': 1., 'count': 2})
    five = minute.groupby(minute.end.dt.ceil('5min')).agg(open=('open', 'first'), high=('high', 'max'),
                 low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'), count=('count', 'sum')).reset_index()
    five['time'] = five.end - pd.Timedelta(minutes=5)
    return minute, five


def test_same_market_minute_aggregation_detects_missing_values_and_price_changes():
    minute, five = source()
    _, valid = compare_minute_bars(minute, five)
    assert valid['matched'] == 600
    changed = five.copy()
    changed.loc[1, 'close'] += .1
    _, broken = compare_minute_bars(minute.drop(index=0), changed)
    assert broken['incomplete'] == 1 and broken['mismatched'] == 1
    with pytest.raises(ValueError, match='OHLC'):
        compare_minute_bars(minute.assign(open=1000), five)


def test_zero_trade_carry_is_not_the_next_active_candle_open_or_range():
    minute, _ = source()
    minute = minute.iloc[:10].copy()
    minute.loc[:4, ['open', 'high', 'low', 'close']] = 150.
    minute.loc[:4, ['volume', 'count']] = 0
    minute.loc[5, ['open', 'high', 'low', 'close']] = 150.
    minute.loc[5, ['volume', 'count']] = 0
    from wonyotti_fr.minute_data import aggregate_minutes
    five = aggregate_minutes(minute)
    five['time'] = five.end - pd.Timedelta(minutes=5)
    assert five.loc[0, 'open'] == 150 and five.loc[0, 'count'] == 0
    assert five.loc[1, 'open'] == minute.loc[6, 'open'] and five.loc[1, 'high'] < 150
    assert compare_minute_bars(minute, five)[1]['mismatched'] == 0
    with pytest.raises(ValueError, match='OHLC'):
        compare_minute_bars(minute.assign(count=0), five)


def test_future_five_minute_prices_do_not_change_earlier_minute_inputs():
    minute, five = source()
    original = attach_confirmed_features(minute, five)
    changed = five.copy()
    changed.loc[400:, ['open', 'high', 'low', 'close']] *= 2
    altered = attach_confirmed_features(minute, changed)
    cutoff = five.loc[400, 'end']
    before = original.end < cutoff
    pd.testing.assert_frame_equal(original.loc[before], altered.loc[before])
    assert not original.loc[~before, MARKET_FEATURES].equals(altered.loc[~before, MARKET_FEATURES])
    at = original[original.end.eq(cutoff)].iloc[0]
    assert at.feature_end == cutoff
    previous = original[original.end.eq(cutoff - pd.Timedelta(minutes=1))].iloc[0]
    assert previous.feature_end == cutoff - pd.Timedelta(minutes=5)
    assert original.iloc[:4][MARKET_FEATURES].isna().all().all()


def test_missing_confirmed_feature_bar_is_not_bridged_and_funding_offsets_are_bounded():
    minute, five = source()
    result = attach_confirmed_features(minute, five.drop(index=400))
    cutoff = five.loc[400, 'end']
    assert result[result.end.ge(cutoff) & result.end.lt(cutoff + pd.Timedelta(minutes=5))].feature_end.isna().all()
    rates = pd.DataFrame({'time': pd.to_datetime(['2020-01-01T00:00:00.500Z']), 'rate': [.001]})
    first, last = pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2020-01-02T00:00Z')
    assert normalized_funding(rates, first, last).time.iloc[0] == first
    with pytest.raises(ValueError, match='펀딩'):
        normalized_funding(rates.assign(time=rates.time + pd.Timedelta(seconds=1)), first, last)
