import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.execution_study import markouts
from wonyotti_fr.timing_study import aggregate_comparison, compare_order_timing, paired_gap_interval


def source():
    end = pd.date_range('2020-01-01T00:01Z', periods=15, freq='1min').as_unit('ns')
    minute = pd.DataFrame({'time': end - pd.Timedelta(minutes=1), 'end': end, 'close': np.arange(101., 116.),
                           'volume': .01, 'contract_volume': 100, 'trades': 2})
    five = minute.iloc[[4, 9, 14]].copy()
    five['time'] = five.end - pd.Timedelta(minutes=5)
    five[['volume', 'contract_volume', 'trades']] *= 5
    return minute, five


def test_aggregation_uses_right_closed_end_and_does_not_hide_incomplete_or_value_difference():
    minute, five = source()
    result, summary = aggregate_comparison(minute, five)
    assert summary['matched_buckets'] == 3
    assert result.close_aggregated.tolist() == [105, 110, 115]
    missing, mismatch = aggregate_comparison(minute.drop(index=3), five.assign(trades=[10, 11, 10]))
    assert mismatch['incomplete_buckets'] == 1
    assert mismatch['value_mismatch_buckets'] == 1
    assert missing.status.tolist() == ['missing_or_incomplete_bucket', 'value_difference', 'matched']


def test_minute_reference_requires_strict_past_and_does_not_bridge_missing_minutes():
    minute, five = source()
    orders = pd.DataFrame({'order_key': ['a'], 'target_time': pd.to_datetime(['2020-01-01T00:06Z']),
                           'side': ['Buy'], 'target': ['increase']})
    fills = pd.DataFrame({'order_key': ['a'], 'lastpx': [104.], 'execcost': [100000], 'execcomm': [-25],
                          'lastliquidityind': ['AddedLiquidity']})
    paired, _ = compare_order_timing(orders, fills, minute, five)
    row = paired[paired.horizon_minutes.eq(5)].iloc[0]
    assert row.reference_price_1m == row.reference_price_5m == 105
    assert row.reference_age_seconds_1m == 60
    assert row.outcome_price_1m == 111 and row.outcome_price_5m == 115
    assert row.outcome_delay_seconds_1m == 0 and row.outcome_delay_seconds_5m == 240
    broken = markouts(orders, fills, minute.drop(index=4), (5,), interval_minutes=1)
    assert broken.reference_price.isna().all()
    with pytest.raises(ValueError, match='봉 간격'):
        markouts(orders, fills, minute, interval_minutes=5)


def test_paired_calendar_blocks_preserve_constant_difference_and_order_weighting():
    times = pd.date_range('2020-01-01', periods=28, freq='1D', tz='UTC')
    frame = pd.DataFrame({'target_time': times, 'reference_difference_bps': -2.,
                          'absolute_reference_gap_reduction_bps': 3.})
    result = paired_gap_interval(frame)
    assert result['reference_difference_bps_ci_low'] == result['reference_difference_bps_ci_high'] == -2
    assert result['absolute_reference_gap_reduction_bps_ci_low'] == result['absolute_reference_gap_reduction_bps_ci_high'] == 3
    extra = frame.iloc[:1].assign(reference_difference_bps=-20., absolute_reference_gap_reduction_bps=30.)
    sample = pd.concat([frame, extra], ignore_index=True)
    result = paired_gap_interval(sample)
    assert result == paired_gap_interval(sample)
    assert result['mean_reference_difference_bps'] == pytest.approx((-2 * 28 - 20) / 29)
    assert result['orders'] == 29 and result['active_days'] == 28
    sparse = paired_gap_interval(frame.iloc[[0, -1]])
    assert sparse['calendar_days'] == 28 and sparse['active_days'] == 2
    assert sparse['reference_difference_bps_ci_low'] is None
