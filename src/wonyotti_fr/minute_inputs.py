from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE_FEATURES = ['minute_ret_1m', 'minute_ret_3m', 'minute_range_fraction', 'minute_volume_ratio_1h']


def minute_features(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.reset_index(drop=True)
    prices = frame[['close', 'high', 'low']]
    if (frame.empty or any(not isinstance(frame[key].dtype, pd.DatetimeTZDtype)
                          or str(frame[key].dtype.tz) != 'UTC' for key in ['time', 'end'])
        or frame.time.isna().any() or not frame.end.sub(frame.time).eq(pd.Timedelta(minutes=1)).all()
        or not frame.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        or (frame.time.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=1).value).any()
        or not np.isfinite(prices).all().all() or prices.le(0).any().any()
        or frame.high.lt(frame.close).any() or frame.low.gt(frame.close).any()
        or not np.isfinite(frame.volume).all() or frame.volume.lt(0).any()):
        raise ValueError('확정 분봉 특징의 시간·가격·거래량 오류')
    result = frame[['end']].astype({'end': 'datetime64[ns, UTC]'}).copy()
    result['minute_ret_1m'] = frame.close.pct_change(1, fill_method=None)
    result['minute_ret_3m'] = frame.close.pct_change(3, fill_method=None)
    result['minute_range_fraction'] = (frame.high - frame.low) / frame.close
    # 현재 분의 거래량은 기준 평균에 포함하지 않는다.
    reference = frame.volume.shift(1).rolling(60, min_periods=60).mean()
    result['minute_volume_ratio_1h'] = frame.volume / reference.replace(0, np.nan)
    result[MINUTE_FEATURES] = result[MINUTE_FEATURES].replace([np.inf, -np.inf], np.nan)
    return result
