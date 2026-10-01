from __future__ import annotations

import numpy as np
import pandas as pd

from .common import records
from .minute_data import validate_minutes
from .minute_repair import BAR_COLUMNS


def verify_unchanged_candles(original: pd.DataFrame, rebuilt: pd.DataFrame, minutes: pd.Series,
                             sources: dict[str, pd.DataFrame]) -> dict:
    if set(sources) != {'daily', 'monthly'} or minutes.empty or minutes.duplicated().any():
        raise ValueError('유지할 분봉의 일별·월별 증거 또는 대상 오류')
    expected = pd.DatetimeIndex(minutes).sort_values().as_unit('ns')
    evidence = {'raw': rebuilt, 'original': original, **sources}
    selected = {}
    for name, frame in evidence.items():
        part = frame[frame.time.isin(expected)].sort_values('time').reset_index(drop=True)
        validate_minutes(part, 1)
        if (not np.array_equal(part.time.dt.as_unit('ns').array.asi8, expected.asi8)
            or not np.isfinite(part[BAR_COLUMNS]).all().all() or part[BAR_COLUMNS].lt(0).any().any()
            or part['count'].le(0).any()):
            raise ValueError('유지할 분봉 증거의 중복·누락·값 오류')
        selected[name] = part
    for name in ['original', 'daily', 'monthly']:
        # 수정이 필요 없는 봉만 허용하며 숫자 허용 오차로 증거 차이를 숨기지 않는다.
        if not np.array_equal(selected['raw'][BAR_COLUMNS].to_numpy(), selected[name][BAR_COLUMNS].to_numpy()):
            raise ValueError('유지할 분봉의 아홉 값이 원체결·기존 입력·일별·월별 자료와 정확히 같지 않습니다.')
    return {'mode': 'unchanged_candle_values', 'columns': BAR_COLUMNS,
            'exact_equal_raw_original_daily_monthly': True, 'minutes': records(selected['raw'][['time', *BAR_COLUMNS]]),
            'individual_trade_corroboration_completed': False,
            'limit': '유지되는 봉 값의 네 경로 일치. 미연결 개별 체결의 진위·원인·전체 시장 완전성 인증이 아님'}
