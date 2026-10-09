from __future__ import annotations

import numpy as np
import pandas as pd

from .close_flow import (
    FLOW_FEATURES,
    FLOW_WINDOWS,
    FlowCloseModel,
    flow_features,
    validate_flow_matrix,
)
from .context_position import CONTEXT_FEATURES, context_features
from .continuation_inputs import ContinuationCloseModel
from .event_features import MARKET_FEATURES, event_features
from .minute_data import validate_minutes


def attach_minute_close_inputs(ledger, bars):
    if (ledger.empty or set(CONTEXT_FEATURES+FLOW_FEATURES) & set(ledger)
        or not set(ContinuationCloseModel.features).issubset(ledger)
        or not isinstance(ledger.decision_time.dtype, pd.DatetimeTZDtype)
        or str(ledger.decision_time.dt.tz) != 'UTC'
        or ledger.decision_time.isna().any() or ledger.decision_time.duplicated().any()
        or not ledger.decision_time.is_monotonic_increasing or not ledger.direction.isin([-1, 1]).all()
        or not np.isfinite(ledger[ContinuationCloseModel.features]).all().all()):
        raise ValueError('분별 청산 추가 입력의 시각·원래 특징·방향 오류')
    times = ledger.decision_time.astype('datetime64[ns, UTC]')
    if (times.array.asi8 % (60*10**9)).any():
        raise ValueError('분별 청산의 분 경계 오류')
    # 현재 판단 뒤의 시세는 추가 특징 계산에서도 제외한다.
    history = bars[bars.end.le(times.max())].reset_index(drop=True)
    if history.empty:
        raise ValueError('분별 청산의 확정 시세 누락')
    validate_minutes(history, 5)
    original = event_features(history).set_index('end')
    original.index = original.index.astype('datetime64[ns, UTC]')
    locations = original.index.get_indexer(times, method='pad')
    if (locations < 0).any():
        raise ValueError('분별 청산의 과거 확정 시세 누락')
    ends = original.index.take(locations)
    elapsed = times.array.asi8-ends.asi8
    if ((elapsed < 0) | (elapsed >= 300*10**9)).any():
        raise ValueError('분별 청산의 미래·오래된 확정 시세 오류')
    pd.testing.assert_frame_equal(original.iloc[locations][MARKET_FEATURES].reset_index(drop=True),
        ledger[MARKET_FEATURES].reset_index(drop=True), check_exact=True)
    context = context_features(history).iloc[locations][CONTEXT_FEATURES].reset_index(drop=True)
    flow = flow_features(history).iloc[locations][list(FLOW_WINDOWS)].reset_index(drop=True)
    for name in FLOW_WINDOWS:
        flow['directional_'+name] = flow[name]*ledger.direction.to_numpy()
    if not np.isfinite(context).all().all() or not np.isfinite(flow).all().all():
        raise ValueError('분별 청산의 연속 시세·준비 기간 지원 부족')
    result = pd.concat([ledger.reset_index(drop=True), context, flow], axis=1)
    validate_flow_matrix(result[FlowCloseModel.features], FlowCloseModel.features)
    linkage = pd.DataFrame({'decision_time': times.reset_index(drop=True),
        'confirmed_feature_end': ends, 'elapsed_seconds': elapsed//10**9})
    return result, linkage
