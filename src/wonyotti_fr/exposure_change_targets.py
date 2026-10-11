import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES


def change_targets(source):
    if (not len(source) or source.end.duplicated().any() or not source.end.is_monotonic_increasing
        or not source.label_end.sub(source.end).eq(pd.Timedelta(minutes=5)).all()
        or not np.isfinite(source[MARKET_FEATURES]).all().all()):
        raise ValueError('노출 변경량 정답의 시각·시장 입력 오류')
    for field in ['net_exposure_supported', 'current_net_exposure_supported']:
        if source[field].dtype != bool:
            raise ValueError('노출 변경량 지원 표시 오류')
    result = source.copy()
    supported = source.net_exposure_supported & source.current_net_exposure_supported
    result['change_supported'] = supported
    result['exposure_change'] = (source.net_exposure-source.current_net_exposure).where(supported, np.nan)
    result['exposure_change_min'] = (source.net_exposure_min-source.current_net_exposure_max).where(supported, np.nan)
    result['exposure_change_max'] = (source.net_exposure_max-source.current_net_exposure_min).where(supported, np.nan)
    reasons = []
    for row in source.itertuples(index=False):
        parts = []
        if not row.current_net_exposure_supported:
            parts.append('current:'+row.current_net_exposure_reason)
        if not row.net_exposure_supported:
            parts.append('target:'+row.net_exposure_reason)
        reasons.append('|'.join(parts) if parts else 'included')
    result['change_reason'] = reasons
    known = result.loc[supported]
    fields = ['exposure_change', 'exposure_change_min', 'exposure_change_max', 'current_net_exposure', 'net_exposure']
    if (not np.isfinite(known[fields]).all().all()
        or (known.exposure_change < known.exposure_change_min-1e-12).any()
        or (known.exposure_change > known.exposure_change_max+1e-12).any()):
        raise ValueError('노출 변경량 정답·가격 민감도 구간 오류')
    return result
