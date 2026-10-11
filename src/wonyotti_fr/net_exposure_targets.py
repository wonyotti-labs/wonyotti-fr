from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES


def boundary_targets(rows):
    values = []
    with localcontext() as ctx:
        ctx.prec = 60
        for row in rows.itertuples(index=False):
            reasons = []
            if not row.all_prices_have_observed_trade:
                reasons.append('unobserved_price_trade')
            if row.external_uncertain_count:
                reasons.append('uncertain_external_transfer')
            if row.unsupported_reasons:
                reasons.append('unsupported_valuation')
            else:
                equity = Decimal(row.equity_min_satoshi)
                upper = Decimal(row.equity_max_satoshi)
                delta = Decimal(row.delta_min_satoshi)
                delta_upper = Decimal(row.delta_max_satoshi)
                stressed = Decimal(row.stress_equity_min_satoshi)
                if not all(v.is_finite() for v in (equity, upper, delta, delta_upper, stressed)):
                    raise ValueError('계좌 노출 정답의 유한성 오류')
                if not row.external_uncertain_count and (equity != upper or delta != delta_upper):
                    raise ValueError('계좌 노출의 확정 입출금 구간 불일치')
                if equity <= 0 or stressed <= 0:
                    reasons.append('nonpositive_equity_lower_bound')
            supported = not reasons
            beta = float(delta/equity) if supported else np.nan
            lo, hi = row.btc_partial_leverage_min, row.btc_partial_leverage_max
            if supported and (not np.isfinite([beta, lo, hi]).all() or not lo-1e-12 <= beta <= hi+1e-12):
                raise ValueError('계좌 노출 정답의 가격 민감도 범위 오류')
            values.append({'net_exposure': beta, 'net_exposure_min': float(lo), 'net_exposure_max': float(hi),
                'net_exposure_supported': supported, 'net_exposure_reason': '|'.join(reasons) if reasons else 'included'})
    return pd.DataFrame(values)


def net_exposure_targets(source, comparisons, part):
    if part not in ('training', 'diagnosis') or not len(source):
        raise ValueError('계좌 노출 정답의 기간·행 오류')
    for column in ['end', 'label_end']:
        times = pd.DatetimeIndex(source[column])
        if times.tz is None or str(times.tz) != 'UTC' or times.hasnans:
            raise ValueError('계좌 노출 정답의 UTC 시각 오류')
    if (not source.end.is_monotonic_increasing or source.end.duplicated().any()
        or not source.label_end.sub(source.end).eq(pd.Timedelta(minutes=5)).all()
        or not np.isfinite(source[MARKET_FEATURES]).all().all()):
        raise ValueError('계좌 노출 정답의 시각·시장 입력 오류')
    out = source.copy().reset_index(drop=True)
    for boundary, time_column, prefix in [('current', 'end', 'current_'), ('target', 'label_end', '')]:
        selected = comparisons.loc[comparisons.part.eq(part) & comparisons.boundary.eq(boundary)].reset_index(drop=True)
        if (len(selected) != len(source) or not np.array_equal(selected.time, source[time_column])
            or not np.array_equal(selected.source_end, source.end)
            or not np.array_equal(selected.source_episode_id, source.episode_id)
            or not np.array_equal(selected.source_target_episode_id, source.position_target_episode_id)):
            raise ValueError('계좌 노출 정답의 전체 경계·원본 포지션 연결 오류')
        target = boundary_targets(selected)
        for name in target:
            out[prefix+name] = target[name].to_numpy()
    return out
