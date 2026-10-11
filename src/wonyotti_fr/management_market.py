from __future__ import annotations

import numpy as np
import pandas as pd

from .event_features import MARKET_FEATURES, event_features
from .minute_inputs import MINUTE_FEATURES, minute_features
from .minute_management import ACTIONS
from .order_history_boost import OrderHistoryBoostModels

FEATURES = list(OrderHistoryBoostModels.features)
OBSERVED = MARKET_FEATURES + MINUTE_FEATURES
REPLACED = [*MARKET_FEATURES, *[f"directional_{n}" for n in MARKET_FEATURES],
            *MINUTE_FEATURES, *[f"directional_{n}" for n in MINUTE_FEATURES]]


def utc_minutes(series):
    if (not isinstance(series.dtype, pd.DatetimeTZDtype) or str(series.dt.tz) != "UTC"
        or series.isna().any() or series.duplicated().any()
        or not series.is_monotonic_increasing
        or (series.astype("datetime64[ns, UTC]").array.asi8 % (60 * 10**9)).any()):
        raise ValueError("관리 시장 입력의 UTC·분 경계·순서 오류")


def observed_tables(minute, five):
    for bars, step in [(minute, 1), (five, 5)]:
        utc_minutes(bars.time)
        utc_minutes(bars.end)
        prices = bars[["open", "high", "low", "close"]]
        if (bars.empty or not bars.end.sub(bars.time).eq(pd.Timedelta(minutes=step)).all()
            or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=step)).all()
            or (bars.time.astype("datetime64[ns, UTC]").array.asi8 % (step * 60 * 10**9)).any()
            or not np.isfinite(prices).all().all() or prices.le(0).any().any()
            or bars.high.lt(bars.close).any() or bars.low.gt(bars.close).any()
            or not np.isfinite(bars.volume).all() or bars.volume.lt(0).any()):
            raise ValueError("관리 시장 입력의 연속 시세·가격·거래량 오류")
        # BitMEX open은 이전 종가이므로 체결 고가·저가 안에 있다고 가정하지 않는다.
    return minute_features(minute), event_features(five)


def connect_market(frame, minute, five):
    utc_minutes(frame.end)
    utc_minutes(minute.end)
    utc_minutes(five.end)
    if (frame.empty or not frame.direction.isin([-1, 1]).all()
        or not np.isfinite(frame[FEATURES]).all().all()
        or not np.isfinite(frame.episode_id).all() or frame.episode_id.le(0).any()
        or frame.episode_id.mod(1).ne(0).any()
        or not frame[[f"y_{a}" for a in ACTIONS]].isin([0, 1]).all().all()):
        raise ValueError("관리 시장 입력의 원본 특징·상태·정답 오류")
    ends = frame.end.reset_index(drop=True)
    minute_index = pd.Index(minute.end).get_indexer(ends)
    five_ends = ends.dt.floor("5min")
    five_index = pd.Index(five.end).get_indexer(five_ends)
    values = np.full((len(frame), len(OBSERVED)), np.nan)
    for index, table, columns, offset in [(five_index, five, MARKET_FEATURES, 0),
                                          (minute_index, minute, MINUTE_FEATURES, len(MARKET_FEATURES))]:
        known = index >= 0
        values[known, offset:offset + len(columns)] = table.iloc[index[known]][columns].to_numpy()
    extra = pd.DataFrame(values, columns=OBSERVED, index=frame.index)
    for name in OBSERVED:
        extra[f"directional_{name}"] = extra[name] * frame.direction
    candidate = frame.copy()
    candidate[REPLACED] = extra[REPLACED]
    pd.testing.assert_frame_equal(candidate.drop(columns=REPLACED), frame.drop(columns=REPLACED),
                                  check_exact=True)
    link = pd.DataFrame({"end": ends, "expected_five_end": five_ends,
                         "minute_found": minute_index >= 0, "five_found": five_index >= 0,
                         "market_finite": np.isfinite(values).all(axis=1)})
    link["supported"] = link.minute_found & link.five_found & link.market_finite
    link["reason"] = np.select([~link.minute_found, ~link.five_found, ~link.market_finite],
                                ["missing_minute", "missing_five", "nonfinite_market"], default="supported")
    return candidate, link


def check_original_market(frame, reconstructed, link):
    if not link.supported.all():
        raise ValueError("관리 시장 입력의 원래 지원 행 재현 누락")
    pd.testing.assert_frame_equal(reconstructed, frame, check_exact=True)


def market_support(frame, candidate, link):
    pd.testing.assert_series_equal(frame.end.reset_index(drop=True), link.end, check_exact=True)
    pd.testing.assert_frame_equal(frame.drop(columns=REPLACED), candidate.drop(columns=REPLACED),
                                  check_exact=True)
    available = link.supported.to_numpy()
    if not np.array_equal(available, np.isfinite(candidate[FEATURES]).all(axis=1)):
        raise ValueError("관리 시장 입력의 공동 지원 표시 오류")
    groups = {}
    for label, selected in [("all", np.ones(len(frame), dtype=bool)),
                             ("included", available), ("excluded", ~available)]:
        part = frame.loc[selected]
        groups[label] = {"rows": len(part), "positive": {a: int(part[f"y_{a}"].sum()) for a in ACTIONS}}
    changes = {}
    for name in REPLACED:
        old, new = frame.loc[available, name].to_numpy(), candidate.loc[available, name].to_numpy()
        delta = np.abs(old - new)
        changes[name] = {"different": int(np.count_nonzero(delta)),
                          "mean_absolute_difference": float(delta.mean()) if len(delta) else None,
                          "maximum_absolute_difference": float(delta.max()) if len(delta) else None}
    enough = all(20 <= n <= groups["included"]["rows"] - 20
                 for n in groups["included"]["positive"].values())
    return {"groups": groups, "reasons": {str(k): int(v) for k, v in link.reason.value_counts().items()},
            "changes": changes, "action_support_sufficient": enough}
