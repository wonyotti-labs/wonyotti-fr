from __future__ import annotations

import numpy as np
import pandas as pd

from .close_flow import FLOW_FEATURES, FLOW_WINDOWS, flow_features
from .management_market import FEATURES, utc_minutes
from .minute_data import compare_minute_bars, validate_minutes
from .minute_first_inputs import MINUTE_FLOW_FEATURES
from .minute_management import ACTIONS

EXTRA_FEATURES = [*FLOW_FEATURES, *MINUTE_FLOW_FEATURES]
EXPANDED_FEATURES = [*FEATURES, *EXTRA_FEATURES]


def flow_tables(minute, five, last):
    # 확정 시각 이후의 봉은 검증·계산에서 함께 제외한다.
    minute = minute[minute.end.le(last)].reset_index(drop=True)
    five = five[five.end.le(last)].reset_index(drop=True)
    for bars, step in [(minute, 1), (five, 5)]:
        utc_minutes(bars.time)
        utc_minutes(bars.end)
        validate_minutes(bars, step)
        if (bars.empty or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=step)).all()
            or not np.isfinite(bars.taker_buy_volume).all() or bars.taker_buy_volume.lt(0).any()
            or bars.taker_buy_volume.gt(bars.volume).any()):
            raise ValueError("관리 체결 방향의 연속성·매수 거래량 오류")
    common = minute[minute.end.le(five.end.max())]
    _, comparison = compare_minute_bars(common, five)
    if comparison["incomplete"] or comparison["mismatched"]:
        raise ValueError("관리 체결 방향의 두 해상도 OHLC·총량 불일치")
    totals = common.groupby(common.end.dt.ceil("5min")).taker_buy_volume.sum()
    actual = five.set_index("end").taker_buy_volume.reindex(totals.index)
    error = np.abs(totals.to_numpy() - actual.to_numpy())
    if len(totals) != len(five) or not np.isfinite(error).all() or (error > 1e-7).any():
        raise ValueError("관리 체결 방향의 두 해상도 매수량 불일치")
    small = minute[["end"]].copy()
    volume, buy = minute.volume.to_numpy(float), minute.taker_buy_volume.to_numpy(float)
    small[MINUTE_FLOW_FEATURES[0]] = np.divide(2*buy, volume, out=np.ones(len(volume)), where=volume != 0)-1
    return small, flow_features(five), {"aggregate": comparison, "buy_volume_buckets": len(totals),
        "maximum_buy_volume_error": float(error.max()), "buy_volume_tolerance": 1e-7,
        "minute_rows": len(minute), "five_rows": len(five)}


def connect_flow(frame, minute, five):
    utc_minutes(frame.end)
    utc_minutes(minute.end)
    utc_minutes(five.end)
    if (frame.empty or set(frame) & set(EXTRA_FEATURES) or not frame.direction.isin([-1, 1]).all()
        or not np.isfinite(frame[FEATURES]).all().all()
        or not frame[["y_"+a for a in ACTIONS]].isin([0, 1]).all().all()):
        raise ValueError("관리 체결 방향의 원본 입력·정답·추가 열 오류")
    result = frame.reset_index(drop=True).copy()
    times = result.end
    small = pd.Index(minute.end).get_indexer(times)
    large = pd.Index(five.end).get_indexer(times.dt.floor("5min"))
    for indexes, source, columns in [(small, minute, MINUTE_FLOW_FEATURES[:1]),
                                     (large, five, list(FLOW_WINDOWS))]:
        values = np.full((len(frame), len(columns)), np.nan)
        known = indexes >= 0
        values[known] = source.iloc[indexes[known]][columns].to_numpy()
        for j, name in enumerate(columns):
            result[name] = values[:, j]
            result["directional_"+name] = values[:, j]*result.direction.to_numpy()
    finite = np.isfinite(result[EXTRA_FEATURES]).all(axis=1).to_numpy()
    if result[EXTRA_FEATURES].abs().gt(1).any().any():
        raise ValueError("관리 체결 방향의 비율 범위 오류")
    link = pd.DataFrame({"end": times, "source_row": np.arange(len(frame)),
        "minute_source_end": times.where(small >= 0),
        "five_source_end": times.dt.floor("5min").where(large >= 0),
        "supported": (small >= 0) & (large >= 0) & finite,
        "reason": np.select([small < 0, large < 0, ~finite],
            ["missing_minute", "missing_five", "unready_flow"], default="supported")})
    pd.testing.assert_frame_equal(result.drop(columns=EXTRA_FEATURES), frame.reset_index(drop=True), check_exact=True)
    return result, link


def flow_support(frame, link):
    pd.testing.assert_series_equal(frame.end.reset_index(drop=True), link.end, check_exact=True)
    groups = {}
    for name, selected in [("all", np.ones(len(frame), bool)),
                            ("included", link.supported.to_numpy()), ("excluded", ~link.supported.to_numpy())]:
        part = frame.loc[selected]
        groups[name] = {"rows": len(part), "positive": {a: int(part["y_"+a].sum()) for a in ACTIONS}}
    included = groups["included"]
    return {"groups": groups, "reasons": {str(k): int(v) for k, v in link.reason.value_counts().items()},
        "all_original_rows_supported": bool(link.supported.all()),
        "action_support_sufficient": all(min(n, included["rows"]-n) >= 20 for n in included["positive"].values())}
