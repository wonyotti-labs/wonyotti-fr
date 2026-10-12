from __future__ import annotations

import numpy as np
import pandas as pd

from .quiet_management import ACTIONS, groups

WINDOWS = {"5m": 1, "15m": 3, "1h": 12, "4h": 48, "1d": 288}
RELATIVE_FEATURES = ["eth_btc_log_return_" + name for name in WINDOWS]
CROSS_FEATURES = RELATIVE_FEATURES + ["directional_" + name for name in RELATIVE_FEATURES]
STEP_NS = 300_000_000_000


def utc_ns(values, label):
    if (not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC"
        or values.isna().any()):
        raise ValueError(label + "의 UTC 시각 오류")
    return values.astype("datetime64[ns, UTC]").array.asi8


def cross_market_table(btc, eth):
    for frame in [btc, eth]:
        if not {"time", "end", "close"}.issubset(frame) or len(frame) <= max(WINDOWS.values()):
            raise ValueError("상대 가격 시세의 열·준비 기간 부족")
        times, ends = utc_ns(frame.time, "시세 시작"), utc_ns(frame.end, "시세 종료")
        prices = frame.close.to_numpy(float)
        if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (times % STEP_NS != 0).any() or not np.all(ends - times == STEP_NS)
            or not np.all(np.diff(times) == STEP_NS)):
            raise ValueError("상대 가격 시세의 가격·5분 연속성 오류")
    if (not np.array_equal(utc_ns(btc.time, "BTC 시작"), utc_ns(eth.time, "ETH 시작"))
        or not np.array_equal(utc_ns(btc.end, "BTC 종료"), utc_ns(eth.end, "ETH 종료"))):
        raise ValueError("상대 가격 두 시장의 시각 불일치")
    result = pd.DataFrame({"market_end": btc.end.to_numpy(), "source_row": np.arange(len(btc))})
    log_btc, log_eth = np.log(btc.close.to_numpy(float)), np.log(eth.close.to_numpy(float))
    for name, size in WINDOWS.items():
        values = np.full(len(btc), np.nan)
        values[size:] = (log_eth[size:] - log_eth[:-size]) - (log_btc[size:] - log_btc[:-size])
        result["eth_btc_log_return_" + name] = values
    return result


def attach_cross_market(frame, table):
    if (not {"end", "episode_id", "direction"}.issubset(frame)
        or not {"market_end", "source_row", *RELATIVE_FEATURES}.issubset(table)
        or frame.columns.duplicated().any() or table.columns.duplicated().any()):
        raise ValueError("상대 가격 연결의 열 누락·중복 오류")
    episodes = frame.episode_id.to_numpy(float)
    if (frame.empty or not frame.index.is_unique or set(CROSS_FEATURES) & set(frame)
        or not frame.direction.isin([-1, 1]).all() or not np.isfinite(episodes).all()
        or ((episodes <= 0) | (episodes >= 2**53) | (episodes % 1 != 0)).any()):
        raise ValueError("상대 가격 연결의 원본 행·방향·입력 중복 오류")
    ends, times = utc_ns(frame.end, "원본 판단"), utc_ns(table.market_end, "상대 가격 종료")
    if (not np.all(np.diff(ends) > 0) or not np.all(np.diff(times) == STEP_NS)
        or not np.array_equal(table.source_row, np.arange(len(table)))):
        raise ValueError("상대 가격 연결의 순서·시세 행 번호 오류")
    selected = np.searchsorted(times, ends, side="right") - 1
    if (selected < max(WINDOWS.values())).any():
        raise ValueError("상대 가격 연결의 과거 창 준비 부족")
    age_ns = ends - times[selected]
    if ((age_ns < 0) | (age_ns >= STEP_NS)).any():
        raise ValueError("상대 가격 연결의 미래·오래된 시세")
    raw = table.iloc[selected][RELATIVE_FEATURES].to_numpy(float)
    if not np.isfinite(raw).all():
        raise ValueError("상대 가격 연결의 입력 누락·비유한 값")
    result = frame.copy()
    result[RELATIVE_FEATURES] = raw
    result[CROSS_FEATURES[len(RELATIVE_FEATURES):]] = raw * frame.direction.to_numpy()[:, None]
    links = frame[["end", "episode_id"]].copy()
    links["market_end"] = table.market_end.iloc[selected].to_numpy()
    links["source_row"] = selected
    links["market_age_seconds"] = age_ns / 1e9
    for name, size in WINDOWS.items():
        links["lag_row_" + name] = selected - size
    pd.testing.assert_frame_equal(result[frame.columns], frame, check_exact=True)
    return result, links


def cross_market_support(frame):
    if not np.isfinite(frame[CROSS_FEATURES]).all().all():
        raise ValueError("상대 가격 지원 집계의 비유한 입력")
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for state, mask in groups(frame, action).items():
            part = frame.loc[mask]
            labels = part["y_" + action].to_numpy()
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("상대 가격 지원 집계의 정답 오류")
            features = {}
            for name in CROSS_FEATURES:
                values = part[name].to_numpy()
                features[name] = None if not len(values) else dict(zip(
                    ["minimum", "q25", "median", "q75", "maximum"],
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist(), strict=True))
            result[action][state] = {"rows": len(part), "positive": int(labels.sum()),
                "negative": int(len(part) - labels.sum()), "features": features}
    return result
