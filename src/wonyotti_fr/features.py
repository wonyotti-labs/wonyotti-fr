from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["return_1h", "return_4h", "return_1d", "trend_4h_16h", "trend_16h_64h",
            "volatility_1d", "volume_ratio", "range_fraction", "body_fraction", "taker_buy_ratio"]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.sort_values("time").reset_index(drop=True)
    if bars.time.duplicated().any() or len(bars) < 257:
        raise ValueError("지표 계산에는 중복 없는 257개 이상의 15분봉이 필요합니다.")
    if not ((bars.end - bars.time) == pd.Timedelta(minutes=15)).all():
        raise ValueError("현재 전략 특징은 15분봉 기준입니다.")
    close = bars.close.astype(float)
    returns = np.log(close).diff()
    result = bars[["time", "end"]].copy()
    for hours, periods in [(1, 4), (4, 16), (24, 96)]:
        result[{1: "return_1h", 4: "return_4h", 24: "return_1d"}[hours]] = close.pct_change(periods, fill_method=None)
    fast = close.ewm(span=16, adjust=False, min_periods=16).mean()
    mid = close.ewm(span=64, adjust=False, min_periods=64).mean()
    slow = close.ewm(span=256, adjust=False, min_periods=256).mean()
    result["trend_4h_16h"] = fast / mid - 1
    result["trend_16h_64h"] = mid / slow - 1
    result["volatility_1d"] = returns.rolling(96, min_periods=96).std() * np.sqrt(96)
    result["volume_ratio"] = bars.volume / bars.volume.rolling(96, min_periods=96).mean().replace(0, np.nan)
    result["range_fraction"] = (bars.high - bars.low) / close
    result["body_fraction"] = (close - bars.open) / bars.open
    result["taker_buy_ratio"] = bars.taker_buy_volume / bars.volume.replace(0, np.nan)
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    # 빈 시세 구간을 지표의 연속 관측값으로 취급하지 않는다.
    breaks = bars.time.diff().ne(pd.Timedelta(minutes=15))
    contaminated = breaks.rolling(256, min_periods=1).max().astype(bool)
    result.loc[contaminated, FEATURES] = np.nan
    return result


def position_labels(features: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    inventory = actions[["time", "after_qty"]].sort_values("time", kind="stable")
    inventory = inventory.drop_duplicates("time", keep="last")
    # 목표는 다음 봉 종료 시점의 트레이더 포지션 방향이다. 미래 값은 학습 정답에만 사용한다.
    query = features.copy()
    query["label_time"] = query.end + pd.Timedelta(minutes=15)
    joined = pd.merge_asof(query.sort_values("label_time"), inventory.rename(columns={"time": "label_time"}),
                           on="label_time", direction="backward")
    joined["label"] = np.sign(joined.after_qty).astype("Int64")
    coverage = (joined.label_time >= actions.time.min()) & (joined.label_time <= actions.time.max())
    joined.loc[~coverage, "label"] = pd.NA
    return joined


def decision_context(actions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    decisions = actions[actions.action.isin(["open", "increase", "reduce", "close", "reverse"])].copy()
    right = features.rename(columns={"time": "bar_open", "end": "feature_time"})
    return pd.merge_asof(decisions.sort_values("time"), right.sort_values("feature_time"),
                         left_on="time", right_on="feature_time", direction="backward",
                         tolerance=pd.Timedelta(minutes=30))
