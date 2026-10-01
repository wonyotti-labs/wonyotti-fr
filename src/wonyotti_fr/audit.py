from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .common import records, save_json, sha256

ZERO_ID = "00000000-0000-0000-0000-000000000000"
COLUMNS = [
    "date", "execid", "orderid", "symbol", "side", "lastqty", "lastpx",
    "lastliquidityind", "orderqty", "exectype", "ordtype", "ordstatus", "cumqty",
    "execcomm", "execcost", "trdmatchid", "transacttime", "timestamp",
]


def load_executions(source: Path, timezone: str = "UTC") -> tuple[pd.DataFrame, dict]:
    files = sorted(source.glob("aoa-execution-*.csv"))
    if not files:
        raise ValueError("aoa-execution-*.csv 입력 파일이 없습니다.")
    frames, manifests = [], []
    for path in files:
        frame = pd.read_csv(path, usecols=COLUMNS, keep_default_na=False)
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(2, len(frame) + 2)
        parsed = pd.to_datetime(frame["transacttime"], format="mixed", errors="raise")
        if parsed.dt.tz is not None:
            raise ValueError("원본 시각 형식이 바뀌었습니다. 시간대 변환 규칙을 확인하세요.")
        frame["time"] = parsed.dt.tz_localize(timezone).dt.tz_convert("UTC")
        manifests.append({
            "file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
            "rows": len(frame), "types": frame["exectype"].value_counts().to_dict(),
            "start": frame["time"].min(), "end": frame["time"].max(),
            "time_inversions": int((frame["time"].diff().dt.total_seconds() < 0).sum()),
        })
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    duplicate = data["execid"].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(f"중복 execid {duplicate.sum()}행: 입력을 확인해야 합니다.")
    for key in ["lastqty", "lastpx", "orderqty", "cumqty", "execcomm", "execcost"]:
        data[key] = pd.to_numeric(data[key], errors="raise")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"유한하지 않은 숫자: {key}")
    for key in ["lastqty", "execcomm", "execcost"]:
        if (data[key] != np.trunc(data[key])).any():
            raise ValueError(f"정수 단위가 아닌 입력: {key}")
        data[key] = data[key].astype("int64")
    trade = data["exectype"].eq("Trade")
    invalid = trade & ((data["lastqty"] <= 0) | (data["lastpx"] <= 0)
                       | ~data["side"].isin(["Buy", "Sell"]))
    if invalid.any():
        raise ValueError(f"유효하지 않은 체결 {invalid.sum()}행: 자동 복원 중단")
    special = data["orderid"].isin(["", ZERO_ID])
    data["order_key"] = data["orderid"].where(~special, "special:" + data["execid"])
    tie_sides = data[trade].groupby(["symbol", "time"])["side"].nunique()
    summary = {
        "files": manifests, "rows": len(data), "duplicate_execid": 0,
        "source_timezone_assumption": timezone, "timezone_independently_verified": False,
        "types": data["exectype"].value_counts().to_dict(),
        "trade_symbols": data.loc[trade, "symbol"].value_counts().to_dict(),
        "trade_order_types": data.loc[trade, "ordtype"].value_counts().to_dict(),
        "liquidity": data.loc[trade, "lastliquidityind"].value_counts().to_dict(),
        "unique_normal_orderids": data.loc[trade & ~special, "orderid"].nunique(),
        "special_orderid_trade_rows": int((trade & special).sum()),
        "opposite_sides_same_timestamp_groups": int((tie_sides > 1).sum()),
    }
    return data.sort_values(["time", "source_file", "source_row"], kind="stable"), summary


def load_wallet(source: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(source.glob("aoa-wallet-*.csv"))
    if len(files) != 1:
        raise ValueError("지갑 원본은 한 개여야 합니다.")
    path = files[0]
    raw = pd.read_csv(path, dtype=str).fillna("")
    data = raw[raw.apply(lambda row: any(str(v).strip() for v in row), axis=1)].copy()
    if data.empty or not data.currency.eq("XBt").all():
        raise ValueError("지갑 입력은 비어 있지 않은 XBt 원장이어야 합니다.")
    if not data.transactstatus.isin(["Completed", "Canceled"]).all():
        raise ValueError("지원하지 않는 지갑 처리 상태가 있습니다.")
    data["source_row"] = data.index + 2
    data["walletbalance_raw"] = data.walletbalance
    data["balance_precision_satoshi"] = data.walletbalance.map(
        lambda v: 10 ** max(Decimal(v).as_tuple().exponent, 0))
    for key in ["amount", "walletbalance"]:
        values = data[key].map(Decimal)
        if any(not v.is_finite() or v != v.to_integral_value() for v in values):
            raise ValueError("지갑 금액은 유한한 정수 사토시여야 합니다.")
        data[key] = values.map(int).astype("int64")
    # 취소 출금은 현금 흐름에 포함하지 않는다.
    data["posted_amount"] = data.amount.where(data.transactstatus.eq("Completed"), 0)
    opening = int(data.iloc[0].walletbalance - data.iloc[0].posted_amount)
    days = pd.to_datetime(data.date, format="%Y-%m-%d", errors="raise")
    daily = data.assign(day=days).sort_values("day", kind="stable").groupby("day").agg(
        posted_amount=("posted_amount", "sum"), recorded_last_balance=("walletbalance", "last"),
        precision=("balance_precision_satoshi", "last"))
    daily["expected_balance"] = daily.posted_amount.cumsum() + opening
    daily["difference"] = daily.recorded_last_balance - daily.expected_balance
    tolerance = daily.precision.where(daily.precision.gt(1), 0) / 2
    unresolved = daily.difference.abs() > tolerance
    summary = {
        "file": path.name, "sha256": sha256(path), "csv_rows": len(raw),
        "nonempty_rows": len(data), "empty_rows": len(raw) - len(data),
        "types": data.transacttype.value_counts().to_dict(),
        "statuses": data.transactstatus.value_counts().to_dict(),
        "currency": data.currency.value_counts().to_dict(),
        "opening_balance_satoshi": opening,
        "closing_balance_satoshi": int(data.iloc[-1].walletbalance),
        "final_posted_balance_difference_satoshi": int(data.iloc[-1].walletbalance - (data.posted_amount.sum() + opening)),
        "date_inversions": int(days.diff().lt(pd.Timedelta(0)).sum()),
        "rounded_balance_rows": int(data.balance_precision_satoshi.gt(1).sum()),
        "daily_snapshot_mismatch_beyond_precision": int(unresolved.sum()),
        "daily_snapshot_unresolved": records(daily[unresolved].reset_index()),
        "daily_snapshot_note": "날짜순 기장액과 각 날짜의 마지막 원본 잔액을 잠정 비교. 출금 신청/완료 날짜와 같은 날의 실제 순서는 미확인. 과학 표기 잔액의 표시 정밀도를 별도로 반영함.",
        "timestamps_complete": False,
        "timestamp_limitation": "지갑 시각이 분:초 형식이므로 일중 순자산/레버리지 복원에 사용할 수 없음",
        "posted_amounts_by_type_satoshi": data.groupby("transacttype").posted_amount.sum().to_dict(),
    }
    return data, summary


def aggregate_events(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    selected = data[data.symbol.eq(symbol)].copy()
    unsupported = set(selected.exectype.unique()) - {"Trade", "Funding"}
    if unsupported:
        raise ValueError(f"{symbol}에서 아직 지원하지 않는 이벤트: {sorted(unsupported)}")
    selected["maker_fills"] = selected.lastliquidityind.eq("AddedLiquidity").astype(int)
    selected["taker_fills"] = selected.lastliquidityind.eq("RemovedLiquidity").astype(int)
    selected["price_qty"] = selected.lastpx * selected.lastqty
    # 같은 주문·시각·방향의 부분 체결만 묶어 의사결정 횟수의 과장을 줄인다.
    grouped = selected.groupby(
        ["time", "order_key", "exectype", "side", "ordtype"], sort=False, dropna=False
    ).agg(
        quantity=("lastqty", "sum"), cost_satoshi=("execcost", "sum"),
        fee_satoshi=("execcomm", "sum"), fills=("execid", "size"),
        maker_fills=("maker_fills", "sum"), taker_fills=("taker_fills", "sum"),
        price_qty=("price_qty", "sum"), source_row=("source_row", "min"),
    ).reset_index()
    grouped["price"] = grouped.price_qty / grouped.quantity.replace(0, np.nan)
    return grouped.sort_values(["time", "source_row"], kind="stable").reset_index(drop=True)


def write_audit(source: Path, destination: Path, timezone: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data, audit = load_executions(source, timezone)
    wallet, audit["wallet"] = load_wallet(source)
    save_json(destination / "audit.json", audit)
    data.to_parquet(destination / "executions.parquet", index=False)
    wallet.to_parquet(destination / "wallet.parquet", index=False)
    special = data[(data.exectype == "Trade") & data.order_key.str.startswith("special:")]
    save_json(destination / "special_executions.json", records(special))
    return data, wallet, audit
