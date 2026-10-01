from __future__ import annotations

import pandas as pd

from .common import records


def reconcile_wallet(actions: pd.DataFrame, wallet: pd.DataFrame, symbol: str = "XBTUSD") -> tuple[pd.DataFrame, dict]:
    posted = wallet[(wallet.address == symbol) & (wallet.transacttype == "RealisedPNL")
                    & wallet.transactstatus.eq("Completed")].copy()
    if posted.empty:
        raise ValueError(f"{symbol} 지갑 실현손익이 없습니다.")
    if not posted.currency.eq("XBt").all():
        raise ValueError("현재 대조는 사토시 원장만 지원합니다.")
    ledger = posted.groupby("date").amount.sum().sort_index() / 1e8
    events = actions.copy()
    # 날짜 D의 원장은 전일 12:00 이상, 당일 12:00 미만 UTC 사건을 집계한다.
    events["posting_date"] = (events.time + pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    events["net_btc"] = events.realized_btc - events.fee_btc
    calculated = events.groupby("posting_date").net_btc.sum()
    daily = pd.concat([ledger.rename("wallet_btc"), calculated.rename("calculated_btc")], axis=1).fillna(0).sort_index()
    daily.index.name = "posting_date"
    daily["difference_satoshi"] = (daily.calculated_btc - daily.wallet_btc) * 1e8
    covered = (daily.index >= ledger.index.min()) & (daily.index <= ledger.index.max())
    tail = events[(events.posting_date < ledger.index.min()) | (events.posting_date > ledger.index.max())]
    residual = float(daily.loc[covered, "difference_satoshi"].sum())
    summary = {
        "symbol": symbol, "settlement_currency": "XBt", "posting_boundary_utc_hour": 12,
        "boundary_inclusion": "previous_noon_inclusive_current_noon_exclusive",
        "wallet_first_posting_date": ledger.index.min(), "wallet_last_posting_date": ledger.index.max(),
        "full_history_difference_btc": float(calculated.sum() - ledger.sum()),
        "outside_wallet_window_btc": float(tail.net_btc.sum()),
        "outside_wallet_window_events": records(tail[["time", "action", "fee_btc", "net_btc", "posting_date"]]),
        "aligned_residual_satoshi": residual,
        "daily_median_abs_difference_satoshi": float(daily.loc[covered, "difference_satoshi"].abs().median()),
        "daily_max_abs_difference_satoshi": float(daily.loc[covered, "difference_satoshi"].abs().max()),
        "aggregate_within_one_satoshi": abs(residual) < 1,
        "source": "https://support.bitmex.com/hc/en-gb/articles/6205277211037-How-do-I-manually-calculate-my-Realised-PNL",
        "note": "가중평균 원가의 일별 배분과 원장 사토시 반올림에 따른 작은 일별 잔차는 유지한다.",
    }
    return daily.reset_index(), summary
