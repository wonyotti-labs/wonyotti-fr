from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .audit import aggregate_events
from .common import new_run, records, save_json, sha256
from .reconciliation import reconcile_wallet
from .reconstruct import reconstruct
from .reports import table


def original_cost_convention(data: pd.DataFrame) -> bool:
    if not data.settlcurrency.eq("XBt").all():
        raise ValueError("사토시 원장이 아닌 계약은 별도 통화 원장이 필요합니다.")
    trades = data[(data.exectype == "Trade") & data.execcost.ne(0)]
    signs = np.sign(trades.execcost) * np.where(trades.side.eq("Buy"), 1, -1)
    if len(set(signs)) != 1:
        raise ValueError("동일 계약에서 원본 비용 방향이 일관되지 않습니다.")
    return bool(signs.iloc[0] < 0)


def reconstruct_portfolio(audit_run: Path, output: Path) -> Path:
    executions_path, wallet_path = audit_run / "executions.parquet", audit_run / "wallet.parquet"
    data, wallet = pd.read_parquet(executions_path), pd.read_parquet(wallet_path)
    if "settlcurrency" not in data:
        raise ValueError("정산 통화를 보존하는 최신 감사 실행이 필요합니다.")
    destination = new_run(output, "portfolio-reconstruction", {
        "executions_sha256": sha256(executions_path), "wallet_sha256": sha256(wallet_path),
        "cost_source": "원본 signed execCost와 정산 통화. 현재 instrument 값을 과거에 소급하지 않음.",
        "initial_position_assumption": 0,
    })
    rows = []
    for symbol, selected in data.groupby("symbol", sort=True):
        inverse = original_cost_convention(selected)
        events = aggregate_events(selected, symbol, allow_settlement=True)
        episodes, actions, summary = reconstruct(events, inverse=inverse)
        target = destination / symbol
        target.mkdir()
        episodes.to_parquet(target / "episodes.parquet", index=False)
        actions.to_parquet(target / "actions.parquet", index=False)
        counterpart = wallet[(wallet.address == symbol) & (wallet.transacttype == "RealisedPNL")]
        reconciliation = None
        if len(counterpart):
            daily, reconciliation = reconcile_wallet(actions, wallet, symbol)
            daily.to_csv(target / "daily_reconciliation.csv", index=False)
            summary["wallet_reconciliation"] = reconciliation
        save_json(target / "summary.json", summary)
        closed = episodes[episodes.closed]
        rows.append({"symbol": symbol, "inverse": inverse, "settlement_currency": "XBt",
                     "trade_fills": int(selected.exectype.eq("Trade").sum()),
                     "settlement_events": int(selected.exectype.eq("Settlement").sum()),
                     "closed_episodes": len(closed), "open_episodes": int((~episodes.closed).sum()),
                     "win_rate": float(closed.net_pnl_btc.gt(0).mean()) if len(closed) else None,
                     "realized_net_btc": summary["realized_net_btc"],
                     "wallet_net_btc": float(counterpart.amount.sum() / 1e8),
                     "aligned_residual_satoshi": reconciliation["aligned_residual_satoshi"] if reconciliation else None,
                     "outside_wallet_window_btc": reconciliation["outside_wallet_window_btc"] if reconciliation else None,
                     "funding_position_mismatches": summary["funding_position_mismatch_count"],
                     "final_contracts": summary["final_position_contracts"]})
        print(f"계약 복원 {symbol}: 종료 구간 {len(closed)}, 펀딩 수량 불일치 {summary['funding_position_mismatch_count']}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(destination / "contracts.csv", index=False)
    summary = {"contracts": len(results), "closed_episodes": int(results.closed_episodes.sum()),
               "total_realized_net_btc": float(results.realized_net_btc.sum()),
               "wallet_symbols_total_btc": float(results.wallet_net_btc.sum()),
               "sum_aligned_residual_satoshi": float(results.aligned_residual_satoshi.sum()),
               "max_abs_contract_residual_satoshi": float(results.aligned_residual_satoshi.abs().max()),
               "funding_position_mismatches": int(results.funding_position_mismatches.sum()),
               "open_contracts": records(results[results.final_contracts.ne(0)][["symbol", "final_contracts"]])}
    save_json(destination / "summary.json", summary)
    (destination / "REPORT.md").write_text(f"""# 원본 전체 계약의 포지션·손익 복원

## 범위

원본 체결의 정산 통화를 직접 확인하고, 모두 XBt인 경우 사토시로 복원한다. 현재 같은 심볼의 정산 통화나 승수가 과거와 같다고 가정하지 않는다. 매수 원본 비용의 부호와 일관성을 검사해 역선물과 선형/콴토 비용 방향을 구분했다. 선형/콴토 원가는 원본 체결 비용을 그대로 쓰므로 현재 승수를 소급 적용하지 않는다.

초기 포지션 0을 가정한다. 만기 Settlement는 원본의 실제 청산 방향·수량·비용으로 처리하며 0 가격 만기 정산도 지원한다. Funding은 원본 지급액으로 처리한다. 이는 전체 기록으로 관찰 가능한 실현손익의 복원이며 계정 소유와 현재 자산에 대한 증명이 아니다.

{table(pd.DataFrame([{k: v for k, v in summary.items() if k != 'open_contracts'}]))}

## 계약별 결과

{table(results)}

지갑은 전일 정오 이상~당일 정오 미만 UTC에 맞춰 비교한다. 원장 마지막 날짜 이후 비용은 별도 기장 구간이다. 계약별 반올림 잔차·기간 차이·펀딩 수량 불일치를 그대로 표시한다. 잔차가 남은 계약은 일치했다고 표시하지 않는다.

## 해석 제한

다른 기간과 계약의 승률을 직접 합쳐 전체 계좌 수익률로 해석하지 않는다. 입출금·미실현 손익·계약 간 동시 위험과 실제 계좌 증거가 없으므로 레버리지나 총재산을 확정할 수 없다. 현재 JSON의 미청산 계약에는 말일 시장 가격으로 표시한 미실현 손익을 넣지 않았다.
""", encoding="utf-8")
    print(f"전체 계약 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination
