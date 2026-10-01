from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import save_json


def table(frame: pd.DataFrame) -> str:
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:,.6g}" if np.isfinite(value) else "해당 없음"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(map(format_value, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def episode_stats(episodes: pd.DataFrame) -> dict:
    closed = episodes[episodes.closed].copy()
    wins = closed.net_pnl_btc > 0
    profit = closed.loc[wins, "net_pnl_btc"].sum()
    loss = -closed.loc[closed.net_pnl_btc < 0, "net_pnl_btc"].sum()
    return {
        "closed_episodes": len(closed), "open_episodes": int((~episodes.closed).sum()),
        "win_rate": float(wins.mean()) if len(closed) else None,
        "net_closed_btc": float(closed.net_pnl_btc.sum()),
        "profit_factor": float(profit / loss) if loss else None,
        "median_hold_minutes": float(closed.hold_minutes.median()) if len(closed) else None,
        "additional_entry_episode_fraction": float((closed.additional_entry_orders > 0).mean()) if len(closed) else None,
    }


def write_reconstruction_report(destination: Path, audit: dict, reconstruction: dict,
                                episodes: pd.DataFrame, actions: pd.DataFrame) -> None:
    stats = episode_stats(episodes)
    save_json(destination / "episode_statistics.json", stats)
    closed = episodes[episodes.closed].copy()
    closed["year"] = closed.exit_time.dt.year
    grouped = closed.groupby(["year", "direction"]).agg(
        episodes=("episode_id", "size"), win_rate=("net_pnl_btc", lambda s: float((s > 0).mean())),
        net_btc=("net_pnl_btc", "sum"), median_hold_min=("hold_minutes", "median"),
        scaled_in_fraction=("additional_entry_orders", lambda s: float((s > 0).mean())),
    ).reset_index()
    selected = ["episode_id", "entry_time", "exit_time", "direction", "initial_qty", "max_qty",
                "additional_entry_orders", "hold_minutes", "net_pnl_btc"]
    worst = closed.nsmallest(15, "net_pnl_btc")[selected]
    best = closed.nlargest(15, "net_pnl_btc")[selected]
    grouped.to_csv(destination / "year_direction.csv", index=False)
    worst.to_csv(destination / "worst_episodes.csv", index=False)
    best.to_csv(destination / "best_episodes.csv", index=False)
    concentration = closed.nlargest(10, "net_pnl_btc").net_pnl_btc.sum()
    normal = closed.groupby(closed.additional_entry_orders.gt(0)).agg(
        episodes=("episode_id", "size"), win_rate=("net_pnl_btc", lambda x: float((x > 0).mean())),
        net_btc=("net_pnl_btc", "sum"), median_pnl_btc=("net_pnl_btc", "median"),
    ).reset_index(names="additional_entry")
    report = f"""# XBTUSD 공개 거래 기록 복원

이 결과는 로컬 연구 자료다. 개별 체결과 추정 전략의 재배포 허락을 의미하지 않는다.

## 범위와 품질

- 전체 실행 기록: {audit['rows']:,}행. 거래 {audit['types'].get('Trade', 0):,}행.
- 정상 주문 ID {audit['unique_normal_orderids']:,}개. 특수 ID 체결 {audit['special_orderid_trade_rows']:,}행은 개별 이벤트로 유지.
- 시간대: {audit['source_timezone_assumption']} 가정. 외부 체결로 독립 확인되지 않음.
- 전체 종목 중 XBTUSD만 포지션 복원. 다른 계약의 승수와 정산은 일반화하지 않음.
- 시작 포지션 0 가정. 펀딩 수량 대조 불일치 {reconstruction['funding_position_mismatch_count']}건.
- 마지막 포지션: {reconstruction['final_position_contracts']:,} 계약.
- 완료된 거래 금액 합계와 마지막 지갑 잔액 차이: {audit['wallet']['final_posted_balance_difference_satoshi']} 사토시. 취소 출금은 기장하지 않음.
- 지갑 날짜 역전 {audit['wallet']['date_inversions']}곳, 표시 정밀도가 낮은 잔액 {audit['wallet']['rounded_balance_rows']}행. 날짜별 마지막 잔액을 잠정 대조하면 표시 정밀도 밖 차이 {audit['wallet']['daily_snapshot_mismatch_beyond_precision']}일. 세부 사항은 audit.json에 기록.
- 지갑 시각은 불완전하여 일중 순자산, 실제 레버리지, 미실현 손익을 포함한 계좌 낙폭을 여기서 계산하지 않음.

## 손익 대조

- 계산한 실현손익(수수료·펀딩 차감): {reconstruction['realized_net_btc']:,.8f} BTC.
- 지갑 XBTUSD 실현손익: {reconstruction['wallet_xbtusd_pnl_btc']:,.8f} BTC.
- 차이(계산값 - 지갑): {reconstruction['wallet_difference_btc']:,.10f} BTC.
- 지갑 날짜는 전일 12:00 이상, 당일 12:00 미만 UTC의 실현손익을 집계한다. 마지막 지갑 날짜 범위 밖의 사건 순손익: {reconstruction['wallet_reconciliation']['outside_wallet_window_btc']:,.10f} BTC.
- 같은 기간에 맞춘 누적 잔차: {reconstruction['wallet_reconciliation']['aligned_residual_satoshi']:,.6f} 사토시. 세부 일별 대조는 wallet_daily_reconciliation.csv와 reconstruction.json에 보존한다.
- 원본 체결 비용을 사용한 가중평균 원가 방식. 표시 단위와 일별 반올림의 작은 잔차를 강제로 0으로 맞추지 않는다.
- 원본 내부 일관성은 자료의 외부 진위나 현재 총재산을 입증하지 않음.

## 매매 행동

{table(pd.DataFrame([stats]))}

{table(grouped)}

추가 진입은 최초 주문 이후 별도 주문으로 포지션을 늘린 경우를 센다. 한 주문의 부분 체결은 추가 진입 의사결정으로 반복 계산하지 않는다. initial_qty는 최초 시각의 체결 묶음이며 최초 주문 전체 수량과 다를 수 있다. 최대 계약 수는 레버리지가 아니다.

{table(normal)}

상위 10개 종료 에피소드 순손익 합계: {concentration:,.8f} BTC. BTC 손익은 계정 성장에 따라 규모가 달라지므로 이후 전략 비교에서 고정 자본으로 정규화해야 한다.

## 가장 큰 손실 15건

{table(worst)}

## 가장 큰 이익 15건

{table(best)}

## 아직 결론낼 수 없는 사항

1. 수익·손실을 보고 거래를 제거하면 사후 선택 편향이다. 손실 직전에 알 수 있었던 특징으로 필터를 만들고 다른 기간에서 평가해야 한다.
2. 지정가 주문에는 메이커와 테이커 체결이 모두 포함될 수 있다. 주문 유형과 유동성 제공 여부를 구분한다.
3. 공개 체결은 최초 주문 시각, 취소 주문, 보지 않은 기회, 뉴스·호가 판단을 모두 설명하지 않는다.
4. 본 보고서의 누적 실현손익 그림은 입출금·미실현손익을 포함한 계좌 수익률 곡선이 아니다.

![Realized PnL](realized_pnl.png)
"""
    (destination / "REPORT.md").write_text(report, encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(closed.exit_time, closed.net_pnl_btc.cumsum(), lw=1)
    axes[0].set(ylabel="Closed-episode net BTC", title="XBTUSD: realized episode PnL (not account equity)")
    axes[1].plot(actions.time, actions.after_qty, lw=0.5)
    axes[1].set(ylabel="Signed contracts", xlabel="Time (assumed UTC)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(destination / "realized_pnl.png", dpi=150)
    plt.close(fig)
