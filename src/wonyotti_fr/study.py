from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, records, save_json, sha256
from .features import build_features, decision_context
from .reports import table
from .research import load_market


def enrich_actions(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if len(actions) != len(events) or not actions.time.equals(events.time) or not actions.order_key.equals(events.order_key):
        raise ValueError("행동과 체결 사건의 순서가 다릅니다.")
    result = actions.copy()
    basis = 0.0
    moves, new_adds = [], []
    seen: set[str] = set()
    for action, event in zip(actions.itertuples(index=False), events.itertuples(index=False), strict=True):
        before, after = action.before_qty, action.after_qty
        entry_price = 1e8 / basis if basis > 0 else np.nan
        moves.append(float(np.sign(before) * (action.price / entry_price - 1)) if before else np.nan)
        new_adds.append(action.action == "increase" and action.order_key not in seen)
        if event.exectype == "Funding":
            continue
        unit_cost = abs(event.cost_satoshi) / event.quantity
        if action.action in {"open", "reverse"}:
            basis, seen = unit_cost, set()
        elif action.action == "increase":
            basis = (basis * abs(before) + unit_cost * abs(after - before)) / abs(after)
        elif action.action == "close":
            basis = 0.0
        seen.add(action.order_key)
    result["favorable_move_before"] = moves
    result["new_increase_order"] = new_adds
    return result


def summarize_groups(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return frame.groupby(by, observed=True).agg(
        episodes=("episode_id", "size"), win_rate=("net_pnl_btc", lambda v: (v > 0).mean()),
        net_btc=("net_pnl_btc", "sum"), median_btc=("net_pnl_btc", "median"),
        median_hold_minutes=("hold_minutes", "median"), p90_hold_minutes=("hold_minutes", lambda v: v.quantile(0.9)),
        scaled_in_fraction=("additional_entry_orders", lambda v: (v > 0).mean()),
        median_additional_orders=("additional_entry_orders", "median"),
        median_max_contracts=("max_qty", "median"),
    ).reset_index()


def run_study(audit_run: Path, market: Path, output: Path) -> Path:
    settings = {"audit_inputs": {name: sha256(audit_run / name) for name in
                                ["episodes.parquet", "actions.parquet", "xbtusd_events.parquet", "audit.json"]},
                "market_manifest_sha256": sha256(market / "manifest-15m.json"),
                "role": "descriptive_only_not_strategy_selection"}
    destination = new_run(output, "behavior-study", settings)
    episodes = pd.read_parquet(audit_run / "episodes.parquet")
    actions = pd.read_parquet(audit_run / "actions.parquet")
    events = pd.read_parquet(audit_run / "xbtusd_events.parquet")
    enriched = enrich_actions(actions, events)
    enriched.to_parquet(destination / "actions_with_prior_position.parquet", index=False)
    first_order = actions[actions.action.isin(["open", "reverse"])].set_index("episode_id").order_key
    entries = actions[actions.action.isin(["open", "increase", "reverse"])].copy()
    entries["opening_qty"] = np.where(entries.action.eq("reverse"), entries.after_qty.abs(),
                                       (entries.after_qty - entries.before_qty).abs())
    first_fills = entries[entries.order_key.eq(entries.episode_id.map(first_order))]
    episodes["first_order_filled_contracts"] = episodes.episode_id.map(first_fills.groupby("episode_id").opening_qty.sum())
    episodes["peak_to_first_order_ratio"] = episodes.max_qty / episodes.first_order_filled_contracts
    closed = episodes[episodes.closed].copy()
    closed["outcome"] = np.where(closed.net_pnl_btc > 0, "profit", "loss_or_flat")
    closed["exit_year"] = closed.exit_time.dt.year
    closed["hold_band"] = pd.cut(closed.hold_minutes, [-1, 5, 30, 120, 1440, 10080, np.inf],
                                 labels=["0-5m", "5-30m", "30m-2h", "2h-1d", "1d-7d", "7d+"])
    outcomes = summarize_groups(closed, ["outcome"])
    hold = summarize_groups(closed, ["hold_band"])
    yearly = summarize_groups(closed, ["exit_year", "direction"])
    adds = enriched[enriched.new_increase_order].copy()
    adds["position_state"] = np.where(adds.favorable_move_before < -1e-8, "adverse",
                                      np.where(adds.favorable_move_before > 1e-8, "favorable", "flat"))
    add_summary = adds.groupby("position_state").agg(
        orders=("order_key", "size"), episodes=("episode_id", "nunique"),
        median_move_before=("favorable_move_before", "median")).reset_index()
    bars, _ = load_market(market, "BTCUSDT")
    features = build_features(bars)
    entry_context = decision_context(actions[actions.action.isin(["open", "reverse"])], features)
    entry_context["direction"] = np.sign(entry_context.after_qty)
    entry_context["directional_prior_1h"] = entry_context.direction * entry_context.return_1h
    entry_context["directional_prior_4h"] = entry_context.direction * entry_context.return_4h
    available = entry_context.dropna(subset=["directional_prior_1h", "directional_prior_4h"])
    entry_description = available.groupby("direction").agg(
        entries=("episode_id", "size"), opposite_prior_1h_fraction=("directional_prior_1h", lambda v: (v < 0).mean()),
        opposite_prior_4h_fraction=("directional_prior_4h", lambda v: (v < 0).mean()),
        median_volume_ratio=("volume_ratio", "median"), median_daily_volatility=("volatility_1d", "median")).reset_index()
    # 가격 경로는 사후 설명에만 쓴다. 포지션 수량 변화나 정확한 평가손익이 아니다.
    excursions = []
    bar_starts = bars.time.astype("datetime64[ns, UTC]")
    bar_ends = bars.end.astype("datetime64[ns, UTC]")
    for episode in closed.itertuples(index=False):
        if episode.entry_time < bars.time.min() or episode.exit_time > bars.end.max():
            continue
        start = bar_starts.searchsorted(episode.entry_time, side="left")
        stop = bar_ends.searchsorted(episode.exit_time, side="right")
        inside = bars.iloc[start:stop]
        if inside.empty:
            continue
        high, low = inside.high.max(), inside.low.min()
        best = (high / episode.entry_price - 1) if episode.direction > 0 else (1 - low / episode.entry_price)
        worst = (low / episode.entry_price - 1) if episode.direction > 0 else (1 - high / episode.entry_price)
        excursions.append({"episode_id": episode.episode_id, "outcome": episode.outcome,
                           "fully_contained_bars": len(inside), "favorable_price_excursion": max(0.0, best),
                           "adverse_price_excursion": min(0.0, worst)})
    paths = pd.DataFrame(excursions)
    path_summary = paths.groupby("outcome").agg(episodes=("episode_id", "size"),
                      median_adverse=("adverse_price_excursion", "median"),
                      median_favorable=("favorable_price_excursion", "median")).reset_index() if len(paths) else pd.DataFrame()
    profits = closed.loc[closed.net_pnl_btc > 0, "net_pnl_btc"]
    losses = -closed.loc[closed.net_pnl_btc < 0, "net_pnl_btc"]
    concentration = {"gross_positive_btc": float(profits.sum()), "gross_negative_btc": float(losses.sum()),
                     "top10_profit_share": float(profits.nlargest(10).sum() / profits.sum()),
                     "top10_loss_share": float(losses.nlargest(10).sum() / losses.sum()),
                     "mean_profit_btc": float(profits.mean()), "mean_loss_btc": float(losses.mean()),
                     "new_increase_orders": len(adds), "adverse_increase_fraction": float(adds.position_state.eq("adverse").mean()),
                     "market_entry_coverage": len(available), "total_entries": len(entry_context)}
    episodes.to_parquet(destination / "episode_details.parquet", index=False)
    entry_context.to_parquet(destination / "entry_context.parquet", index=False)
    paths.to_parquet(destination / "price_excursions.parquet", index=False)
    for name, frame in [("outcomes", outcomes), ("holding_period", hold), ("year_direction", yearly),
                        ("scale_in_state", add_summary), ("entry_behavior", entry_description), ("price_paths", path_summary)]:
        frame.to_csv(destination / f"{name}.csv", index=False)
    save_json(destination / "findings.json", {**concentration, "outcomes": records(outcomes),
                                              "entry_behavior": records(entry_description)})
    report = f"""# XBTUSD 행동·손실 심층 관찰

본 보고서는 원본 기록을 설명하는 사후 분석이다. 아래 손익·보유 시간·경로는 진입 때 알 수 없으며 필터 학습에 투입하지 않았다. 통계는 원본 시간대 UTC와 시작 포지션 0을 가정한다. 기록의 진위나 실제 판단 의도를 인증하지 않는다.

## 승률과 손실 크기

{table(outcomes)}

이익 구간의 평균 순손익은 {concentration['mean_profit_btc']:.6f} BTC, 손실 구간의 평균 손실은 {concentration['mean_loss_btc']:.6f} BTC다. 큰 이익 10개의 이익 총액 비중은 {concentration['top10_profit_share']:.2%}, 큰 손실 10개의 손실 총액 비중은 {concentration['top10_loss_share']:.2%}다. 계좌 규모가 변하므로 BTC 금액 비교를 고정 자본 수익률로 해석하지 않는다.

## 보유 시간과 추가 진입

{table(hold)}

보유 시간이 긴 손실을 관찰했다고 해당 거래를 제거하지 않는다. 시간 제한은 승리한 장기 거래도 함께 끊으므로 별도 기간에서 비용과 기회손실을 검증해야 한다.

첫 진입 주문의 체결량은 최초 시각의 일부 체결량과 구분했다. `episode_details.parquet`의 `first_order_filled_contracts`는 해당 에피소드의 최초 주문으로 실제 진입한 수량이며 제출한 주문 전체 수량과는 다를 수 있다. `peak_to_first_order_ratio`도 계좌 레버리지가 아니다.

## 추가 진입 직전의 상태

{table(add_summary)}

별도 추가 진입 주문 {len(adds):,}건 중 기존 가중평균 진입 가격 대비 불리한 가격에서 증가한 비율은 {concentration['adverse_increase_fraction']:.2%}다. 원본 역선물 체결 원가로 직전 평균 진입 가격을 추적했다. 미실현 손익에 수수료·펀딩은 넣지 않은 가격 방향 비교이며, 추가 진입의 의도를 확인한 것은 아니다. 같은 주문의 후속 체결은 독립 주문으로 세지 않았다.

## 진입 직전 시장 움직임

전체 진입 {len(entry_context):,}건 중 특징을 연결한 표본은 {len(available):,}건이다. 2020년 이전은 이번 바이낸스 대용 시세 범위에 없다. `opposite_prior_*_fraction`은 직전 상승에 매도하거나 하락에 매수한 비율이다.

{table(entry_description)}

관찰된 단기 반대 방향 진입은 역추세 가설의 근거가 될 수 있지만, 호가·뉴스·다른 거래소 상황·취소 주문이 없어 실제 전략이라고 확정할 수 없다. 모든 15분봉의 보유 방향을 모사하는 첫 모델은 거래 중인 시간에 표본이 몰리는 한계가 있다. 다음 버전에서는 진입·추가·축소·청산을 따로 예측하고 미거래 시점의 비교 표본을 명시한다.

## 보유 중 가격 경로

{table(path_summary)}

기간 안에 완전히 포함된 15분봉만 사용했다. 최초 체결 가격 대비 바이낸스 가격 경로이며 실제 BitMEX 미실현 손익이 아니다. 짧은 매매와 부분 봉은 제외돼 선택 편향이 있다. 최초 이후 변하는 포지션 크기와 원가를 반영하지 않아 실제 손절 가능 가격이나 손절 후 성과로 해석할 수 없다.

## 연도와 방향

{table(yearly)}

## 다음에 검증할 가설

1. 단기 하락 후 매수·상승 후 매도라는 진입 가설을 사건 단위 모델로 분리한다. 단순 역추세 기준과도 비교한다.
2. 추가 진입을 먼저 금지한 기준과 제한적으로 허용한 기준을 비교하고, 손실 때 늘어난 위험이 다른 기간에서도 반복되는지 검증한다.
3. 최대 보유 시간과 변동성 제한을 각각 제거하는 실험으로 어느 장치가 손실과 이익을 함께 줄였는지 확인한다.
4. 체결의 다수가 지정가이므로 호가 자료 없이는 대기열 이점을 모사할 수 없다. 현재 시장가 연구 후보를 본인의 전략 복제라고 부르지 않는다.

현재 표를 보고 정한 새 규칙은 기존에 관찰한 평가 구간에서 다시 검증하더라도 탐색 결과로 기록한다. 외부 체결과 시간 확인, 지갑 대조 잔차 해결은 계속 미완료다.
"""
    (destination / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"행동 분석 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination
