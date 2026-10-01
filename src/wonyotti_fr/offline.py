from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .features import FEATURES, build_features
from .model import DirectionModel
from .research import check_funding_coverage, load_market
from .simulator import RiskConfig, simulate


def write_simulation(destination, bars, signals, funding, risk):
    curve, trades, fills, metrics = simulate(bars, signals, funding, risk)
    curve.to_parquet(destination / "equity.parquet", index=False)
    trades.to_parquet(destination / "trades.parquet", index=False)
    fills.to_parquet(destination / "fills.parquet", index=False)
    save_json(destination / "metrics.json", metrics)
    save_json(destination / "final_state.json", {"cash": float(curve.equity.iloc[-1]), "quantity": 0,
                                               "last_bar_end": bars.end.iloc[-1],
                                               "state": "completed_and_liquidated", "resume_supported": False})
    (destination / "REPORT.md").write_text(
        f"# 오프라인 시세 재생\n\n기간: {bars.time.iloc[0]} ~ {bars.end.iloc[-1]}. "
        f"{len(bars):,}개 봉, 종료 거래 {metrics['closed_trades']:,}개.\n\n"
        f"총수익 {metrics['total_return']:.2%}, 관측 최대 낙폭 {metrics['max_drawdown']:.2%}. "
        f"회계 잔차 {metrics['accounting_residual']:.8g} USDT.\n\n"
        "확정 봉에서 만든 신호를 다음 봉 시가에 실행했다. 마지막에는 비용을 포함해 청산했다. "
        "실제 주문이나 인증 API 호출은 없다. 실행 도중 재시작 복원과 실시간 모의매매는 지원하지 않는다. "
        "이는 선택된 가정에 따른 과거 시세 재생이며 수익성 인증이 아니다.\n", encoding="utf-8")
    print(f"오프라인 재생 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination


def replay(research_run: Path, market: Path, symbol: str, start: str, end: str, output: Path) -> Path:
    selected = json.loads((research_run / "frozen_selection.json").read_text())
    model_path = research_run / "model.json"
    if sha256(model_path) != selected["model_sha256"]:
        raise ValueError("선택한 모델의 체크섬이 다릅니다.")
    model = DirectionModel.from_dict(json.loads(model_path.read_text()))
    risk = RiskConfig(**selected["risk"])
    risk.validate()
    begin, finish = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if begin >= finish or begin != begin.floor("15min") or finish != finish.floor("15min"):
        raise ValueError("시작과 종료를 15분 경계에 오름차순으로 지정하세요.")
    bars, funding = load_market(market, symbol)
    check_funding_coverage(market, symbol, start, end)
    features = build_features(bars)
    signals = model.signals(features, selected["threshold"], selected["volatility_cap"])
    mask = (bars.time >= begin) & (bars.time < finish)
    sliced = bars[mask].reset_index(drop=True)
    expected = pd.date_range(begin, finish, freq="15min", inclusive="left")
    if len(sliced) != len(expected) or not np.array_equal(sliced.time.to_numpy(), expected.to_numpy()):
        raise ValueError("재생 기간에 누락 시세가 있습니다.")
    settings = {"symbol": symbol, "start": start, "end_exclusive": end, "risk": selected["risk"],
                "model_sha256": sha256(model_path), "selection_sha256": sha256(research_run / "frozen_selection.json"),
                "market_manifest_sha256": sha256(market / "manifest-15m.json")}
    destination = new_run(output, "replay", settings)
    return write_simulation(destination, sliced, signals[mask], funding, risk)


def demo(output: Path) -> Path:
    rng = np.random.default_rng(37)
    size = 4000
    closing = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, size)))
    opening = np.r_[closing[0], closing[:-1]]
    volume = rng.uniform(1000, 5000, size)
    bars = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=size, freq="15min", tz="UTC"),
                         "open": opening, "close": closing,
                         "high": np.maximum(opening, closing) * 1.001,
                         "low": np.minimum(opening, closing) * 0.999,
                         "volume": volume, "taker_buy_volume": volume * 0.5})
    bars["end"] = bars.time + pd.Timedelta(minutes=15)
    feat = build_features(bars)
    signals = np.sign(feat.trend_4h_16h.fillna(0)).to_numpy(dtype=int)
    signals[feat[FEATURES].isna().any(axis=1)] = 0
    funding = pd.DataFrame({"time": bars.time.iloc[::32], "rate": 0.0001})
    destination = new_run(output, "synthetic-demo", {"seed": 37, "synthetic": True,
                                                   "strategy": "EMA demonstration, not trader imitation"})
    return write_simulation(destination, bars, signals, funding, RiskConfig())
