from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, records, save_json, sha256
from .features import build_features
from .model import DirectionModel
from .reports import table
from .research import check_funding_coverage, load_market
from .simulator import RiskConfig, simulate


def block_interval(returns: np.ndarray, seed: int = 41, samples: int = 1000, block_days: int = 30) -> dict:
    if len(returns) < 2 * block_days or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("블록 재표집에 충분한 유효 일별 수익률이 필요합니다.")
    rng = np.random.default_rng(seed)
    blocks = (len(returns) + block_days - 1) // block_days
    starts = rng.integers(0, len(returns), size=(samples, blocks))
    indices = ((starts[:, :, None] + np.arange(block_days)) % len(returns)).reshape(samples, -1)[:, :len(returns)]
    annual = np.expm1(np.log1p(returns[indices]).sum(axis=1) * 365.25 / len(returns))
    low, middle, high = np.quantile(annual, [0.025, 0.5, 0.975])
    return {"annual_return_p025": float(low), "annual_return_p50": float(middle),
            "annual_return_p975": float(high), "fraction_above_cash": float((annual > 0).mean()),
            "days": len(returns), "samples": samples, "block_days": block_days, "seed": seed}


def run_robustness(research_run: Path, market: Path, output: Path) -> Path:
    selection_path = research_run / "frozen_selection.json"
    selected = json.loads(selection_path.read_text())
    model_path = research_run / "model.json"
    if sha256(model_path) != selected["model_sha256"]:
        raise ValueError("고정 모델 체크섬 불일치")
    model = DirectionModel.from_dict(json.loads(model_path.read_text()))
    results = pd.read_csv(research_run / "test_results.csv")
    risk = RiskConfig(**selected["risk"])
    settings = {"selection_sha256": sha256(selection_path), "model_sha256": sha256(model_path),
                "market_manifest_sha256": sha256(market / "manifest-15m.json"),
                "role": "observed_evaluation_followup_no_retuning", "block_days": 30, "bootstrap_samples": 1000}
    destination = new_run(output, "robustness", settings)
    uncertainty = []
    curve_hashes = {}
    for row in results.itertuples(index=False):
        path = research_run / row.symbol / row.strategy / "equity.parquet"
        curve_hashes[f"{row.symbol}/{row.strategy}"] = sha256(path)
        curve = pd.read_parquet(path)
        daily = curve.groupby((curve.time - pd.Timedelta(nanoseconds=1)).dt.date).equity.last()
        returns = (daily / daily.shift(1, fill_value=risk.initial_equity) - 1).to_numpy()
        uncertainty.append({"symbol": row.symbol, "strategy": row.strategy, **block_interval(returns)})
    save_json(destination / "input_curve_hashes.json", curve_hashes)
    annual = []
    for symbol in results.symbol.unique():
        bars, funding = load_market(market, symbol)
        signals = model.signals(build_features(bars), selected["threshold"], selected["volatility_cap"])
        for year in range(2022, 2026):
            start, end = f"{year}-01-01", f"{year + 1}-01-01"
            check_funding_coverage(market, symbol, start, end)
            mask = (bars.time >= start) & (bars.time < end)
            _, trades, _, metrics = simulate(bars[mask].reset_index(drop=True), signals[mask], funding, risk)
            annual.append({"symbol": symbol, "year": year,
                           **{k: metrics[k] for k in ["total_return", "max_drawdown", "closed_trades", "permanent_halt"]}})
            target = destination / f"{symbol}-{year}"
            target.mkdir()
            trades.to_parquet(target / "trades.parquet", index=False)
            save_json(target / "metrics.json", metrics)
    uncertainty = pd.DataFrame(uncertainty)
    annual = pd.DataFrame(annual)
    uncertainty.to_csv(destination / "block_bootstrap.csv", index=False)
    annual.to_csv(destination / "annual_restart.csv", index=False)
    save_json(destination / "results.json", {"annual_restart": records(annual), "uncertainty": records(uncertainty),
                                             "candidate_decision": "수익성 있는 전략으로 채택하지 않는다."})
    (destination / "REPORT.md").write_text(f"""# 첫 후보의 후속 강건성 점검

2022~2025 결과를 이미 관찰한 뒤 수행한 후속 진단이다. 모델·신뢰도·위험 설정을 수정하지 않았다. 이 결과를 새 미사용 평가라고 부르지 않는다. 전략이 손실이면 이를 기록하고 수익성 있는 전략으로 채택하지 않는다.

## 연도별 독립 재시작

{table(annual)}

매년 자본 10,000 USDT와 중지 상태를 초기화했다. 다년 실행이 누적 낙폭 한도로 일찍 멈춘 효과를 구분하기 위한 진단이다. 매년 새 돈을 넣는 연속 운용 성과도 아니고, 재학습 워크포워드도 아니다. 원자료에 2022년 이후 본인의 행동 정답이 없어 방향 모사 모델을 그 기간에 재학습하지 않았다.

## 일별 수익률 블록 재표집

{table(uncertainty)}

30일 연속 블록을 순환 방식으로 1,000회 재표집했다. 연환산 수익률의 2.5%, 50%, 97.5% 분위와 현금 수익 0보다 큰 표본 비중이다. 고정한 모델과 관찰한 한 경로에 조건부인 요약으로, 미래 수익 확률이나 통계적 유의성 인증이 아니다. 여러 후보를 선택한 불확실성과 시장의 구조 변화는 포함하지 못한다. 영구 중지 뒤 현금 기간도 포함한다.

## 해석

위험 제어로 손실이 줄었다고 전략 자체의 양의 기대수익이 입증된 것은 아니다. 실제 체결에서의 우위를 복원하려면 시간대 확인, 사건별 판단 모델, 비용·유동성 조건과 새로운 평가 자료가 필요하다. 다음 실험 계획에서 이미 관찰한 자료와 새 자료를 명시한다.
""", encoding="utf-8")
    print(f"후속 검증 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination
