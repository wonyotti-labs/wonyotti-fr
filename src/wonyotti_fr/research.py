from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .common import new_run, records, save_json, sha256
from .features import FEATURES, build_features, decision_context, position_labels
from .model import DirectionModel
from .reports import table
from .simulator import RiskConfig, simulate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_market(market: Path, symbol: str, interval: str = "15m") -> tuple[pd.DataFrame, pd.DataFrame]:
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("지원하지 않는 시세 간격")
    manifest = json.loads((market / f"manifest-{interval}.json").read_text())
    summary = manifest["summary"][symbol]
    paths = [(market / summary[k]["file"]).resolve() for k in ["klines", "fundingRate"]]
    for key, path in zip(["klines", "fundingRate"], paths, strict=True):
        if not path.is_relative_to(market.resolve()):
            raise ValueError("시장 자료 경로가 지정한 폴더를 벗어납니다.")
        if sha256(path) != summary[key]["sha256"]:
            raise ValueError(f"정규화 시장 파일이 변경되었습니다: {path.name}")
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def check_funding_coverage(market: Path, symbol: str, start: str, end: str, interval: str = "15m"):
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("지원하지 않는 시세 간격")
    manifest = json.loads((market / f"manifest-{interval}.json").read_text())
    required = set(str(p) for p in pd.period_range(start[:7], (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m"), freq="M"))
    available = {r["month"] for r in manifest["successful"] if r["symbol"] == symbol and r["kind"] == "fundingRate"}
    if required - available:
        raise ValueError(f"{symbol} 펀딩 자료가 없는 평가 기간: {sorted(required - available)}")


def run_research(audit_run: Path, market: Path, output: Path, symbols: list[str]) -> Path:
    if len(set(symbols)) != len(symbols) or not symbols or any(s not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"} for s in symbols):
        raise ValueError("연구 지원 심볼은 BTCUSDT, ETHUSDT, SOLUSDT입니다.")
    settings = {
        "train": ["2020-01-01", "2020-12-31"],
        "validation": ["2021-01-01", "2022-01-01"],
        "test": ["2022-01-01", "2026-01-01"], "symbols": symbols,
        "purge": "학습 말일 24시간 제외. 정답은 15분 뒤 보유 방향.",
        "features": FEATURES, "model": "standardized balanced logistic regression C=0.1",
        "selection": "2021 BTC 순수익 - 0.5*최대낙폭 절댓값, 종료 거래 20개 이상",
        "execution": "15분봉 확정 후 다음 봉 시가 시장가, 수수료 5bp + 슬리피지 3bp 편도",
        "funding": "공식 펀딩률, 시가를 마크 가격 대용으로 사용, 1초 미만 정산 기록 지연 정규화",
        "warning": "BitMEX 시간대 독립 확인 전 UTC 가정. Binance는 시장 특징의 대용 데이터.",
        "audit_input_sha256": {name: sha256(audit_run / name) for name in ["actions.parquet", "episodes.parquet", "audit.json"]},
        "market_manifest_sha256": sha256(market / "manifest-15m.json"),
    }
    destination = new_run(output, "research", settings)
    (destination / "market_manifest.json").write_bytes((market / "manifest-15m.json").read_bytes())
    print(f"연구 시작: {destination}", flush=True)
    actions = pd.read_parquet(audit_run / "actions.parquet")
    episodes = pd.read_parquet(audit_run / "episodes.parquet")
    btc, btc_funding = load_market(market, "BTCUSDT")
    features = build_features(btc)
    labeled = position_labels(features, actions)
    train = labeled[(labeled.end >= settings["train"][0]) & (labeled.label_time < settings["train"][1])].dropna(subset=FEATURES + ["label"])
    model = DirectionModel.fit(train)
    save_json(destination / "model.json", model.to_dict())
    distribution = train.label.value_counts().sort_index().to_dict()
    coefficients = pd.DataFrame(model.coefficients.T, index=FEATURES,
                                columns=[f"class_{v}" for v in model.classes] if len(model.classes) == 3 else ["binary_logit"])
    coefficients.to_csv(destination / "coefficients.csv")
    validation_labels = labeled[(labeled.end >= "2021-01-01") & (labeled.label_time < "2022-01-01")].dropna(subset=FEATURES + ["label"])
    predicted = model.signals(validation_labels, 0)
    imitation_metrics = {"train_rows": len(train), "train_class_counts": {str(k): int(v) for k, v in distribution.items()},
                         "validation_rows": len(validation_labels),
                         "validation_balanced_accuracy": float(balanced_accuracy_score(validation_labels.label.to_numpy(dtype=int), predicted)),
                         "confusion_matrix_labels": [-1, 0, 1],
                         "confusion_matrix": confusion_matrix(validation_labels.label.to_numpy(dtype=int), predicted, labels=[-1, 0, 1]).tolist()}
    save_json(destination / "imitation.json", imitation_metrics)
    context = decision_context(actions, features)
    context.to_parquet(destination / "decision_context.parquet", index=False)
    entry = context[context.action.isin(["open", "reverse"])].merge(
        episodes[["episode_id", "closed", "net_pnl_btc", "hold_minutes"]], on="episode_id", how="left")
    entry["outcome"] = np.where(entry.closed & (entry.net_pnl_btc > 0), "profit",
                                np.where(entry.closed, "loss_or_flat", "open"))
    entry["direction"] = np.sign(entry.after_qty)
    entry_summary = entry.dropna(subset=FEATURES).groupby(["direction", "outcome"])[FEATURES].median().reset_index()
    entry_summary.to_csv(destination / "entry_context_medians.csv", index=False)
    entry_counts = entry.groupby(["direction", "outcome"]).size().reset_index(name="count")
    vol_cap = float(train.volatility_1d.quantile(0.90))
    candidates = []
    for threshold in [0.45, 0.60]:
        for stop in [0.02, 0.04]:
            for cap in [None, vol_cap]:
                candidates.append({"threshold": threshold, "volatility_cap": cap,
                                   "risk": asdict(RiskConfig(stop_fraction=stop))})
    save_json(destination / "candidate_plan.json", {"candidates": candidates, "count": len(candidates),
                                                   "evaluation_order": "validation, freeze selection, test"})
    check_funding_coverage(market, "BTCUSDT", "2021-01-01", "2022-01-01")
    val_mask = (btc.time >= "2021-01-01") & (btc.time < "2022-01-01")
    val_bars = btc[val_mask].reset_index(drop=True)
    validation_results = []
    for index, candidate in enumerate(candidates):
        signals = model.signals(features, candidate["threshold"], candidate["volatility_cap"])
        _, _, _, metrics = simulate(val_bars, signals[val_mask], btc_funding, RiskConfig(**candidate["risk"]))
        eligible = metrics["closed_trades"] >= 20
        score = metrics["total_return"] - 0.5 * abs(metrics["max_drawdown"])
        validation_results.append({"candidate": index, "eligible": eligible, "score": score, **metrics})
        print(f"개발 후 검증 후보 {index + 1}/{len(candidates)}: 순수익 {metrics['total_return']:.2%}", flush=True)
    eligible_results = [r for r in validation_results if r["eligible"]]
    if not eligible_results:
        raise ValueError("검증 거래 수가 충분한 후보가 없습니다. 평가 기준을 사후 변경하지 않습니다.")
    chosen = max(eligible_results, key=lambda value: value["score"])
    selected = candidates[chosen["candidate"]]
    save_json(destination / "validation.json", validation_results)
    save_json(destination / "frozen_selection.json", {"candidate": chosen["candidate"], **selected,
                                                      "model_sha256": sha256(destination / "model.json"),
                                                      "selected_on": "BTCUSDT 2021 only", "test_opened_at_freeze": False})
    save_json(destination / "evaluation_observed.json", {"test_period": settings["test"], "symbols": symbols,
                                                         "status": "evaluation_started", "do_not_reuse_as_unseen_holdout": True})
    rows, stress_rows, yearly_rows = [], [], []
    chart, axes = plt.subplots(len(symbols), 1, figsize=(12, 3.5 * len(symbols)), squeeze=False, layout="constrained")
    for axis, symbol in zip(axes[:, 0], symbols, strict=True):
        bars, funding = load_market(market, symbol)
        check_funding_coverage(market, symbol, "2022-01-01", "2026-01-01")
        feat = build_features(bars)
        mask = (bars.time >= "2022-01-01") & (bars.time < "2026-01-01")
        test_bars = bars[mask].reset_index(drop=True)
        raw_signals = model.signals(feat, selected["threshold"])
        protected = model.signals(feat, selected["threshold"], selected["volatility_cap"])
        trend = np.where((feat.trend_4h_16h > 0) & (feat.trend_16h_64h > 0), 1,
                         np.where((feat.trend_4h_16h < 0) & (feat.trend_16h_64h < 0), -1, 0))
        trend[feat[FEATURES].isna().any(axis=1)] = 0
        unprotected = RiskConfig(stop_fraction=0, max_hold_bars=0, cooldown_bars=0,
                                 daily_loss_limit=1, max_drawdown=1)
        strategies = {"imitation_risk": (protected, RiskConfig(**selected["risk"])),
                      "imitation_without_risk_filters": (raw_signals, unprotected),
                      "ema_baseline": (trend, RiskConfig(**selected["risk"])),
                      "half_capital_long_hold": (np.ones(len(bars), dtype=int), unprotected)}
        for name, (signals, risk) in strategies.items():
            curve, trades, fills, metrics = simulate(test_bars, signals[mask], funding, risk)
            target = destination / symbol / name
            target.mkdir(parents=True)
            curve.to_parquet(target / "equity.parquet", index=False)
            trades.to_parquet(target / "trades.parquet", index=False)
            fills.to_parquet(target / "fills.parquet", index=False)
            save_json(target / "metrics.json", metrics)
            rows.append({"symbol": symbol, "strategy": name, **{k: metrics[k] for k in ["total_return", "annualized_return", "max_drawdown", "daily_sharpe", "closed_trades", "win_rate", "profit_factor", "fees", "funding_cost", "average_exposure", "permanent_halt"]}})
            previous_equity = risk.initial_equity
            for year, subset in curve.groupby((curve.time - pd.Timedelta(nanoseconds=1)).dt.year):
                final = subset.equity.iloc[-1]
                yearly_rows.append({"symbol": symbol, "strategy": name, "year": int(year), "return": float(final / previous_equity - 1)})
                previous_equity = final
            axis.plot(curve.time, curve.equity / risk.initial_equity, label=name, lw=0.8)
            print(f"표본 외 {symbol}/{name}: {metrics['total_return']:.2%}, 낙폭 {metrics['max_drawdown']:.2%}", flush=True)
        for multiplier in [2, 3]:
            config = RiskConfig(**{**selected["risk"], "fee_bps": 5 * multiplier, "slippage_bps": 3 * multiplier})
            _, _, _, metrics = simulate(test_bars, protected[mask], funding, config)
            stress_rows.append({"symbol": symbol, "cost_multiplier": multiplier,
                                "total_return": metrics["total_return"], "max_drawdown": metrics["max_drawdown"],
                                "closed_trades": metrics["closed_trades"]})
        axis.axhline(1, color="gray", ls="--", lw=0.6, label="cash")
        axis.set(title=f"{symbol}: fixed model, 2022-2025", ylabel="Equity / initial")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    chart.savefig(destination / "out_of_sample.png", dpi=150)
    plt.close(chart)
    results = pd.DataFrame(rows)
    stress = pd.DataFrame(stress_rows)
    results.to_csv(destination / "test_results.csv", index=False)
    stress.to_csv(destination / "cost_stress.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(destination / "yearly_results.csv", index=False)
    save_json(destination / "results.json", {"test": records(results), "stress": records(stress), "status": "research_candidate_not_approved_for_live_trading"})
    val_summary = pd.DataFrame([{k: r[k] for k in ["candidate", "score", "total_return", "max_drawdown", "closed_trades", "eligible"]} for r in validation_results])
    selected_rows = results[results.strategy == "imitation_risk"]
    positive_count = int((selected_rows.total_return > 0).sum())
    report = f"""# 공개 포지션에서 추정한 매매 후보의 첫 검증

## 실험의 결론 범위

이 모델은 공개 XBTUSD 보유 방향을 시장 특징으로 모사한 통계적 후보다. 원래 트레이더의 진입 이유, 주문 대기열, 실제 위험 한도를 복원한 모델이 아니다. BitMEX 거래를 바이낸스에서 그대로 실행한 결과도 아니다. 자료가 진짜인지, 현재 총재산이 얼마인지를 이 실험으로 판정할 수 없다.

고정한 위험 제어 후보의 2022~2025 순수익이 양수인 시장은 {positive_count}/{len(symbols)}개다. 전체 결과는 아래 표와 로컬 원시 실행 기록에 남겼다. 수익성·실거래 준비 완료를 보장하지 않는다.

## 데이터와 시간 분리

- 전체 원본 거래를 감사한 결과와 별개로 모델 학습은 2020년의 시세와 포지션 정답만 사용했다.
- 학습 말일 24시간을 제외하고 미래 15분의 보유 방향을 정답으로 삼았다. 미래 정보는 정답으로만 사용하고 특징 계산에는 사용하지 않았다.
- 2021년 BTC로 8개 위험 설정을 비교한 뒤 선택을 파일로 고정했다. 이후 2022~2025 BTC·ETH·SOL을 평가했다.
- 2022~2025 결과는 이번 실행에서 이미 관찰한 평가 구간이다. 다음 실험에서 수정한 규칙에 대해 다시 미사용 최종 평가라고 부를 수 없다.
- BTC·ETH 공식 자료는 이번 수집에서 2020년부터 확보됐다. 2018~2019년의 시세 설명은 하지 않았다.
- BitMEX 원본 시간대를 UTC로 가정했다. 독립 체결 샘플 검증 전까지 시장 특징 연결과 전략 해석은 잠정적이다.

## 방향 모사 성능

{table(pd.DataFrame([{k: v for k, v in imitation_metrics.items() if k not in ['train_class_counts', 'confusion_matrix', 'confusion_matrix_labels']}]))}

학습 정답 분포: {json.dumps(imitation_metrics['train_class_counts'], ensure_ascii=False)}. 방향 모사 정확도는 수익률과 다른 지표다.

## 실제 매매 시점의 특징

표의 손익 분류는 설명용이다. 그 정보를 입력 특징이나 손실 거래 제거 규칙에 넣지 않았다. 진입 시각 직전에 완료된 최대 30분 이내 봉만 연결했다. 동일 주문의 후속 부분 체결을 독립 판단으로 해석하지 않는다.

{table(entry_counts)}

{table(entry_summary)}

유효한 시세 특징이 연결된 진입/반전 표본은 {len(entry.dropna(subset=FEATURES)):,}/{len(entry):,}개다. 매매하지 않은 시점을 포함한 전체 15분봉의 포지션 방향을 학습 표본으로 사용했다.

## 개발 후 검증과 선택

{table(val_summary)}

선택한 설정: `{json.dumps(selected, ensure_ascii=False)}`

손절·노출 상한·보유 시간·변동성 필터는 손실 억제를 위해 추가한 연구 규칙이며 본인이 사용했다고 확인된 규칙이 아니다. 손실 거래를 지워서 성과를 계산하지 않았다.

## 표본 외 결과

총수익과 낙폭은 소수 비율이다. 예: 0.1은 10%다. 자본은 10,000 USDT, 초기 진입은 자본의 최대 50%, 시장가 편도 수수료 5bp와 슬리피지 3bp를 가정했다. 실제 계정 수수료를 주장하는 수치가 아니다. 보유 후 가격 변동으로 노출 비율은 변할 수 있다. 현금 기준의 수익은 0이다.

{table(results)}

위험 제어 제거 실험은 같은 모델·신뢰도 기준을 쓰되 변동성 필터, 손절, 보유 제한, 일중 중지와 누적 손실 중지를 끈다. 장기 보유 기준은 같은 초기 50% 자본으로 진입하고 중간 재조정하지 않는다. 노출 시간이 달라 수익률만으로 우열을 단정하지 않는다.

## 비용 스트레스

{table(stress)}

![Out of sample](out_of_sample.png)

## 실행 가정과 미해결 사항

1. 다음 봉 시가 체결을 사용한다. 지정가 체결 확률, 대기열, 주문량에 따른 시장 충격은 재현하지 않았다.
2. 공식 펀딩률을 적용했지만 실제 마크 가격 대신 봉 시가를 사용했다. 1초 미만 펀딩 기록 지연은 해당 봉 경계로 맞췄다.
3. 최대 낙폭은 봉 종료 순자산 기준이다. 봉 내부 낙폭과 갭으로 위험 한도가 초과될 수 있다.
4. 일중 손실 한도는 다음 관측 시가에서 검사한다. 누적 낙폭 한도에 걸리면 남은 기간 중지한다. 거래 수와 중지 여부를 함께 확인해야 한다.
5. 모델은 설명 가능한 최소 기준이다. 체결 미시구조, 추가 진입의 의도, 자산 규모 변화는 아직 모사하지 않았다.
6. 검증 기간과 모델을 더 많이 시도할수록 과최적화 위험이 증가한다. 새로운 후보는 실험 계획과 관찰 이력을 함께 남긴다.

## 다음 연구

- 원본 시간대와 외부 체결 일치를 확인하고 지갑 손익 대조 잔차의 원인을 분해한다.
- 1분봉·틱 자료에서 진입/추가 진입/청산을 구분하는 사건 중심 모델을 검토한다.
- 단순 추세 기준을 안정적으로 이기는지, 위험 제어가 다른 기간에서도 손실을 줄이는지 조사한다.
- 워크포워드와 블록 재표집 불확실성을 추가한다. 같은 평가 기간에서 재튜닝한 결과를 새로운 증거처럼 제시하지 않는다.
- 실거래 연결은 하지 않는다. 새로운 실시간 자료에 대한 모의매매는 별도 검증 단계다.
"""
    (destination / "REPORT.md").write_text(report, encoding="utf-8")
    save_json(destination / "evaluation_observed.json", {"test_period": settings["test"], "symbols": symbols,
                                                         "do_not_reuse_as_unseen_holdout": True})
    print(f"연구 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination
