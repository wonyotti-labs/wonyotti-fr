from __future__ import annotations

import argparse
from pathlib import Path

from .audit import aggregate_events, write_audit
from .bitmex_history import fetch_bitmex_history
from .common import new_run, save_json
from .event_diagnostics import run_event_diagnostics
from .event_evaluation import run_event_evaluation
from .event_replay import run_event_replay
from .event_research import run_event_selection
from .event_study import run_event_study
from .market import fetch_market, repair_gaps
from .offline import demo, replay
from .portfolio import reconstruct_portfolio
from .reconciliation import reconcile_wallet
from .reconstruct import reconstruct
from .reports import write_reconstruction_report
from .research import run_research
from .robustness import run_robustness
from .study import run_study
from .verification import verify_samples


def run_audit(args) -> None:
    source = args.source.resolve(strict=True)
    destination = new_run(args.output, "audit", {"source": str(source), "timezone": args.timezone})
    print(f"감사 시작: {destination}", flush=True)
    data, wallet, audit = write_audit(source, destination, args.timezone)
    events = aggregate_events(data, "XBTUSD")
    events.to_parquet(destination / "xbtusd_events.parquet", index=False)
    episodes, actions, summary = reconstruct(events)
    wallet_pnl = int(wallet.loc[(wallet.address == "XBTUSD") & (wallet.transacttype == "RealisedPNL"), "amount"].sum())
    summary["wallet_xbtusd_pnl_btc"] = wallet_pnl / 1e8
    summary["wallet_difference_btc"] = summary["realized_net_btc"] - wallet_pnl / 1e8
    daily, summary["wallet_reconciliation"] = reconcile_wallet(actions, wallet)
    daily.to_csv(destination / "wallet_daily_reconciliation.csv", index=False)
    save_json(destination / "reconstruction.json", summary)
    episodes.to_parquet(destination / "episodes.parquet", index=False)
    actions.to_parquet(destination / "actions.parquet", index=False)
    write_reconstruction_report(destination, audit, summary, episodes, actions)
    print(f"보고서: {destination / 'REPORT.md'}", flush=True)
    print(f"에피소드 {len(episodes):,}개 / 같은 기간 대조 잔차 {summary['wallet_reconciliation']['aligned_residual_satoshi']:.6f} 사토시", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wonyotti: offline research only")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="전체 입력 검사 및 XBTUSD 복원")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--timezone", default="UTC", help="독립 검증 전까지 명시적 가정")
    audit.add_argument("--output", type=Path, default=Path("artifacts"))
    audit.set_defaults(func=run_audit)
    portfolio = commands.add_parser("portfolio", help="원본 정산 통화와 비용으로 전체 계약 복원")
    portfolio.add_argument("--audit-run", type=Path, required=True)
    portfolio.add_argument("--output", type=Path, default=Path("artifacts"))
    portfolio.set_defaults(func=lambda a: reconstruct_portfolio(a.audit_run, a.output))
    verify = commands.add_parser("verify", help="공식 공개 체결로 원본 표본을 외부 대조")
    verify.add_argument("--audit-run", type=Path, required=True)
    verify.add_argument("--output", type=Path, default=Path("artifacts"))
    verify.add_argument("--cache", type=Path, default=Path("data/bitmex-verification"))
    verify.set_defaults(func=lambda a: verify_samples(a.audit_run, a.output, a.cache))
    history = commands.add_parser("bitmex-history", help="원거래소 공식 5분봉을 특징 연구용으로 수집")
    history.add_argument("--start", default="2018-03-01")
    history.add_argument("--end", default="2022-01-01")
    history.add_argument("--output", type=Path, default=Path("data/bitmex-history"))
    history.set_defaults(func=lambda a: fetch_bitmex_history(a.output, a.start, a.end))
    market = commands.add_parser("market", help="공식 시세/펀딩 자료를 체크섬 검증 후 저장")
    market.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    market.add_argument("--start", default="2019-09")
    market.add_argument("--end", default="2025-12")
    market.add_argument("--interval", default="15m", choices=["1m", "5m", "15m", "1h"])
    market.add_argument("--output", type=Path, default=Path("data/market"))
    market.set_defaults(func=lambda a: fetch_market(a.output, a.symbols, a.start, a.end, a.interval))
    repair = commands.add_parser("market-repair", help="월별 자료의 내부 결측을 공식 일별 자료로 보완")
    repair.add_argument("--market", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--interval", default="15m", choices=["1m", "5m", "15m", "1h"])
    repair.set_defaults(func=lambda a: repair_gaps(a.market, a.output, a.interval))
    research = commands.add_parser("research", help="학습·검증·평가를 분리하여 모사 후보를 비교")
    research.add_argument("--audit-run", type=Path, required=True)
    research.add_argument("--market", type=Path, default=Path("data/market"))
    research.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    research.add_argument("--output", type=Path, default=Path("artifacts"))
    research.set_defaults(func=lambda a: run_research(a.audit_run, a.market, a.output, a.symbols))
    study = commands.add_parser("study", help="추가 진입·보유 시간·시장 맥락의 사후 행동 분석")
    study.add_argument("--audit-run", type=Path, required=True)
    study.add_argument("--market", type=Path, default=Path("data/market-complete"))
    study.add_argument("--output", type=Path, default=Path("artifacts"))
    study.set_defaults(func=lambda a: run_study(a.audit_run, a.market, a.output))
    events = commands.add_parser("event-study", help="독립 주문의 진입·관리 학습 자료와 모델 생성")
    events.add_argument("--audit-run", type=Path, required=True)
    events.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    events.add_argument("--output", type=Path, default=Path("artifacts"))
    events.set_defaults(func=lambda a: run_event_study(a.audit_run, a.history, a.output))
    selection = commands.add_parser("event-select", help="2020년에서 사건별 정책 선택 후 설정 고정")
    selection.add_argument("--study-run", type=Path, required=True)
    selection.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    selection.add_argument("--output", type=Path, default=Path("artifacts"))
    selection.set_defaults(func=lambda a: run_event_selection(a.study_run, a.market, a.output))
    evaluation = commands.add_parser("event-evaluate", help="고정 사건별 후보의 다년·다시장·위험 제거 평가")
    evaluation.add_argument("--selection-run", type=Path, required=True)
    evaluation.add_argument("--market", type=Path, required=True)
    evaluation.add_argument("--period", choices=["observed", "new"], required=True)
    evaluation.add_argument("--output", type=Path, default=Path("artifacts"))
    evaluation.set_defaults(func=lambda a: run_event_evaluation(a.selection_run, a.market, a.output, a.period))
    diagnostics = commands.add_parser("event-diagnose", help="완료한 사건별 평가의 매매 빈도·가격 손익·비용 분해")
    diagnostics.add_argument("--study-run", type=Path, required=True)
    diagnostics.add_argument("--evaluation-runs", type=Path, nargs="+", required=True)
    diagnostics.add_argument("--output", type=Path, default=Path("artifacts"))
    diagnostics.set_defaults(func=lambda a: run_event_diagnostics(a.study_run, a.evaluation_runs, a.output))
    journal = commands.add_parser("event-replay", help="영속 저널을 이용한 사건별 봇 재생과 중단 복원")
    journal.add_argument("--selection-run", type=Path, required=True)
    journal.add_argument("--market", type=Path, required=True)
    journal.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT", "SOLUSDT"], default="BTCUSDT")
    journal.add_argument("--start", required=True)
    journal.add_argument("--end", required=True)
    journal.add_argument("--journal", type=Path, required=True)
    journal.add_argument("--max-bars", type=int)
    journal.add_argument("--halt", action="store_true")
    journal.add_argument("--verify-memory", action="store_true")
    journal.add_argument("--output", type=Path, default=Path("artifacts"))
    journal.set_defaults(func=lambda a: run_event_replay(a.selection_run, a.market, a.symbol, a.start, a.end,
                                                       a.journal, a.output, a.max_bars, a.halt, a.verify_memory))
    robustness = commands.add_parser("robustness", help="고정 후보의 연도별 재시작과 블록 재표집 진단")
    robustness.add_argument("--research-run", type=Path, required=True)
    robustness.add_argument("--market", type=Path, default=Path("data/market-complete"))
    robustness.add_argument("--output", type=Path, default=Path("artifacts"))
    robustness.set_defaults(func=lambda a: run_robustness(a.research_run, a.market, a.output))
    playback = commands.add_parser("replay", help="고정한 모델로 로컬 시세를 오프라인 재생")
    playback.add_argument("--research-run", type=Path, required=True)
    playback.add_argument("--market", type=Path, default=Path("data/market-complete"))
    playback.add_argument("--symbol", default="BTCUSDT")
    playback.add_argument("--start", required=True, help="UTC 시작, 포함")
    playback.add_argument("--end", required=True, help="UTC 종료, 미포함")
    playback.add_argument("--output", type=Path, default=Path("artifacts"))
    playback.set_defaults(func=lambda a: replay(a.research_run, a.market, a.symbol, a.start, a.end, a.output))
    sample = commands.add_parser("demo", help="원본과 네트워크 없이 합성 시세로 동작 확인")
    sample.add_argument("--output", type=Path, default=Path("artifacts"))
    sample.set_defaults(func=lambda a: demo(a.output))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
