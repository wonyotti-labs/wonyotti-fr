from __future__ import annotations

import argparse
from pathlib import Path

from .audit import aggregate_events, write_audit
from .common import new_run, save_json
from .market import fetch_market, repair_gaps
from .offline import demo, replay
from .reconstruct import reconstruct
from .reports import write_reconstruction_report
from .research import run_research
from .robustness import run_robustness
from .study import run_study


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
    save_json(destination / "reconstruction.json", summary)
    episodes.to_parquet(destination / "episodes.parquet", index=False)
    actions.to_parquet(destination / "actions.parquet", index=False)
    write_reconstruction_report(destination, audit, summary, episodes, actions)
    print(f"보고서: {destination / 'REPORT.md'}", flush=True)
    print(f"에피소드 {len(episodes):,}개 / 손익 대조 차이 {summary['wallet_difference_btc']:.10f} BTC", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wonyotti: offline research only")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="전체 입력 검사 및 XBTUSD 복원")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--timezone", default="UTC", help="독립 검증 전까지 명시적 가정")
    audit.add_argument("--output", type=Path, default=Path("artifacts"))
    audit.set_defaults(func=run_audit)
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
    repair.set_defaults(func=lambda a: repair_gaps(a.market, a.output))
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
