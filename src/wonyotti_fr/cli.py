from __future__ import annotations

import argparse
from pathlib import Path

from .audit import aggregate_events, write_audit
from .common import new_run, save_json
from .reconstruct import reconstruct
from .reports import write_reconstruction_report


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
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
