from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .entry_path_audit import occupancy_summary, verify_holding_intervals, verify_occupancy_counts
from .entry_wait_reference import verify_wait_episodes
from .holding_stress import read_json, seal
from .minute_data import validate_minutes
from .pullback_diagnostics import waiting_diagnostics
from .repeat_direct_research import CASES, checked_repeat_reference
from .research import load_market
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot


def checked_entry_path_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("진입 경로의 완료 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (
        proof.get("complete") is not True
        or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True
        or summary.get("profitability_accepted") is not False
    ):
        raise ValueError("진입 경로의 기존 계좌 완료 미충족")
    settings = read_json(root / "manifest.json")["settings"]
    spec, frozen = checked_repeat_reference(
        Path(settings["reference"]), settings["reference_sha256"]
    )
    runs = read_json(root / "runs.json")
    if set(runs) != set(CASES) or frozen["risk"]["signal_delay_bars"] != 0:
        raise ValueError("진입 경로의 다섯 계좌·기본 지연 설정 오류")
    for name, digest in runs.items():
        folder = root / name
        checked_files(folder)
        if (
            sha256(folder / "files.json") != digest
            or read_json(folder / "config.json") != frozen["risk"]
        ):
            raise ValueError("진입 경로의 실제 계좌·위험 변경")
    return root, spec, frozen


def run_entry_path_audit(reference, expected, output):
    source, spec, frozen = checked_entry_path_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V118.md")
    out = new_run(
        output,
        "entry-path-audit",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "protocol_sha256": sha256(protocol),
            "source": str(source),
            "cases": CASES,
            "new_models_fitted": False,
            "new_accounts_executed": False,
            "all_periods_already_observed": True,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    summaries, checks, runs = {}, {}, {}
    try:
        for name, (symbol, bounds) in CASES.items():
            original, folder = source / name, out / name
            folder.mkdir()
            curve = pd.read_parquet(original / "equity.parquet")
            fills = pd.read_parquet(original / "fills.parquet")
            trades = pd.read_parquet(original / "trades.parquet")
            bars, _ = load_market(Path(spec["cases"][name]["market"]), symbol, "1m")
            first, last = (pd.Timestamp(t, tz="UTC") for t in bounds)
            bars = bars[bars.time.ge(first) & bars.time.lt(last)].reset_index(drop=True)
            validate_minutes(bars, 1)
            expected_times = pd.date_range(first, last, freq="min", inclusive="left").as_unit("ns")
            if not pd.DatetimeIndex(bars.time).as_unit("ns").equals(expected_times):
                raise ValueError("진입 경로의 분봉 기간 누락")
            for filename in ["equity.parquet", "fills.parquet"]:
                copy_snapshot(original / filename, folder / filename)
                if sha256(folder / filename) != sha256(original / filename):
                    raise ValueError("진입 경로의 기존 계좌 사본 변경")
            waiting = waiting_diagnostics(folder, bars, 0, management_state=True)
            episodes = pd.read_parquet(folder / "waiting_episodes.parquet")
            wait_check = verify_wait_episodes(curve, fills, bars, episodes)
            holding = verify_holding_intervals(curve, trades, fills)
            occupancy = occupancy_summary(curve, episodes)
            verify_occupancy_counts(curve, episodes, occupancy)
            if (
                waiting["completed_waits"] != len(episodes)
                or waiting["matched_entry_fills"] != wait_check["matched_entry_fills"]
            ):
                raise ValueError("진입 경로의 대기 집계 불일치")
            summaries[name] = {
                "symbol": symbol,
                "bounds": bounds,
                "occupancy": occupancy,
                "waiting": waiting,
                "holding": holding,
                "original_account_metrics": read_json(original / "metrics.json"),
            }
            checks[name] = {
                **wait_check,
                **holding,
                "all_occupancy_months_and_counts_independently_verified": True,
                "source_account_outputs_reused_unchanged": True,
            }
            save_json(folder / "summary.json", summaries[name])
            save_json(folder / "verification.json", checks[name])
            seal(folder)
            runs[name] = sha256(folder / "files.json")
            save_json(out / "runs.json", runs)
            print(name + ": 보유 구간·전체 대기·월별 집계 검산 완료", flush=True)
        if checked_entry_path_reference(reference, expected) != (source, spec, frozen):
            raise ValueError("진입 경로 검산 중 원본 변경")
        manifest = read_json(out / "manifest.json")
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != value
            for name, value in manifest["source_sha256"].items()
        ):
            raise ValueError("진입 경로 검산 중 코드·계획 변경")
        save_json(out / "summaries.json", summaries)
        save_json(out / "verification.json", checks)
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "cases": len(summaries),
                "all_rows": sum(v["occupancy"]["total"]["minutes"] for v in summaries.values()),
                "all_closed_trades": sum(v["holding"]["closed_trades"] for v in summaries.values()),
                "all_wait_episodes": sum(
                    v["waiting"]["completed_waits"] for v in summaries.values()
                ),
                "independent_verification_complete": True,
                "new_models_fitted": False,
                "new_accounts_executed": False,
                "profitability_accepted": False,
            },
        )
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="기존 다섯 계좌의 진입 대기·보유 경로 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_entry_path_audit(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
