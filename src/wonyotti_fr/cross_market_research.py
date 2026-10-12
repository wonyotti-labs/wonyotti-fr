from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .cross_market import (
    CROSS_FEATURES,
    WINDOWS,
    attach_cross_market,
    cross_market_support,
    cross_market_table,
)
from .cross_market_reference import verify_cross_market, verify_cross_support
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .retained_weight_close_reference import checked_files
from .uncapped_management import FEATURES as PREVIOUS_FEATURES
from .uncapped_management_learning import checked_uncapped_reference

MARKETS = {
    "BTCUSDT": ("data/market-5m-btc-candle-verified", "687a6b1214f3d94114192d3775970b7078233b6323d883ecd1a4403b4a2521c3"),
    "ETHUSDT": ("data/market-5m-eth-sol-verified", "af755cd2eed681f45cf0f9f7cf777ce151f07a95027db6764ce3b17b6dc6b089"),
}


def checked_cross_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("상대 가격 문맥의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"])
        or summary.get("uncapped_count_admitted") is not False
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_accounts_executed", "profitability_accepted"])):
        raise ValueError("상대 가격 문맥의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_uncapped_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"] or settings.get("features") != PREVIOUS_FEATURES:
        raise ValueError("상대 가격 문맥의 원본·기존 입력 연결 오류")
    return {**sources, "uncapped": str(root)}


def checked_cross_market_source(folder, symbol, expected):
    manifest = folder / "manifest-5m.json"
    if folder.is_symlink() or manifest.is_symlink() or sha256(manifest) != expected:
        raise ValueError("상대 가격 시세 출처의 manifest 지문 오류")
    metadata = read_json(manifest)
    if symbol not in {"BTCUSDT", "ETHUSDT"} or metadata.get("interval") != "5m":
        raise ValueError("상대 가격 시세의 종목·주기 오류")
    entry = metadata["summary"][symbol]["klines"]
    paired = metadata["paired_reconciliation"]
    paths = [manifest]
    for name, digest in [(entry["file"], entry["sha256"]), (paired["evidence_file"], paired["evidence_sha256"])]:
        path = folder / name
        if Path(name).name != name or path.is_symlink() or sha256(path) != digest:
            raise ValueError("상대 가격 시세의 파일·보정 증거 오류")
        paths.append(path)
    reconciliation = read_json(paths[2])
    if (reconciliation["source_sha256"] != paired["source_sha256"]
        or symbol not in reconciliation["requested_symbols"]):
        raise ValueError("상대 가격 시세의 보정 출처 연결 오류")
    frame = pd.read_parquet(paths[1], columns=["time", "end", "close"])
    if (len(frame) != entry["rows"] or entry.get("gaps") != 0
        or frame.time.min() != pd.Timestamp(entry["first"]) or frame.time.max() != pd.Timestamp(entry["last"])):
        raise ValueError("상대 가격 시세의 저장 범위·행 수 불일치")
    return frame, {"symbol": symbol, "files": {str(p): sha256(p) for p in paths},
        "source_rows": len(frame), "source_first_end": frame.end.min(), "source_last_end": frame.end.max(),
        "recorded_reconciliation_after": reconciliation["symbols"][symbol]["after"],
        "reconciliation_covers_entire_training_period_claimed": False}


def run_cross_market_inputs(reference, expected, output):
    sources = checked_cross_reference(reference, expected)
    teacher = Path(sources["teacher"])
    protocol = Path("docs/EXPERIMENT_V136.md")
    markets, provenance = {}, {}
    for symbol, (folder, digest) in MARKETS.items():
        markets[symbol], provenance[symbol] = checked_cross_market_source(Path(folder), symbol, digest)
    paths = [reference, protocol, teacher / "files.json"]
    paths += [Path(sources["uncapped"]) / name for name in ["files.json", "manifest.json", "summary.json"]]
    paths += [teacher / name for name in checked_files(teacher)]
    paths += [Path(name) for value in provenance.values() for name in value["files"]]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "cross-market-inputs", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "markets": MARKETS, "features": CROSS_FEATURES,
        "windows": WINDOWS, "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        last = max(frame.end.max() for frame in rows.values())
        markets = {s: f.loc[f.end.le(last)].reset_index(drop=True) for s, f in markets.items()}
        btc, eth = markets["BTCUSDT"], markets["ETHUSDT"]
        for symbol, frame in markets.items():
            provenance[symbol].update(used_rows=len(frame), used_first_end=frame.end.min(), used_last_end=frame.end.max())
        table = cross_market_table(btc, eth)
        support, monthly, verification = {}, {}, {}
        for phase, frame in rows.items():
            expanded, links = attach_cross_market(frame, table)
            verification[phase] = verify_cross_market(frame, btc, eth, expanded, links)
            support[phase] = cross_market_support(expanded)
            verify_cross_support(expanded, support[phase])
            monthly[phase] = {}
            for month, part in expanded.groupby(expanded.end.dt.strftime("%Y-%m"), sort=True):
                monthly[phase][month] = cross_market_support(part)
                verify_cross_support(part, monthly[phase][month])
            save_frame(out / (phase + "_features.parquet"), expanded[["end", "episode_id", *CROSS_FEATURES]])
            save_frame(out / (phase + "_links.parquet"), links)
            verification[phase].update(all_support_and_distributions_verified=True, parquet_readback_exact=True)
            print(phase + ": " + str(len(frame)) + "행의 상대 가격·모든 창·과거 시각·전체 지원 검산 완료", flush=True)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("상대 가격 입력 검산 중 입력·코드·계획 변경")
        for name, value in [("provenance", provenance), ("support", support), ("monthly_support", monthly), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "input_values_checked": sum(len(v) for v in rows.values()) * 10,
            "original_columns_exact": True, "all_inputs_and_code_unchanged": True, "input_reconstruction_verified": True,
            "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True})
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_cross_market_inputs(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
