from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .holding_stress import seal
from .management_market import (
    REPLACED,
    check_original_market,
    connect_market,
    market_support,
    observed_tables,
)
from .management_market_reference import verify_market_connection, verify_market_support
from .management_score_research import checked_source
from .minute_data import compare_minute_bars
from .research import load_market
from .timing_study import read_minute_history


def read(path):
    return json.loads(path.read_text())


def checked_market_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 시장 정렬의 출처 지문 오류")
    spec = read(path)
    required = {spec["previous_proof"]}
    for prefix, filename in [("native_minute", "manifest.json"), ("native_five", "manifest.json"),
                             ("target_minute", "manifest-1m.json"), ("target_five", "manifest-5m.json")]:
        root = Path(spec[prefix])
        manifest_path = root / filename
        manifest = read(manifest_path)
        if prefix.startswith("native"):
            name = manifest["file"]
        else:
            name = manifest["summary"]["BTCUSDT"]["klines"]["file"]
        data = root / name
        if not data.resolve().is_relative_to(root.resolve()):
            raise ValueError("관리 시장 정렬의 정규화 시세 경로 오류")
        required.update([str(manifest_path), str(data)])
    if not required <= set(spec["sealed_files"]) or spec.get("profitability_accepted") is not False:
        raise ValueError("관리 시장 정렬의 필수 봉인 입력 누락")
    for name, digest in spec["sealed_files"].items():
        item = Path(name)
        if item.is_symlink() or not item.is_file() or sha256(item) != digest:
            raise ValueError("관리 시장 정렬의 봉인 입력 변경: " + name)
    _, previous = checked_source(Path(spec["previous_proof"]), spec["sealed_files"][spec["previous_proof"]])
    return spec, previous


def run_market_support(reference, expected, output):
    spec, previous = checked_market_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V116.md")
    out = new_run(output, "management-market-support", {"reference": str(reference),
        "reference_sha256": expected, "protocol_sha256": sha256(protocol), "spec": spec,
        "teacher": previous["teacher"], "changed_features": REPLACED, "new_models_fitted": False,
        "all_periods_already_observed": True, "profitability_accepted": False})
    print(out, flush=True)
    try:
        native_minute, native_hashes = read_minute_history(Path(spec["native_minute"]), Path(spec["native_five"]))
        native_manifest = read(Path(spec["native_five"]) / "manifest.json")
        native_path = Path(spec["native_five"]) / native_manifest["file"]
        if sha256(native_path) != native_manifest["sha256"]:
            raise ValueError("관리 시장 정렬의 원거래소 5분 지문 오류")
        teacher = Path(previous["teacher"])
        settings = read(teacher / "manifest.json")["settings"]
        labels = Path(settings["labels"])
        if sha256(labels / "files.json") != settings["labels_files_sha256"]:
            raise ValueError("관리 시장 정렬의 기존 분별 입력 연결 오류")
        if sha256(labels / "manifest.json") != read(labels / "files.json")["manifest.json"]:
            raise ValueError("관리 시장 정렬의 기존 분별 입력 메타데이터 변경")
        original = read(labels / "manifest.json")["settings"]
        if any(original[k] != v for k, v in native_hashes.items()):
            raise ValueError("관리 시장 정렬의 원거래소 분봉 출처 변경")
        native_five = pd.read_parquet(native_path)
        target_minute, _ = load_market(Path(spec["target_minute"]), "BTCUSDT", "1m")
        target_five, _ = load_market(Path(spec["target_five"]), "BTCUSDT", "5m")
        cutoff = pd.Timestamp("2022-01-01", tz="UTC")
        target_minute = target_minute[target_minute.end.le(cutoff)].reset_index(drop=True)
        target_five = target_five[target_five.end.le(cutoff)].reset_index(drop=True)
        _, comparison = compare_minute_bars(target_minute, target_five)
        if comparison["incomplete"] or comparison["mismatched"]:
            raise ValueError("관리 시장 정렬의 Binance 해상도 집계 불일치")
        native = observed_tables(native_minute, native_five)
        target = observed_tables(target_minute, target_five)
        summaries, proofs = {}, {}
        episodes = {}
        for phase, (start, stop) in PERIODS.items():
            frame = pd.read_parquet(teacher / f"{phase}_used.parquet")
            if (not frame.end.ge(pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=1)).all()
                or not frame.entry_time.ge(pd.Timestamp(start, tz="UTC")).all()
                or not frame.label_end.lt(pd.Timestamp(stop, tz="UTC") - pd.Timedelta(days=1)).all()):
                raise ValueError("관리 시장 정렬의 기존 시간 분할 경계 변경")
            episodes[phase] = set(frame.episode_id)
            old, old_link = connect_market(frame, *native)
            check_original_market(frame, old, old_link)
            verify_market_connection(frame, old, old_link, native_minute, *native)
            candidate, link = connect_market(frame, *target)
            proof = verify_market_connection(frame, candidate, link, target_minute, *target)
            summaries[phase] = market_support(frame, candidate, link)
            verify_market_support(frame, candidate, link, summaries[phase])
            candidate[["end", *REPLACED]].to_parquet(out / f"{phase}_market.parquet", index=False)
            link.to_parquet(out / f"{phase}_linkage.parquet", index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(out / f"{phase}_market.parquet"),
                                          candidate[["end", *REPLACED]].reset_index(drop=True), check_exact=True)
            pd.testing.assert_frame_equal(pd.read_parquet(out / f"{phase}_linkage.parquet"), link, check_exact=True)
            proofs[phase] = {**proof, "original_market_reconstructed_exactly": True,
                             "independent_support_counts_verified": True, "parquet_readback_exact": True}
            print(phase + ": " + str(summaries[phase]["groups"]), flush=True)
            del frame, old, old_link, candidate, link
        if any(episodes[a] & episodes[b] for a in episodes for b in episodes if a < b):
            raise ValueError("관리 시장 정렬의 구간 사이 포지션 중복")
        if checked_market_reference(reference, expected) != (spec, previous):
            raise ValueError("관리 시장 정렬 중 입력 변경")
        manifest = read(out / "manifest.json")
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != digest for name, digest in manifest["source_sha256"].items()):
            raise ValueError("관리 시장 정렬 중 계획·코드 변경")
        save_json(out / "source_verification.json", {"native": native_hashes, "target_aggregation": comparison})
        save_json(out / "support.json", summaries)
        save_json(out / "verification.json", proofs)
        save_json(out / "summary.json", {"complete": True,
            "paired_model_research_eligible": all(s["action_support_sufficient"] for s in summaries.values()),
            "new_models_fitted": False, "original_rows_preserved": True, "episode_intersection": 0,
            "profitability_accepted": False})
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="관리 모형의 거래소 시장 입력 정렬과 지원 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_market_support(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
