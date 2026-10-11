from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_flow import EXTRA_FEATURES, connect_flow, flow_support, flow_tables
from .management_flow_reference import (
    direct_flow_table,
    verify_flow_connection,
    verify_flow_support,
)
from .management_market_learning import checked_learning_reference
from .management_market_models import validate_paired_rows
from .research import load_market
from .retained_weight_close_reference import checked_files


def checked_flow_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 체결 방향의 출처 지문 오류")
    reference = read_json(path)
    for key in ["input_proof", "model_proof"]:
        item = Path(reference[key])
        if item.is_symlink() or sha256(item) != reference["sealed_files"][str(item)]:
            raise ValueError("관리 체결 방향의 선행 완료 증거 변경")
    input_proof = Path(reference["input_proof"])
    inputs, teacher = checked_learning_reference(input_proof, reference["sealed_files"][str(input_proof)])
    proof = read_json(Path(reference["model_proof"]))
    models = Path(proof["run"])
    checked_files(models)
    summary = read_json(models/"summary.json")
    settings = read_json(models/"manifest.json")["settings"]
    verified = read_json(models/"verification.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(models/"files.json") != proof["files_sha256"]
        or sha256(models/"summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("profitability_accepted") is not False
        or settings["input_run"] != str(inputs) or settings["teacher"] != str(teacher)
        or settings["reference"] != str(input_proof)
        or settings["reference_sha256"] != reference["sealed_files"][str(input_proof)]
        or verified.get("native_trees_equal_stored") is not True
        or verified.get("stored_diagnosis_predictions_exact") is not True):
        raise ValueError("관리 체결 방향의 이전 모형·원본 연결 검산 누락")
    spec = read_json(inputs/"manifest.json")["settings"]["spec"]
    return {"teacher": str(teacher), "target_minute": spec["target_minute"],
        "target_five": spec["target_five"], "input_run": str(inputs), "previous_model_run": str(models)}


def full_teacher_rows(teacher):
    rows = {}
    for phase, (first, last) in PERIODS.items():
        frame = pd.read_parquet(teacher/f"{phase}_used.parquet")
        if (not frame.end.ge(pd.Timestamp(first, tz="UTC")+pd.Timedelta(days=1)).all()
            or not frame.entry_time.ge(pd.Timestamp(first, tz="UTC")).all()
            or not frame.label_end.lt(pd.Timestamp(last, tz="UTC")-pd.Timedelta(days=1)).all()
            or "source_row" in frame):
            raise ValueError("관리 체결 방향의 원래 전체 행·시간 분리 오류")
        rows[phase] = frame
    validate_paired_rows(rows, rows)
    return rows


def run_flow_support(reference, expected, output):
    sources = checked_flow_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V117.md")
    out = new_run(output, "management-flow-support", {"reference": str(reference),
        "reference_sha256": expected, "protocol_sha256": sha256(protocol), "sources": sources,
        "extra_features": EXTRA_FEATURES, "periods": PERIODS, "new_models_fitted": False,
        "all_periods_already_observed": True, "profitability_accepted": False})
    print(out, flush=True)
    try:
        rows = full_teacher_rows(Path(sources["teacher"]))
        last = max(frame.end.max() for frame in rows.values())
        minute, _ = load_market(Path(sources["target_minute"]), "BTCUSDT", "1m")
        five, _ = load_market(Path(sources["target_five"]), "BTCUSDT", "5m")
        small, large, source_check = flow_tables(minute, five, last)
        minute, five = (bars[bars.end.le(last)].reset_index(drop=True) for bars in [minute, five])
        print("두 해상도 체결량 대조·독립 창 합 계산", flush=True)
        direct = direct_flow_table(minute, five)
        support, verification = {}, {}
        for phase, frame in rows.items():
            expanded, link = connect_flow(frame, small, large)
            support[phase] = flow_support(frame, link)
            verification[phase] = verify_flow_connection(frame, expanded, link, minute, five, direct)
            verify_flow_support(frame, link, support[phase])
            values = expanded[["end", *EXTRA_FEATURES]]
            for label, table in [("flow", values), ("linkage", link)]:
                path = out/f"{phase}_{label}.parquet"
                table.to_parquet(path, index=False)
                pd.testing.assert_frame_equal(pd.read_parquet(path), table, check_exact=True)
            verification[phase].update(support_counts_verified=True, parquet_readback_exact=True)
            print(phase+": "+str(support[phase]["groups"]), flush=True)
        if checked_flow_reference(reference, expected) != sources:
            raise ValueError("관리 체결 방향의 실행 중 원본 변경")
        manifest = read_json(out/"manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent/name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("관리 체결 방향의 실행 중 계획·코드 변경")
        for name, value in [("support", support), ("verification", verification), ("source_verification", source_check)]:
            save_json(out/(name+".json"), value)
        save_json(out/"summary.json", {"complete": True, "original_rows_preserved": True,
            "input_values_independently_verified": sum(len(v) for v in rows.values())*len(EXTRA_FEATURES),
            "paired_model_research_eligible": all(v["all_original_rows_supported"] and v["action_support_sufficient"] for v in support.values()),
            "new_models_fitted": False, "profitability_accepted": False})
        seal(out)
    except Exception as error:
        save_json(out/"failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="원본 관리 행동의 체결 방향 입력·지원 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_flow_support(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
