from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .causal_management import (
    causal_admission,
    causal_predictions,
    fit_causal_calibration,
    map_management_history,
    mapping_support,
    reference_distributions,
)
from .causal_management_reference import (
    verify_causal_admission,
    verify_causal_calibration,
    verify_causal_predictions,
    verify_management_mapping,
    verify_mapping_support,
)
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .management_rank_research import CONTEXT, checked_score_reference
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet
from .quiet_management_reference import verify_quiet_evaluation
from .retained_weight_close_reference import checked_files


def checked_causal_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("과거 분포 보정의 선행 증거 지문 오류")
    proof = read_json(reference)
    audit = Path(proof["run"])
    checked_files(audit)
    summary = read_json(audit / "summary.json")
    if (proof.get("complete") is not True or sha256(audit / "files.json") != proof["files_sha256"]
        or sha256(audit / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete"])
        or any(summary.get(k) is not False or proof.get(k) is not False
               for k in ["new_accounts_executed", "new_policy_selected", "profitability_accepted"])
        or summary.get("combinations") != 48 or summary.get("monthly_combinations") != 384):
        raise ValueError("과거 분포 보정의 선행 진단·미채택 근거 오류")
    settings = read_json(audit / "manifest.json")["settings"]
    root, previous, native, teacher = checked_score_reference(Path(settings["reference"]), settings["reference_sha256"])
    if any(str(value) != settings[key] for value, key in [(root, "root"), (previous, "previous"),
                                                         (native, "native"), (teacher, "teacher")]):
        raise ValueError("과거 분포 보정의 원본·모형 출처 오류")
    return audit, previous, teacher


def run_causal_management(reference, expected, output):
    audit, previous, teacher = checked_causal_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V130.md")
    paths = [reference, protocol, audit / "files.json", previous / "files.json", teacher / "files.json",
             previous / "metrics.json", previous / "monthly.json"]
    for phase in ["training", "calibration", "diagnosis"]:
        paths += [previous / (phase + "_predictions.parquet"), teacher / (phase + "_used.parquet")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "causal-management-distribution", {"reference": str(reference),
        "reference_sha256": expected, "audit": str(audit), "previous": str(previous), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "window_days": 30, "minimum_history": 1000,
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        original = {phase: pd.read_parquet(previous / (phase + "_predictions.parquet")) for phase in rows}
        old_metrics, old_monthly = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        for phase, frame in original.items():
            pd.testing.assert_frame_equal(rows[phase][CONTEXT], frame[CONTEXT], check_exact=True)
            compare_statistics(evaluate_quiet(frame), old_metrics[phase])
        references = reference_distributions(original["training"])
        save_json(out / "reference_quantiles.json", {a: {s: {m: v.tolist() for m, v in models.items()}
            for s, models in states.items()} for a, states in references.items()})
        print("원본·고정 모델 결과·2020년 기준 분포 검산 완료, 과거 창 계산 시작", flush=True)
        ledgers = map_management_history(original, references)
        print("과거 창 계산 완료, 별도 누적 수·선형 보간 검산 시작", flush=True)
        mapping_verification = verify_management_mapping(original, references, ledgers)
        print("모든 과거 n/L/E·지원·분위수 검산 완료", flush=True)
        support = {phase: mapping_support(frame, ledgers[phase]) for phase, frame in original.items()}
        for phase, frame in original.items():
            verify_mapping_support(frame, ledgers[phase], support[phase])
        save_json(out / "support.json", support)
        offsets, thresholds, details, constants = fit_causal_calibration(original["calibration"], ledgers["calibration"])
        verify_causal_calibration(original["calibration"], ledgers["calibration"], offsets, thresholds, details, constants)
        for name, value in [("offsets", offsets), ("thresholds", thresholds), ("threshold_selection", details), ("constants", constants)]:
            save_json(out / (name + ".json"), value)
        metrics, monthly, monthly_support = {}, {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        for phase, frame in original.items():
            ledger = ledgers[phase]
            candidate = causal_predictions(frame, ledger, offsets, thresholds, constants)
            preserved = verify_causal_predictions(frame, ledger, candidate, offsets, thresholds, constants)
            unchanged = {k: unchanged[k] and preserved[k] for k in unchanged}
            metrics[phase] = evaluate_quiet(candidate)
            verify_quiet_evaluation(candidate, metrics[phase])
            monthly[phase], monthly_support[phase] = {}, {}
            for month, indices in candidate.groupby(candidate.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                old, part, state = frame.iloc[indices], candidate.iloc[indices], ledger.iloc[indices]
                compare_statistics(evaluate_quiet(old), old_monthly[phase][month])
                monthly[phase][month] = evaluate_quiet(part)
                verify_quiet_evaluation(part, monthly[phase][month])
                monthly_support[phase][month] = mapping_support(old, state)
                verify_mapping_support(old, state, monthly_support[phase][month])
            save_frame(out / (phase + "_history.parquet"), ledger)
            save_frame(out / (phase + "_predictions.parquet"), candidate)
            print(phase + ": 원본·미지원·청산·recent 보존 및 모든 상태·월 검산 완료", flush=True)
        decision = causal_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged)
        verify_causal_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("과거 분포 보정 중 입력·코드·계획 변경")
        for name, value in [("metrics", metrics), ("monthly", monthly), ("monthly_support", monthly_support),
                            ("verification", mapping_verification), ("decision", decision)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {p: len(f) for p, f in rows.items()}, "checks": decision["checks"],
            "causal_distribution_admitted": decision["causal_distribution_admitted"],
            "all_inputs_and_code_unchanged": True, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/35 조건 통과", flush=True)
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
    run_causal_management(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
