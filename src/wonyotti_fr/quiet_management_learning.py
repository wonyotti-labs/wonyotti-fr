from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .position_boost_verification import compare_statistics
from .quiet_management import (
    ACTIONS,
    evaluate_quiet,
    fit_quiet_models,
    quiet_admission,
    quiet_mask,
    quiet_predictions,
    validate_quiet_rows,
)
from .quiet_management_reference import (
    verify_quiet_admission,
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_refits,
    verify_quiet_support,
)
from .request_timing_research import CONTEXT, checked_timing_reference, verify_saved_requests
from .retained_weight_close_reference import checked_files


def checked_quiet_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("최근 행동 분리의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "prior_exact_metrics_preserved"])
        or summary.get("combinations") != 27
        or summary.get("new_accounts_executed") is not False):
        raise ValueError("최근 행동 분리의 선행 완료 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    models, teacher = checked_timing_reference(Path(settings["reference"]), settings["reference_sha256"])
    if str(models) != settings["root"] or str(teacher) != settings["teacher"]:
        raise ValueError("최근 행동 분리의 정답·모형 출처 변경")
    return models, teacher


def run_quiet_management(reference, expected, output):
    previous, teacher = checked_quiet_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V126.md")
    paths = [reference, protocol, previous / "files.json", previous / "metrics.json",
             previous / "thresholds.json", teacher / "files.json"]
    for phase in ["training", "calibration", "diagnosis"]:
        paths += [teacher / (phase + "_used.parquet"), previous / (phase + "_predictions.parquet")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "quiet-management-learning", {"reference": str(reference),
        "reference_sha256": expected, "previous": str(previous), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "new_accounts_executed": False,
        "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        save_json(out / "support.json", support)
        print("quiet/recent 및 최초·재시작 상태 지원 검산 완료", flush=True)
        models, offsets, thresholds, details, constants = fit_quiet_models(rows["training"], rows["calibration"])
        for name, values in [("models", models), ("offsets", offsets), ("thresholds", thresholds),
                             ("threshold_selection", details), ("constants", constants)]:
            save_json(out / (name + ".json"), values)
        print("새 512개 트리·두 선형 적합 완료, 독립 재학습 검산 시작", flush=True)
        verification = verify_quiet_refits(rows, models)
        old_metrics = read_json(previous / "metrics.json")
        old_thresholds = read_json(previous / "thresholds.json")
        metrics, monthly = {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                original = pd.read_parquet(previous / (phase + "_predictions.parquet"))
                pd.testing.assert_frame_equal(frame[CONTEXT], original[CONTEXT], check_exact=True)
                verify_saved_requests(original, old_thresholds)
                predicted = quiet_predictions(frame, original, models, offsets, thresholds, constants)
                verify_quiet_predictions(frame, original, predicted, models, offsets, thresholds, constants)
                if phase == "calibration":
                    verify_quiet_calibration(predicted, offsets, thresholds, details, constants)
                metrics[phase] = evaluate_quiet(predicted)
                verify_quiet_evaluation(predicted, metrics[phase])
                for action in ACTIONS:
                    for group in ["all", "first"]:
                        compare_statistics(metrics[phase][action][group]["baseline"],
                            old_metrics[phase][action][group]["native_calibrated"])
                    recent = ~quiet_mask(frame, action)
                    unchanged["recent_" + action] &= bool(
                        np.array_equal(predicted.loc[recent, action + "_tree_score"], original.loc[recent, action + "_native_calibrated"])
                        and np.array_equal(predicted.loc[recent, action + "_tree_request"], original.loc[recent, action + "_native_calibrated_request"]))
                unchanged["exit"] &= bool(np.array_equal(predicted.exit_score, original.exit_native_calibrated)
                    and np.array_equal(predicted.exit_request, original.exit_native_calibrated_request))
                monthly[phase] = {}
                for month, part in predicted.groupby(predicted.end.dt.strftime("%Y-%m"), sort=True):
                    values = evaluate_quiet(part)
                    verify_quiet_evaluation(part, values)
                    monthly[phase][month] = values
                save_frame(out / (phase + "_predictions.parquet"), predicted)
                print(phase + ": 전체 확률·요청·5개 상태·모든 월 검산 완료", flush=True)
        decision = quiet_admission(metrics["diagnosis"], unchanged)
        verify_quiet_admission(metrics["diagnosis"], unchanged, decision)
        verification.update(original_rows_and_scores_preserved=True,
            all_states_and_months_verified=True, offsets_thresholds_and_decision_verified=True,
            unchanged=unchanged, rows={k: len(v) for k, v in rows.items()})
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("최근 행동 분리의 실행 중 입력·계획·코드 변경")
        for name, values in [("metrics", metrics), ("monthly", monthly), ("verification", verification),
                             ("decision", decision)]:
            save_json(out / (name + ".json"), values)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "quiet_models_admitted": decision["quiet_models_admitted"], "checks": decision["checks"],
            "rows": {k: len(v) for k, v in rows.items()}, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True,
            "all_inputs_and_code_unchanged": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/27 조건 통과", flush=True)
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
    run_quiet_management(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
