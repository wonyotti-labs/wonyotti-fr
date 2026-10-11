from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_experts import expert_admission, expert_predictions, expert_support, fit_experts
from .management_experts_reference import (
    verify_expert_admission,
    verify_expert_calibration,
    verify_expert_predictions,
    verify_expert_refits,
    verify_expert_support,
)
from .management_flow_research import full_teacher_rows
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet, quiet_predictions
from .quiet_management_reference import verify_quiet_evaluation, verify_quiet_predictions
from .quiet_state_calibration import state_predictions
from .quiet_state_learning import checked_state_reference
from .quiet_state_reference import verify_state_predictions
from .retained_weight_close_reference import checked_files


def checked_expert_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 분리 학습의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("state_calibration_admitted") is not False
        or summary.get("new_accounts_executed") is not False):
        raise ValueError("관리 분리 학습의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    quiet, native, teacher = checked_state_reference(Path(settings["reference"]), settings["reference_sha256"])
    if any(str(value) != settings[key] for value, key in [(quiet, "previous"), (native, "native"), (teacher, "teacher")]):
        raise ValueError("관리 분리 학습의 원본·모형 출처 변경")
    return root, quiet, native, teacher


def run_experts(reference, expected, output):
    previous, quiet, native, teacher = checked_expert_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V128.md")
    paths = [reference, protocol, teacher / "files.json", native / "files.json"]
    for root, names in [(previous, ["files", "offsets", "thresholds", "control_thresholds", "metrics", "monthly"]),
                        (quiet, ["files", "models", "offsets", "thresholds", "constants", "support"])]:
        paths += [root / (name + ".json") for name in names]
    for phase in ["training", "calibration", "diagnosis"]:
        paths += [root / (phase + suffix) for root, suffix in [
            (previous, "_predictions.parquet"), (quiet, "_predictions.parquet"),
            (native, "_predictions.parquet"), (teacher, "_used.parquet")]]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "management-state-experts", {"reference": str(reference),
        "reference_sha256": expected, "previous": str(previous), "quiet": str(quiet),
        "native": str(native), "teacher": str(teacher), "protocol_sha256": sha256(protocol),
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = expert_support(rows)
        verify_expert_support(rows, support)
        assert support == read_json(quiet / "support.json")
        save_json(out / "support.json", support)
        print("최초·재시작 4개 학습 집단의 지원·원본 보존 검산 완료", flush=True)
        models, offsets, thresholds, details, constants = fit_experts(rows["training"], rows["calibration"])
        for name, value in [("models", models), ("offsets", offsets), ("thresholds", thresholds),
                            ("threshold_selection", details), ("constants", constants)]:
            save_json(out / (name + ".json"), value)
        print("새 1,024개 트리·4개 선형 학습 완료, 독립 재적합 검산 시작", flush=True)
        verification = verify_expert_refits(rows, models)
        old_models, old_offsets, old_thresholds, old_constants = [read_json(quiet / (name + ".json"))
            for name in ["models", "offsets", "thresholds", "constants"]]
        state_offsets, state_thresholds, state_controls = [read_json(previous / (name + ".json"))
            for name in ["offsets", "thresholds", "control_thresholds"]]
        old_metrics, old_monthly = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        metrics, monthly = {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                base = pd.read_parquet(native / (phase + "_predictions.parquet"))
                prior_quiet = pd.read_parquet(quiet / (phase + "_predictions.parquet"))
                recreated = quiet_predictions(frame, base, old_models, old_offsets, old_thresholds, old_constants)
                pd.testing.assert_frame_equal(recreated, prior_quiet, check_exact=True)
                verify_quiet_predictions(frame, base, prior_quiet, old_models, old_offsets, old_thresholds, old_constants)
                old = pd.read_parquet(previous / (phase + "_predictions.parquet"))
                recreated, control = state_predictions(prior_quiet, state_offsets, state_thresholds, state_controls)
                pd.testing.assert_frame_equal(recreated, old, check_exact=True)
                verify_state_predictions(prior_quiet, old, control, state_offsets, state_thresholds, state_controls)
                compare_statistics(evaluate_quiet(old), old_metrics[phase])
                saved = expert_predictions(frame, old, models, offsets, thresholds, constants)
                preserved = verify_expert_predictions(frame, old, saved, models, offsets, thresholds, constants)
                unchanged = {key: unchanged[key] and preserved[key] for key in unchanged}
                if phase == "calibration":
                    verify_expert_calibration(saved, offsets, thresholds, details, constants)
                metrics[phase] = evaluate_quiet(saved)
                verify_quiet_evaluation(saved, metrics[phase])
                monthly[phase] = {}
                for month, indices in saved.groupby(saved.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = saved.iloc[indices]
                    compare_statistics(evaluate_quiet(old.iloc[indices]), old_monthly[phase][month])
                    monthly[phase][month] = evaluate_quiet(part)
                    verify_quiet_evaluation(part, monthly[phase][month])
                save_frame(out / (phase + "_predictions.parquet"), saved)
                print(phase + ": 기존 대조·전체 확률·결합·상태·월별 지표 검산 완료", flush=True)
        decision = expert_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged)
        verify_expert_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("관리 분리 학습 중 입력·계획·코드 변경")
        verification.update(original_rows_and_reference_scores_exact=True,
            all_offsets_thresholds_constants_groups_months_and_checks_verified=True,
            unchanged=unchanged, rows={p: len(f) for p, f in rows.items()})
        for name, value in [("metrics", metrics), ("monthly", monthly), ("verification", verification), ("decision", decision)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "state_experts_admitted": decision["state_experts_admitted"], "checks": decision["checks"],
            "rows": verification["rows"], "new_accounts_executed": False, "profitability_accepted": False,
            "all_inputs_and_code_unchanged": True, "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/31 조건 통과", flush=True)
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
    run_experts(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
