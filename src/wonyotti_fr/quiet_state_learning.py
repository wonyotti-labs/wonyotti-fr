from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet, quiet_predictions, validate_quiet_rows
from .quiet_management_learning import checked_quiet_reference
from .quiet_management_reference import (
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_support,
)
from .quiet_state_calibration import (
    fit_state_calibration,
    state_admission,
    state_means,
    state_predictions,
    validate_state_support,
)
from .quiet_state_reference import (
    verify_state_admission,
    verify_state_calibration,
    verify_state_means,
    verify_state_predictions,
)
from .retained_weight_close_reference import checked_files


def checked_state_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("상태별 보정의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("quiet_models_admitted") is not False
        or summary.get("new_accounts_executed") is not False):
        raise ValueError("상태별 보정의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    native, teacher = checked_quiet_reference(Path(settings["reference"]), settings["reference_sha256"])
    if str(native) != settings["previous"] or str(teacher) != settings["teacher"]:
        raise ValueError("상태별 보정의 원본·대조 출처 변경")
    return root, native, teacher


def run_state_calibration(reference, expected, output):
    previous, native, teacher = checked_state_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V127.md")
    names = ["files", "models", "offsets", "thresholds", "threshold_selection", "constants", "metrics", "monthly", "support"]
    paths = [reference, protocol, teacher / "files.json", native / "files.json"]
    paths += [previous / (name + ".json") for name in names]
    for phase in ["training", "calibration", "diagnosis"]:
        paths += [root / (phase + suffix) for root, suffix in [
            (previous, "_predictions.parquet"), (native, "_predictions.parquet"), (teacher, "_used.parquet")]]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "quiet-state-calibration", {"reference": str(reference),
        "reference_sha256": expected, "previous": str(previous), "native": str(native),
        "teacher": str(teacher), "protocol_sha256": sha256(protocol),
        "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(previous / "support.json")
        originals = {phase: pd.read_parquet(previous / (phase + "_predictions.parquet")) for phase in rows}
        state_support = {phase: validate_state_support(f) for phase, f in originals.items()}
        save_json(out / "support.json", state_support)
        models, old_offsets, old_thresholds, constants = [read_json(previous / (name + ".json"))
            for name in ["models", "offsets", "thresholds", "constants"]]
        offsets, thresholds, controls, details, fitted_support = fit_state_calibration(originals["calibration"])
        verify_state_calibration(originals["calibration"], offsets, thresholds, controls, details, fitted_support)
        for name, value in [("offsets", offsets), ("thresholds", thresholds), ("control_thresholds", controls), ("threshold_selection", details)]:
            save_json(out / (name + ".json"), value)
        print("상반기 8개 상태별 절편·16개 문턱 검산 완료", flush=True)
        prior_metrics, prior_months = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        metrics, control_metrics, monthly, control_monthly, means = {}, {}, {}, {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                old = originals[phase]
                base = pd.read_parquet(native / (phase + "_predictions.parquet"))
                reproduced = quiet_predictions(frame, base, models, old_offsets, old_thresholds, constants)
                pd.testing.assert_frame_equal(reproduced, old, check_exact=True)
                verify_quiet_predictions(frame, base, old, models, old_offsets, old_thresholds, constants)
                compare_statistics(evaluate_quiet(old), prior_metrics[phase])
                candidate, control = state_predictions(old, offsets, thresholds, controls)
                preserved = verify_state_predictions(old, candidate, control, offsets, thresholds, controls)
                unchanged = {key: unchanged[key] and preserved[key] for key in unchanged}
                metrics[phase], control_metrics[phase] = evaluate_quiet(candidate), evaluate_quiet(control)
                verify_quiet_evaluation(candidate, metrics[phase])
                verify_quiet_evaluation(control, control_metrics[phase])
                means[phase] = state_means(old, candidate)
                verify_state_means(old, candidate, means[phase])
                monthly[phase], control_monthly[phase] = {}, {}
                for month, indices in candidate.groupby(candidate.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    original_part = old.iloc[indices]
                    compare_statistics(evaluate_quiet(original_part), prior_months[phase][month])
                    for saved, destination in [(candidate, monthly), (control, control_monthly)]:
                        part = saved.iloc[indices]
                        destination[phase][month] = evaluate_quiet(part)
                        verify_quiet_evaluation(part, destination[phase][month])
                save_frame(out / (phase + "_predictions.parquet"), candidate)
                columns = ["end", "episode_id", *[a + "_" + m + "_request" for a in ["reduce", "increase"] for m in ["tree", "linear"]]]
                save_frame(out / (phase + "_threshold_only_requests.parquet"), control[columns])
                print(phase + ": 고정 모델·확률·동일 요청·상태·월별 지표 검산 완료", flush=True)
        decision = state_admission(metrics["diagnosis"], prior_metrics["diagnosis"], unchanged)
        verify_state_admission(metrics["diagnosis"], prior_metrics["diagnosis"], unchanged, decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("상태별 보정 중 입력·계획·코드 변경")
        for name, value in [("metrics", metrics), ("control_metrics", control_metrics), ("monthly", monthly),
                            ("control_monthly", control_monthly), ("means", means), ("decision", decision)]:
            save_json(out / (name + ".json"), value)
        verification = {"fixed_models_and_all_original_predictions_exact": True,
            "independent_numeric_tree_and_linear_predictions_verified": True,
            "all_offsets_thresholds_groups_months_means_and_checks_verified": True,
            "threshold_only_and_split_calibration_requests_equal": True,
            "unchanged": unchanged, "rows": {p: len(f) for p, f in rows.items()}}
        save_json(out / "verification.json", verification)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "state_calibration_admitted": decision["state_calibration_admitted"], "checks": decision["checks"],
            "rows": verification["rows"], "new_models_fitted": False, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True,
            "all_inputs_and_code_unchanged": True})
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
    run_state_calibration(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
