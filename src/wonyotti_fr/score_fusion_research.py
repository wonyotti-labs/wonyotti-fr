from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_management import mapping_support
from .causal_management_reference import (
    verify_causal_predictions,
    verify_management_mapping,
    verify_mapping_support,
)
from .causal_management_research import checked_causal_reference
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .management_rank_research import CONTEXT
from .management_score_fusion import fit_score_fusion, fusion_admission, fusion_predictions
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet
from .quiet_management_reference import verify_quiet_evaluation
from .retained_weight_close_reference import checked_files
from .score_fusion_reference import (
    verify_fusion_admission,
    verify_fusion_predictions,
    verify_fusion_refits,
)


def checked_fusion_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("관리 결합 보정의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete"])
        or summary.get("causal_distribution_admitted") is not False
        or any(summary.get(k) is not False or proof.get(k) is not False
               for k in ["new_accounts_executed", "profitability_accepted"])):
        raise ValueError("관리 결합 보정의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    audit, previous, teacher = checked_causal_reference(Path(settings["reference"]), settings["reference_sha256"])
    if (settings.get("window_days") != 30 or settings.get("minimum_history") != 1000
        or any(str(value) != settings[key] for value, key in [(audit, "audit"), (previous, "previous"), (teacher, "teacher")])):
        raise ValueError("관리 결합 보정의 원본·이력 설정 오류")
    return root, previous, teacher


def run_score_fusion(reference, expected, output):
    root, previous, teacher = checked_fusion_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V131.md")
    paths = [reference, protocol, teacher / "files.json"]
    for origin in [root, previous]:
        paths += [origin / (name + ".json") for name in ["files", "metrics", "monthly"]]
    paths += [root / (name + ".json") for name in ["reference_quantiles", "support", "monthly_support", "offsets", "thresholds", "constants"]]
    for phase in ["training", "calibration", "diagnosis"]:
        paths += [root / (phase + "_predictions.parquet"), root / (phase + "_history.parquet"),
                  previous / (phase + "_predictions.parquet"), teacher / (phase + "_used.parquet")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "management-score-fusion", {"reference": str(reference), "reference_sha256": expected,
        "relative": str(root), "previous": str(previous), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        original = {phase: pd.read_parquet(previous / (phase + "_predictions.parquet")) for phase in rows}
        ledgers = {phase: pd.read_parquet(root / (phase + "_history.parquet")) for phase in rows}
        pools = {a: {s: {m: np.array(v) for m, v in models.items()} for s, models in states.items()}
                 for a, states in read_json(root / "reference_quantiles.json").items()}
        for phase, frame in original.items():
            pd.testing.assert_frame_equal(rows[phase][CONTEXT], frame[CONTEXT], check_exact=True)
        mapping_verified = verify_management_mapping(original, pools, ledgers)
        support = read_json(root / "support.json")
        for phase, frame in original.items():
            assert mapping_support(frame, ledgers[phase]) == support[phase]
            verify_mapping_support(frame, ledgers[phase], support[phase])
        save_json(out / "support.json", support)
        print("원본·기준 분포·모든 과거 n/L/E·지원 검산 완료, 단독·결합 보정기 적합 시작", flush=True)
        fitted = fit_score_fusion(original["calibration"], ledgers["calibration"])
        save_json(out / "calibrators.json", fitted)
        verification = verify_fusion_refits(original["calibration"], ledgers["calibration"], fitted)
        print("16개 보정기 독립 재적합 완료", flush=True)
        previous_metrics, previous_monthly = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        relative_metrics, relative_monthly = read_json(root / "metrics.json"), read_json(root / "monthly.json")
        old_parameters = [read_json(root / (name + ".json")) for name in ["offsets", "thresholds", "constants"]]
        old_support_months = read_json(root / "monthly_support.json")
        metrics, monthly, monthly_support = {}, {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        for phase, frame in original.items():
            ledger = ledgers[phase]
            prior = pd.read_parquet(root / (phase + "_predictions.parquet"))
            verify_causal_predictions(frame, ledger, prior, *old_parameters)
            compare_statistics(evaluate_quiet(frame), previous_metrics[phase])
            compare_statistics(evaluate_quiet(prior), relative_metrics[phase])
            metrics[phase], monthly[phase], monthly_support[phase] = {}, {}, {}
            for kind in ["absolute", "joint"]:
                saved = fusion_predictions(frame, ledger, fitted, kind)
                preserved = verify_fusion_predictions(frame, ledger, saved, fitted, kind, calibration=phase == "calibration")
                unchanged = {k: unchanged[k] and preserved[k] for k in unchanged}
                metrics[phase][kind] = evaluate_quiet(saved)
                verify_quiet_evaluation(saved, metrics[phase][kind])
                for month, indices in saved.groupby(saved.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = saved.iloc[indices]
                    monthly[phase].setdefault(month, {})[kind] = evaluate_quiet(part)
                    verify_quiet_evaluation(part, monthly[phase][month][kind])
                    if kind == "absolute":
                        compare_statistics(evaluate_quiet(frame.iloc[indices]), previous_monthly[phase][month])
                        compare_statistics(evaluate_quiet(prior.iloc[indices]), relative_monthly[phase][month])
                        value = mapping_support(frame.iloc[indices], ledger.iloc[indices])
                        assert value == old_support_months[phase][month]
                        verify_mapping_support(frame.iloc[indices], ledger.iloc[indices], value)
                        monthly_support[phase][month] = value
                save_frame(out / (phase + "_" + kind + "_predictions.parquet"), saved)
            print(phase + ": 단독·결합 전체 확률·문턱·보존·기존 대조·모든 월 검산 완료", flush=True)
        decision = fusion_admission(metrics["diagnosis"]["joint"], metrics["diagnosis"]["absolute"], previous_metrics["diagnosis"], unchanged)
        verify_fusion_admission(metrics["diagnosis"]["joint"], metrics["diagnosis"]["absolute"], previous_metrics["diagnosis"], unchanged, decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("관리 결합 보정 중 입력·코드·계획 변경")
        verification.update(mapping=mapping_verified, previous_predictions_groups_and_months_preserved=True,
                            all_new_calibration_and_predictions_verified=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("monthly_support", monthly_support),
                            ("verification", verification), ("decision", decision)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {p: len(f) for p, f in rows.items()}, "new_calibrators": 16,
            "checks": decision["checks"], "score_fusion_admitted": decision["score_fusion_admitted"],
            "all_inputs_and_code_unchanged": True, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/39 조건 통과", flush=True)
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
    run_score_fusion(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
