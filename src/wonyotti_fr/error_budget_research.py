from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .add_count_audit import attach_add_counts, audit_count_strata
from .add_count_reference import verify_add_count_links, verify_count_strata
from .common import new_run, save_json, sha256
from .count_region_learning import checked_count_region_reference
from .count_region_management import (
    FEATURES,
    REGIONS,
    count_region_admission,
    count_region_quiet_predictions,
    evaluate_count_regions,
    pre_limit_restart_metrics,
)
from .count_region_reference import (
    verify_count_region_admission,
    verify_count_region_quiet_predictions,
    verify_count_regions,
    verify_pre_limit_restart,
    verify_routing_ledger,
)
from .error_budget import GROUPS, MODELS, evaluate_budget, fit_budget_cutoffs, focal_retention
from .error_budget_reference import (
    verify_budget_cutoffs,
    verify_budget_evaluation,
    verify_focal_retention,
)
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet, quiet_predictions, validate_quiet_rows
from .quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_support,
)
from .quiet_state_calibration import state_predictions
from .quiet_state_reference import verify_state_calibration, verify_state_predictions
from .retained_weight_close_reference import checked_files


def checked_budget_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("오신호 한도 진단의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (
        proof.get("complete") is not True
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(
            summary.get(k) is not True
            for k in [
                "complete",
                "independent_verification_complete",
                "all_inputs_and_code_unchanged",
            ]
        )
        or summary.get("count_region_admitted") is not False
        or any(
            summary.get(k) is not False or proof.get(k) is not False
            for k in ["new_accounts_executed", "profitability_accepted"]
        )
    ):
        raise ValueError("오신호 한도 진단의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_count_region_reference(
        Path(settings["reference"]), settings["reference_sha256"]
    )
    if (
        sources != settings["sources"]
        or settings.get("features") != FEATURES
        or settings.get("regions") != list(REGIONS)
        or settings.get("iterations_per_region") != 128
        or settings.get("total_new_trees") != 512
        or settings.get("bot_max_adds_unchanged") != 5
    ):
        raise ValueError("오신호 한도 진단의 원본·모형·분기 연결 오류")
    return {**sources, "region_candidate": str(root)}


def run_error_budget(reference, expected, output):
    sources = checked_budget_reference(reference, expected)
    teacher, previous, quiet, native, audit, counts, candidate = [
        Path(sources[k])
        for k in ["teacher", "previous", "quiet", "native", "audit", "counts", "region_candidate"]
    ]
    protocol = Path("docs/EXPERIMENT_V138.md")
    paths = [reference, protocol, audit / "actions.parquet", audit / "xbtusd_events.parquet"]
    for folder in [teacher, previous, quiet, native, counts, candidate]:
        paths += [folder / name for name in checked_files(folder)] + [folder / "files.json"]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(
        output,
        "error-budget-audit",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "sources": sources,
            "protocol_sha256": sha256(protocol),
            "groups": GROUPS,
            "models": MODELS,
            "budget_source": "v127_original_calibration_false_positive_count",
            "new_models_fitted": False,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(quiet / "support.json") == read_json(candidate / "support.json")
        old_fit = [
            read_json(quiet / (k + ".json"))
            for k in ["models", "offsets", "thresholds", "threshold_selection", "constants"]
        ]
        old_state = [
            read_json(previous / (k + ".json"))
            for k in [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "support",
            ]
        ]
        new_fit = [
            read_json(candidate / (k + ".json"))
            for k in [
                "models",
                "global_offsets",
                "global_thresholds",
                "global_threshold_selection",
                "constants",
            ]
        ]
        new_state = [
            read_json(candidate / (k + ".json"))
            for k in [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "calibration_support",
            ]
        ]
        actions, events = (
            pd.read_parquet(audit / "actions.parquet"),
            pd.read_parquet(audit / "xbtusd_events.parquet"),
        )
        saved = {
            k: read_json(candidate / (k + ".json"))
            for k in [
                "metrics",
                "monthly",
                "control_metrics",
                "global_metrics",
                "strata",
                "monthly_strata",
                "pre_limit",
                "monthly_pre_limit",
                "regions",
                "monthly_regions",
                "decision",
            ]
        }
        old_metrics, old_months = (
            read_json(previous / "metrics.json"),
            read_json(previous / "monthly.json"),
        )
        predictions, verification = {}, {}
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        for phase, frame in rows.items():
            links = attach_add_counts(frame, actions, events)
            verification[phase] = verify_add_count_links(frame, actions, events, links)
            pd.testing.assert_frame_equal(
                links, pd.read_parquet(counts / (phase + "_count_links.parquet")), check_exact=True
            )
            verify_routing_ledger(frame, pd.read_parquet(candidate / (phase + "_routing.parquet")))
            base = pd.read_parquet(native / (phase + "_predictions.parquet"))
            old_global = quiet_predictions(frame, base, *old_fit[:3], old_fit[4])
            verify_quiet_predictions(frame, base, old_global, *old_fit[:3], old_fit[4])
            pd.testing.assert_frame_equal(
                old_global,
                pd.read_parquet(quiet / (phase + "_predictions.parquet")),
                check_exact=True,
            )
            old, old_control = state_predictions(old_global, *old_state[:3])
            verify_state_predictions(old_global, old, old_control, *old_state[:3])
            pd.testing.assert_frame_equal(
                old, pd.read_parquet(previous / (phase + "_predictions.parquet")), check_exact=True
            )
            shared = count_region_quiet_predictions(frame, old, *new_fit[:3], new_fit[4])
            fixed = verify_count_region_quiet_predictions(
                frame, old, shared, *new_fit[:3], new_fit[4]
            )
            unchanged = {k: unchanged[k] and fixed[k] for k in unchanged}
            new, control = state_predictions(shared, *new_state[:3])
            verify_state_predictions(shared, new, control, *new_state[:3])
            pd.testing.assert_frame_equal(
                new, pd.read_parquet(candidate / (phase + "_predictions.parquet")), check_exact=True
            )
            stored_control = pd.read_parquet(
                candidate / (phase + "_threshold_only_requests.parquet")
            )
            pd.testing.assert_frame_equal(
                control[stored_control.columns], stored_control, check_exact=True
            )
            if phase == "calibration":
                verify_quiet_calibration(old_global, *old_fit[1:])
                verify_state_calibration(old_global, *old_state[:4], old_state[4][phase])
                verify_quiet_calibration(shared, *new_fit[1:])
                verify_state_calibration(shared, *new_state)
            for value, original in [
                (old, old_metrics[phase]),
                (new, saved["metrics"][phase]),
                (shared, saved["global_metrics"][phase]),
                (control, saved["control_metrics"][phase]),
            ]:
                metrics = evaluate_quiet(value)
                compare_statistics(metrics, original)
                verify_quiet_evaluation(value, metrics)
            pair = {"v127": old, "v137": new}
            partitions = [(None, frame, pair)]
            partitions += [
                (month, frame.iloc[indices], {k: v.iloc[indices] for k, v in pair.items()})
                for month, indices in frame.groupby(
                    frame.end.dt.strftime("%Y-%m"), sort=True
                ).indices.items()
            ]
            for month, part, values in partitions:
                for name, function, verifier in [
                    ("strata", audit_count_strata, verify_count_strata),
                    ("regions", evaluate_count_regions, verify_count_regions),
                    ("pre_limit", pre_limit_restart_metrics, verify_pre_limit_restart),
                ]:
                    result = (
                        function(part, values, MODELS)
                        if name == "strata"
                        else function(part, values)
                    )
                    original = (
                        saved[name][phase]
                        if month is None
                        else saved["monthly_" + name][phase][month]
                    )
                    compare_statistics(result, original)
                    if name == "strata":
                        verifier(part, values, result, MODELS)
                    else:
                        verifier(part, values, result)
                if month is not None:
                    compare_statistics(evaluate_quiet(values["v127"]), old_months[phase][month])
                    compare_statistics(
                        evaluate_quiet(values["v137"]), saved["monthly"][phase][month]
                    )
            predictions[phase] = pair
            verification[phase].update(
                original_probabilities_requests_and_metrics_exact=True,
                original_count_routing_exact=True,
            )
            print(phase + ": 원본·횟수·고정 모형의 모든 확률·요청·지표 재구성 완료", flush=True)
        decision = count_region_admission(
            saved["metrics"]["diagnosis"],
            old_metrics["diagnosis"],
            unchanged,
            saved["pre_limit"]["diagnosis"],
        )
        verify_count_region_admission(
            saved["metrics"]["diagnosis"],
            old_metrics["diagnosis"],
            unchanged,
            saved["pre_limit"]["diagnosis"],
            decision,
        )
        compare_statistics(decision, saved["decision"])
        cutoffs = fit_budget_cutoffs(rows["calibration"], predictions["calibration"])
        verify_budget_cutoffs(rows["calibration"], predictions["calibration"], cutoffs)
        save_json(out / "cutoffs.json", cutoffs)
        metrics, monthly, retention, monthly_retention = {}, {}, {}, {}
        for phase, frame in rows.items():
            pair = predictions[phase]
            metrics[phase], ledger = evaluate_budget(frame, pair, cutoffs)
            verify_budget_evaluation(frame, pair, cutoffs, metrics[phase], ledger)
            retention[phase] = focal_retention(frame, pair, cutoffs, ledger)
            verify_focal_retention(frame, pair, cutoffs, ledger, retention[phase])
            monthly[phase], monthly_retention[phase] = {}, {}
            for month, positions in frame.groupby(
                frame.end.dt.strftime("%Y-%m"), sort=True
            ).indices.items():
                part, values = (
                    frame.iloc[positions],
                    {k: v.iloc[positions] for k, v in pair.items()},
                )
                monthly[phase][month], local = evaluate_budget(part, values, cutoffs)
                verify_budget_evaluation(part, values, cutoffs, monthly[phase][month], local)
                pd.testing.assert_frame_equal(local, ledger.iloc[positions], check_exact=True)
                monthly_retention[phase][month] = focal_retention(part, values, cutoffs, local)
                verify_focal_retention(
                    part, values, cutoffs, local, monthly_retention[phase][month]
                )
            save_frame(out / (phase + "_requests.parquet"), ledger)
            print(phase + ": 고정 예산·모든 집단·월·요청 교집합·새 적중 유지 검산 완료", flush=True)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("오신호 한도 진단 중 입력·코드·계획 변경")
        for name, value in [
            ("metrics", metrics),
            ("monthly", monthly),
            ("focal_retention", retention),
            ("monthly_focal_retention", monthly_retention),
            ("verification", verification),
            ("original_decision", decision),
        ]:
            save_json(out / (name + ".json"), value)
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "independent_verification_complete": True,
                "rows": {k: len(v) for k, v in rows.items()},
                "all_inputs_and_code_unchanged": True,
                "original_predictions_and_metrics_exact": True,
                "all_groups_months_cutoffs_and_requests_verified": True,
                "new_models_fitted": False,
                "new_accounts_executed": False,
                "profitability_accepted": False,
                "original_candidate_admitted": False,
                "all_source_periods_already_observed": True,
            },
        )
        seal(out)
        print("오신호 한도 진단 독립 검산 완료", flush=True)
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
    run_error_budget(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
