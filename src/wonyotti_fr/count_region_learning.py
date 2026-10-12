from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .add_count_audit import attach_add_counts, audit_count_strata
from .add_count_reference import verify_add_count_links, verify_count_strata
from .common import new_run, save_json, sha256
from .count_region_management import (
    COMPARISONS,
    FEATURES,
    REGIONS,
    count_region_admission,
    count_region_quiet_predictions,
    evaluate_count_regions,
    fit_count_region_models,
    pre_limit_restart_metrics,
    region_masks,
    validate_count_region_inputs,
    validate_region_support,
)
from .count_region_reference import (
    verify_count_region_admission,
    verify_count_region_quiet_predictions,
    verify_count_region_refits,
    verify_count_regions,
    verify_pre_limit_restart,
    verify_region_support,
    verify_routing_ledger,
)
from .cross_market_learning import checked_cross_input_reference
from .cross_market_models import FEATURES as PREVIOUS_FEATURES
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet, quiet_predictions, validate_quiet_rows
from .quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_refits,
    verify_quiet_support,
)
from .quiet_state_calibration import fit_state_calibration, state_predictions
from .quiet_state_reference import verify_state_calibration, verify_state_predictions
from .retained_weight_close_reference import checked_files


def checked_count_region_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("횟수 분리 학습의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"])
        or summary.get("cross_market_admitted") is not False
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_accounts_executed", "profitability_accepted"])):
        raise ValueError("횟수 분리 학습의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_cross_input_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"] or settings.get("features") != PREVIOUS_FEATURES:
        raise ValueError("횟수 분리 학습의 원본·선행 입력 연결 오류")
    return {**sources, "cross_candidate": str(root)}


def run_count_region_learning(reference, expected, output):
    sources = checked_count_region_reference(reference, expected)
    teacher, previous, quiet, native, audit, counts, prices = [Path(sources[k]) for k in
        ["teacher", "previous", "quiet", "native", "audit", "counts", "candidate"]]
    protocol = Path("docs/EXPERIMENT_V137.md")
    paths = [reference, protocol, audit / "actions.parquet", audit / "xbtusd_events.parquet"]
    for folder in [teacher, previous, quiet, native, counts, prices, Path(sources["cross_candidate"])]:
        paths += [folder / name for name in checked_files(folder)]
        paths.append(folder / "files.json")
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "count-region-learning", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "features": FEATURES,
        "regions": REGIONS, "iterations_per_region": 128, "total_new_trees": 512, "bot_max_adds_unchanged": 5,
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(quiet / "support.json")
        save_json(out / "support.json", support)
        regional_support = validate_region_support(rows)
        for phase, frame in rows.items():
            verify_region_support(frame, regional_support[phase])
        save_json(out / "region_support.json", regional_support)
        old_fit = [read_json(quiet / (k + ".json")) for k in ["models", "offsets", "thresholds", "threshold_selection", "constants"]]
        verification = {"previous_refits": verify_quiet_refits(rows, old_fit[0]), "count_reconstruction": {}}
        old_state = [read_json(previous / (k + ".json")) for k in ["offsets", "thresholds", "control_thresholds", "threshold_selection", "support"]]
        actions, events = pd.read_parquet(audit / "actions.parquet"), pd.read_parquet(audit / "xbtusd_events.parquet")
        saved_counts, saved_monthly = read_json(counts / "metrics.json"), read_json(counts / "monthly.json")
        originals = {}
        for phase, frame in rows.items():
            links = attach_add_counts(frame, actions, events)
            verification["count_reconstruction"][phase] = verify_add_count_links(frame, actions, events, links)
            pd.testing.assert_frame_equal(links, pd.read_parquet(counts / (phase + "_count_links.parquet")), check_exact=True)
            validate_count_region_inputs(frame)
            ledger = frame[["end", "episode_id"]].copy()
            for action in ["reduce", "increase"]:
                routes = np.full(len(frame), "recent", dtype=object)
                for region, mask in region_masks(frame, action).items():
                    routes[mask] = region
                ledger[action + "_region"] = routes
            verify_routing_ledger(frame, ledger)
            save_frame(out / (phase + "_routing.parquet"), ledger)
            base = pd.read_parquet(native / (phase + "_predictions.parquet"))
            old = quiet_predictions(frame, base, *old_fit[:3], old_fit[4])
            pd.testing.assert_frame_equal(old, pd.read_parquet(quiet / (phase + "_predictions.parquet")), check_exact=True)
            verify_quiet_predictions(frame, base, old, *old_fit[:3], old_fit[4])
            original, control = state_predictions(old, *old_state[:3])
            verify_state_predictions(old, original, control, *old_state[:3])
            pd.testing.assert_frame_equal(original, pd.read_parquet(previous / (phase + "_predictions.parquet")), check_exact=True)
            if phase == "calibration":
                verify_quiet_calibration(old, *old_fit[1:])
                verify_state_calibration(old, *old_state[:4], old_state[4][phase])
            fixed = {"v127": original, "v133": pd.read_parquet(prices / (phase + "_predictions.parquet"))}
            reproduced = audit_count_strata(frame, fixed)
            compare_statistics(reproduced, saved_counts[phase])
            verify_count_strata(frame, fixed, reproduced)
            for month, positions in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                part, values = frame.iloc[positions], {k: v.iloc[positions] for k, v in fixed.items()}
                reproduced = audit_count_strata(part, values)
                compare_statistics(reproduced, saved_monthly[phase][month])
                verify_count_strata(part, values, reproduced)
            originals[phase] = original
            print(phase + ": 기존 50개·횟수 분리·v134 전체 집계·기존 v127 예측 재현 완료", flush=True)
        fitted = fit_count_region_models(rows["training"], rows["calibration"])
        for name, value in zip(["models", "global_offsets", "global_thresholds", "global_threshold_selection", "constants"], fitted, strict=True):
            save_json(out / (name + ".json"), value)
        print("한도 전후 50개 입력의 새 512개 트리·네 선형 학습 완료, 독립 재적합 시작", flush=True)
        verification["new_refits"] = verify_count_region_refits(rows, fitted[0])
        calibration = count_region_quiet_predictions(rows["calibration"], originals["calibration"], *fitted[:3], fitted[4])
        verify_count_region_quiet_predictions(rows["calibration"], originals["calibration"], calibration, *fitted[:3], fitted[4])
        verify_quiet_calibration(calibration, *fitted[1:])
        state = fit_state_calibration(calibration)
        verify_state_calibration(calibration, *state)
        for name, value in zip(["offsets", "thresholds", "control_thresholds", "threshold_selection", "calibration_support"], state, strict=True):
            save_json(out / (name + ".json"), value)
        metrics, monthly, controls, global_metrics = {}, {}, {}, {}
        strata, monthly_strata, pre_limit, monthly_pre_limit = {}, {}, {}, {}
        regions, monthly_regions = {}, {}
        old_metrics, old_months = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                old = originals[phase]
                shared = count_region_quiet_predictions(frame, old, *fitted[:3], fitted[4])
                preserved = verify_count_region_quiet_predictions(frame, old, shared, *fitted[:3], fitted[4])
                candidate, control = state_predictions(shared, *state[:3])
                verify_state_predictions(shared, candidate, control, *state[:3])
                unchanged = {k: unchanged[k] and preserved[k] for k in unchanged}
                for saved, destination in [(candidate, metrics), (control, controls), (shared, global_metrics)]:
                    destination[phase] = evaluate_quiet(saved)
                    verify_quiet_evaluation(saved, destination[phase])
                compare_statistics(evaluate_quiet(old), old_metrics[phase])
                predictions = {"v127": old, "v137": candidate}
                regions[phase] = evaluate_count_regions(frame, predictions)
                verify_count_regions(frame, predictions, regions[phase])
                monthly_regions[phase] = {}
                strata[phase] = audit_count_strata(frame, predictions, COMPARISONS)
                verify_count_strata(frame, predictions, strata[phase], COMPARISONS)
                pre_limit[phase] = pre_limit_restart_metrics(frame, predictions)
                verify_pre_limit_restart(frame, predictions, pre_limit[phase])
                monthly[phase], monthly_strata[phase], monthly_pre_limit[phase] = {}, {}, {}
                for month, positions in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = candidate.iloc[positions]
                    monthly[phase][month] = evaluate_quiet(part)
                    verify_quiet_evaluation(part, monthly[phase][month])
                    compare_statistics(evaluate_quiet(old.iloc[positions]), old_months[phase][month])
                    source, values = frame.iloc[positions], {k: v.iloc[positions] for k, v in predictions.items()}
                    monthly_regions[phase][month] = evaluate_count_regions(source, values)
                    verify_count_regions(source, values, monthly_regions[phase][month])
                    monthly_strata[phase][month] = audit_count_strata(source, values, COMPARISONS)
                    verify_count_strata(source, values, monthly_strata[phase][month], COMPARISONS)
                    monthly_pre_limit[phase][month] = pre_limit_restart_metrics(source, values)
                    verify_pre_limit_restart(source, values, monthly_pre_limit[phase][month])
                save_frame(out / (phase + "_predictions.parquet"), candidate)
                fields = ["end", "episode_id", *[a + "_" + m + "_request" for a in ["reduce", "increase"] for m in ["tree", "linear"]]]
                save_frame(out / (phase + "_threshold_only_requests.parquet"), control[fields])
                print(phase + ": 전체 확률·월·횟수 구간·제한 전 재시작 직접 지표 검산 완료", flush=True)
        decision = count_region_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, pre_limit["diagnosis"])
        verify_count_region_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, pre_limit["diagnosis"], decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("횟수 분리 학습 중 입력·코드·계획 변경")
        verification.update(unchanged=unchanged, original_rows_and_predictions_exact=True,
            original_fifty_features_unchanged=True, region_memberships_and_metrics_verified=True, all_offsets_thresholds_constants_months_buckets_verified=True,
            pre_limit_restart_metrics_direct=True, threshold_only_requests_exact=True)
        outputs = {"metrics": metrics, "monthly": monthly, "control_metrics": controls, "global_metrics": global_metrics,
            "strata": strata, "monthly_strata": monthly_strata, "pre_limit": pre_limit, "monthly_pre_limit": monthly_pre_limit,
            "regions": regions, "monthly_regions": monthly_regions, "decision": decision, "verification": verification}
        for name, value in outputs.items():
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "checks": decision["checks"],
            "count_region_admitted": decision["count_region_admitted"], "new_trees": 512, "new_linear_models": 4,
            "all_inputs_and_code_unchanged": True, "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/40 조건 통과", flush=True)
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
    run_count_region_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
