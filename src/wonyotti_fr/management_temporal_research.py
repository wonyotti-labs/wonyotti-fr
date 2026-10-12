from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .error_budget import MODELS, budget_masks, evaluate_budget
from .error_budget_reference import verify_budget_evaluation
from .error_budget_research import checked_budget_reference
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .management_temporal import HORIZONS, match_temporal, temporal_months, temporal_summary
from .management_temporal_reference import verify_temporal, verify_temporal_summaries
from .position_boost_verification import compare_statistics
from .retained_weight_close_reference import checked_files


def checked_temporal_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("관리 시간 대응의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (
        proof.get("complete") is not True
        or type(proof.get("process_exit_code")) is not int
        or proof.get("process_exit_code") != 0
        or sha256(root / "files.json") != proof.get("files_sha256")
        or sha256(root / "summary.json") != proof.get("summary_sha256")
        or any(
            summary.get(k) is not True
            for k in [
                "complete",
                "independent_verification_complete",
                "all_inputs_and_code_unchanged",
                "original_predictions_and_metrics_exact",
                "all_groups_months_cutoffs_and_requests_verified",
            ]
        )
        or any(
            summary.get(k) is not False or proof.get(k) is not False
            for k in [
                "new_models_fitted",
                "new_accounts_executed",
                "profitability_accepted",
            ]
        )
        or summary.get("original_candidate_admitted") is not False
    ):
        raise ValueError("관리 시간 대응의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_budget_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"]:
        raise ValueError("관리 시간 대응의 원본 출처 연결 오류")
    return root, sources


def audit_temporal(frame, predictions, cutoffs, previous, previous_ledger, months):
    metrics, ledger = evaluate_budget(frame, predictions, cutoffs)
    verify_budget_evaluation(frame, predictions, cutoffs, metrics, ledger)
    compare_statistics(metrics, previous)
    pd.testing.assert_frame_equal(ledger, previous_ledger, check_exact=True)
    results, requests_out, targets_out = [], [], []
    for group, mask in budget_masks(frame).items():
        action = group.split("__")[0]
        targets = mask & frame["y_" + action].eq(1).to_numpy()
        for model in MODELS:
            for mode in ["original", "budget_matched"]:
                supported = (
                    mode == "original" or cutoffs[group]["models"][model]["status"] == "supported"
                )
                if not supported:
                    for horizon in HORIZONS:
                        results.append(
                            {
                                "group": group,
                                "model": model,
                                "mode": mode,
                                "horizon_minutes": horizon,
                                "status": "no_calibration_rows",
                                "total": None,
                                "monthly": {month: None for month in months},
                            }
                        )
                    continue
                requested = mask & (
                    predictions[model][action + "_tree_request"].to_numpy()
                    if mode == "original"
                    else ledger[model + "__" + group].fillna(False).to_numpy(bool)
                )
                exact_count = None
                for horizon in HORIZONS:
                    rr, ee = match_temporal(frame, requested, targets, horizon)
                    verify_temporal(frame, requested, targets, horizon, rr, ee)
                    total = temporal_summary(rr, ee, horizon)
                    monthly = temporal_months(rr, ee, horizon, months)
                    verify_temporal_summaries(rr, ee, horizon, total, monthly)
                    if horizon == 1:
                        old = previous[group]["models"][model][mode]
                        exact_count = total["matched_requests"]
                        if old is None:
                            assert total["requests"] == total["targets"] == 0
                        else:
                            assert exact_count == old["true_positive"]
                            assert total["unmatched_requests"] == old["false_positive"]
                            assert total["unmatched_targets"] == old["false_negative"]
                    else:
                        assert total["matched_requests"] >= exact_count
                    tags = {
                        "group": group,
                        "model": model,
                        "mode": mode,
                        "horizon_minutes": horizon,
                    }
                    results.append(
                        {**tags, "status": "supported", "total": total, "monthly": monthly}
                    )
                    for table, output in [(rr, requests_out), (ee, targets_out)]:
                        output.append(table.assign(**tags))
    return (
        results,
        pd.concat(requests_out, ignore_index=True),
        pd.concat(targets_out, ignore_index=True),
    )


def run_temporal_audit(reference, expected, output):
    parent, sources = checked_temporal_reference(reference, expected)
    teacher, previous, candidate, counts = (
        Path(sources[key]) for key in ["teacher", "previous", "region_candidate", "counts"]
    )
    protocol = Path("docs/EXPERIMENT_V139.md")
    paths = [reference, protocol, *sorted(Path(__file__).parent.glob("*.py"))]
    for folder in [parent, teacher, previous, candidate, counts]:
        paths.extend(folder / name for name in checked_files(folder))
        paths.append(folder / "files.json")
    before = {str(path): sha256(path) for path in paths}
    out = new_run(
        output,
        "management-temporal-audit",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "sources": sources,
            "protocol_sha256": sha256(protocol),
            "horizons": list(HORIZONS),
            "matching": "same_episode_retained_group_forward_half_open_one_to_one",
            "new_models_fitted": False,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        cutoffs, old_metrics = (
            read_json(parent / (name + ".json")) for name in ["cutoffs", "metrics"]
        )
        rows = full_teacher_rows(teacher)
        results, verification = {}, {}
        for phase, frame in rows.items():
            links = pd.read_parquet(counts / (phase + "_count_links.parquet"))
            context = [
                "end",
                "episode_id",
                "state_time",
                "entry_time",
                "direction",
                "adds",
                "adds_capped",
            ]
            pd.testing.assert_frame_equal(frame[context], links[context], check_exact=True)
            pair = {
                name: pd.read_parquet(folder / (phase + "_predictions.parquet"))
                for name, folder in [("v127", previous), ("v137", candidate)]
            }
            old_ledger = pd.read_parquet(parent / (phase + "_requests.parquet"))
            first, last = PERIODS[phase]
            months = (
                pd.date_range(first, last, freq="MS", inclusive="left").strftime("%Y-%m").tolist()
            )
            print(phase + ": 고정 요청·일대일 대응·이분 그래프 검산 시작", flush=True)
            values, requests, targets = audit_temporal(
                frame, pair, cutoffs, old_metrics[phase], old_ledger, months
            )
            results[phase] = values
            save_frame(out / (phase + "_requests.parquet"), requests)
            save_frame(out / (phase + "_targets.parquet"), targets)
            verification[phase] = {
                "rows": len(frame),
                "conditions": len(values),
                "months": months,
                "source_predictions_counts_and_budget_requests_exact": True,
                "one_minute_confusions_exact": True,
                "all_graph_cardinalities_verified": True,
                "all_monthly_and_total_summaries_verified": True,
                "parquet_readback_exact": True,
            }
            save_json(out / (phase + "_results.json"), values)
            print(phase + ": 144개 조건·모든 월·관측 부족 검산 완료", flush=True)
        if any(sha256(Path(path)) != digest for path, digest in before.items()):
            raise ValueError("관리 시간 대응의 실행 중 입력·코드·계획 변경")
        save_json(out / "verification.json", verification)
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "independent_verification_complete": True,
                "rows": {phase: len(frame) for phase, frame in rows.items()},
                "all_inputs_and_code_unchanged": True,
                "all_source_periods_already_observed": True,
                "new_models_fitted": False,
                "new_accounts_executed": False,
                "profitability_accepted": False,
                "original_candidate_admitted": False,
            },
        )
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    print("관리 시간 대응 독립 검산 완료", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="관리 요청과 후속 양성 분의 일대일 대응 진단")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_temporal_audit(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
