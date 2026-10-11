from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .add_count_audit import BUCKETS, MODELS, attach_add_counts, audit_count_strata
from .add_count_reference import verify_add_count_links, verify_count_strata
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .order_price_learning import checked_price_learning_reference
from .position_boost_verification import compare_statistics
from .retained_weight_close_reference import checked_files


def checked_count_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("추가 횟수 진단의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"])
        or summary.get("order_price_admitted") is not False
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_accounts_executed", "profitability_accepted"])):
        raise ValueError("추가 횟수 진단의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_price_learning_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"]:
        raise ValueError("추가 횟수 진단의 기존 원본 연결 오류")
    return {**sources, "candidate": str(root)}


def verify_original_metrics(strata, originals):
    for name in MODELS:
        for action in ["reduce", "increase"]:
            for state in ["all", "quiet", "recent", "first", "restart"]:
                compare_statistics(strata[action][state]["all"]["models"][name], originals[name][action][state]["tree"])


def run_add_count_research(reference, expected, output):
    sources = checked_count_reference(reference, expected)
    teacher, audit = [Path(sources[k]) for k in ["teacher", "audit"]]
    models = {"v127": Path(sources["previous"]), "v133": Path(sources["candidate"])}
    protocol = Path("docs/EXPERIMENT_V134.md")
    paths = [reference, protocol, audit / "actions.parquet", audit / "xbtusd_events.parquet"]
    for folder in [teacher, *models.values()]:
        paths += [folder / name for name in checked_files(folder)]
        paths.append(folder / "files.json")
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "add-count-audit", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "buckets": BUCKETS,
        "models": MODELS, "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        actions = pd.read_parquet(audit / "actions.parquet")
        events = pd.read_parquet(audit / "xbtusd_events.parquet")
        originals = {name: read_json(folder / "metrics.json") for name, folder in models.items()}
        original_months = {name: read_json(folder / "monthly.json") for name, folder in models.items()}
        metrics, monthly, verification = {}, {}, {}
        for phase, frame in rows.items():
            links = attach_add_counts(frame, actions, events)
            verification[phase] = verify_add_count_links(frame, actions, events, links)
            predictions = {name: pd.read_parquet(folder / (phase + "_predictions.parquet")) for name, folder in models.items()}
            metrics[phase] = audit_count_strata(frame, predictions)
            verify_count_strata(frame, predictions, metrics[phase])
            verify_original_metrics(metrics[phase], {name: values[phase] for name, values in originals.items()})
            monthly[phase] = {}
            for month, positions in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                part = frame.iloc[positions]
                saved = {name: values.iloc[positions] for name, values in predictions.items()}
                monthly[phase][month] = audit_count_strata(part, saved)
                verify_count_strata(part, saved, monthly[phase][month])
                verify_original_metrics(monthly[phase][month], {name: values[phase][month] for name, values in original_months.items()})
            save_frame(out / (phase + "_count_links.parquet"), links)
            verification[phase].update(all_bucket_metrics_and_sums_verified=True, original_state_and_month_metrics_exact=True)
            print(phase + ": " + str(len(frame)) + "행의 원본 횟수·두 예측·일곱 구간·모든 월 검산 완료", flush=True)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("추가 횟수 진단 중 입력·코드·계획 변경")
        for name, value in [("metrics", metrics), ("monthly", monthly), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "months": {k: len(v) for k, v in monthly.items()},
            "all_inputs_and_code_unchanged": True, "input_reconstruction_verified": True,
            "original_predictions_and_state_metrics_exact": True, "all_buckets_and_months_verified": True,
            "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True})
        seal(out)
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
    run_add_count_research(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
