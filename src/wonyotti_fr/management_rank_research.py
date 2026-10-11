from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_experts_learning import checked_expert_reference
from .management_flow_research import full_teacher_rows
from .management_rank_audit import audit_scores, tail_cutoffs
from .management_rank_reference import reference_cutoffs, verify_score_audit
from .quiet_management import groups
from .quiet_management_reference import independent_groups
from .retained_weight_close_reference import checked_files

PHASES = ("training", "calibration", "diagnosis")
ACTIONS = ("reduce", "increase")
STATES = ("first", "restart")
MODELS = ("native", "capacity", "shared", "experts")
CONTEXT = ["end", "episode_id", "y_exit", "y_reduce", "y_increase", "past_reduce_exists",
           "past_increase_exists", "past_reduce_log_count_15m", "past_increase_log_count_15m"]


def checked_score_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("점수 진단의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("new_accounts_executed") is not False
        or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete"])
        or any(summary.get(k) is not False for k in ["state_experts_admitted", "new_accounts_executed", "profitability_accepted"])):
        raise ValueError("점수 진단의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    previous, quiet, native, teacher = checked_expert_reference(Path(settings["reference"]), settings["reference_sha256"])
    if any(str(value) != settings[key] for value, key in [
        (previous, "previous"), (quiet, "quiet"), (native, "native"), (teacher, "teacher")]):
        raise ValueError("점수 진단의 원본·모형 출처 변경")
    return root, previous, native, teacher


def verified_context(source, current, previous):
    for frame in [current, previous]:
        pd.testing.assert_frame_equal(source[CONTEXT], frame[CONTEXT], check_exact=True)
    pd.testing.assert_frame_equal(current[["exit_score", "exit_request"]],
                                   previous[["exit_score", "exit_request"]], check_exact=True)
    for action in ACTIONS:
        masks, independent = groups(source, action), independent_groups(source, action)
        for key in masks:
            np.testing.assert_array_equal(masks[key], independent[key])
        columns = [action + "_" + model + suffix for model in ["baseline", "capacity"]
                   for suffix in ["_score", "_request"]]
        pd.testing.assert_frame_equal(current[columns], previous[columns], check_exact=True)
        columns = [c for c in current if c.startswith(action + "_")]
        pd.testing.assert_frame_equal(current.loc[masks["recent"], columns],
                                       previous.loc[masks["recent"], columns], check_exact=True)


def views(current, previous, native_thresholds, shared_thresholds, expert_thresholds):
    result, thresholds = {}, {}
    for action in ACTIONS:
        for state in STATES:
            mask = groups(current, action)[state]
            for model in MODELS:
                key = action + "/" + state + "/" + model
                origin, stem = (previous, "tree") if model == "shared" else (current, "baseline" if model == "native" else "tree" if model == "experts" else model)
                prefix = action + "_" + stem
                frame = origin.loc[mask, ["end", "episode_id", "y_" + action, prefix + "_score", prefix + "_request"]].copy()
                frame.columns = ["end", "episode_id", "label", "score", "request"]
                result[key] = frame
                if model in {"native", "capacity"}:
                    threshold = native_thresholds["first" if state == "first" else "full"][model + "_calibrated"][action]
                else:
                    threshold = (shared_thresholds if model == "shared" else expert_thresholds)[action][state]["tree"]
                thresholds[key] = threshold
    return result, thresholds


def verify_original(result, metric):
    if metric is None:
        assert result["original"]["rows"] == 0
        return
    for current, original in [("rows", "rows"), ("positive", "positive"), ("requests", "predicted_positive"),
                              ("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative")]:
        if result["original"][current] != metric[original]:
            raise ValueError("점수 진단과 기존 혼동표 불일치")


def run_score_audit(reference, expected, output):
    root, previous, native, teacher = checked_score_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V129.md")
    paths = [reference, protocol, native / "files.json", native / "thresholds.json", teacher / "files.json"]
    for origin in [root, previous]:
        paths += [origin / (name + ".json") for name in ["files", "summary", "thresholds", "metrics", "monthly"]]
    for phase in PHASES:
        paths += [root / (phase + "_predictions.parquet"), previous / (phase + "_predictions.parquet"),
                  teacher / (phase + "_used.parquet")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "management-rank-audit", {"reference": str(reference), "reference_sha256": expected,
        "root": str(root), "previous": str(previous), "native": str(native), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        thresholds = [read_json(origin / "thresholds.json") for origin in [native, previous, root]]
        prepared, limits = {}, None
        for phase in PHASES:
            current, old = [pd.read_parquet(origin / (phase + "_predictions.parquet")) for origin in [root, previous]]
            verified_context(rows[phase], current, old)
            prepared[phase], phase_limits = views(current, old, *thresholds)
            if limits is not None and phase_limits != limits:
                raise ValueError("점수 진단의 기간별 기존 문턱 변경")
            limits = phase_limits
        calibration = prepared["calibration"]
        cutoffs = {key: tail_cutoffs(part.score) for key, part in calibration.items()}
        for key, value in cutoffs.items():
            assert value == reference_cutoffs(calibration[key].score)
        save_json(out / "calibration_cutoffs.json", cutoffs)
        print("원본·기존 문턱·대조·상반기 절단점 검산 완료", flush=True)
        results, monthly, verified = {}, {}, {}
        old_metrics = {kind: read_json(origin / "metrics.json") for kind, origin in [("experts", root), ("shared", previous)]}
        old_monthly = {kind: read_json(origin / "monthly.json") for kind, origin in [("experts", root), ("shared", previous)]}
        for phase in PHASES:
            results[phase], monthly[phase], verified[phase] = {}, {}, {}
            ledgers = {}
            all_months = sorted(rows[phase].end.dt.strftime("%Y-%m").unique())
            for key, part in prepared[phase].items():
                action, state, model = key.split("/")
                kind, metric_name = ("shared", "tree") if model == "shared" else ("experts", "baseline" if model == "native" else "tree" if model == "experts" else model)
                result, ledger = audit_scores(part, limits[key], cutoffs[key])
                verified[phase][key] = verify_score_audit(part, limits[key], calibration[key].score, result, ledger)
                verify_original(result, old_metrics[kind][phase][action][state][metric_name])
                results[phase][key], ledgers[key] = result, ledger
                month_keys = part.end.dt.strftime("%Y-%m")
                for month in all_months:
                    subset = part.loc[month_keys.eq(month)]
                    value, month_ledger = audit_scores(subset, limits[key], cutoffs[key])
                    verify_score_audit(subset, limits[key], calibration[key].score, value, month_ledger)
                    verify_original(value, old_monthly[kind][phase][month][action][state][metric_name])
                    monthly[phase].setdefault(month, {})[key] = value
                    ledgers[key].setdefault("months", {})[month] = month_ledger
            ledger_path = out / (phase + "_ledger.json")
            save_json(ledger_path, ledgers)
            if read_json(ledger_path) != ledgers:
                raise ValueError("점수 진단 원장의 저장 후 불일치")
            print(phase + ": 16개 집단·모든 월의 순위·동점·혼동표·포지션 검산 완료", flush=True)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("점수 진단 중 입력·코드·계획 변경")
        for name, value in [("results", results), ("monthly", monthly), ("verification", verified)]:
            save_json(out / (name + ".json"), value)
        combinations = sum(len(v) for v in results.values())
        month_combinations = sum(len(v) for months in monthly.values() for v in months.values())
        if combinations != 48 or month_combinations != 384:
            raise ValueError("점수 진단의 기간·월별 집단 누락")
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "combinations": combinations, "monthly_combinations": month_combinations,
            "source_rows": {p: len(f) for p, f in rows.items()}, "all_inputs_and_code_unchanged": True,
            "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True, "new_policy_selected": False})
        seal(out)
        print("48개 기간·384개 월 조합 완료: 새 정책·수익성 미채택 유지", flush=True)
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
    run_score_audit(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
