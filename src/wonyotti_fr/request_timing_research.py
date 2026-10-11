from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .joint_history_models import joint_requests
from .minute_management import ACTIONS
from .request_timing import audit_action
from .request_timing_reference import verify_action
from .retained_weight_close_reference import checked_files

MODELS = ("native_calibrated", "capacity_calibrated", "aged_calibrated")
PHASES = ("training", "calibration", "diagnosis")
CONTEXT = ["end", "episode_id", "past_increase_exists", "past_reduce_exists",
           "y_exit", "y_reduce", "y_increase"]


def checked_timing_reference(reference, expected):
    if (reference.is_symlink() or reference.stat().st_size > 2**20
        or sha256(reference) != expected):
        raise ValueError("시간 진단의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or proof.get("new_accounts_executed") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete"])
        or any(summary.get(k) is not False for k in ["extrema_age_admitted", "new_accounts_executed",
                                                    "profitability_accepted"])):
        raise ValueError("시간 진단의 선행 완료·미채택 근거 오류")
    spec = read_json(root / "manifest.json")["settings"]["spec"]
    teacher = Path(spec["teacher"])
    checked_files(teacher)
    if sha256(teacher / "files.json") != spec["teacher_files_sha256"]:
        raise ValueError("시간 진단의 원본 정답 출처 변경")
    return root, teacher


def verify_saved_requests(frame, thresholds):
    for model in MODELS:
        probabilities = frame[[a + "_" + model for a in ACTIONS]].to_numpy(float)
        requests = joint_requests(frame, probabilities, thresholds["full"][model],
                                  thresholds["first"][model])
        for index, action in enumerate(ACTIONS):
            np.testing.assert_array_equal(requests[:, index], frame[action + "_" + model + "_request"])


def run_timing(reference, expected, output):
    root, teacher = checked_timing_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V125.md")
    paths = [reference, protocol, root / "files.json", root / "summary.json",
             root / "thresholds.json", root / "metrics.json", teacher / "files.json"]
    for phase in PHASES:
        paths += [root / (phase + "_predictions.parquet"), teacher / (phase + "_used.parquet")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "request-timing-audit", {"reference": str(reference),
        "reference_sha256": expected, "root": str(root), "teacher": str(teacher),
        "models": MODELS, "protocol_sha256": sha256(protocol),
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        thresholds, original = read_json(root / "thresholds.json"), read_json(root / "metrics.json")
        results, verification = {}, {}
        for phase in PHASES:
            frame = pd.read_parquet(root / (phase + "_predictions.parquet"))
            source = pd.read_parquet(teacher / (phase + "_used.parquet"), columns=CONTEXT)
            pd.testing.assert_frame_equal(frame[CONTEXT], source, check_exact=True)
            verify_saved_requests(frame, thresholds)
            results[phase], verification[phase] = {}, {}
            for model in MODELS:
                results[phase][model], verification[phase][model] = {}, {}
                for action in ACTIONS:
                    request, label = action + "_" + model + "_request", "y_" + action
                    result, ledger = audit_action(frame, request, label)
                    evidence = verify_action(frame, request, label, result, ledger)
                    old = original[phase][action]["all"][model]
                    for current, previous in [("rows", "rows"), ("positives", "positive"),
                        ("requests", "predicted_positive"), ("tp", "true_positive"),
                        ("fp", "false_positive"), ("fn", "false_negative")]:
                        if result[current] != old[previous]:
                            raise ValueError("시간 진단과 기존 혼동 수 불일치")
                    results[phase][model][action] = result
                    verification[phase][model][action] = evidence
                    path = out / (phase + "_" + model + "_" + action + "_ledger.json")
                    save_json(path, ledger)
                    if read_json(path) != ledger:
                        raise ValueError("시간 진단 원장의 저장 후 불일치")
            print(phase + ": 9개 조합의 요청·군집·최대 일대일 대응 검산 완료", flush=True)
        after = {str(p): sha256(p) for p in paths}
        if before != after:
            raise ValueError("시간 진단 중 입력·계획·코드 변경")
        save_json(out / "results.json", results)
        save_json(out / "verification.json", verification)
        save_json(out / "summary.json", {"complete": True, "combinations": 27,
            "source_rows": {p: results[p][MODELS[0]][ACTIONS[0]]["rows"] for p in PHASES},
            "all_inputs_and_code_unchanged": True, "independent_verification_complete": True,
            "prior_exact_metrics_preserved": True, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True})
        seal(out)
        print("전체 27개 조합 완료: 수익성 미채택 유지", flush=True)
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
    run_timing(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
