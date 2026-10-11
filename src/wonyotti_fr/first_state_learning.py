from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .entry_path_research import checked_entry_path_reference
from .event_research import load_selection
from .first_state_models import (
    evaluate_frame,
    first_state_admission,
    fit_first_models,
    prediction_frame,
)
from .first_state_reference import (
    refit_first_models,
    verify_first_admission,
    verify_first_calibration,
    verify_first_evaluation,
    verify_first_frame,
)
from .holding_stress import read_json, seal
from .management_flow_research import full_teacher_rows
from .retained_weight_close_reference import checked_files


def checked_first_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("첫 상태 학습의 참조 지문 오류")
    spec = read_json(path)
    for key in ["path_proof", "model_proof"]:
        item = Path(spec[key])
        if item.is_symlink() or sha256(item) != spec["proof_sha256"][key]:
            raise ValueError("첫 상태 학습의 선행 완료 증거 변경")
        proof = read_json(item)
        root = Path(proof["run"])
        checked_files(root)
        summary = read_json(root / "summary.json")
        if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
            or sha256(root / "files.json") != proof["files_sha256"]
            or sha256(root / "summary.json") != proof["summary_sha256"]
            or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True):
            raise ValueError("첫 상태 학습의 선행 검산 미완료")
        settings = read_json(root / "manifest.json")["settings"]
        if key == "path_proof":
            _, account, _ = checked_entry_path_reference(Path(settings["reference"]), settings["reference_sha256"])
            if account["selection"] != spec["selection"]:
                raise ValueError("첫 상태 학습의 기존 정책 연결 변경")
        else:
            if settings["teacher"] != spec["teacher"]:
                raise ValueError("첫 상태 학습의 원본 입력 연결 변경")
            verified = read_json(root / "verification.json")
            if not all(verified.get(k) is True for k in ["original_trees_and_offsets_exact", "original_diagnosis_predictions_exact"]):
                raise ValueError("첫 상태 학습의 기존 모형 재현 증거 부족")
    teacher = Path(spec["teacher"])
    checked_files(teacher)
    if sha256(teacher / "files.json") != spec["teacher_files_sha256"]:
        raise ValueError("첫 상태 학습의 원본 행·모델 변경")
    frozen, policy = load_selection(Path(spec["selection"]))
    if (frozen["protocol"] != "exit_state_v48"
        or sha256(Path(spec["selection"]) / "frozen_selection.json") != spec["selection_sha256"]
        or policy.manager.model.to_dict() != read_json(teacher / "models.json")["histogram"]
        or policy.manager.offset.to_dict() != read_json(teacher / "offsets.json")["histogram"]):
        raise ValueError("첫 상태 학습의 공통 대조 변경")
    return spec, policy


def verify_support(rows, support):
    for phase, frame in rows.items():
        for action in ["reduce", "increase"]:
            pairs = list(zip(frame[f"past_{action}_exists"], frame["y_" + action], strict=True))
            for name, flag in [("first", 0), ("repeat", 1)]:
                positive = sum(x == flag and y == 1 for x, y in pairs)
                negative = sum(x == flag and y == 0 for x, y in pairs)
                assert support[phase][action][name] == {"rows": positive + negative,
                    "positive": positive, "negative": negative}


def run_first_state_research(reference, expected, output):
    spec, policy = checked_first_reference(reference, expected)
    teacher, protocol = Path(spec["teacher"]), Path("docs/EXPERIMENT_V119.md")
    out = new_run(output, "first-state-learning", {"reference": str(reference),
        "reference_sha256": expected, "spec": spec, "protocol_sha256": sha256(protocol),
        "first_state_only": True, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        rows = full_teacher_rows(teacher)
        print("첫 추가·축소의 고정 부스팅·선형 적합 및 상반기 보정", flush=True)
        models, offsets, thresholds, details, support = fit_first_models(rows)
        verify_support(rows, support)
        verification = refit_first_models(rows, models)
        for name, value in [("models", models), ("offsets", offsets), ("thresholds", thresholds),
                            ("threshold_selection", details), ("support", support)]:
            save_json(out / (name + ".json"), value)
        constants = {a: float(rows["calibration"].loc[rows["calibration"][f"past_{a}_exists"].eq(0), "y_" + a].mean())
                     for a in ["reduce", "increase"]}
        metrics, monthly = {}, {}
        unchanged = {"exit": True, "repeat_reduce": True, "repeat_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                scores = prediction_frame(frame, policy, models, offsets, thresholds, constants)
                verify_first_frame(frame, scores, policy, models, offsets, thresholds, constants)
                if phase == "calibration":
                    verify_first_calibration(scores, offsets, thresholds, details, policy)
                if phase == "diagnosis":
                    previous = pd.read_parquet(teacher / "predictions.parquet")
                    pd.testing.assert_frame_equal(scores[["end", "episode_id"]], previous[["end", "episode_id"]], check_exact=True)
                    for action in ["exit", "reduce", "increase"]:
                        column = "exit_score" if action == "exit" else action + "_baseline_score"
                        np.testing.assert_array_equal(scores[column], previous[action + "_histogram_calibrated"])
                metrics[phase] = evaluate_frame(scores)
                verify_first_evaluation(scores, metrics[phase])
                monthly[phase] = {}
                for month, part in scores.groupby(scores.end.dt.strftime("%Y-%m"), sort=True):
                    monthly[phase][month] = evaluate_frame(part)
                    verify_first_evaluation(part, monthly[phase][month])
                path = out / (phase + "_predictions.parquet")
                scores.to_parquet(path, index=False)
                pd.testing.assert_frame_equal(scores, pd.read_parquet(path), check_exact=True)
                print(phase + ": 확률·보존 요청·모든 월 지표 검산 완료", flush=True)
        decision = first_state_admission(metrics["diagnosis"], unchanged)
        verify_first_admission(metrics["diagnosis"], unchanged, decision)
        verification.update(all_prediction_rows=sum(len(v) for v in rows.values()),
            support_counts_verified=True, original_diagnosis_predictions_exact=True,
            all_probabilities_offsets_thresholds_and_months_verified=True,
            exit_and_repeat_scores_and_requests_unchanged=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("decision", decision), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        if checked_first_reference(reference, expected)[0] != spec:
            raise ValueError("첫 상태 학습 중 기존 입력 변경")
        manifest = read_json(out / "manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent / name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("첫 상태 학습 중 계획·코드 변경")
        save_json(out / "summary.json", {"complete": True, "rows": {k: len(v) for k, v in rows.items()},
            "independent_verification_complete": True, "new_accounts_executed": False, **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="첫 추가·축소의 시간순 분리 학습과 독립 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_first_state_research(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
