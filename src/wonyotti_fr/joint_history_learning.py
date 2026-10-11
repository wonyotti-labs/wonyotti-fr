from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .first_state_learning import checked_first_reference
from .first_state_models import validate_first_rows
from .holding_stress import read_json, seal
from .joint_history_models import (
    EXTRA_FEATURES,
    attach_joint_history,
    evaluate_joint,
    fit_joint_models,
    joint_admission,
    joint_feature_usage,
    joint_predictions,
    joint_thresholds,
    verify_joint_inputs,
)
from .joint_history_reference import (
    refit_joint_models,
    verify_joint_admission,
    verify_joint_evaluation,
    verify_joint_predictions,
    verify_joint_thresholds,
    verify_joint_usage,
)
from .management_flow_research import full_teacher_rows
from .management_market_model_reference import verify_offsets
from .minute_management import ACTIONS
from .retained_weight_close_reference import checked_files


def checked_joint_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("이력 조합 학습의 완료 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("first_state_models_admitted") is not False or summary.get("new_accounts_executed") is not False):
        raise ValueError("이력 조합 학습의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    spec, policy = checked_first_reference(Path(settings["reference"]), settings["reference_sha256"])
    if spec != settings["spec"]:
        raise ValueError("이력 조합 학습의 원본 출처 변경")
    return spec, policy


def run_joint_history_learning(reference, expected, output):
    spec, policy = checked_joint_reference(reference, expected)
    teacher, protocol = Path(spec["teacher"]), Path("docs/EXPERIMENT_V120.md")
    out = new_run(output, "joint-history-learning", {"reference": str(reference),
        "reference_sha256": expected, "spec": spec, "protocol_sha256": sha256(protocol),
        "extra_features": EXTRA_FEATURES, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        native = full_teacher_rows(teacher)
        expanded = attach_joint_history(native)
        input_count = verify_joint_inputs(native, expanded)
        support = validate_first_rows(native)
        print("네 이력 조합 입력 대조 및 전체 행의 고정 모델 학습", flush=True)
        models, offsets, training = fit_joint_models(native, expanded)
        if (models["native"].to_dict() != read_json(teacher / "models.json")["histogram"]
            or offsets["native"].to_dict() != read_json(teacher / "offsets.json")["histogram"]):
            raise ValueError("이력 조합 학습의 기존 모델·절편 재현 불일치")
        for name, value in [("models", models), ("offsets", offsets)]:
            save_json(out / (name + ".json"), {k: v.to_dict() for k, v in value.items()})
        save_json(out / "training_support.json", training)
        save_json(out / "support.json", support)
        print("독립 재적합의 384개 트리와 선형 계수 대조", flush=True)
        verification = refit_joint_models(native, expanded, models)
        cy = native["calibration"][["y_" + a for a in ACTIONS]].to_numpy(int)
        constant = cy.mean(axis=0)
        metrics, monthly = {}, {}
        with threadpool_limits(limits=1):
            calibration = joint_predictions(native["calibration"], expanded["calibration"], models, offsets, constant)
            verify_offsets(calibration, cy, offsets)
            thresholds, details = joint_thresholds(native["calibration"], calibration)
            verify_joint_thresholds(native["calibration"], calibration, thresholds, details)
            if (thresholds["full"]["native_calibrated"] != policy.thresholds
                or thresholds["first"]["native_calibrated"] != policy.first_thresholds):
                raise ValueError("이력 조합의 원래 첫·반복 문턱 재현 불일치")
            save_json(out / "thresholds.json", thresholds)
            save_json(out / "threshold_selection.json", details)
            for phase, frame in native.items():
                scores = calibration if phase == "calibration" else joint_predictions(frame, expanded[phase], models, offsets, constant)
                verify_joint_predictions(frame, expanded[phase], models, offsets, constant, scores)
                metrics[phase], requests = evaluate_joint(frame, scores, thresholds)
                verify_joint_evaluation(frame, scores, thresholds, metrics[phase], requests)
                monthly[phase] = {}
                for month, indices in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = frame.iloc[indices]
                    values = {name: p[indices] for name, p in scores.items()}
                    monthly[phase][month], decisions = evaluate_joint(part, values, thresholds)
                    verify_joint_evaluation(part, values, thresholds, monthly[phase][month], decisions)
                saved = frame[["end", "episode_id", "past_increase_exists", "past_reduce_exists", *["y_" + a for a in ACTIONS]]].copy()
                for name, p in scores.items():
                    for index, action in enumerate(ACTIONS):
                        saved[action + "_" + name] = p[:, index]
                        saved[action + "_" + name + "_request"] = requests[name][:, index]
                if phase == "diagnosis":
                    previous = pd.read_parquet(teacher / "predictions.parquet")
                    pd.testing.assert_frame_equal(frame[["end", "episode_id"]], previous[["end", "episode_id"]], check_exact=True)
                    for suffix in ["raw", "calibrated"]:
                        np.testing.assert_array_equal(scores["native_" + suffix], previous[[a + "_histogram_" + suffix for a in ACTIONS]].to_numpy())
                path = out / (phase + "_predictions.parquet")
                saved.to_parquet(path, index=False)
                pd.testing.assert_frame_equal(saved, pd.read_parquet(path), check_exact=True)
                print(phase + ": 전체 확률·첫 문턱·네 조합·모든 월 검산 완료", flush=True)
        decision = joint_admission(metrics["diagnosis"])
        verify_joint_admission(metrics["diagnosis"], decision)
        usage = joint_feature_usage(models["joint"])
        verify_joint_usage(models["joint"], usage)
        verification.update(original_trees_and_offsets_exact=True, original_diagnosis_predictions_exact=True,
            original_first_and_full_thresholds_exact=True, all_prediction_rows=sum(len(v) for v in native.values()),
            all_four_input_values_verified=input_count, all_groups_months_and_requests_verified=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("decision", decision),
                            ("feature_usage", usage), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        if checked_joint_reference(reference, expected)[0] != spec:
            raise ValueError("이력 조합 학습 중 기존 입력 변경")
        manifest = read_json(out / "manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent / name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("이력 조합 학습 중 코드·계획 변경")
        save_json(out / "summary.json", {"complete": True, "rows": {k: len(v) for k, v in native.items()},
            "independent_verification_complete": True, "new_accounts_executed": False, **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="추가·축소 이력 조합의 시간순 학습·요청 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_joint_history_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
