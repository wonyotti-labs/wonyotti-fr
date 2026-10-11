from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .balanced_management import balanced_admission, fit_balanced_management
from .balanced_management_reference import (
    verify_balanced_admission,
    verify_balanced_predictions,
    verify_balanced_refit,
)
from .common import new_run, save_json, sha256
from .first_state_models import validate_first_rows
from .holding_stress import read_json, seal
from .joint_history_models import evaluate_joint, joint_thresholds
from .joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from .management_calibration import ManagementOffset
from .management_capacity import LABELS, capacity_predictions, load_capacity_model
from .management_capacity_learning import checked_capacity_reference, save_frame
from .management_capacity_reference import verify_capacity_predictions
from .management_flow_research import full_teacher_rows
from .management_market_model_reference import verify_offsets
from .minute_management import ACTIONS
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels
from .position_boost_verification import compare_statistics
from .retained_weight_close_reference import checked_files


def checked_balanced_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("이진 균형 학습의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("capacity_admitted") is not False or summary.get("new_accounts_executed") is not False
        or summary.get("selected") != "depth2_iter256"):
        raise ValueError("이진 균형 학습의 선행 완료·선택·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    spec, policy = checked_capacity_reference(Path(settings["reference"]), settings["reference_sha256"])
    if spec != settings["spec"]:
        raise ValueError("이진 균형 학습의 원본 출처 변경")
    return root, spec, policy


def load_reference_models(root):
    raw, offsets = (read_json(root / (name + ".json")) for name in ["models", "offsets"])
    if set(raw) != {"native", "capacity", "linear"} or set(offsets) != set(raw):
        raise ValueError("이진 균형 학습의 대조 모형 누락")
    models = {"native": OrderHistoryBoostModels.from_dict(raw["native"]),
        "capacity": load_capacity_model(raw["capacity"]), "linear": OrderHistoryModels.from_dict(raw["linear"])}
    if models["capacity"].format != "management_capacity_depth2_iter256_v1":
        raise ValueError("이진 균형 학습의 직접 대조 크기 오류")
    return models, {name: ManagementOffset.from_dict(v) for name, v in offsets.items()}


def verify_reference_thresholds(root, thresholds, details):
    for filename, actual in [("thresholds", thresholds), ("threshold_selection", details)]:
        expected = read_json(root / (filename + ".json"))
        original = {group: {name: value for name, value in models.items() if not name.startswith("balanced_")}
                    for group, models in actual.items()}
        assert original == expected


def verify_reference_metrics(root, metrics, monthly):
    def original(data):
        return {a: {g: {k: v for k, v in models.items() if not k.startswith("balanced_")}
                    for g, models in groups.items()} for a, groups in data.items()}
    expected = read_json(root / "metrics.json")
    compare_statistics({p: original(v) for p, v in metrics.items()}, expected)
    compare_statistics({p: {month: original(v) for month, v in months.items()} for p, months in monthly.items()},
                       read_json(root / "monthly.json"))


def run_balanced_management(reference, expected, output):
    previous, spec, policy = checked_balanced_reference(reference, expected)
    teacher, protocol = Path(spec["teacher"]), Path("docs/EXPERIMENT_V122.md")
    out = new_run(output, "balanced-management-learning", {"reference": str(reference),
        "reference_sha256": expected, "spec": spec, "protocol_sha256": sha256(protocol),
        "previous": str(previous), "weight_rule": "binary_training_inverse_frequency",
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        rows = full_teacher_rows(teacher)
        support = validate_first_rows(rows)
        models, offsets = load_reference_models(previous)
        print("세 이진 행동의 빈도 가중치 계산 및 동일 256회 학습", flush=True)
        model, offset, weights, training = fit_balanced_management(rows)
        models["balanced"], offsets["balanced"] = model, offset
        for name, value in [("models", models), ("offsets", offsets)]:
            save_json(out / (name + ".json"), {k: v.to_dict() for k, v in value.items()})
        save_json(out / "training_support.json", training)
        save_json(out / "support.json", support)
        saved_weights = rows["training"][["end", "episode_id", *LABELS]].copy()
        saved_weights.insert(0, "source_row", np.arange(len(saved_weights)))
        for action in ACTIONS:
            saved_weights[action + "_weight"] = weights[action]
        save_frame(out / "training_weights.parquet", saved_weights)
        print("행별 가중치·768개 트리 및 전체 확률 독립 검산", flush=True)
        verification = verify_balanced_refit(rows, model, weights, training)
        cy = rows["calibration"][LABELS].to_numpy(int)
        constant = cy.mean(axis=0)
        metrics, monthly = {}, {}
        with threadpool_limits(limits=1):
            calibration = capacity_predictions(rows["calibration"], models, offsets, constant)
            verify_offsets(calibration, cy, offsets)
            thresholds, details = joint_thresholds(rows["calibration"], calibration)
            verify_joint_thresholds(rows["calibration"], calibration, thresholds, details)
            verify_reference_thresholds(previous, thresholds, details)
            if (thresholds["full"]["native_calibrated"] != policy.thresholds
                or thresholds["first"]["native_calibrated"] != policy.first_thresholds):
                raise ValueError("이진 균형 학습의 원래 첫·반복 문턱 불일치")
            save_json(out / "thresholds.json", thresholds)
            save_json(out / "threshold_selection.json", details)
            for phase, frame in rows.items():
                scores = calibration if phase == "calibration" else capacity_predictions(frame, models, offsets, constant)
                refs = {k: v for k, v in models.items() if k != "balanced"}
                verify_capacity_predictions(frame, refs, offsets, constant, scores)
                verify_balanced_predictions(frame, model, offset, scores)
                metrics[phase], requests = evaluate_joint(frame, scores, thresholds)
                verify_joint_evaluation(frame, scores, thresholds, metrics[phase], requests)
                monthly[phase] = {}
                for month, indices in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = frame.iloc[indices]
                    values = {name: p[indices] for name, p in scores.items()}
                    monthly[phase][month], decisions = evaluate_joint(part, values, thresholds)
                    verify_joint_evaluation(part, values, thresholds, monthly[phase][month], decisions)
                saved = frame[["end", "episode_id", "past_increase_exists", "past_reduce_exists", *LABELS]].copy()
                for name, p in scores.items():
                    for index, action in enumerate(ACTIONS):
                        saved[action + "_" + name] = p[:, index]
                        saved[action + "_" + name + "_request"] = requests[name][:, index]
                original = pd.read_parquet(previous / (phase + "_predictions.parquet"))
                pd.testing.assert_frame_equal(saved[original.columns], original, check_exact=True)
                save_frame(out / (phase + "_predictions.parquet"), saved)
                print(phase + ": 대조·전체 확률·첫 문턱·네 조합·모든 월 검산 완료", flush=True)
        verify_reference_metrics(previous, metrics, monthly)
        decision = balanced_admission(metrics["diagnosis"])
        verify_balanced_admission(metrics["diagnosis"], decision)
        verification.update(original_models_offsets_and_all_scores_requests_exact=True,
            original_first_and_full_thresholds_exact=True, original_all_period_and_month_metrics_preserved=True,
            all_prediction_rows=sum(len(v) for v in rows.values()), all_training_weights=len(rows["training"])*3,
            all_groups_months_and_requests_verified=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("decision", decision), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        after_root, after_spec, _ = checked_balanced_reference(reference, expected)
        if after_root != previous or after_spec != spec:
            raise ValueError("이진 균형 학습 중 기존 입력 변경")
        manifest = read_json(out / "manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent / name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("이진 균형 학습 중 코드·계획 변경")
        save_json(out / "summary.json", {"complete": True, "rows": {k: len(v) for k, v in rows.items()},
            "independent_verification_complete": True, "new_accounts_executed": False, **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="원본 관리 행동의 이진 빈도 균형 학습·독립 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_balanced_management(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
