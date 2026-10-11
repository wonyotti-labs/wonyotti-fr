from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .first_state_models import validate_first_rows
from .holding_stress import read_json, seal
from .joint_history_learning import checked_joint_reference
from .joint_history_models import evaluate_joint, joint_thresholds
from .joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from .management_capacity import (
    GRID,
    LABELS,
    capacity_admission,
    capacity_predictions,
    fit_inner_models,
    fit_outer_models,
    load_capacity_model,
    select_capacity,
    split_inner,
)
from .management_capacity_reference import (
    verify_capacity_admission,
    verify_capacity_predictions,
    verify_inner_models,
    verify_inner_split,
    verify_outer_models,
    verify_selection,
)
from .management_flow_research import full_teacher_rows
from .management_market_model_reference import verify_offsets
from .minute_management import ACTIONS
from .retained_weight_close_reference import checked_files


def checked_capacity_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 복잡도의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("independent_verification_complete") is not True
        or summary.get("joint_history_admitted") is not False or summary.get("new_accounts_executed") is not False):
        raise ValueError("관리 복잡도의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    spec, policy = checked_joint_reference(Path(settings["reference"]), settings["reference_sha256"])
    if spec != settings["spec"]:
        raise ValueError("관리 복잡도의 원본 출처 변경")
    return spec, policy


def save_frame(path, frame):
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(path), check_exact=True)


def run_capacity_learning(reference, expected, output):
    spec, policy = checked_capacity_reference(reference, expected)
    teacher, protocol = Path(spec["teacher"]), Path("docs/EXPERIMENT_V121.md")
    out = new_run(output, "management-capacity-learning", {"reference": str(reference),
        "reference_sha256": expected, "spec": spec, "protocol_sha256": sha256(protocol), "grid": GRID,
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        rows = full_teacher_rows(teacher)
        support = validate_first_rows(rows)
        inner, ledger, inner_support = split_inner(rows["training"])
        verification = verify_inner_split(rows["training"], inner, ledger, inner_support)
        if (inner_support["fit"]["rows"] != 258703 or inner_support["selection"]["rows"] != 242373
            or inner_support["fit"]["episodes"] != 137 or inner_support["selection"]["episodes"] != 135
            or list(inner_support["fit"]["positive"].values()) != [142, 752, 759]
            or list(inner_support["selection"]["positive"].values()) != [160, 579, 615]
            or inner_support["crossing_episodes"] != 1):
            raise ValueError("관리 복잡도의 사전 관찰 지원 재현 불일치")
        save_frame(out / "inner_membership.parquet", ledger)
        save_json(out / "inner_support.json", inner_support)
        save_json(out / "support.json", support)
        print("2020년 내부 네 후보 학습 및 원래 확률 보존", flush=True)
        candidates, predictions, fit_support = fit_inner_models(inner)
        selection = select_capacity(inner, predictions)
        verify_selection(inner, predictions, selection)
        save_json(out / "inner_models.json", {k: v.to_dict() for k, v in candidates.items()})
        save_json(out / "inner_training_support.json", fit_support)
        save_json(out / "inner_selection.json", selection)
        for phase, frame in inner.items():
            saved = frame[["end", "episode_id", *LABELS]].copy()
            saved.insert(0, "source_row", frame.index)
            for name, phases in predictions.items():
                for i, action in enumerate(ACTIONS):
                    saved[name + "_" + action] = phases[phase][:, i]
            save_frame(out / ("inner_" + phase + "_predictions.parquet"), saved)
        print("내부 1,920개 트리·전체 확률·선택 점수 독립 검산", flush=True)
        verification["independent_inner_trees"] = verify_inner_models(inner, candidates, predictions)
        assert verification["independent_inner_trees"] == 1920
        for model in candidates.values():
            assert load_capacity_model(model.to_dict()).to_dict() == model.to_dict()
        selected = selection["selected"]
        print("내부 선택 완료: " + selected + "; 전체 2020년 재학습", flush=True)
        models, offsets, training = fit_outer_models(rows, selected)
        if (models["native"].to_dict() != read_json(teacher / "models.json")["histogram"]
            or offsets["native"].to_dict() != read_json(teacher / "offsets.json")["histogram"]):
            raise ValueError("관리 복잡도의 기존 모델·절편 재현 불일치")
        for name, value in [("models", models), ("offsets", offsets)]:
            save_json(out / (name + ".json"), {k: v.to_dict() for k, v in value.items()})
        save_json(out / "training_support.json", training)
        print("선택·기존 전체 재학습 및 선형의 독립 재적합", flush=True)
        verification.update(verify_outer_models(rows, models, selected))
        cy = rows["calibration"][LABELS].to_numpy(int)
        constant = cy.mean(axis=0)
        metrics, monthly = {}, {}
        with threadpool_limits(limits=1):
            calibration = capacity_predictions(rows["calibration"], models, offsets, constant)
            verify_offsets(calibration, cy, offsets)
            thresholds, details = joint_thresholds(rows["calibration"], calibration)
            verify_joint_thresholds(rows["calibration"], calibration, thresholds, details)
            if (thresholds["full"]["native_calibrated"] != policy.thresholds
                or thresholds["first"]["native_calibrated"] != policy.first_thresholds):
                raise ValueError("관리 복잡도의 원래 첫·반복 문턱 재현 불일치")
            save_json(out / "thresholds.json", thresholds)
            save_json(out / "threshold_selection.json", details)
            for phase, frame in rows.items():
                scores = calibration if phase == "calibration" else capacity_predictions(frame, models, offsets, constant)
                verify_capacity_predictions(frame, models, offsets, constant, scores)
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
                if phase == "diagnosis":
                    previous = pd.read_parquet(teacher / "predictions.parquet")
                    pd.testing.assert_frame_equal(frame[["end", "episode_id"]], previous[["end", "episode_id"]], check_exact=True)
                    for suffix in ["raw", "calibrated"]:
                        np.testing.assert_array_equal(scores["native_" + suffix], previous[[a + "_histogram_" + suffix for a in ACTIONS]].to_numpy())
                save_frame(out / (phase + "_predictions.parquet"), saved)
                print(phase + ": 전체 확률·첫 문턱·네 조합·모든 월 검산 완료", flush=True)
        decision = capacity_admission(metrics["diagnosis"])
        verify_capacity_admission(metrics["diagnosis"], decision)
        verification.update(original_trees_and_offsets_exact=True, original_diagnosis_predictions_exact=True,
            original_first_and_full_thresholds_exact=True, all_prediction_rows=sum(len(v) for v in rows.values()),
            inner_prediction_rows=sum(len(v) for v in inner.values()) * 4, all_groups_months_and_requests_verified=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("decision", decision), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        if checked_capacity_reference(reference, expected)[0] != spec:
            raise ValueError("관리 복잡도 학습 중 기존 입력 변경")
        manifest = read_json(out / "manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent / name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("관리 복잡도 학습 중 코드·계획 변경")
        save_json(out / "summary.json", {"complete": True, "rows": {k: len(v) for k, v in rows.items()},
            "selected": selected, "independent_verification_complete": True, "new_accounts_executed": False, **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="원본 관리 모델의 내부 시간순 크기 선택·외부 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_capacity_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
