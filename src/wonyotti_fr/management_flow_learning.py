from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_calibration import ManagementOffset
from .management_flow import EXTRA_FEATURES
from .management_flow_model_reference import (
    refit_flow_models,
    verify_flow_admission,
    verify_flow_predictions,
    verify_flow_usage,
)
from .management_flow_models import (
    fit_flow_models,
    flow_feature_usage,
    flow_model_admission,
    flow_predictions,
    validate_flow_rows,
)
from .management_flow_research import checked_flow_reference, full_teacher_rows
from .management_market_model_reference import verify_market_evaluation, verify_offsets
from .minute_management import ACTIONS
from .order_history_boost import OrderHistoryBoostModels
from .paired_management_evaluation import choose_thresholds, evaluate
from .retained_weight_close_reference import checked_files


def checked_flow_learning_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 체결 학습의 입력 검산 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root/"summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or proof.get("paired_model_research_eligible") is not True
        or sha256(root/"files.json") != proof["files_sha256"]
        or sha256(root/"summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("original_rows_preserved") is not True
        or summary.get("paired_model_research_eligible") is not True
        or summary.get("new_models_fitted") is not False):
        raise ValueError("관리 체결 학습의 선행 전체 입력 지원 미충족")
    settings = read_json(root/"manifest.json")["settings"]
    support, verification = (read_json(root/(n+".json")) for n in ["support", "verification"])
    flags = ["all_twelve_values_independently_verified", "source_times_and_membership_verified",
             "original_columns_and_labels_exact", "support_counts_verified", "parquet_readback_exact"]
    if (set(support) != set(PERIODS) or set(verification) != set(PERIODS)
        or any(not all(verification[p].get(flag) is True for flag in flags)
            or verification[p]["rows"] != support[p]["groups"]["all"]["rows"]
            or support[p].get("all_original_rows_supported") is not True
            or support[p].get("action_support_sufficient") is not True for p in PERIODS)
        or settings["extra_features"] != EXTRA_FEATURES
        or settings["protocol_sha256"] != sha256(Path("docs/EXPERIMENT_V117.md"))):
        raise ValueError("관리 체결 학습의 전체 독립 검산·계획 변경")
    sources = checked_flow_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"]:
        raise ValueError("관리 체결 학습의 원본 연결 변경")
    return root, Path(sources["teacher"])


def flow_learning_rows(root, teacher):
    native = full_teacher_rows(teacher)
    expanded, memberships = {}, {}
    support = read_json(root/"support.json")
    for phase, frame in native.items():
        flow = pd.read_parquet(root/f"{phase}_flow.parquet")
        link = pd.read_parquet(root/f"{phase}_linkage.parquet")
        pd.testing.assert_series_equal(frame.end.reset_index(drop=True), flow.end, check_exact=True)
        pd.testing.assert_series_equal(frame.end.reset_index(drop=True), link.end, check_exact=True)
        np.testing.assert_array_equal(link.source_row, np.arange(len(frame)))
        if (not link.supported.all() or not link.reason.eq("supported").all()
            or len(frame) != support[phase]["groups"]["all"]["rows"]):
            raise ValueError("관리 체결 학습의 원래 전체 지원 행 변경")
        expanded[phase] = frame.copy()
        expanded[phase][EXTRA_FEATURES] = flow[EXTRA_FEATURES].to_numpy()
        memberships[phase] = link
    validate_flow_rows(native, expanded)
    return native, expanded, memberships


def run_flow_learning(reference, expected, output):
    root, teacher = checked_flow_learning_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V117.md")
    out = new_run(output, "management-flow-learning", {"reference": str(reference),
        "reference_sha256": expected, "input_run": str(root), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "periods": PERIODS,
        "new_model_families": ["native_histogram", "flow_histogram", "flow_logistic"],
        "profitability_accepted": False, "trading_returns_evaluated": False})
    print(out, flush=True)
    try:
        native, expanded, memberships = flow_learning_rows(root, teacher)
        old_model = OrderHistoryBoostModels.from_dict(read_json(teacher/"models.json")["histogram"])
        old_offset = ManagementOffset.from_dict(read_json(teacher/"offsets.json")["histogram"])
        print("전체 동일 행의 세 모형 적합·상반기 보정 시작", flush=True)
        models, offsets, support = fit_flow_models(native, expanded)
        if models["native"].to_dict() != old_model.to_dict() or offsets["native"].to_dict() != old_offset.to_dict():
            raise ValueError("관리 체결 학습의 원본 대조 트리·절편 재현 불일치")
        for name, values in [("models", models), ("offsets", offsets)]:
            save_json(out/(name+".json"), {k: v.to_dict() for k, v in values.items()})
        save_json(out/"training_support.json", support)
        print("독립 재학습의 384개 트리·선형 계수 대조 시작", flush=True)
        verification = refit_flow_models(native, expanded, models)
        cy = native["calibration"][["y_"+a for a in ACTIONS]].to_numpy(int)
        constant = cy.mean(axis=0)
        metrics, monthly = {}, {}
        with threadpool_limits(limits=1):
            calibration = flow_predictions(native["calibration"], expanded["calibration"], models, offsets, constant)
            verify_offsets(calibration, cy, offsets)
            thresholds, selections = choose_thresholds(cy, calibration)
            save_json(out/"thresholds.json", thresholds)
            save_json(out/"threshold_selection.json", selections)
            for phase in PERIODS:
                a, b = native[phase], expanded[phase]
                scores = calibration if phase == "calibration" else flow_predictions(a, b, models, offsets, constant)
                verify_flow_predictions(a, b, models, offsets, constant, scores)
                y = a[["y_"+action for action in ACTIONS]].to_numpy(int)
                metrics[phase], _ = evaluate(y, scores, thresholds)
                verify_market_evaluation(y, scores, thresholds, metrics[phase], selections if phase == "calibration" else None)
                monthly[phase] = {}
                months = a.end.dt.strftime("%Y-%m").to_numpy()
                for month in sorted(set(months)):
                    mask = months == month
                    sub = {name: values[mask] for name, values in scores.items()}
                    monthly[phase][month], _ = evaluate(y[mask], sub, thresholds)
                    verify_market_evaluation(y[mask], sub, thresholds, monthly[phase][month])
                saved = a[["end", "episode_id", *["y_"+action for action in ACTIONS]]].copy()
                for name, values in scores.items():
                    for j, action in enumerate(ACTIONS):
                        saved[action+"_"+name] = values[:, j]
                if phase == "diagnosis":
                    previous = pd.read_parquet(teacher/"predictions.parquet")
                    pd.testing.assert_frame_equal(previous[["end", "episode_id"]], a[["end", "episode_id"]], check_exact=True)
                    for suffix in ["raw", "calibrated"]:
                        np.testing.assert_array_equal(scores["native_"+suffix],
                            previous[[action+"_histogram_"+suffix for action in ACTIONS]].to_numpy())
                for suffix, frame in [("predictions", saved), ("membership", memberships[phase])]:
                    path = out/f"{phase}_{suffix}.parquet"
                    frame.to_parquet(path, index=False)
                    pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
                print(phase+": 전체 확률·문턱·월별 지표 검산 완료", flush=True)
        decision = flow_model_admission(metrics["diagnosis"])
        verify_flow_admission(metrics["diagnosis"], decision)
        usage = flow_feature_usage(models["flow"])
        verify_flow_usage(models["flow"], usage)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("decision", decision), ("flow_feature_usage", usage)]:
            save_json(out/(name+".json"), value)
        verification.update(original_trees_and_offsets_exact=True, original_diagnosis_predictions_exact=True,
            all_probability_rows_verified=sum(len(v) for v in native.values()),
            offsets_thresholds_metrics_and_admission_verified=True, flow_split_usage_verified=True)
        save_json(out/"verification.json", verification)
        if checked_flow_learning_reference(reference, expected) != (root, teacher):
            raise ValueError("관리 체결 학습 중 기존 입력 변경")
        manifest = read_json(out/"manifest.json")
        if (sha256(protocol) != manifest["settings"]["protocol_sha256"]
            or any(sha256(Path(__file__).parent/name) != value for name, value in manifest["source_sha256"].items())):
            raise ValueError("관리 체결 학습 중 계획·코드 변경")
        save_json(out/"summary.json", {"complete": True, "rows": {k: len(v) for k, v in native.items()},
            "independent_verification_complete": True, "flow_feature_splits": {k: sum(v.values()) for k, v in usage.items()},
            **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out/"failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="원본 관리 행동의 체결 방향 모형 비교·독립 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_flow_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
