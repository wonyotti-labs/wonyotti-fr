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
from .management_market import REPLACED, market_support
from .management_market_model_reference import (
    refit_market_models,
    verify_market_admission,
    verify_market_evaluation,
    verify_market_predictions,
    verify_offsets,
)
from .management_market_models import fit_market_models, market_model_admission, market_predictions
from .management_market_reference import verify_market_support
from .management_market_research import checked_market_reference
from .minute_management import ACTIONS
from .order_history_boost import OrderHistoryBoostModels
from .paired_management_evaluation import choose_thresholds, evaluate
from .retained_weight_close_reference import checked_files


def checked_learning_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 시장 학습의 입력 검산 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True or summary.get("paired_model_research_eligible") is not True
        or summary.get("new_models_fitted") is not False or summary.get("original_rows_preserved") is not True
        or summary.get("episode_intersection") != 0):
        raise ValueError("관리 시장 학습의 선행 입력 검산·지원 미충족")
    settings = read_json(root / "manifest.json")["settings"]
    verification = read_json(root / "verification.json")
    support = read_json(root / "support.json")
    required = ["all_connections_and_directional_values_verified", "direct_minute_formulas_verified",
                "original_state_and_labels_unchanged", "original_market_reconstructed_exactly",
                "independent_support_counts_verified", "parquet_readback_exact"]
    if (set(verification) != set(PERIODS) or set(support) != set(PERIODS)
        or any(not all(verification[p].get(k) is True for k in required)
               or verification[p].get("rows") != support[p]["groups"]["all"]["rows"]
               or support[p].get("action_support_sufficient") is not True for p in PERIODS)):
        raise ValueError("관리 시장 학습의 전체 연결·입력 독립 검산 누락")
    if settings["protocol_sha256"] != sha256(Path("docs/EXPERIMENT_V116.md")):
        raise ValueError("관리 시장 학습의 사전 계획 변경")
    _, previous = checked_market_reference(Path(settings["reference"]), settings["reference_sha256"])
    if settings["teacher"] != previous["teacher"] or settings["changed_features"] != REPLACED:
        raise ValueError("관리 시장 학습의 원본·특징 연결 변경")
    checked_files(Path(previous["teacher"]))
    return root, Path(previous["teacher"])


def learning_rows(root, teacher):
    native, aligned, membership = {}, {}, {}
    saved_support = read_json(root / "support.json")
    for phase, (first, last) in PERIODS.items():
        full = pd.read_parquet(teacher / f"{phase}_used.parquet")
        market = pd.read_parquet(root / f"{phase}_market.parquet")
        link = pd.read_parquet(root / f"{phase}_linkage.parquet")
        pd.testing.assert_series_equal(full.end.reset_index(drop=True), market.end, check_exact=True)
        if (not full.end.ge(pd.Timestamp(first, tz="UTC") + pd.Timedelta(days=1)).all()
            or not full.entry_time.ge(pd.Timestamp(first, tz="UTC")).all()
            or not full.label_end.lt(pd.Timestamp(last, tz="UTC") - pd.Timedelta(days=1)).all()
            or "source_row" in full):
            raise ValueError("관리 시장 학습의 원래 행·시간 경계 변경")
        target = full.copy()
        target[REPLACED] = market[REPLACED].to_numpy()
        current = market_support(full, target, link)
        if current != saved_support[phase]:
            raise ValueError("관리 시장 학습의 봉인 지원 집계 변경")
        verify_market_support(full, target, link, current)
        selected = link.supported.to_numpy()
        membership[phase] = link.copy()
        membership[phase]["source_row"] = np.arange(len(full))
        for rows, output in [(full, native), (target, aligned)]:
            part = rows.loc[selected].copy()
            part["source_row"] = np.flatnonzero(selected)
            output[phase] = part.reset_index(drop=True)
    return native, aligned, membership


def run_market_learning(reference, expected, output):
    root, teacher = checked_learning_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V116.md")
    out = new_run(output, "management-market-learning", {"reference": str(reference),
        "reference_sha256": expected, "input_run": str(root), "teacher": str(teacher),
        "protocol_sha256": sha256(protocol), "periods": PERIODS,
        "new_model_families": ["native_histogram", "aligned_histogram", "aligned_logistic"],
        "profitability_accepted": False, "trading_returns_evaluated": False})
    print(out, flush=True)
    try:
        native, aligned, membership = learning_rows(root, teacher)
        stored_model = OrderHistoryBoostModels.from_dict(read_json(teacher / "models.json")["histogram"])
        stored_offset = ManagementOffset.from_dict(read_json(teacher / "offsets.json")["histogram"])
        print("동일 학습 행의 세 모형 적합·상반기 보정 시작", flush=True)
        models, offsets, support = fit_market_models(native, aligned)
        identical_train = bool(membership["training"].supported.all())
        identical_calibration = bool(membership["calibration"].supported.all())
        if identical_train and models["native"].to_dict() != stored_model.to_dict():
            raise ValueError("동일 학습 원본 대조와 기존 v35 트리 불일치")
        if identical_train and identical_calibration and offsets["native"].to_dict() != stored_offset.to_dict():
            raise ValueError("동일 보정 원본 대조와 기존 v35 절편 불일치")
        save_json(out / "models.json", {k: v.to_dict() for k, v in models.items()})
        save_json(out / "offsets.json", {k: v.to_dict() for k, v in offsets.items()})
        save_json(out / "training_support.json", support)
        print("독립 라이브러리 재학습·트리·선형 계수 대조 시작", flush=True)
        verification = refit_market_models(native, aligned, models)
        cy = native["calibration"][["y_" + a for a in ACTIONS]].to_numpy(int)
        constant = cy.mean(axis=0)
        with threadpool_limits(limits=1):
            calibration = market_predictions(native["calibration"], aligned["calibration"], models,
                                              offsets, stored_model, stored_offset, constant)
            verify_offsets(calibration, cy, offsets)
            thresholds, selections = choose_thresholds(cy, calibration)
            save_json(out / "thresholds.json", thresholds)
            save_json(out / "threshold_selection.json", selections)
            metrics, monthly = {}, {}
            for phase in PERIODS:
                a, b = native[phase], aligned[phase]
                scores = calibration if phase == "calibration" else market_predictions(a, b, models,
                    offsets, stored_model, stored_offset, constant)
                verify_market_predictions(a, b, models, offsets, stored_model, stored_offset, constant, scores)
                y = a[["y_" + action for action in ACTIONS]].to_numpy(int)
                metrics[phase], _ = evaluate(y, scores, thresholds)
                verify_market_evaluation(y, scores, thresholds, metrics[phase], selections if phase == "calibration" else None)
                monthly[phase] = {}
                months = a.end.dt.strftime("%Y-%m").to_numpy()
                for month in sorted(set(months)):
                    mask = months == month
                    sub = {name: values[mask] for name, values in scores.items()}
                    monthly[phase][month], _ = evaluate(y[mask], sub, thresholds)
                    verify_market_evaluation(y[mask], sub, thresholds, monthly[phase][month])
                saved = a[["end", "episode_id", "source_row", *["y_" + action for action in ACTIONS]]].copy()
                for name, values in scores.items():
                    for index, action in enumerate(ACTIONS):
                        saved[f"{action}_{name}"] = values[:, index]
                if phase == "diagnosis":
                    old = pd.read_parquet(teacher / "predictions.parquet").iloc[a.source_row].reset_index(drop=True)
                    pd.testing.assert_frame_equal(old[["end", "episode_id"]], a[["end", "episode_id"]], check_exact=True)
                    for suffix in ["raw", "calibrated"]:
                        np.testing.assert_array_equal(scores["stored_" + suffix],
                            old[[f"{action}_histogram_{suffix}" for action in ACTIONS]].to_numpy())
                saved.to_parquet(out / f"{phase}_predictions.parquet", index=False)
                membership[phase].to_parquet(out / f"{phase}_membership.parquet", index=False)
                pd.testing.assert_frame_equal(pd.read_parquet(out / f"{phase}_predictions.parquet"), saved, check_exact=True)
                pd.testing.assert_frame_equal(pd.read_parquet(out / f"{phase}_membership.parquet"), membership[phase], check_exact=True)
                print(phase + ": 전체 확률·문턱·월별 집계 검산 완료", flush=True)
        decision = market_model_admission(metrics["diagnosis"])
        verify_market_admission(metrics["diagnosis"], decision)
        for name, data in [("metrics", metrics), ("monthly", monthly), ("decision", decision)]:
            save_json(out / (name + ".json"), data)
        verification.update(all_probability_rows_verified=sum(len(a) for a in native.values()),
            stored_diagnosis_predictions_exact=True, offsets_thresholds_metrics_and_admission_verified=True,
            native_training_identical=identical_train, native_calibration_identical=identical_calibration,
            native_trees_equal_stored=models["native"].to_dict() == stored_model.to_dict(),
            native_offset_equal_stored=offsets["native"].to_dict() == stored_offset.to_dict())
        save_json(out / "verification.json", verification)
        if checked_learning_reference(reference, expected) != (root, teacher):
            raise ValueError("관리 시장 학습 중 기존 입력 변경")
        manifest = read_json(out / "manifest.json")
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != digest for name, digest in manifest["source_sha256"].items()):
            raise ValueError("관리 시장 학습 중 계획·코드 변경")
        save_json(out / "summary.json", {"complete": True, "rows": {k: len(v) for k, v in native.items()},
            "independent_verification_complete": True, **decision})
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="동일 원본 행의 거래소 관리 모형 적합·독립 검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_market_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
