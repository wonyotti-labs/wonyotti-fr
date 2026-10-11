from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .order_price_history import PRICE_FEATURES, attach_order_prices, price_support
from .order_price_model_reference import (
    verify_price_admission,
    verify_price_quiet_predictions,
    verify_price_refits,
)
from .order_price_models import fit_price_models, price_admission, price_quiet_predictions
from .order_price_reference import verify_order_prices, verify_price_support
from .order_price_research import checked_price_reference
from .position_boost_verification import compare_statistics
from .quiet_management import evaluate_quiet, quiet_predictions, validate_quiet_rows
from .quiet_management_reference import (
    verify_quiet_calibration,
    verify_quiet_evaluation,
    verify_quiet_predictions,
    verify_quiet_refits,
    verify_quiet_support,
)
from .quiet_state_calibration import fit_state_calibration, state_predictions
from .quiet_state_reference import verify_state_calibration, verify_state_predictions
from .retained_weight_close_reference import checked_files


def checked_price_learning_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("가격 관리 학습의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "input_reconstruction_verified", "original_columns_exact", "all_inputs_and_code_unchanged"])
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_models_fitted", "new_accounts_executed", "profitability_accepted"])):
        raise ValueError("가격 관리 학습의 입력 검산 완료 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_price_reference(Path(settings["reference"]), settings["reference_sha256"])
    if sources != settings["sources"] or settings.get("features") != PRICE_FEATURES:
        raise ValueError("가격 관리 학습의 원본·특징 연결 오류")
    previous = Path(sources["previous"])
    state_settings = read_json(previous / "manifest.json")["settings"]
    quiet, native = Path(state_settings["previous"]), Path(state_settings["native"])
    for folder in [quiet, native]:
        checked_files(folder)
    if state_settings["teacher"] != sources["teacher"]:
        raise ValueError("가격 관리 학습의 기존 모형 원본 불일치")
    return {**sources, "prices": str(root), "quiet": str(quiet), "native": str(native)}


def run_order_price_learning(reference, expected, output):
    sources = checked_price_learning_reference(reference, expected)
    prices, teacher, previous, quiet, native, audit = [Path(sources[k]) for k in ["prices", "teacher", "previous", "quiet", "native", "audit"]]
    protocol = Path("docs/EXPERIMENT_V133.md")
    paths = [reference, protocol, audit / "actions.parquet"]
    for folder in [prices, teacher, previous, quiet, native]:
        paths += [folder / name for name in checked_files(folder)]
        paths.append(folder / "files.json")
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "order-price-management-learning", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(quiet / "support.json")
        save_json(out / "support.json", support)
        actions = pd.read_parquet(audit / "actions.parquet")
        verification = {"prices": {}}
        price_supports = read_json(prices / "support.json")
        for phase, frame in rows.items():
            attached, links, ledger = attach_order_prices(frame, actions)
            verification["prices"][phase] = verify_order_prices(frame, actions, attached, links, ledger)
            for name, values in [("prices", attached[["end", "episode_id", *PRICE_FEATURES]]), ("links", links)]:
                pd.testing.assert_frame_equal(values, pd.read_parquet(prices / (phase + "_" + name + ".parquet")), check_exact=True)
            pd.testing.assert_frame_equal(ledger, pd.read_parquet(prices / "first_price_ledger.parquet"), check_exact=True)
            compare_statistics(price_support(attached), price_supports[phase])
            verify_price_support(attached, price_supports[phase])
            rows[phase] = attached
        print("원래 전체 행·기존 입력·네 가격 입력 재복원 검산 완료", flush=True)
        old_fitted = [read_json(quiet / (name + ".json")) for name in ["models", "offsets", "thresholds", "threshold_selection", "constants"]]
        verification["previous_refits"] = verify_quiet_refits(rows, old_fitted[0])
        previous_state = [read_json(previous / (name + ".json")) for name in ["offsets", "thresholds", "control_thresholds", "threshold_selection", "support"]]
        originals = {}
        for phase, frame in rows.items():
            base = pd.read_parquet(native / (phase + "_predictions.parquet"))
            old = quiet_predictions(frame, base, *old_fitted[:3], old_fitted[4])
            pd.testing.assert_frame_equal(old, pd.read_parquet(quiet / (phase + "_predictions.parquet")), check_exact=True)
            verify_quiet_predictions(frame, base, old, *old_fitted[:3], old_fitted[4])
            original, control = state_predictions(old, *previous_state[:3])
            verify_state_predictions(old, original, control, *previous_state[:3])
            pd.testing.assert_frame_equal(original, pd.read_parquet(previous / (phase + "_predictions.parquet")), check_exact=True)
            if phase == "calibration":
                verify_quiet_calibration(old, *old_fitted[1:])
                verify_state_calibration(old, *previous_state[:4], previous_state[4][phase])
            originals[phase] = original
        print("기존 50개 입력의 512개 트리·선형·v126/v127 전체 예측 재현 완료", flush=True)
        fitted = fit_price_models(rows["training"], rows["calibration"])
        for name, values in zip(["models", "global_offsets", "global_thresholds", "global_threshold_selection", "constants"], fitted, strict=True):
            save_json(out / (name + ".json"), values)
        print("54개 입력의 새 512개 트리·두 선형 학습 완료, 독립 재적합 검산 시작", flush=True)
        verification["new_refits"] = verify_price_refits(rows, fitted[0])
        calibrated = price_quiet_predictions(rows["calibration"], originals["calibration"], *fitted[:3], fitted[4])
        verify_price_quiet_predictions(rows["calibration"], originals["calibration"], calibrated, *fitted[:3], fitted[4])
        verify_quiet_calibration(calibrated, *fitted[1:])
        state = fit_state_calibration(calibrated)
        verify_state_calibration(calibrated, *state)
        for name, values in zip(["offsets", "thresholds", "control_thresholds", "threshold_selection", "calibration_support"], state, strict=True):
            save_json(out / (name + ".json"), values)
        metrics, monthly, controls, global_metrics = {}, {}, {}, {}
        old_metrics, old_months = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                old = originals[phase]
                global_prediction = price_quiet_predictions(frame, old, *fitted[:3], fitted[4])
                preserved = verify_price_quiet_predictions(frame, old, global_prediction, *fitted[:3], fitted[4])
                candidate, control = state_predictions(global_prediction, *state[:3])
                verify_state_predictions(global_prediction, candidate, control, *state[:3])
                unchanged = {k: unchanged[k] and preserved[k] for k in unchanged}
                for saved, destination in [(candidate, metrics), (control, controls), (global_prediction, global_metrics)]:
                    destination[phase] = evaluate_quiet(saved)
                    verify_quiet_evaluation(saved, destination[phase])
                compare_statistics(evaluate_quiet(old), old_metrics[phase])
                monthly[phase] = {}
                for month, positions in candidate.groupby(candidate.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = candidate.iloc[positions]
                    monthly[phase][month] = evaluate_quiet(part)
                    verify_quiet_evaluation(part, monthly[phase][month])
                    compare_statistics(evaluate_quiet(old.iloc[positions]), old_months[phase][month])
                save_frame(out / (phase + "_predictions.parquet"), candidate)
                fields = ["end", "episode_id", *[a + "_" + m + "_request" for a in ["reduce", "increase"] for m in ["tree", "linear"]]]
                save_frame(out / (phase + "_threshold_only_requests.parquet"), control[fields])
                print(phase + ": 전체 확률·원시 및 상태 문턱·기존 판단·모든 월 검산 완료", flush=True)
        decision = price_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged)
        verify_price_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("가격 관리 학습 중 입력·코드·계획 변경")
        verification.update(unchanged=unchanged, original_rows_and_predictions_exact=True,
            all_offsets_thresholds_constants_and_months_verified=True, threshold_only_requests_exact=True)
        for name, value in [("metrics", metrics), ("monthly", monthly), ("control_metrics", controls), ("global_metrics", global_metrics),
                            ("decision", decision), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "checks": decision["checks"],
            "order_price_admitted": decision["order_price_admitted"], "new_trees": 512, "new_linear_models": 2,
            "all_inputs_and_code_unchanged": True, "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/37 조건 통과", flush=True)
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
    run_order_price_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
