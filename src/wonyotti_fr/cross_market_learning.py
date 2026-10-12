from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .add_count_audit import attach_add_counts, audit_count_strata
from .add_count_reference import verify_add_count_links, verify_count_strata
from .common import new_run, save_json, sha256
from .cross_market import (
    CROSS_FEATURES,
    WINDOWS,
    attach_cross_market,
    cross_market_support,
    cross_market_table,
)
from .cross_market_model_reference import (
    verify_cross_market_admission,
    verify_cross_market_quiet_predictions,
    verify_cross_market_refits,
    verify_pre_limit_restart,
)
from .cross_market_models import (
    BASE_FEATURES,
    COMPARISONS,
    FEATURES,
    cross_market_admission,
    cross_market_quiet_predictions,
    fit_cross_market_models,
    pre_limit_restart_metrics,
    validate_cross_market_inputs,
)
from .cross_market_reference import verify_cross_market, verify_cross_support
from .cross_market_research import MARKETS, checked_cross_market_source, checked_cross_reference
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
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


def checked_cross_input_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("상대 가격 학습의 입력 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    required = ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged",
        "input_reconstruction_verified", "original_columns_exact"]
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in required)
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_models_fitted", "new_accounts_executed", "profitability_accepted"])):
        raise ValueError("상대 가격 학습의 입력 완료·검산 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_cross_reference(Path(settings["reference"]), settings["reference_sha256"])
    if (sources != settings["sources"] or settings.get("features") != CROSS_FEATURES
        or settings.get("windows") != WINDOWS or settings.get("markets") != {k: list(v) for k, v in MARKETS.items()}
        or settings.get("protocol_sha256") != sha256(Path("docs/EXPERIMENT_V136.md"))):
        raise ValueError("상대 가격 학습의 원본·시세·창·계획 연결 오류")
    return {**sources, "cross_inputs": str(root)}


def run_cross_market_learning(reference, expected, output):
    sources = checked_cross_input_reference(reference, expected)
    teacher, previous, quiet, native, audit, counts, prices = [Path(sources[k]) for k in
        ["teacher", "previous", "quiet", "native", "audit", "counts", "candidate"]]
    cross_inputs = Path(sources["cross_inputs"])
    markets, provenance = {}, {}
    for symbol, (folder, digest) in MARKETS.items():
        markets[symbol], provenance[symbol] = checked_cross_market_source(Path(folder), symbol, digest)
    protocol = Path("docs/EXPERIMENT_V136.md")
    paths = [reference, protocol, audit / "actions.parquet", audit / "xbtusd_events.parquet"]
    for folder in [teacher, previous, quiet, native, counts, prices, cross_inputs]:
        paths += [folder / name for name in checked_files(folder)]
        paths.append(folder / "files.json")
    paths += [Path(p) for value in provenance.values() for p in value["files"]]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "cross-market-learning", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "features": FEATURES,
        "additional_features": CROSS_FEATURES, "base_features": BASE_FEATURES, "bot_max_adds_unchanged": 5,
        "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(quiet / "support.json")
        save_json(out / "support.json", support)
        old_fit = [read_json(quiet / (k + ".json")) for k in ["models", "offsets", "thresholds", "threshold_selection", "constants"]]
        verification = {"previous_refits": verify_quiet_refits(rows, old_fit[0]), "count_reconstruction": {}}
        old_state = [read_json(previous / (k + ".json")) for k in ["offsets", "thresholds", "control_thresholds", "threshold_selection", "support"]]
        actions, events = pd.read_parquet(audit / "actions.parquet"), pd.read_parquet(audit / "xbtusd_events.parquet")
        saved_counts, saved_monthly = read_json(counts / "metrics.json"), read_json(counts / "monthly.json")
        last = max(f.end.max() for f in rows.values())
        markets = {s: f.loc[f.end.le(last)].reset_index(drop=True) for s, f in markets.items()}
        btc, eth = markets["BTCUSDT"], markets["ETHUSDT"]
        table = cross_market_table(btc, eth)
        input_support = read_json(cross_inputs / "support.json")
        input_monthly = read_json(cross_inputs / "monthly_support.json")
        verification["cross_market_reconstruction"] = {}
        originals = {}
        for phase, frame in rows.items():
            links = attach_add_counts(frame, actions, events)
            verification["count_reconstruction"][phase] = verify_add_count_links(frame, actions, events, links)
            pd.testing.assert_frame_equal(links, pd.read_parquet(counts / (phase + "_count_links.parquet")), check_exact=True)
            expanded, cross_links = attach_cross_market(frame, table)
            verification["cross_market_reconstruction"][phase] = verify_cross_market(frame, btc, eth, expanded, cross_links)
            pd.testing.assert_frame_equal(cross_links, pd.read_parquet(cross_inputs / (phase + "_links.parquet")), check_exact=True)
            pd.testing.assert_frame_equal(expanded[["end", "episode_id", *CROSS_FEATURES]],
                pd.read_parquet(cross_inputs / (phase + "_features.parquet")), check_exact=True)
            pd.testing.assert_frame_equal(expanded[BASE_FEATURES], frame[BASE_FEATURES], check_exact=True)
            reconstructed = cross_market_support(expanded)
            compare_statistics(reconstructed, input_support[phase])
            verify_cross_support(expanded, reconstructed)
            for month, part in expanded.groupby(expanded.end.dt.strftime("%Y-%m"), sort=True):
                reconstructed = cross_market_support(part)
                compare_statistics(reconstructed, input_monthly[phase][month])
                verify_cross_support(part, reconstructed)
            validate_cross_market_inputs(expanded)
            rows[phase] = frame = expanded
            base = pd.read_parquet(native / (phase + "_predictions.parquet"))
            old = quiet_predictions(frame, base, *old_fit[:3], old_fit[4])
            pd.testing.assert_frame_equal(old, pd.read_parquet(quiet / (phase + "_predictions.parquet")), check_exact=True)
            verify_quiet_predictions(frame, base, old, *old_fit[:3], old_fit[4])
            original, control = state_predictions(old, *old_state[:3])
            verify_state_predictions(old, original, control, *old_state[:3])
            pd.testing.assert_frame_equal(original, pd.read_parquet(previous / (phase + "_predictions.parquet")), check_exact=True)
            if phase == "calibration":
                verify_quiet_calibration(old, *old_fit[1:])
                verify_state_calibration(old, *old_state[:4], old_state[4][phase])
            fixed = {"v127": original, "v133": pd.read_parquet(prices / (phase + "_predictions.parquet"))}
            reproduced = audit_count_strata(frame, fixed)
            compare_statistics(reproduced, saved_counts[phase])
            verify_count_strata(frame, fixed, reproduced)
            for month, positions in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                part, values = frame.iloc[positions], {k: v.iloc[positions] for k, v in fixed.items()}
                reproduced = audit_count_strata(part, values)
                compare_statistics(reproduced, saved_monthly[phase][month])
                verify_count_strata(part, values, reproduced)
            originals[phase] = original
            print(phase + ": 기존 50개·상대 입력 10개·v134 전체 집계·기존 v127 예측 재현 완료", flush=True)
        fitted = fit_cross_market_models(rows["training"], rows["calibration"])
        for name, value in zip(["models", "global_offsets", "global_thresholds", "global_threshold_selection", "constants"], fitted, strict=True):
            save_json(out / (name + ".json"), value)
        print("상대 가격 60개 입력의 새 512개 트리·두 선형 학습 완료, 독립 재적합 시작", flush=True)
        verification["new_refits"] = verify_cross_market_refits(rows, fitted[0])
        calibration = cross_market_quiet_predictions(rows["calibration"], originals["calibration"], *fitted[:3], fitted[4])
        verify_cross_market_quiet_predictions(rows["calibration"], originals["calibration"], calibration, *fitted[:3], fitted[4])
        verify_quiet_calibration(calibration, *fitted[1:])
        state = fit_state_calibration(calibration)
        verify_state_calibration(calibration, *state)
        for name, value in zip(["offsets", "thresholds", "control_thresholds", "threshold_selection", "calibration_support"], state, strict=True):
            save_json(out / (name + ".json"), value)
        metrics, monthly, controls, global_metrics = {}, {}, {}, {}
        strata, monthly_strata, pre_limit, monthly_pre_limit = {}, {}, {}, {}
        old_metrics, old_months = read_json(previous / "metrics.json"), read_json(previous / "monthly.json")
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        with threadpool_limits(limits=1):
            for phase, frame in rows.items():
                old = originals[phase]
                shared = cross_market_quiet_predictions(frame, old, *fitted[:3], fitted[4])
                preserved = verify_cross_market_quiet_predictions(frame, old, shared, *fitted[:3], fitted[4])
                candidate, control = state_predictions(shared, *state[:3])
                verify_state_predictions(shared, candidate, control, *state[:3])
                unchanged = {k: unchanged[k] and preserved[k] for k in unchanged}
                for saved, destination in [(candidate, metrics), (control, controls), (shared, global_metrics)]:
                    destination[phase] = evaluate_quiet(saved)
                    verify_quiet_evaluation(saved, destination[phase])
                compare_statistics(evaluate_quiet(old), old_metrics[phase])
                predictions = {"v127": old, "v136": candidate}
                strata[phase] = audit_count_strata(frame, predictions, COMPARISONS)
                verify_count_strata(frame, predictions, strata[phase], COMPARISONS)
                pre_limit[phase] = pre_limit_restart_metrics(frame, predictions)
                verify_pre_limit_restart(frame, predictions, pre_limit[phase])
                monthly[phase], monthly_strata[phase], monthly_pre_limit[phase] = {}, {}, {}
                for month, positions in frame.groupby(frame.end.dt.strftime("%Y-%m"), sort=True).indices.items():
                    part = candidate.iloc[positions]
                    monthly[phase][month] = evaluate_quiet(part)
                    verify_quiet_evaluation(part, monthly[phase][month])
                    compare_statistics(evaluate_quiet(old.iloc[positions]), old_months[phase][month])
                    source, values = frame.iloc[positions], {k: v.iloc[positions] for k, v in predictions.items()}
                    monthly_strata[phase][month] = audit_count_strata(source, values, COMPARISONS)
                    verify_count_strata(source, values, monthly_strata[phase][month], COMPARISONS)
                    monthly_pre_limit[phase][month] = pre_limit_restart_metrics(source, values)
                    verify_pre_limit_restart(source, values, monthly_pre_limit[phase][month])
                save_frame(out / (phase + "_predictions.parquet"), candidate)
                fields = ["end", "episode_id", *[a + "_" + m + "_request" for a in ["reduce", "increase"] for m in ["tree", "linear"]]]
                save_frame(out / (phase + "_threshold_only_requests.parquet"), control[fields])
                print(phase + ": 전체 확률·월·횟수 구간·제한 전 재시작 직접 지표 검산 완료", flush=True)
        decision = cross_market_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, pre_limit["diagnosis"])
        verify_cross_market_admission(metrics["diagnosis"], old_metrics["diagnosis"], unchanged, pre_limit["diagnosis"], decision)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("상대 가격 학습 중 입력·코드·계획 변경")
        verification.update(unchanged=unchanged, original_rows_and_predictions_exact=True,
            only_ten_cross_market_inputs_added=True, all_offsets_thresholds_constants_months_buckets_verified=True,
            pre_limit_restart_metrics_direct=True, threshold_only_requests_exact=True)
        outputs = {"metrics": metrics, "monthly": monthly, "control_metrics": controls, "global_metrics": global_metrics,
            "strata": strata, "monthly_strata": monthly_strata, "pre_limit": pre_limit, "monthly_pre_limit": monthly_pre_limit,
            "decision": decision, "verification": verification}
        for name, value in outputs.items():
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "checks": decision["checks"],
            "cross_market_admitted": decision["cross_market_admitted"], "new_trees": 512, "new_linear_models": 2,
            "all_inputs_and_code_unchanged": True, "new_accounts_executed": False, "profitability_accepted": False,
            "all_source_periods_already_observed": True})
        seal(out)
        print("독립 검산 완료: " + str(sum(decision["checks"].values())) + "/40 조건 통과", flush=True)
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
    run_cross_market_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
