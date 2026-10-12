from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .add_count_audit import attach_add_counts, audit_count_strata
from .add_count_reference import verify_add_count_links, verify_count_strata
from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .count_region_reference import verify_pre_limit_restart
from .first_state_models import metrics
from .fixed_bin_management import MODELS, fit_fixed_bins, fixed_bin_admission
from .fixed_bin_reference import verify_fixed_admission, verify_fixed_refits
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .paired_error_budget import (
    evaluate_paired_budget,
    fit_paired_budget,
    verify_paired_budget,
    verify_paired_cutoffs,
)
from .position_boost_verification import compare_statistics
from .quiet_management import (
    FEATURES,
    evaluate_quiet,
    groups,
    quiet_predictions,
    validate_quiet_rows,
)
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
from .tail_weighted_learning import checked_tail_reference
from .tail_weighted_management import (
    WEIGHT_RULE,
    calibrate_tail_models,
    training_weight_groups,
)
from .tail_weighted_reference import verify_weight_groups


def checked_fixed_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("고정 구간 학습의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (
        proof.get("complete") is not True
        or type(proof.get("process_exit_code")) is not int
        or proof["process_exit_code"] != 0
        or sha256(root / "files.json") != proof.get("files_sha256")
        or sha256(root / "summary.json") != proof.get("summary_sha256")
        or any(
            summary.get(k) is not True
            for k in [
                "complete",
                "independent_verification_complete",
                "all_inputs_and_code_unchanged",
                "new_models_fitted",
            ]
        )
        or proof.get("new_models_fitted") is not True
        or any(
            summary.get(k) is not False or proof.get(k) is not False
            for k in ["new_accounts_executed", "profitability_accepted"]
        )
        or summary.get("tail_weighted_admitted") is not False
        or summary.get("new_trees") != 512
        or summary.get("new_linear_models") != 2
    ):
        raise ValueError("고정 구간 학습의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_tail_reference(Path(settings["reference"]), settings["reference_sha256"])
    if (
        settings.get("sources") != sources
        or settings.get("features") != FEATURES
        or settings.get("weight_rule") != WEIGHT_RULE
        or settings.get("models") != ["v127", "v140"]
    ):
        raise ValueError("고정 구간 학습의 원본 출처·가중치 연결 오류")
    return {**sources, "tail": str(root)}


def dependency_paths():
    import sklearn.ensemble._hist_gradient_boosting.binning as binning
    import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as boosting

    return [Path(binning.__file__), Path(boosting.__file__), Path("uv.lock")]


def pre_limit_metrics(frame, predictions):
    mask = groups(frame, "increase")["restart"] & frame.adds.lt(5).to_numpy()
    return {
        name: metrics(
            frame.loc[mask, "y_increase"].to_numpy(),
            values.loc[mask, "increase_tree_score"].to_numpy(),
            values.loc[mask, "increase_tree_request"].to_numpy(),
            0.5,
        )
        for name, values in predictions.items()
    }


def run_fixed_learning(reference, expected, output):
    sources = checked_fixed_reference(reference, expected)
    teacher, previous, quiet, native, counts, audit, budget_parent = (
        Path(sources[k])
        for k in ["teacher", "previous", "quiet", "native", "counts", "audit", "budget"]
    )
    protocol = Path("docs/EXPERIMENT_V141.md")
    paths = [reference, protocol, audit / "actions.parquet", audit / "xbtusd_events.parquet"]
    for folder in [
        teacher,
        previous,
        quiet,
        native,
        counts,
        budget_parent,
        Path(sources["temporal"]),
        Path(sources["tail"]),
    ]:
        paths += [folder / name for name in checked_files(folder)] + [folder / "files.json"]
    paths += sorted(Path(__file__).parent.glob("*.py")) + dependency_paths()
    before = {str(p): sha256(p) for p in paths}
    out = new_run(
        output,
        "fixed-bin-management",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "sources": sources,
            "protocol_sha256": sha256(protocol),
            "features": FEATURES,
            "models": MODELS,
            "weight_rule": WEIGHT_RULE,
            "total_new_trees": 512,
            "new_linear_models": 0,
            "preserved_linear_models": 2,
            "equivalence_control_trees": 512,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(teacher)
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(quiet / "support.json")
        save_json(out / "support.json", support)
        old_fit = [
            read_json(quiet / (k + ".json"))
            for k in ["models", "offsets", "thresholds", "threshold_selection", "constants"]
        ]
        tail = Path(sources["tail"])
        weighted_fit = [
            read_json(tail / (k + ".json"))
            for k in [
                "models",
                "global_offsets",
                "global_thresholds",
                "global_threshold_selection",
                "constants",
            ]
        ]
        weighted_state = [
            read_json(tail / (k + ".json"))
            for k in [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "calibration_support",
            ]
        ]
        weighted_metrics, weighted_monthly = (
            read_json(tail / "metrics.json"),
            read_json(tail / "monthly.json"),
        )
        weighted_predictions = {}
        old_state = [
            read_json(previous / (k + ".json"))
            for k in [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "support",
            ]
        ]
        print("기존 무가중 512개 트리·두 선형 독립 재적합 시작", flush=True)
        verification = {"previous_refits": verify_quiet_refits(rows, old_fit[0]), "counts": {}}
        actions, events = (
            pd.read_parquet(audit / "actions.parquet"),
            pd.read_parquet(audit / "xbtusd_events.parquet"),
        )
        originals, bases = {}, {}
        old_metrics, old_monthly = (
            read_json(previous / "metrics.json"),
            read_json(previous / "monthly.json"),
        )
        for phase, frame in rows.items():
            links = attach_add_counts(frame, actions, events)
            verification["counts"][phase] = verify_add_count_links(frame, actions, events, links)
            pd.testing.assert_frame_equal(
                links, pd.read_parquet(counts / (phase + "_count_links.parquet")), check_exact=True
            )
            base = pd.read_parquet(native / (phase + "_predictions.parquet"))
            global_old = quiet_predictions(frame, base, *old_fit[:3], old_fit[4])
            verify_quiet_predictions(frame, base, global_old, *old_fit[:3], old_fit[4])
            pd.testing.assert_frame_equal(
                global_old,
                pd.read_parquet(quiet / (phase + "_predictions.parquet")),
                check_exact=True,
            )
            old, control = state_predictions(global_old, *old_state[:3])
            verify_state_predictions(global_old, old, control, *old_state[:3])
            pd.testing.assert_frame_equal(
                old, pd.read_parquet(previous / (phase + "_predictions.parquet")), check_exact=True
            )
            computed = evaluate_quiet(old)
            verify_quiet_evaluation(old, computed)
            compare_statistics(computed, old_metrics[phase])
            for month, indices in old.groupby(
                old.end.dt.strftime("%Y-%m"), sort=True
            ).indices.items():
                compare_statistics(evaluate_quiet(old.iloc[indices]), old_monthly[phase][month])
            if phase == "calibration":
                verify_quiet_calibration(global_old, *old_fit[1:])
                verify_state_calibration(global_old, *old_state[:4], old_state[4][phase])
            shared_weighted = quiet_predictions(frame, base, *weighted_fit[:3], weighted_fit[4])
            verify_quiet_predictions(
                frame, base, shared_weighted, *weighted_fit[:3], weighted_fit[4]
            )
            wp, wc = state_predictions(shared_weighted, *weighted_state[:3])
            verify_state_predictions(shared_weighted, wp, wc, *weighted_state[:3])
            pd.testing.assert_frame_equal(
                wp, pd.read_parquet(tail / (phase + "_predictions.parquet")), check_exact=True
            )
            compare_statistics(evaluate_quiet(wp), weighted_metrics[phase])
            for month, indices in wp.groupby(
                wp.end.dt.strftime("%Y-%m"), sort=True
            ).indices.items():
                compare_statistics(evaluate_quiet(wp.iloc[indices]), weighted_monthly[phase][month])
            if phase == "calibration":
                verify_quiet_calibration(shared_weighted, *weighted_fit[1:])
                verify_state_calibration(shared_weighted, *weighted_state)
            weighted_predictions[phase] = wp
            originals[phase], bases[phase] = old, base
            print(phase + ": 원본·과거 횟수·기존 전체 점수·요청 재현 완료", flush=True)
        prior_cuts = read_json(tail / "budget_cutoffs.json")
        prior_budgets = read_json(tail / "budget_metrics.json")
        old_pair = {p: {"v127": originals[p], "v140": weighted_predictions[p]} for p in rows}
        verify_paired_cutoffs(rows["calibration"], old_pair["calibration"], "v127", prior_cuts)
        for phase, frame in rows.items():
            previous_budget, previous_ledger = evaluate_paired_budget(
                frame, old_pair[phase], prior_cuts
            )
            verify_paired_budget(
                frame, old_pair[phase], prior_cuts, previous_budget, previous_ledger
            )
            compare_statistics(previous_budget, prior_budgets[phase])
            pd.testing.assert_frame_equal(
                previous_ledger,
                pd.read_parquet(tail / (phase + "_budget_requests.parquet")),
                check_exact=True,
            )
        weight_ledger = pd.read_parquet(tail / "training_weights.parquet")
        details = read_json(tail / "weight_details.json")
        weight_groups = training_weight_groups(rows["training"], weight_ledger)
        verify_weight_groups(rows["training"], weight_ledger, weight_groups)
        compare_statistics(weight_groups, read_json(tail / "weight_groups.json"))
        fixed = fit_fixed_bins(
            rows["training"], old_fit[0], weighted_fit[0], weight_ledger, details
        )
        models, bins, equivalence, encoded, levels = fixed
        for name, value in [
            ("models", models),
            ("bins", bins),
            ("equivalence_models", equivalence),
            ("encoded_models", encoded),
            ("level_boundaries", levels),
            ("weight_details", details),
            ("weight_groups", weight_groups),
        ]:
            save_json(out / (name + ".json"), value)
        save_frame(out / "training_weights.parquet", weight_ledger)
        print("무가중 동등성·고정 구간 가중 학습 완료, 독립 재적합 검산 시작", flush=True)
        verification["fixed_refits"] = verify_fixed_refits(
            rows, old_fit[0], weighted_fit[0], weight_ledger, details, fixed
        )
        fitted = [models, *calibrate_tail_models(rows["calibration"], models)]
        for name, value in zip(
            ["global_offsets", "global_thresholds", "global_threshold_selection", "constants"],
            fitted[1:],
            strict=True,
        ):
            save_json(out / (name + ".json"), value)
        calibration = quiet_predictions(
            rows["calibration"], bases["calibration"], *fitted[:3], fitted[4]
        )
        verify_quiet_predictions(
            rows["calibration"], bases["calibration"], calibration, *fitted[:3], fitted[4]
        )
        verify_quiet_calibration(calibration, *fitted[1:])
        state = fit_state_calibration(calibration)
        verify_state_calibration(calibration, *state)
        for name, value in zip(
            [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "calibration_support",
            ],
            state,
            strict=True,
        ):
            save_json(out / (name + ".json"), value)
        calibrated, cal_control = state_predictions(calibration, *state[:3])
        verify_state_predictions(calibration, calibrated, cal_control, *state[:3])
        cal_pair = {"v127": originals["calibration"], "v141": calibrated}
        cutoffs = fit_paired_budget(rows["calibration"], cal_pair, "v127")
        verify_paired_cutoffs(rows["calibration"], cal_pair, "v127", cutoffs)
        old_cuts = read_json(budget_parent / "cutoffs.json")
        for key, value in cutoffs.items():
            assert (
                value["support"] == old_cuts[key]["support"]
                and value["budget"] == old_cuts[key]["budget"]
            )
            assert value["models"]["v127"] == old_cuts[key]["models"]["v127"]
        save_json(out / "budget_cutoffs.json", cutoffs)
        outputs = {
            k: {}
            for k in [
                "metrics",
                "monthly",
                "global_metrics",
                "control_metrics",
                "strata",
                "monthly_strata",
                "pre_limit",
                "monthly_pre_limit",
                "budget_metrics",
                "monthly_budget",
            ]
        }
        old_budgets = read_json(budget_parent / "metrics.json")
        unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
        for phase, frame in rows.items():
            shared = quiet_predictions(frame, bases[phase], *fitted[:3], fitted[4])
            verify_quiet_predictions(frame, bases[phase], shared, *fitted[:3], fitted[4])
            candidate, control = state_predictions(shared, *state[:3])
            verify_state_predictions(shared, candidate, control, *state[:3])
            np.testing.assert_array_equal(candidate.exit_score, originals[phase].exit_score)
            np.testing.assert_array_equal(candidate.exit_request, originals[phase].exit_request)
            for action in ["reduce", "increase"]:
                recent = groups(frame, action)["recent"]
                for suffix in ["score", "request"]:
                    np.testing.assert_array_equal(
                        candidate.loc[recent, action + "_tree_" + suffix],
                        originals[phase].loc[recent, action + "_tree_" + suffix],
                    )
            for label, value in [
                ("metrics", candidate),
                ("global_metrics", shared),
                ("control_metrics", control),
            ]:
                evaluated = evaluate_quiet(value)
                verify_quiet_evaluation(value, evaluated)
                outputs[label][phase] = evaluated
            for action in ["reduce", "increase"]:
                for name in ["linear_raw", "linear", "constant", "baseline", "capacity"]:
                    for suffix in ["score", "request"]:
                        column = action + "_" + name + "_" + suffix
                        np.testing.assert_array_equal(
                            candidate[column], weighted_predictions[phase][column]
                        )
            pair = {"v127": originals[phase], "v141": candidate}
            budget, ledger = evaluate_paired_budget(frame, pair, cutoffs)
            verify_paired_budget(frame, pair, cutoffs, budget, ledger)
            for key, value in budget.items():
                compare_statistics(
                    value["models"]["v127"], old_budgets[phase][key]["models"]["v127"]
                )
            outputs["budget_metrics"][phase] = budget
            outputs["strata"][phase] = audit_count_strata(frame, pair, MODELS)
            verify_count_strata(frame, pair, outputs["strata"][phase], MODELS)
            outputs["pre_limit"][phase] = pre_limit_metrics(frame, pair)
            verify_pre_limit_restart(frame, pair, outputs["pre_limit"][phase])
            for label in ["monthly", "monthly_strata", "monthly_pre_limit", "monthly_budget"]:
                outputs[label][phase] = {}
            first, last = PERIODS[phase]
            for month in pd.date_range(first, last, freq="MS", inclusive="left").strftime("%Y-%m"):
                mask = frame.end.dt.strftime("%Y-%m").eq(month).to_numpy()
                part, values = frame.loc[mask], {k: v.loc[mask] for k, v in pair.items()}
                m = evaluate_quiet(values["v141"])
                verify_quiet_evaluation(values["v141"], m)
                outputs["monthly"][phase][month] = m
                m = audit_count_strata(part, values, MODELS)
                verify_count_strata(part, values, m, MODELS)
                outputs["monthly_strata"][phase][month] = m
                m = pre_limit_metrics(part, values)
                verify_pre_limit_restart(part, values, m)
                outputs["monthly_pre_limit"][phase][month] = m
                m, ml = evaluate_paired_budget(part, values, cutoffs)
                verify_paired_budget(part, values, cutoffs, m, ml)
                pd.testing.assert_frame_equal(ml, ledger.loc[mask], check_exact=True)
                outputs["monthly_budget"][phase][month] = m
            save_frame(out / (phase + "_predictions.parquet"), candidate)
            save_frame(out / (phase + "_budget_requests.parquet"), ledger)
            fields = [
                "end",
                "episode_id",
                *[
                    a + "_" + m + "_request"
                    for a in ["reduce", "increase"]
                    for m in ["tree", "linear"]
                ],
            ]
            save_frame(out / (phase + "_threshold_only_requests.parquet"), control[fields])
            print(phase + ": 전체·월·횟수·한도 전·공통 예산 검산 완료", flush=True)
        decision = fixed_bin_admission(
            outputs["metrics"]["diagnosis"],
            old_metrics["diagnosis"],
            unchanged,
            outputs["pre_limit"]["diagnosis"],
            outputs["budget_metrics"]["diagnosis"],
        )
        verify_fixed_admission(
            outputs["metrics"]["diagnosis"],
            old_metrics["diagnosis"],
            unchanged,
            outputs["pre_limit"]["diagnosis"],
            outputs["budget_metrics"]["diagnosis"],
            decision,
        )
        if any(sha256(Path(path)) != digest for path, digest in before.items()):
            raise ValueError("꼬리 가중 학습 중 입력·코드·계획 변경")
        verification.update(
            original_rows_and_predictions_exact=True,
            all_weights_and_groups_verified=True,
            v140_all_predictions_metrics_months_and_linears_preserved=True,
            all_calibrations_thresholds_months_strata_and_budgets_verified=True,
            unchanged=unchanged,
        )
        for name, value in {**outputs, "decision": decision, "verification": verification}.items():
            save_json(out / (name + ".json"), value)
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "independent_verification_complete": True,
                "rows": {k: len(v) for k, v in rows.items()},
                "checks": decision["checks"],
                "fixed_bins_admitted": decision["fixed_bins_admitted"],
                "new_models_fitted": True,
                "new_trees": 512,
                "new_linear_models": 0,
                "preserved_linear_models": 2,
                "equivalence_control_trees": 512,
                "all_inputs_and_code_unchanged": True,
                "new_accounts_executed": False,
                "profitability_accepted": False,
                "all_source_periods_already_observed": True,
            },
        )
        seal(out)
        print(
            "고정 구간 학습 독립 검산 완료: "
            + str(sum(decision["checks"].values()))
            + "/42 조건 통과",
            flush=True,
        )
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="구간 경계를 고정한 가중 관리 학습")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_fixed_learning(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
