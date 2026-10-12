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
from .fixed_bin_learning import dependency_paths, pre_limit_metrics
from .fixed_bin_management import fixed_bin_admission
from .fixed_bin_reference import verify_fixed_admission
from .forward_final_models import (
    calibrate_forward_models,
    fit_forward_models,
    forward_final_admission,
)
from .forward_final_reference import verify_forward_final_admission, verify_forward_final_refits
from .forward_management import ARMS, FOLDS, WEIGHT_RULE, forward_weight_groups
from .forward_management_reference import (
    verify_forward_auxiliaries,
    verify_forward_weight_groups,
    verify_forward_weights,
)
from .forward_weight_learning import checked_forward_reference
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
    verify_quiet_support,
)
from .quiet_state_calibration import fit_state_calibration, state_predictions
from .quiet_state_reference import verify_state_calibration, verify_state_predictions
from .retained_weight_close_reference import checked_files


def checked_forward_weight_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("최종 과거 예측 학습의 비중 증거 지문 오류")
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
            for k in [
                "experiment_complete",
                "final_models_fitted",
                "new_accounts_executed",
                "profitability_accepted",
            ]
        )
        or summary.get("scope") != "auxiliary_and_weight_stage"
        or proof.get("scope") != "auxiliary_and_weight_stage"
        or summary.get("auxiliary_trees") != 1536
        or summary.get("new_final_trees") != 0
        or summary.get("new_linear_models") != 0
    ):
        raise ValueError("최종 과거 예측 학습의 비중 단계 완료 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_forward_reference(Path(settings["reference"]), settings["reference_sha256"])
    if (
        settings.get("sources") != sources
        or settings.get("features") != FEATURES
        or settings.get("arms") != list(ARMS)
        or settings.get("folds") != {k: list(v) for k, v in FOLDS.items()}
        or settings.get("weight_rule") != WEIGHT_RULE
        or settings.get("protocol_sha256") != sha256(Path("docs/EXPERIMENT_V142.md"))
    ):
        raise ValueError("최종 과거 예측 학습의 가중치 계획·출처 연결 오류")
    return {**sources, "forward_weights": str(root)}


def read_fields(folder, names):
    return [read_json(folder / (name + ".json")) for name in names]


def replay_prior_models(rows, sources):
    paths = {k: Path(v) for k, v in sources.items()}
    versions = {"v127": paths["previous"], "v140": paths["tail"], "v141": paths["fixed"]}
    originals, bases = {name: {} for name in versions}, {}
    actions, events = [
        pd.read_parquet(paths["audit"] / name)
        for name in ["actions.parquet", "xbtusd_events.parquet"]
    ]
    verification = {"counts": {}, "legacy_42_conditions_exact": {}}
    for phase, frame in rows.items():
        links = attach_add_counts(frame, actions, events)
        verification["counts"][phase] = verify_add_count_links(frame, actions, events, links)
        pd.testing.assert_frame_equal(
            links,
            pd.read_parquet(paths["counts"] / (phase + "_count_links.parquet")),
            check_exact=True,
        )
        bases[phase] = pd.read_parquet(paths["native"] / (phase + "_predictions.parquet"))
    for name, root in versions.items():
        global_root = paths["quiet"] if name == "v127" else root
        global_names = (
            ["models", "offsets", "thresholds", "threshold_selection", "constants"]
            if name == "v127"
            else [
                "models",
                "global_offsets",
                "global_thresholds",
                "global_threshold_selection",
                "constants",
            ]
        )
        fitted = read_fields(global_root, global_names)
        state = read_fields(
            root,
            [
                "offsets",
                "thresholds",
                "control_thresholds",
                "threshold_selection",
                "support" if name == "v127" else "calibration_support",
            ],
        )
        expected_metrics, expected_months = read_fields(root, ["metrics", "monthly"])
        for phase, frame in rows.items():
            raw = quiet_predictions(frame, bases[phase], *fitted[:3], fitted[4])
            verify_quiet_predictions(frame, bases[phase], raw, *fitted[:3], fitted[4])
            candidate, control = state_predictions(raw, *state[:3])
            verify_state_predictions(raw, candidate, control, *state[:3])
            pd.testing.assert_frame_equal(
                candidate,
                pd.read_parquet(root / (phase + "_predictions.parquet")),
                check_exact=True,
            )
            metrics = evaluate_quiet(candidate)
            verify_quiet_evaluation(candidate, metrics)
            compare_statistics(metrics, expected_metrics[phase])
            for month, ids in frame.groupby(
                frame.end.dt.strftime("%Y-%m"), sort=True
            ).indices.items():
                compare_statistics(
                    evaluate_quiet(candidate.iloc[ids]), expected_months[phase][month]
                )
            if phase == "calibration":
                verify_quiet_calibration(raw, *fitted[1:])
                verify_state_calibration(
                    raw, *state[:4], state[4][phase] if name == "v127" else state[4]
                )
            originals[name][phase] = candidate
        print(name + ": 기존 모든 기간의 모형·점수·보정·요청·집계 재현", flush=True)
    unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
    for name in ["v140", "v141"]:
        root = versions[name]
        cutoffs, expected_budgets = read_fields(root, ["budget_cutoffs", "budget_metrics"])
        pairs = {p: {"v127": originals["v127"][p], name: originals[name][p]} for p in rows}
        verify_paired_cutoffs(rows["calibration"], pairs["calibration"], "v127", cutoffs)
        for phase, frame in rows.items():
            budgets, ledger = evaluate_paired_budget(frame, pairs[phase], cutoffs)
            verify_paired_budget(frame, pairs[phase], cutoffs, budgets, ledger)
            compare_statistics(budgets, expected_budgets[phase])
            pd.testing.assert_frame_equal(
                ledger,
                pd.read_parquet(root / (phase + "_budget_requests.parquet")),
                check_exact=True,
            )
        candidate, previous = [evaluate_quiet(originals[n]["diagnosis"]) for n in [name, "v127"]]
        pre = pre_limit_metrics(rows["diagnosis"], pairs["diagnosis"])
        verify_pre_limit_restart(rows["diagnosis"], pairs["diagnosis"], pre)
        decision = fixed_bin_admission(
            candidate, previous, unchanged, pre, expected_budgets["diagnosis"], model_name=name
        )
        verify_fixed_admission(
            candidate,
            previous,
            unchanged,
            pre,
            expected_budgets["diagnosis"],
            decision,
            model_name=name,
        )
        if decision["checks"] != read_json(root / "decision.json")["checks"]:
            raise ValueError("최종 과거 예측 학습의 기존 42개 조건 변경")
        verification["legacy_42_conditions_exact"][name] = True
    return originals, bases, verification


def evaluate_final_arm(rows, models, arm, bases, originals, parent_cutoffs, out):
    out.mkdir(exist_ok=False)
    fitted = [models, *calibrate_forward_models(rows["calibration"], models, arm)]
    calibration = quiet_predictions(
        rows["calibration"], bases["calibration"], *fitted[:3], fitted[4]
    )
    verify_quiet_calibration(calibration, *fitted[1:])
    state = fit_state_calibration(calibration)
    verify_state_calibration(calibration, *state)
    candidate = state_predictions(calibration, *state[:3])[0]
    cal_pair = {"v127": originals["calibration"], arm: candidate}
    cuts = fit_paired_budget(rows["calibration"], cal_pair, "v127")
    verify_paired_cutoffs(rows["calibration"], cal_pair, "v127", cuts)
    for key, item in cuts.items():
        assert (
            item["support"] == parent_cutoffs[key]["support"]
            and item["budget"] == parent_cutoffs[key]["budget"]
        )
        assert item["models"]["v127"] == parent_cutoffs[key]["models"]["v127"]
    for name, value in zip(
        [
            "models",
            "global_offsets",
            "global_thresholds",
            "global_threshold_selection",
            "constants",
        ],
        fitted,
        strict=True,
    ):
        save_json(out / (name + ".json"), value)
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
    save_json(out / "budget_cutoffs.json", cuts)
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
                for name in ["baseline", "capacity", "constant"]:
                    column = action + "_" + name + "_" + suffix
                    np.testing.assert_array_equal(candidate[column], originals[phase][column])
        for label, value in [
            ("metrics", candidate),
            ("global_metrics", shared),
            ("control_metrics", control),
        ]:
            m = evaluate_quiet(value)
            verify_quiet_evaluation(value, m)
            outputs[label][phase] = m
        pair = {"v127": originals[phase], arm: candidate}
        budget, ledger = evaluate_paired_budget(frame, pair, cuts)
        verify_paired_budget(frame, pair, cuts, budget, ledger)
        outputs["budget_metrics"][phase] = budget
        m = audit_count_strata(frame, pair, ("v127", arm))
        verify_count_strata(frame, pair, m, ("v127", arm))
        outputs["strata"][phase] = m
        outputs["pre_limit"][phase] = pre_limit_metrics(frame, pair)
        verify_pre_limit_restart(frame, pair, outputs["pre_limit"][phase])
        for label in ["monthly", "monthly_strata", "monthly_pre_limit", "monthly_budget"]:
            outputs[label][phase] = {}
        months = frame.end.dt.strftime("%Y-%m")
        for month in pd.date_range(*PERIODS[phase], freq="MS", inclusive="left").strftime("%Y-%m"):
            mask = months.eq(month).to_numpy()
            part, values = frame.loc[mask], {n: v.loc[mask] for n, v in pair.items()}
            m = evaluate_quiet(values[arm])
            verify_quiet_evaluation(values[arm], m)
            outputs["monthly"][phase][month] = m
            m = audit_count_strata(part, values, ("v127", arm))
            verify_count_strata(part, values, m, ("v127", arm))
            outputs["monthly_strata"][phase][month] = m
            m = pre_limit_metrics(part, values)
            verify_pre_limit_restart(part, values, m)
            outputs["monthly_pre_limit"][phase][month] = m
            m, ml = evaluate_paired_budget(part, values, cuts)
            verify_paired_budget(part, values, cuts, m, ml)
            pd.testing.assert_frame_equal(ml, ledger.loc[mask], check_exact=True)
            outputs["monthly_budget"][phase][month] = m
        save_frame(out / (phase + "_predictions.parquet"), candidate)
        save_frame(out / (phase + "_budget_requests.parquet"), ledger)
        fields = [
            "end",
            "episode_id",
            *[a + "_" + m + "_request" for a in ["reduce", "increase"] for m in ["tree", "linear"]],
        ]
        save_frame(out / (phase + "_threshold_only_requests.parquet"), control[fields])
        print(arm + "/" + phase + ": 전체·월·횟수·공통 예산 검산 완료", flush=True)
    unchanged = {"exit": True, "recent_reduce": True, "recent_increase": True}
    previous = evaluate_quiet(originals["diagnosis"])
    decision = fixed_bin_admission(
        outputs["metrics"]["diagnosis"],
        previous,
        unchanged,
        outputs["pre_limit"]["diagnosis"],
        outputs["budget_metrics"]["diagnosis"],
        model_name=arm,
    )
    verify_fixed_admission(
        outputs["metrics"]["diagnosis"],
        previous,
        unchanged,
        outputs["pre_limit"]["diagnosis"],
        outputs["budget_metrics"]["diagnosis"],
        decision,
        model_name=arm,
    )
    for name, value in {**outputs, "decision_42": decision}.items():
        save_json(out / (name + ".json"), value)
    seal(out)
    return outputs, decision


def run_forward_final(reference, expected, output):
    sources = checked_forward_weight_reference(reference, expected)
    paths = {k: Path(v) for k, v in sources.items()}
    protocol = Path("docs/EXPERIMENT_V142.md")
    inputs = [reference, protocol, *dependency_paths()]
    for key in [
        "teacher",
        "previous",
        "quiet",
        "native",
        "counts",
        "budget",
        "tail",
        "fixed",
        "forward_weights",
    ]:
        folder = paths[key]
        inputs += [folder / name for name in checked_files(folder)] + [folder / "files.json"]
    inputs += [paths["audit"] / name for name in ["actions.parquet", "xbtusd_events.parquet"]]
    inputs += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in inputs}
    out = new_run(
        output,
        "forward-management-final",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "sources": sources,
            "protocol_sha256": sha256(protocol),
            "features": FEATURES.copy(),
            "arms": ARMS,
            "weight_rule": WEIGHT_RULE,
            "new_final_trees": 1024,
            "new_linear_models": 4,
            "preserved_auxiliary_trees": 1536,
            "equivalence_control_trees": 512,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        rows = full_teacher_rows(paths["teacher"])
        support = validate_quiet_rows(rows)
        verify_quiet_support(rows, support)
        assert support == read_json(paths["quiet"] / "support.json")
        originals, bases, verification = replay_prior_models(rows, sources)
        stage = paths["forward_weights"]
        auxiliary = (
            read_json(stage / "auxiliary_models.json"),
            pd.read_parquet(stage / "forward_scores.parquet"),
            read_json(stage / "fold_support.json"),
        )
        ledgers = {
            arm: pd.read_parquet(stage / (arm + "_training_weights.parquet")) for arm in ARMS
        }
        details, overlap = read_fields(stage, ["weight_details", "hard_overlap"])
        weighted = ledgers, details, overlap
        parents, bins = (
            read_json(paths["quiet"] / "models.json"),
            read_json(paths["fixed"] / "bins.json"),
        )
        print("보조 1536개 트리·기존 가중치 독립 재검산 시작", flush=True)
        verification["auxiliaries"] = verify_forward_auxiliaries(rows["training"], *auxiliary)
        verification["weights"] = verify_forward_weights(
            rows["training"], parents, auxiliary[1], *weighted
        )
        expected_groups = read_json(stage / "weight_groups.json")
        for arm in ARMS:
            grouped = forward_weight_groups(rows["training"], ledgers[arm])
            verify_forward_weight_groups(rows["training"], ledgers[arm], grouped)
            compare_statistics(grouped, expected_groups[arm])
        print("두 방식의 최종 1024개 트리·선형 네 개 학습 시작", flush=True)
        fitted = fit_forward_models(rows["training"], parents, bins, auxiliary, weighted)
        for name, value in zip(
            ["models", "equivalence_models", "encoded_models", "level_boundaries"],
            fitted,
            strict=True,
        ):
            save_json(out / (name + ".json"), value)
        save_json(out / "bins.json", bins)
        print("최종 학습 완료, 모든 기간·모형 독립 재적합 시작", flush=True)
        verification["final_refits"] = verify_forward_final_refits(
            rows, parents, bins, ledgers, fitted
        )
        parent_cuts = read_json(paths["fixed"] / "budget_cutoffs.json")
        results, decisions = {}, {}
        for arm in ARMS:
            results[arm], decisions[arm] = evaluate_final_arm(
                rows, fitted[0][arm], arm, bases, originals["v127"], parent_cuts, out / arm
            )
        forward, matched = results["v142forward"], results["v142matched"]
        arguments = (
            forward["metrics"]["diagnosis"],
            matched["metrics"]["diagnosis"],
            evaluate_quiet(originals["v127"]["diagnosis"]),
            {"exit": True, "recent_reduce": True, "recent_increase": True},
            forward["pre_limit"]["diagnosis"],
            forward["budget_metrics"]["diagnosis"],
            matched["budget_metrics"]["diagnosis"],
        )
        decision = forward_final_admission(*arguments)
        verify_forward_final_admission(*arguments, decision)
        # 하위 실행의 봉인을 별도 지문으로 연결한다.
        child_seals = {arm: sha256(out / arm / "files.json") for arm in ARMS}
        save_json(out / "child_seals.json", child_seals)
        for arm in ARMS:
            checked_files(out / arm)
        if any(sha256(Path(p)) != h for p, h in before.items()):
            raise ValueError("최종 과거 예측 학습 중 입력·코드·계획 변경")
        verification.update(
            all_original_predictions_preserved=True,
            all_weights_preserved=True,
            all_calibrations_months_strata_budgets_verified=True,
        )
        save_json(out / "verification.json", verification)
        save_json(out / "decision.json", decision)
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "experiment_complete": True,
                "independent_verification_complete": True,
                "all_inputs_and_code_unchanged": True,
                "rows": {p: len(f) for p, f in rows.items()},
                "new_models_fitted": True,
                "final_models_fitted": True,
                "new_final_trees": 1024,
                "new_linear_models": 4,
                "preserved_auxiliary_trees": 1536,
                "equivalence_control_trees": 512,
                "base_checks": {arm: value["checks"] for arm, value in decisions.items()},
                "checks": decision["checks"],
                "forward_weight_admitted": decision["forward_weight_admitted"],
                "new_accounts_executed": False,
                "profitability_accepted": False,
                "all_source_periods_already_observed": True,
            },
        )
        seal(out)
        print(
            "최종 과거 예측 비교 완료: " + str(sum(decision["checks"].values())) + "/48 조건 통과",
            flush=True,
        )
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(
        description="과거 분기 점수 기반의 두 고정 구간 최종 모델 비교"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_forward_final(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
