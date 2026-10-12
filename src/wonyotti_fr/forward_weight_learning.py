from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .fixed_bin_learning import checked_fixed_reference, dependency_paths
from .forward_management import (
    ARMS,
    FOLDS,
    WEIGHT_RULE,
    build_forward_weights,
    fit_forward_auxiliaries,
    forward_partition,
    forward_weight_groups,
    require_forward_support,
)
from .forward_management_reference import (
    reference_forward_partition,
    verify_forward_auxiliaries,
    verify_forward_weight_groups,
    verify_forward_weights,
)
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .quiet_management import FEATURES, validate_quiet_rows
from .quiet_management_reference import verify_quiet_support
from .retained_weight_close_reference import checked_files
from .tail_weighted_management import WEIGHT_RULE as PARENT_RULE


def checked_forward_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("과거 예측 학습의 선행 증거 지문 오류")
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
        or summary.get("fixed_bins_admitted") is not False
        or summary.get("new_trees") != 512
        or summary.get("new_linear_models") != 0
        or summary.get("preserved_linear_models") != 2
        or summary.get("equivalence_control_trees") != 512
    ):
        raise ValueError("과거 예측 학습의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    sources = checked_fixed_reference(Path(settings["reference"]), settings["reference_sha256"])
    if (
        settings.get("sources") != sources
        or settings.get("features") != FEATURES
        or settings.get("weight_rule") != PARENT_RULE
        or settings.get("models") != ["v127", "v141"]
    ):
        raise ValueError("과거 예측 학습의 기존 자료·입력·모델 연결 오류")
    return {**sources, "fixed": str(root)}


def run_forward_weights(reference, expected, output):
    sources = checked_forward_reference(reference, expected)
    teacher, fixed, quiet = (Path(sources[k]) for k in ["teacher", "fixed", "quiet"])
    protocol = Path("docs/EXPERIMENT_V142.md")
    paths = [reference, protocol, *dependency_paths()]
    for folder in [teacher, fixed, quiet]:
        paths += [folder / name for name in checked_files(folder)] + [folder / "files.json"]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(
        output,
        "forward-management-weights",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "sources": sources,
            "protocol_sha256": sha256(protocol),
            "features": FEATURES.copy(),
            "arms": ARMS,
            "folds": FOLDS,
            "weight_rule": WEIGHT_RULE,
            "scope": "auxiliary_and_weight_stage",
            "planned_auxiliary_trees": 1536,
            "final_models_fitted": False,
            "experiment_complete": False,
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
        training = rows["training"]
        partition, fold_support = forward_partition(training)
        reference_forward_partition(training, partition, fold_support)
        save_frame(out / "partition.parquet", partition)
        save_json(out / "fold_support.json", fold_support)
        require_forward_support(fold_support)
        print("모든 분기 표본·포지션·시각 경계 검산, 보조 1536개 트리 학습 시작", flush=True)
        models, scores, fitted_support = fit_forward_auxiliaries(training)
        assert fitted_support == fold_support
        pd.testing.assert_frame_equal(partition, scores[partition.columns], check_exact=True)
        save_json(out / "auxiliary_models.json", models)
        save_frame(out / "forward_scores.parquet", scores)
        print("과거 보조 학습 완료, 모든 트리·구간·확률 독립 재적합 시작", flush=True)
        verification = {
            "auxiliaries": verify_forward_auxiliaries(training, models, scores, fold_support)
        }
        parents = read_json(quiet / "models.json")
        ledgers, details, overlap = build_forward_weights(training, parents, models, scores)
        verification["weights"] = verify_forward_weights(
            training, parents, scores, ledgers, details, overlap
        )
        verification["groups"] = {}
        groups = {}
        for arm in ARMS:
            groups[arm] = forward_weight_groups(training, ledgers[arm])
            verification["groups"][arm] = verify_forward_weight_groups(
                training, ledgers[arm], groups[arm]
            )
            save_frame(out / (arm + "_training_weights.parquet"), ledgers[arm])
        for name, value in [
            ("weight_details", details),
            ("hard_overlap", overlap),
            ("weight_groups", groups),
            ("verification", verification),
        ]:
            save_json(out / (name + ".json"), value)
        for filename, frame in [
            ("partition.parquet", partition),
            ("forward_scores.parquet", scores),
            *[(arm + "_training_weights.parquet", ledgers[arm]) for arm in ARMS],
        ]:
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out / filename), check_exact=True)
        if any(sha256(Path(p)) != digest for p, digest in before.items()):
            raise ValueError("과거 예측·가중치 생성 중 입력·코드·계획 변경")
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "scope": "auxiliary_and_weight_stage",
                "experiment_complete": False,
                "independent_verification_complete": True,
                "all_inputs_and_code_unchanged": True,
                "rows": {p: len(f) for p, f in rows.items()},
                "auxiliary_trees": 1536,
                "new_models_fitted": True,
                "final_models_fitted": False,
                "new_final_trees": 0,
                "new_linear_models": 0,
                "new_accounts_executed": False,
                "profitability_accepted": False,
                "all_source_periods_already_observed": True,
                "pending": [
                    "final_fixed_bin_models",
                    "weighted_linears",
                    "calibration",
                    "all_period_metrics_and_budgets",
                    "42_and_48_conditions",
                ],
            },
        )
        seal(out)
        print(
            "보조 1536개 트리와 두 방식 가중치 검산 완료; 최종 모델·성능 평가는 미실행", flush=True
        )
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="과거 분기 예측과 동일 지원 대조의 가중치 생성")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_forward_weights(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
