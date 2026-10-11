from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .balanced_management_learning import load_reference_models
from .common import new_run, save_json, sha256
from .extrema_age import (
    EXTRA_FEATURES,
    SOURCE_COLUMNS,
    age_groups,
    attach_extrema_age,
    extrema_clock_table,
)
from .extrema_age_model_reference import (
    verify_age_admission,
    verify_age_predictions,
    verify_age_refits,
    verify_age_usage,
)
from .extrema_age_models import age_admission, age_feature_usage, age_predictions, fit_age_models
from .extrema_age_reference import verify_age_groups, verify_extrema_inputs
from .first_state_models import validate_first_rows
from .holding_stress import read_json, seal
from .joint_history_models import evaluate_joint, joint_thresholds
from .joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from .management_capacity import LABELS
from .management_capacity_learning import save_frame
from .management_capacity_reference import verify_capacity_predictions
from .management_flow_research import full_teacher_rows
from .management_market_model_reference import verify_offsets
from .minute_management import ACTIONS
from .position_boost_verification import compare_statistics
from .retained_weight_close_reference import checked_files
from .volatility_path_learning import checked_volatility_reference


def checked_age_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("극값 시계의 선행 증거 지문 오류")
    proof = read_json(path)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (
        proof.get("complete") is not True
        or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or summary.get("complete") is not True
        or summary.get("independent_verification_complete") is not True
        or summary.get("volatility_path_admitted") is not False
        or summary.get("new_accounts_executed") is not False
    ):
        raise ValueError("극값 시계의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    previous, spec, policy = checked_volatility_reference(
        Path(settings["reference"]), settings["reference_sha256"]
    )
    if spec != settings["spec"] or str(previous) != settings["previous"]:
        raise ValueError("극값 시계의 원본·대조 출처 변경")
    return previous, spec, policy


def load_extrema_source(teacher):
    checked_files(teacher)
    root, hashes = teacher, {}
    for key, digest_key in [
        ("labels", "labels_files_sha256"),
        ("inventory_labels", "inventory_files_sha256"),
        ("path_labels", "path_files_sha256"),
    ]:
        settings = read_json(root / "manifest.json")["settings"]
        for name in ["manifest.json", "files.json"]:
            hashes[str(root / name)] = sha256(root / name)
        root = Path(settings[key])
        if root.is_symlink() or sha256(root / "files.json") != settings[digest_key]:
            raise ValueError("극값 시계의 원본 경로 지문 연결 오류")
        checked_files(root)
    for name in ["manifest.json", "files.json", "events.parquet"]:
        hashes[str(root / name)] = sha256(root / name)
    frame = pd.read_parquet(root / "events.parquet", columns=SOURCE_COLUMNS)
    return frame, {"path_source": str(root), "files": hashes}


def verify_age_reference_thresholds(previous, thresholds, details):
    for name, value in [("thresholds", thresholds), ("threshold_selection", details)]:
        original = {
            g: {k: v for k, v in models.items() if not k.startswith("aged_")}
            for g, models in value.items()
        }
        assert original == read_json(previous / (name + ".json"))


def verify_age_reference_metrics(previous, metrics, monthly):
    def original(value):
        return {
            a: {
                g: {k: v for k, v in models.items() if not k.startswith("aged_")}
                for g, models in groups.items()
            }
            for a, groups in value.items()
        }

    compare_statistics(
        {p: original(v) for p, v in metrics.items()}, read_json(previous / "metrics.json")
    )
    compare_statistics(
        {p: {m: original(v) for m, v in months.items()} for p, months in monthly.items()},
        read_json(previous / "monthly.json"),
    )


def run_extrema_age(reference, expected, output):
    previous, spec, policy = checked_age_reference(reference, expected)
    teacher, protocol = Path(spec["teacher"]), Path("docs/EXPERIMENT_V124.md")
    out = new_run(
        output,
        "extrema-age-learning",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "spec": spec,
            "previous": str(previous),
            "protocol_sha256": sha256(protocol),
            "extra_features": EXTRA_FEATURES,
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        rows = full_teacher_rows(teacher)
        support = validate_first_rows(rows)
        source, source_proof = load_extrema_source(teacher)
        save_json(out / "input_sources.json", source_proof)
        print("전체 원본 경로의 엄격한 극값 갱신·시간순 상태 검산", flush=True)
        table = extrema_clock_table(source)
        expanded, links = attach_extrema_age(rows, table)
        input_verification = verify_extrema_inputs(source, table, rows, expanded, links)
        del source, table
        for phase in rows:
            save_frame(
                out / (phase + "_new_inputs.parquet"),
                expanded[phase][["end", "episode_id", *EXTRA_FEATURES]],
            )
            save_frame(out / (phase + "_linkage.parquet"), links[phase])
        save_json(out / "input_verification.json", input_verification)
        save_json(out / "support.json", support)
        models, offsets = load_reference_models(previous)
        print("두 경과 시간 입력 검산 및 52개 입력 고정 모델 학습", flush=True)
        new_models, new_offsets, training = fit_age_models(expanded)
        models.update(new_models)
        offsets.update(new_offsets)
        for name, values in [("models", models), ("offsets", offsets)]:
            save_json(out / (name + ".json"), {k: v.to_dict() for k, v in values.items()})
        save_json(out / "training_support.json", training)
        print("새 768개 트리·선형 계수·모든 확률 독립 검산", flush=True)
        verification = verify_age_refits(expanded, models)
        cy = rows["calibration"][LABELS].to_numpy(int)
        constant = cy.mean(axis=0)
        metrics, monthly, ages = {}, {}, {}
        with threadpool_limits(limits=1):
            calibration = age_predictions(
                rows["calibration"], expanded["calibration"], models, offsets, constant
            )
            verify_offsets(calibration, cy, offsets)
            thresholds, details = joint_thresholds(rows["calibration"], calibration)
            verify_joint_thresholds(rows["calibration"], calibration, thresholds, details)
            verify_age_reference_thresholds(previous, thresholds, details)
            if (
                thresholds["full"]["native_calibrated"] != policy.thresholds
                or thresholds["first"]["native_calibrated"] != policy.first_thresholds
            ):
                raise ValueError("극값 시계의 원래 첫·반복 문턱 불일치")
            save_json(out / "thresholds.json", thresholds)
            save_json(out / "threshold_selection.json", details)
            for phase, frame in rows.items():
                scores = (
                    calibration
                    if phase == "calibration"
                    else age_predictions(frame, expanded[phase], models, offsets, constant)
                )
                refs = {k: v for k, v in models.items() if k in {"native", "capacity", "linear"}}
                verify_capacity_predictions(frame, refs, offsets, constant, scores)
                verify_age_predictions(expanded[phase], models, offsets, scores)
                metrics[phase], requests = evaluate_joint(frame, scores, thresholds)
                verify_joint_evaluation(frame, scores, thresholds, metrics[phase], requests)
                monthly[phase], ages[phase] = {}, {"best": {}, "worst": {}}
                memberships = [
                    ("month", month, indices)
                    for month, indices in frame.groupby(
                        frame.end.dt.strftime("%Y-%m"), sort=True
                    ).indices.items()
                ]
                groups = {}
                for label in ["best", "worst"]:
                    groups[label] = age_groups(links[phase], label)
                    verify_age_groups(links[phase], label, groups[label])
                    memberships += [
                        (label, str(i), np.flatnonzero(groups[label] == i)) for i in range(5)
                    ]
                for kind, key, indices in memberships:
                    part = frame.iloc[indices]
                    values = {name: p[indices] for name, p in scores.items()}
                    result, decisions = evaluate_joint(part, values, thresholds)
                    verify_joint_evaluation(part, values, thresholds, result, decisions)
                    destination = monthly[phase] if kind == "month" else ages[phase][kind]
                    destination[key] = result
                saved = frame[
                    ["end", "episode_id", "past_increase_exists", "past_reduce_exists", *LABELS]
                ].copy()
                for label, values in groups.items():
                    saved[label + "_age_group"] = values
                for name, values in scores.items():
                    for i, action in enumerate(ACTIONS):
                        saved[action + "_" + name] = values[:, i]
                        saved[action + "_" + name + "_request"] = requests[name][:, i]
                original = pd.read_parquet(previous / (phase + "_predictions.parquet"))
                pd.testing.assert_frame_equal(saved[original.columns], original, check_exact=True)
                save_frame(out / (phase + "_predictions.parquet"), saved)
                print(phase + ": 대조·전체 확률·월별·경과 시간 구간 검산 완료", flush=True)
        verify_age_reference_metrics(previous, metrics, monthly)
        decision = age_admission(metrics["diagnosis"])
        verify_age_admission(metrics["diagnosis"], decision)
        usage = age_feature_usage(models["aged"])
        verify_age_usage(models["aged"], usage)
        verification.update(
            original_models_offsets_and_all_scores_requests_exact=True,
            original_first_and_full_thresholds_exact=True,
            original_all_period_and_month_metrics_preserved=True,
            all_prediction_rows=sum(len(v) for v in rows.values()),
            all_new_input_values=sum(
                v["new_values"] for v in input_verification["periods"].values()
            ),
            all_months_history_groups_and_age_groups_verified=True,
        )
        for name, value in [
            ("metrics", metrics),
            ("monthly", monthly),
            ("age_metrics", ages),
            ("decision", decision),
            ("feature_usage", usage),
            ("verification", verification),
        ]:
            save_json(out / (name + ".json"), value)
        after_previous, after_spec, _ = checked_age_reference(reference, expected)
        if after_previous != previous or after_spec != spec:
            raise ValueError("극값 시계 학습 중 기존 입력 변경")
        for name, digest in source_proof["files"].items():
            if sha256(Path(name)) != digest:
                raise ValueError("극값 시계 학습 중 원본 경로 변경")
        manifest = read_json(out / "manifest.json")
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != value
            for name, value in manifest["source_sha256"].items()
        ):
            raise ValueError("극값 시계 학습 중 코드·계획 변경")
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "rows": {k: len(v) for k, v in rows.items()},
                "independent_verification_complete": True,
                "new_accounts_executed": False,
                **decision,
            },
        )
        seal(out)
        print(decision, flush=True)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="보유 극값 갱신 후 경과 시간의 시간순 학습·검산")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_extrema_age(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
