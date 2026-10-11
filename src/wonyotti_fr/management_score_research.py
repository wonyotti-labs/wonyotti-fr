from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .action_model import select_threshold
from .common import new_run, save_json, sha256
from .event_research import load_selection
from .first_management import BETAS
from .holding_stress import seal
from .management_score_audit import repeat_admission, score_frame, summarize_scores
from .management_score_reference import verify_scores, verify_summary
from .management_support import FEATURES
from .management_support_research import CASES, checked_reference
from .minute_management import ACTIONS
from .position_boost_verification import compare_statistics
from .retained_weight_close_reference import checked_files
from .streaming_backtest import ParquetRows


def checked_source(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError("관리 점수 진단의 입력 검산 지문 오류")
    proof = json.loads(path.read_text())
    root = Path(proof["run"])
    checked_files(root)
    if (
        proof.get("complete") is not True
        or proof.get("profitability_accepted") is not False
        or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
    ):
        raise ValueError("관리 점수 진단의 이전 검산 완료·봉인 오류")
    manifest = json.loads((root / "manifest.json").read_text())
    settings = manifest["settings"]
    spec = checked_reference(Path(settings["reference"]), settings["reference_sha256"])
    runs = json.loads((root / "runs.json").read_text())
    if set(runs) != set(CASES):
        raise ValueError("관리 점수 진단의 두 실제 경로 누락")
    for name, digest in runs.items():
        checked_files(root / name)
        if sha256(root / name / "files.json") != digest:
            raise ValueError("관리 점수 진단의 실제 입력 파일 변경")
    return root, spec


def reproduce_thresholds(calibration, policy, selection):
    common = json.loads((selection / "history_thresholds.json").read_text())
    first = json.loads((selection / "first_policy_thresholds.json").read_text())
    results = {}
    for action in ACTIONS:
        threshold, support = select_threshold(
            calibration[f"y_{action}"], calibration[f"{action}_score"], BETAS[action]
        )
        if threshold != policy.thresholds[action]:
            raise ValueError("관리 점수 진단의 기존 공통 문턱 재현 오류")
        compare_statistics(support, common["support"][action])
        results[action] = {"common": {"threshold": threshold, "support": support}}
        if action != "exit":
            part = calibration.loc[calibration[f"past_{action}_exists"] == 0]
            threshold, support = select_threshold(
                part[f"y_{action}"], part[f"{action}_score"], BETAS[action]
            )
            if threshold != policy.first_thresholds[action]:
                raise ValueError("관리 점수 진단의 기존 첫 문턱 재현 오류")
            compare_statistics(support, first["support"][action])
            results[action]["first"] = {"threshold": threshold, "support": support}
    return results


def independently_check_admission(summaries, decision):
    expected = True
    for action in ["reduce", "increase"]:
        groups = summaries["diagnosis"]["actions"][action]
        r = groups["repeat"]
        a, b = r["original"]["raw_metrics"], r["direct"]["raw_metrics"]
        conditions = {
            "support": r["positive"] >= 20 and r["rows"] - r["positive"] >= 20,
            "repeat_recall_improved": bool(a and b and b["recall"] - a["recall"] > 1e-12),
            "repeat_f_improved": bool(a and b and b["f_beta"] - a["f_beta"] > 1e-12),
            "all_f_preserved": groups["all"]["direct"]["raw_metrics"]["f_beta"]
            - groups["all"]["original"]["raw_metrics"]["f_beta"]
            >= -1e-12,
        }
        assert decision["checks"][action] == conditions
        expected = expected and all(conditions.values())
    same = all(summaries["diagnosis"]["unchanged_masks"].values())
    changed = (
        sum(
            summaries[k]["changed_choices"] + summaries[k]["changed_management_wait"] for k in CASES
        )
        > 0
    )
    assert decision["unchanged_masks"] == same and decision["bot_decisions_changed"] == changed
    assert decision["further_account_research_eligible"] == (expected and same and changed)
    assert not decision["profitability_accepted"] and not decision["new_models_fitted"]


def run_management_score_audit(reference, expected, output):
    previous, spec = checked_source(reference, expected)
    selection, teacher = Path(spec["selection"]), Path(spec["teacher"])
    protocol = Path("docs/EXPERIMENT_V114.md")
    out = new_run(
        output,
        "management-score-audit",
        {
            "previous": str(previous),
            "reference": str(reference),
            "reference_sha256": expected,
            "protocol_sha256": sha256(protocol),
            "new_models_fitted": False,
            "new_account_replayed": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        _, policy = load_selection(selection)
        sources = {
            name: teacher / f"{name}_used.parquet"
            for name in ["training", "calibration", "diagnosis"]
        }
        sources.update({name: previous / name / "management_inputs.parquet" for name in CASES})
        summaries = {}
        for name, path in sources.items():
            columns = [
                *FEATURES,
                *[
                    k
                    for k in [
                        "end",
                        "episode_id",
                        "entry_time",
                        "y_exit",
                        "y_reduce",
                        "y_increase",
                        "intent",
                        "event",
                    ]
                    if k in pq.ParquetFile(path).schema.names
                ],
            ]
            writer = ParquetRows(out / f"{name}_scores.parquet", 8192)
            pieces = []
            try:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns):
                    source = batch.to_pandas().reset_index(drop=True)
                    scored = score_frame(source, policy)
                    verify_scores(source, scored, policy)
                    pieces.append(scored)
                    for row in scored.to_dict("records"):
                        writer.append(row)
            finally:
                writer.close()
            saved = pd.read_parquet(out / f"{name}_scores.parquet")
            pd.testing.assert_frame_equal(
                saved, pd.concat(pieces, ignore_index=True), check_exact=True
            )
            del pieces
            summaries[name] = summarize_scores(saved)
            verify_summary(saved, summaries[name])
            if name == "calibration":
                save_json(
                    out / "threshold_reproduction.json",
                    reproduce_thresholds(saved, policy, selection),
                )
            if name == "diagnosis":
                old = pd.read_parquet(teacher / "predictions.parquet")
                np.testing.assert_array_equal(saved.end, old.end)
                for action in ACTIONS:
                    np.testing.assert_array_equal(
                        saved[f"{action}_score"], old[f"{action}_histogram_calibrated"]
                    )
            save_json(out / "summaries.json", summaries)
            print(f"{name}: {len(saved):,}행 점수·문턱·의도·저장 후 집계 검산 완료", flush=True)
            del saved
        decision = repeat_admission(
            summaries["diagnosis"], {name: summaries[name] for name in CASES}
        )
        independently_check_admission(summaries, decision)
        save_json(out / "decision.json", decision)
        if checked_source(reference, expected) != (previous, spec):
            raise ValueError("관리 점수 진단 중 기존 입력 변경")
        manifest = json.loads((out / "manifest.json").read_text())
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != digest
            for name, digest in manifest["source_sha256"].items()
        ):
            raise ValueError("관리 점수 진단 중 코드·계획 변경")
        save_json(
            out / "summary.json",
            {
                "complete": True,
                "cohorts": len(sources),
                "rows": sum(v["rows"] for v in summaries.values()),
                "all_scores_and_requests_verified": True,
                "previous_bot_intents_and_events_exact": True,
                "old_diagnosis_scores_exact": True,
                "threshold_selection_reproduced": True,
                "all_summary_and_admission_independently_verified": True,
                "new_models_fitted": False,
                "new_account_replayed": False,
                "profitability_accepted": False,
            },
        )
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="고정 관리 점수와 반복 문턱의 요청 억제 진단")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_management_score_audit(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
