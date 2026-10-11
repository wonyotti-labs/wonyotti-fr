from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .management_capacity_learning import save_frame
from .management_flow_research import full_teacher_rows
from .order_price_history import PRICE_FEATURES, attach_order_prices, price_support
from .order_price_reference import (
    verify_execution_prices,
    verify_order_prices,
    verify_price_support,
)
from .retained_weight_close_reference import checked_files
from .score_fusion_research import checked_fusion_reference


def checked_price_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("과거 주문 가격의 선행 증거 지문 오류")
    proof = read_json(reference)
    root = Path(proof["run"])
    checked_files(root)
    summary = read_json(root / "summary.json")
    if (proof.get("complete") is not True or sha256(root / "files.json") != proof["files_sha256"]
        or sha256(root / "summary.json") != proof["summary_sha256"]
        or any(summary.get(k) is not True for k in ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"])
        or summary.get("score_fusion_admitted") is not False
        or any(summary.get(k) is not False or proof.get(k) is not False for k in ["new_accounts_executed", "profitability_accepted"])):
        raise ValueError("과거 주문 가격의 선행 완료·미채택 근거 오류")
    settings = read_json(root / "manifest.json")["settings"]
    relative, previous, teacher = checked_fusion_reference(Path(settings["reference"]), settings["reference_sha256"])
    if any(str(value) != settings.get(key) for value, key in [(relative, "relative"), (previous, "previous"), (teacher, "teacher")]):
        raise ValueError("과거 주문 가격의 원본 연결 오류")
    metadata = read_json(teacher / "manifest.json")["settings"]
    history = Path(metadata["history"])
    if sha256(history / "files.json") != metadata["history_files_sha256"]:
        raise ValueError("과거 주문 가격의 기존 이력 지문 오류")
    checked_files(history)
    source = read_json(history / "manifest.json")["settings"]
    audit = Path(source["audit"])
    for name, digest in source["audit_sha256"].items():
        if Path(name).name != name or (audit / name).is_symlink() or sha256(audit / name) != digest:
            raise ValueError("과거 주문 가격의 원본 체결·행동 지문 오류")
    return {"root": str(root), "previous": str(previous), "teacher": str(teacher),
            "history": str(history), "audit": str(audit)}


def run_order_price_research(reference, expected, output):
    sources = checked_price_reference(reference, expected)
    teacher, history, audit = [Path(sources[k]) for k in ["teacher", "history", "audit"]]
    protocol = Path("docs/EXPERIMENT_V132.md")
    paths = [reference, protocol, teacher / "files.json", teacher / "manifest.json", history / "files.json", history / "manifest.json"]
    paths += [audit / name for name in ["executions.parquet", "actions.parquet", "episodes.parquet", "xbtusd_events.parquet"]]
    paths += [teacher / (phase + "_used.parquet") for phase in ["training", "calibration", "diagnosis"]]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p): sha256(p) for p in paths}
    out = new_run(output, "order-price-history", {"reference": str(reference), "reference_sha256": expected,
        "sources": sources, "protocol_sha256": sha256(protocol), "features": PRICE_FEATURES,
        "price_definition": "first_observed_order_event_quantity_weighted_price",
        "new_models_fitted": False, "new_accounts_executed": False, "profitability_accepted": False})
    print(out, flush=True)
    try:
        save_json(out / "input_sources.json", before)
        actions = pd.read_parquet(audit / "actions.parquet")
        events = pd.read_parquet(audit / "xbtusd_events.parquet")
        raw = pd.read_parquet(audit / "executions.parquet", columns=["symbol", "execid", "time", "order_key", "exectype", "side", "ordtype", "lastqty", "lastpx", "source_row"])
        provenance = verify_execution_prices(raw, events, actions)
        save_json(out / "price_provenance.json", provenance)
        del raw, events
        print("원본 체결의 수량 가중 가격·사건 순서·행동 가격 검산 완료", flush=True)
        rows = full_teacher_rows(teacher)
        support, monthly, verification = {}, {}, {}
        saved_ledger = None
        for phase, frame in rows.items():
            attached, links, ledger = attach_order_prices(frame, actions)
            verification[phase] = verify_order_prices(frame, actions, attached, links, ledger)
            if saved_ledger is None:
                save_frame(out / "first_price_ledger.parquet", ledger)
                saved_ledger = ledger
            else:
                pd.testing.assert_frame_equal(saved_ledger, ledger, check_exact=True)
            support[phase] = price_support(attached)
            verify_price_support(attached, support[phase])
            monthly[phase] = {}
            for month, part in attached.groupby(attached.end.dt.strftime("%Y-%m"), sort=True):
                monthly[phase][month] = price_support(part)
                verify_price_support(part, monthly[phase][month])
            save_frame(out / (phase + "_prices.parquet"), attached[["end", "episode_id", *PRICE_FEATURES]])
            save_frame(out / (phase + "_links.parquet"), links)
            print(phase + ": " + str(len(frame)) + "행의 과거 가격·기존 이력·모든 상태·월 검산 완료", flush=True)
        if before != {str(p): sha256(p) for p in paths}:
            raise ValueError("과거 주문 가격 복원 중 입력·코드·계획 변경")
        for name, value in [("support", support), ("monthly_support", monthly), ("verification", verification)]:
            save_json(out / (name + ".json"), value)
        save_json(out / "summary.json", {"complete": True, "independent_verification_complete": True,
            "rows": {k: len(v) for k, v in rows.items()}, "features_checked": sum(len(v) * 4 for v in rows.values()),
            "original_columns_exact": True, "all_inputs_and_code_unchanged": True,
            "input_reconstruction_verified": True, "new_models_fitted": False, "new_accounts_executed": False,
            "profitability_accepted": False, "all_source_periods_already_observed": True})
        seal(out)
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
    run_order_price_research(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
