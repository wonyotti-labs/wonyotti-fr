from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .first_exit_verification import verify_execution
from .holding_stress import seal, stress_details
from .management_score_research import checked_source, independently_check_admission
from .minute_data import prepare_minute_period
from .repeat_direct import ReferenceRepeatPolicy, RepeatDirectPolicy
from .retained_weight_close_reference import checked_files

CASES = {
    "development": ("BTCUSDT", ["2021-01-01", "2022-01-01"]),
    "confirmation": ("BTCUSDT", ["2022-01-01", "2023-01-01"]),
    "observed": ("BTCUSDT", ["2023-01-01", "2026-01-01"]),
    "recent": ("BTCUSDT", ["2026-01-01", "2026-10-01"]),
    "eth_observed": ("ETHUSDT", ["2023-01-01", "2026-01-01"]),
}
RUN_FILES = [
    "config.json",
    "metrics.json",
    "equity.parquet",
    "trades.parquet",
    "fills.parquet",
    "final_state.json",
]


def read(path):
    return json.loads(path.read_text())


def checked_repeat_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 2**20 or sha256(reference) != expected:
        raise ValueError("반복 직접 계좌의 출처 지문 오류")
    spec = read(reference)
    proof_path = Path(spec["score_verification"])
    if (
        sha256(proof_path) != spec["score_verification_sha256"]
        or set(spec["cases"]) != set(CASES)
        or spec.get("profitability_accepted") is not False
    ):
        raise ValueError("반복 직접 계좌의 고정 사례·검산 연결 오류")
    proof = read(proof_path)
    source = Path(proof["run"])
    checked_files(source)
    if (
        proof.get("complete") is not True
        or proof.get("profitability_accepted") is not False
        or sha256(source / "files.json") != proof["files_sha256"]
        or sha256(source / "summary.json") != proof["summary_sha256"]
    ):
        raise ValueError("반복 직접 계좌의 선행 점수 검산 미완료")
    settings = read(source / "manifest.json")["settings"]
    _, previous_spec = checked_source(Path(settings["reference"]), settings["reference_sha256"])
    decision = read(source / "decision.json")
    independently_check_admission(read(source / "summaries.json"), decision)
    if (
        not decision["further_account_research_eligible"]
        or spec["selection"] != previous_spec["selection"]
    ):
        raise ValueError("반복 직접 계좌의 사전 진단 미통과·부모 변경")
    frozen, _ = load_selection(Path(spec["selection"]))
    if frozen["protocol"] != "exit_state_v48":
        raise ValueError("반복 직접 계좌의 고정 부모 오류")
    required = {str(proof_path), str(Path(spec["selection"]) / "frozen_selection.json")}
    for key, (symbol, bounds) in CASES.items():
        case = spec["cases"][key]
        if case["symbol"] != symbol or case["bounds"] != bounds:
            raise ValueError("반복 직접 계좌의 시장·기간 변경")
        required.update(str(Path(case["baseline"]) / name) for name in RUN_FILES)
        required.update(
            str(Path(case[k]) / name)
            for k, name in [("market", "manifest-1m.json"), ("features", "manifest-5m.json")]
        )
        if read(Path(case["baseline"]) / "config.json") != frozen["risk"]:
            raise ValueError("반복 직접 계좌의 부모 위험 설정 불일치")
    if not required <= set(spec["sealed_files"]):
        raise ValueError("반복 직접 계좌의 필수 입력 봉인 누락")
    for name, digest in spec["sealed_files"].items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("반복 직접 계좌의 봉인 입력 변경: " + name)
    return spec, frozen


def holding_details(folder):
    trades = pd.read_parquet(folder / "trades.parquet")
    if trades.empty:
        return {"closed_trades": 0, "hold_minutes": None, "exit_reasons": {}}
    return {
        "closed_trades": len(trades),
        "hold_minutes": {
            "mean": float(trades.hold_bars.mean()),
            "median": float(trades.hold_bars.median()),
            "maximum": int(trades.hold_bars.max()),
        },
        "exit_reasons": {
            str(name): {"trades": len(part), "net_pnl": float(part.net_pnl.sum())}
            for name, part in trades.groupby("exit_reason")
        },
    }


def account_admission(rows):
    if len(rows) != len(CASES) or {row["case"] for row in rows} != set(CASES):
        raise ValueError("반복 직접 계좌의 다섯 실행 누락·중복")
    checks = {
        row["case"]: {
            "positive": row["total_return"] > 0,
            "at_least_30_trades": row["closed_trades"] >= 30,
            "no_halt": not row["permanent_halt"],
        }
        for row in rows
    }
    return {
        "complete": True,
        "runs": len(rows),
        "checks": checks,
        "cost_and_market_expansion_eligible": all(
            all(checks[name].values()) for name in ["confirmation", "observed", "recent"]
        ),
        "existing_parent_results_reused": True,
        "new_models_fitted": False,
        "all_periods_already_observed": True,
        "profitability_accepted": False,
    }


def run_repeat_direct(reference, expected, output):
    spec, frozen = checked_repeat_reference(reference, expected)
    protocol = Path("docs/EXPERIMENT_V115.md")
    out = new_run(
        output,
        "repeat-direct-account",
        {
            "reference": str(reference),
            "reference_sha256": expected,
            "spec": spec,
            "risk": frozen["risk"],
            "protocol_sha256": sha256(protocol),
            "repeat_multiplier": 1.0,
            "original_repeat_multiplier": 1.5,
            "new_models_fitted": False,
            "profitability_accepted": False,
        },
    )
    print(out, flush=True)
    try:
        save_json(
            out / "baseline_results.json",
            {
                name: read(Path(case["baseline"]) / "metrics.json")
                for name, case in spec["cases"].items()
            },
        )
        rows, runs = [], {}
        for name in CASES:
            case = spec["cases"][name]
            bars, checks = prepare_minute_period(
                Path(case["market"]),
                Path(case["features"]),
                case["symbol"],
                *case["bounds"],
                minute_inputs=True,
            )
            folder = out / name
            _, parent = load_selection(Path(spec["selection"]))
            print(name + ": 후보 자체 계좌 시작", flush=True)
            metrics = backtest(
                bars, RepeatDirectPolicy(parent), EngineConfig(**frozen["risk"]), folder
            )
            save_json(folder / "input_verification.json", checks)
            print(name + ": 별도 정책의 전체 계좌 검산 시작", flush=True)
            _, original = load_selection(Path(spec["selection"]))
            verification = verify_execution(folder, bars, ReferenceRepeatPolicy(original))
            save_json(folder / "execution_verification.json", verification)
            details = stress_details(folder, frozen["risk"]["initial_equity"])
            details["holding"] = holding_details(folder)
            save_json(folder / "details.json", details)
            seal(folder)
            runs[name] = sha256(folder / "files.json")
            rows.append({"case": name, "symbol": case["symbol"], **metrics, **details})
            save_json(out / "results.json", rows)
            save_json(out / "runs.json", runs)
            print(
                f"{name}: {metrics['total_return']:+.6%}, {metrics['closed_trades']}거래, 중지 {metrics['permanent_halt']}, 전체 검산 완료",
                flush=True,
            )
            del bars, parent, original
        if checked_repeat_reference(reference, expected) != (spec, frozen):
            raise ValueError("반복 직접 계좌 실행 중 기존 입력 변경")
        manifest = read(out / "manifest.json")
        if sha256(protocol) != manifest["settings"]["protocol_sha256"] or any(
            sha256(Path(__file__).parent / name) != digest
            for name, digest in manifest["source_sha256"].items()
        ):
            raise ValueError("반복 직접 계좌 실행 중 코드·계획 변경")
        save_json(out / "summary.json", account_admission(rows))
        seal(out)
    except Exception as error:
        save_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description="반복 관리 가산 제거의 자체 계좌 대조")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run_repeat_direct(args.reference, args.sha256, args.output)


if __name__ == "__main__":
    main()
