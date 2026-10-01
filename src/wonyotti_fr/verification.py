from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from .common import new_run, save_json, sha256
from .market import safe_get
from .reports import table


def select_samples(executions: pd.DataFrame) -> pd.DataFrame:
    trades = executions[(executions.exectype == "Trade") & (executions.symbol == "XBTUSD")].sort_values("time")
    picked = []
    for _, group in trades.groupby([trades.time.dt.year, trades.time.dt.quarter]):
        positions = sorted({0, len(group) // 2, len(group) - 1})
        picked.append(group.iloc[positions])
    return pd.concat(picked, ignore_index=True)


def compare_trade(private: dict, public: list[dict]) -> dict:
    matched = [row for row in public if row.get("trdMatchID") == private["trdmatchid"]]
    result = {"execid": private["execid"], "trdmatchid": private["trdmatchid"], "source_time": private["time"],
              "liquidity": private["lastliquidityind"], "found_count": len(matched), "matched": False}
    if len(matched) != 1:
        return result
    row = matched[0]
    difference = (pd.Timestamp(row["timestamp"]) - pd.Timestamp(private["time"])).total_seconds()
    expected_side = private["side"]
    if private["lastliquidityind"] == "AddedLiquidity":
        expected_side = "Sell" if expected_side == "Buy" else "Buy"
    checks = {"symbol_equal": row["symbol"] == private["symbol"],
              "size_equal": int(row["size"]) == int(private["lastqty"]),
              "price_equal": float(row["price"]) == float(private["lastpx"]),
              "aggressor_side_equal": row["side"] == expected_side,
              "gross_value_equal": int(row["grossValue"]) == abs(int(private["execcost"])),
              "utc_time_within_1ms": abs(difference) < 0.001000001}
    return {**result, **checks, "public_time": row["timestamp"], "time_delta_seconds": difference,
            "matched": all(checks.values())}


def verify_samples(audit_run: Path, output: Path, cache: Path) -> Path:
    source = audit_run / "executions.parquet"
    executions = pd.read_parquet(source, columns=["time", "execid", "trdmatchid", "symbol", "exectype",
                                                 "lastqty", "lastpx", "side", "lastliquidityind", "execcost"])
    selected = select_samples(executions)
    destination = new_run(output, "external-verification", {
        "source_sha256": sha256(source), "sample_rule": "각 연도·분기 XBTUSD 체결의 첫째·중간·마지막",
        "sample_count": len(selected), "identity_claim": "공개 익명 체결 일부와의 일치만 검사한다.",
        "official_docs": "https://docs.bitmex.com/api-explorer/get-trade.html",
    })
    selected.to_parquet(destination / "selected_samples.parquet", index=False)
    cache.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for index, row in enumerate(selected.to_dict("records")):
        center = pd.Timestamp(row["time"]).floor("ms")
        params = {"symbol": "XBTUSD", "count": 1000, "reverse": "false",
                  "startTime": (center - pd.Timedelta(milliseconds=1)).isoformat(),
                  "endTime": (center + pd.Timedelta(milliseconds=2)).isoformat()}
        url = "https://www.bitmex.com/api/v1/trade?" + urlencode(params)
        key = hashlib.sha256(url.encode()).hexdigest()
        path, metadata_path = cache / f"{key}.json", cache / f"{key}.meta.json"
        if path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["url"] != url or metadata["sha256"] != sha256(path):
                raise ValueError("공개 체결 캐시가 변경되었습니다.")
        else:
            payload = safe_get(url, 2 * 1024 * 1024)
            value = json.loads(payload)
            if not isinstance(value, list):
                raise ValueError("공개 체결 응답 형식 오류")
            path.write_bytes(payload)
            metadata = {"url": url, "sha256": sha256(path), "checksum_kind": "locally_recorded_not_provider_signed"}
            save_json(metadata_path, metadata)
            time.sleep(0.8)
        public = json.loads(path.read_text())
        result = compare_trade(row, public)
        result["response_rows"] = len(public)
        result["query_may_be_truncated"] = len(public) >= 1000
        rows.append(result)
        sources.append(metadata)
        if (index + 1) % 8 == 0:
            print(f"외부 대조 {index + 1}/{len(selected)}개: 일치 {sum(r['matched'] for r in rows)}개", flush=True)
    result = pd.DataFrame(rows)
    result.to_parquet(destination / "sample_results.parquet", index=False)
    save_json(destination / "sources.json", sources)
    summary = {"samples": len(rows), "matched": int(result.matched.sum()),
               "unmatched": int((~result.matched).sum()),
               "max_abs_time_delta_seconds": float(result.time_delta_seconds.abs().max()) if "time_delta_seconds" in result else None,
               "scope": "XBTUSD 분기별 표본. 전체 체결·계정 소유·재산을 입증하지 않음."}
    save_json(destination / "summary.json", summary)
    result["year"] = pd.to_datetime(result.source_time, utc=True).dt.year
    by_year = result.groupby("year").agg(samples=("matched", "size"), matches=("matched", "sum")).reset_index()
    (destination / "REPORT.md").write_text(f"""# 원거래소 공개 체결과 표본 대조

## 결과

{table(pd.DataFrame([summary]))}

{table(by_year)}

각 분기의 첫째·중간·마지막 XBTUSD 체결을 가격이나 성공 여부를 보고 바꾸지 않고 선택했다. 공개 API의 체결 ID·종목·수량·가격·공격 주문 방향·체결 가치·시각을 대조했다. 메이커 체결의 공개 매수/매도는 상대 테이커 방향이므로 반대 방향을 비교한다. 시각은 공개 응답의 밀리초 정밀도 차이를 1ms 이내로 확인한다.

일치한 표본은 원본 시각을 UTC로 해석하는 근거다. 표본 밖의 모든 행이 정확하다고 인증하지 않는다. 공개 체결은 익명이므로 특정인의 계정 소유·전체 순자산을 증명하지 않는다. 파일을 공개 체결에 맞춰 작성할 가능성까지 배제하는 검사가 아니다.

## 출처와 무결성

[공식 공개 체결 API](https://docs.bitmex.com/api-explorer/get-trade.html), [공식 자료 목록](https://public.bitmex.com/). 공식 목록이 사용하는 저장소는 `https://s3-eu-west-1.amazonaws.com/public.bitmex.com/`다. 이전 `public.bitmex.com/data/...` 경로의 404를 자료 전체의 부재로 해석하지 않는다.

응답은 인증 없이 공개 읽기 API에서 받았다. 응답 해시는 로컬 재현용이며 공급자가 서명한 체크섬이 아니다. 원본 표본·전체 응답·요청 URL·검사 결과는 로컬에 보존한다.

## 미일치 표본

{table(result[~result.matched]) if (~result.matched).any() else '선택한 표본에는 미일치가 없다.'}
""", encoding="utf-8")
    print(f"외부 체결 대조 보고서: {destination / 'REPORT.md'}", flush=True)
    return destination
