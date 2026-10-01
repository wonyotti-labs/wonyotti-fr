from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .common import save_json, sha256
from .market import safe_get


def cached_public_json(url: str, cache: Path) -> tuple[list, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    path, meta_path = cache / f"{key}.json", cache / f"{key}.meta.json"
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["url"] != url or sha256(path) != metadata["sha256"]:
            raise ValueError("공개 자료 캐시 무결성 오류")
    else:
        content = safe_get(url, 5 * 1024 * 1024)
        value = json.loads(content)
        if not isinstance(value, list):
            raise ValueError("공개 자료 응답이 배열이 아닙니다.")
        temporary = path.with_suffix(".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        metadata = {"url": url, "sha256": sha256(path), "file": path.name,
                    "hash_type": "local_sha256_not_provider_signature"}
        save_json(meta_path, metadata)
        time.sleep(0.75)
    return json.loads(path.read_text()), metadata


def fetch_bitmex_history(output: Path, start: str = "2018-03-01", end: str = "2022-01-01") -> Path:
    begin, finish = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if begin >= finish or (finish - begin).days > 1500:
        raise ValueError("수집 기간은 1~1500일이어야 합니다.")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["start"] != start or existing["end_exclusive"] != end:
            raise ValueError("다른 수집 범위는 새 폴더를 사용하세요.")
        path = output / existing["file"]
        if sha256(path) != existing["sha256"]:
            raise ValueError("완료된 시장 자료가 변경되었습니다.")
        return path
    cursor = begin + pd.Timedelta(minutes=5)
    chunks, sources = [], []
    while cursor <= finish:
        url = "https://www.bitmex.com/api/v1/trade/bucketed?" + urlencode({
            "symbol": "XBTUSD", "binSize": "5m", "partial": "false", "count": 1000,
            "reverse": "false", "startTime": cursor.isoformat(), "endTime": finish.isoformat(),
        })
        values, metadata = cached_public_json(url, output / "responses")
        if not values:
            raise ValueError(f"요청 기간 종료 전 공개 자료가 비었습니다: {cursor}")
        frame = pd.DataFrame(values)
        frame["end"] = pd.to_datetime(frame.timestamp, utc=True)
        if frame.end.duplicated().any() or not frame.end.is_monotonic_increasing or frame.end.min() < cursor:
            raise ValueError("캔들 API 순서·경계 오류")
        chunks.append(frame)
        sources.append(metadata)
        cursor = frame.end.max() + pd.Timedelta(milliseconds=1)
        if len(sources) % 20 == 0:
            print(f"BitMEX 5분봉 {sum(len(f) for f in chunks):,}개, 현재 {frame.end.max()}", flush=True)
        if len(sources) > 500:
            raise ValueError("시장 수집 페이지 상한 초과")
    frame = pd.concat(chunks, ignore_index=True)
    frame["time"] = frame.end - pd.Timedelta(minutes=5)
    frame = frame[(frame.time >= begin) & (frame.end <= finish)].copy()
    numeric = ["open", "high", "low", "close", "homeNotional", "volume", "trades"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    invalid = ~np.isfinite(frame[numeric]).all(axis=1) | frame[["open", "high", "low", "close"]].le(0).any(axis=1)
    invalid_count = int(invalid.sum())
    frame = frame[~invalid].copy()
    if frame.time.duplicated().any():
        raise ValueError("중복 캔들")
    expected = pd.date_range(begin, finish, freq="5min", inclusive="left")
    missing = expected.difference(frame.time)
    result = frame[["time", "end", "open", "high", "low", "close", "homeNotional", "volume", "trades"]].rename(
        columns={"homeNotional": "volume", "volume": "contract_volume"})
    path = output / "XBTUSD-5m.parquet"
    result.to_parquet(path, index=False)
    manifest = {"source": "BitMEX official public bucketed trades API", "start": start, "end_exclusive": end,
                "symbol": "XBTUSD", "interval": "5m", "rows": len(result), "missing_bars": len(missing),
                "missing_sample": [str(v) for v in missing[:30]], "invalid_price_rows_excluded": invalid_count,
                "file": path.name, "sha256": sha256(path), "responses": sources,
                "open_price_note": "BitMEX bucketed open은 이전 봉 종가다. 실제 다음 봉 첫 체결가로 사용하지 않는다.",
                "volume_unit": "BTC homeNotional", "official_docs": "https://docs.bitmex.com/api-explorer/get-trade-bucketed.html"}
    save_json(manifest_path, manifest)
    print(f"BitMEX 과거 시세: {path}, 결측 {len(missing)}봉", flush=True)
    return path
