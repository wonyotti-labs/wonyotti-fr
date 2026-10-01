from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd

from .common import save_json, sha256

BASE = "https://data.binance.vision/data/futures/um/monthly"
ALLOWED_HOSTS = {"data.binance.vision", "public.bitmex.com", "www.bitmex.com", "s3-eu-west-1.amazonaws.com"}
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote", "ignore"]


def safe_get(url: str, limit: int) -> bytes:
    if type(limit) is not int or not 0 < limit <= 256 * 1024 * 1024:
        raise ValueError("다운로드 용량 제한 설정 오류")
    if any(ord(character) < 32 or ord(character) == 127 for character in url) or "\\" in url:
        raise ValueError("허용되지 않은 URL 문자")
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None or parsed.password is not None):
        raise ValueError("허용되지 않은 다운로드 주소")
    if parsed.port not in (None, 443) or parsed.fragment:
        raise ValueError("허용되지 않은 포트 또는 URL 조각")
    # 클라이언트와 서버의 경로 정규화 차이로 허용 범위를 벗어나지 못하게 한다.
    if "%" in parsed.path or parsed.params or any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ValueError("허용되지 않은 다운로드 경로 표현")
    if parsed.hostname == "www.bitmex.com" and parsed.path not in {
        "/api/v1/trade", "/api/v1/trade/bucketed", "/api/v1/funding", "/api/v1/instrument",
    }:
        raise ValueError("공개 시세 읽기 경로만 허용합니다.")
    if parsed.hostname == "s3-eu-west-1.amazonaws.com" and not parsed.path.startswith("/public.bitmex.com/data/"):
        raise ValueError("공식 BitMEX 자료 경로만 허용합니다.")
    for attempt in range(3):
        try:
            with httpx.Client(timeout=httpx.Timeout(45, connect=15), follow_redirects=False,
                              trust_env=False) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("다운로드 용량 제한 초과")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 0
            if status and status < 500 and status != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("다운로드 실패")


def verified_archive(url: str, cache: Path) -> tuple[Path, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.zip", name):
        raise ValueError("잘못된 아카이브 이름")
    path = cache / name
    checksum_path = cache / (name + ".CHECKSUM")
    checksum = checksum_path.read_bytes() if checksum_path.exists() else safe_get(url + ".CHECKSUM", 4096)
    parts = checksum.decode("ascii").split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) or parts[1].lstrip("*") != name:
        raise ValueError("체크섬 파일 형식 또는 대상 파일명 오류")
    expected = parts[0].lower()
    if not path.exists():
        content = safe_get(url, 100 * 1024 * 1024)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("시장 자료 체크섬 불일치")
        temporary = path.with_suffix(".zip.part")
        temporary.write_bytes(content)
        temporary.replace(path)
        checksum_path.write_bytes(checksum)
    if sha256(path) != expected:
        raise ValueError(f"캐시 파일 체크섬 불일치: {name}")
    return path, {"file": name, "url": url, "sha256": expected, "checksum_verified": True,
                  "bytes": path.stat().st_size}


def archive_csv(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise ValueError("단일 CSV 아카이브만 지원합니다.")
        member = members[0]
        if member.filename != path.stem + ".csv" or member.file_size > 512 * 1024 * 1024:
            raise ValueError("아카이브 내부 파일 또는 용량이 올바르지 않습니다.")
        # 파일 시스템에 추출하지 않아 경로 탈출을 방지한다.
        with archive.open(member) as handle:
            content = handle.read(512 * 1024 * 1024 + 1)
        if len(content) > 512 * 1024 * 1024:
            raise ValueError("CSV 용량 제한 초과")
        return content


def parse_klines(content: bytes, interval: str) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(content), header=None, names=KLINE_COLUMNS, dtype=str)
    if len(raw) and raw.iloc[0].open_time == "open_time":
        raw = raw.iloc[1:].copy()
    for key in KLINE_COLUMNS:
        raw[key] = pd.to_numeric(raw[key], errors="raise")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("시장 데이터에 유한하지 않은 숫자가 있습니다.")
    raw["time"] = pd.to_datetime(raw.open_time.astype("int64"), unit="ms", utc=True)
    raw["end"] = raw.time + pd.Timedelta(interval)
    # pandas 시각 해상도와 무관하게 밀리초 값으로 대조한다.
    expected_close = np.array([t.value // 10**6 for t in raw.end]) - 1
    if not np.array_equal(raw.close_time.to_numpy(), expected_close):
        raise ValueError("캔들 종료 시각/간격 불일치")
    invalid = ((raw.high < raw[["open", "close", "low"]].max(axis=1))
               | (raw.low > raw[["open", "close", "high"]].min(axis=1))
               | (raw[["open", "high", "low", "close"]] <= 0).any(axis=1)
               | (raw.volume < 0))
    if invalid.any():
        raise ValueError("유효하지 않은 OHLCV")
    return raw.drop(columns=["ignore", "open_time", "close_time"])


def fetch_market(root: Path, symbols: list[str], start: str, end: str,
                 interval: str = "15m", workers: int = 4) -> dict:
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("지원하지 않는 캔들 간격")
    if any(not re.fullmatch(r"[A-Z0-9]{3,20}", s) for s in symbols):
        raise ValueError("잘못된 심볼")
    months = pd.period_range(start, end, freq="M")
    if len(months) == 0 or len(months) > 120 or len(symbols) > 5:
        raise ValueError("한 번에 120개월, 5개 종목까지 지원합니다.")
    root.mkdir(parents=True, exist_ok=True)
    requests = []
    for symbol in symbols:
        for month in months:
            for kind in ["klines", "fundingRate"]:
                filename = f"{symbol}-{interval}-{month}.zip" if kind == "klines" else f"{symbol}-fundingRate-{month}.zip"
                folder = f"klines/{symbol}/{interval}" if kind == "klines" else f"fundingRate/{symbol}"
                requests.append((symbol, str(month), kind, f"{BASE}/{folder}/{filename}"))
    successful, missing = [], []
    frames = {s: {"klines": [], "fundingRate": []} for s in symbols}

    def fetch(item):
        symbol, month, kind, url = item
        path, record = verified_archive(url, root / "archives")
        content = archive_csv(path)
        if kind == "klines":
            frame = parse_klines(content, interval)
        else:
            frame = pd.read_csv(io.BytesIO(content))
            required = {"calc_time", "last_funding_rate"}
            if not required.issubset(frame.columns):
                raise ValueError("펀딩 스키마 변경")
            frame["time"] = pd.to_datetime(frame.calc_time, unit="ms", utc=True)
            frame["rate"] = pd.to_numeric(frame.last_funding_rate, errors="raise")
            if not np.isfinite(frame.rate).all():
                raise ValueError("유효하지 않은 펀딩률")
            frame = frame[["time", "rate"]]
        return symbol, kind, frame, {**record, "symbol": symbol, "month": month, "kind": kind, "rows": len(frame)}

    with ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
        pending = {pool.submit(fetch, item): item for item in requests}
        for number, future in enumerate(as_completed(pending), 1):
            item = pending[future]
            try:
                symbol, kind, frame, record = future.result()
                frames[symbol][kind].append(frame)
                successful.append(record)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                missing.append({"symbol": item[0], "month": item[1], "kind": item[2], "url": item[3], "status": 404})
            if number % 24 == 0 or number == len(requests):
                print(f"공식 시장 자료 {number}/{len(requests)}개 처리, 없는 자료 {len(missing)}개", flush=True)
    market_summary = {}
    for symbol in symbols:
        market_summary[symbol] = {}
        for kind, chunks in frames[symbol].items():
            if not chunks:
                market_summary[symbol][kind] = {"rows": 0}
                continue
            frame = pd.concat(chunks, ignore_index=True).sort_values("time")
            if frame.time.duplicated().any():
                raise ValueError(f"중복 시장 시각: {symbol}/{kind}")
            path = root / f"{symbol}-{interval if kind == 'klines' else 'funding'}.parquet"
            frame.to_parquet(path, index=False)
            gaps = int((frame.time.diff() > pd.Timedelta(interval)).sum()) if kind == "klines" else None
            market_summary[symbol][kind] = {"rows": len(frame), "first": frame.time.min(), "last": frame.time.max(),
                                            "gaps": gaps, "file": path.name, "sha256": sha256(path)}
    manifest = {"retrieved_at": datetime.now(UTC), "source": "Binance USD-M futures official monthly archives",
                "start_month": start, "end_month": end, "interval": interval,
                "successful": successful, "missing": missing, "summary": market_summary}
    save_json(root / f"manifest-{interval}.json", manifest)
    return manifest


def repair_gaps(source: Path, output: Path, interval: str = "15m") -> dict:
    import json
    import shutil

    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("지원하지 않는 봉 간격")
    manifest_path = source / f"manifest-{interval}.json"
    manifest = json.loads(manifest_path.read_text())
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("보완 자료는 원본 밖의 새 폴더에 저장해야 합니다.")
    shutil.copytree(source, output)
    manifest["parent_manifest_sha256"] = sha256(manifest_path)
    repairs = []
    for symbol, summary in manifest["summary"].items():
        metadata = summary["klines"]
        path = output / metadata["file"]
        if not path.resolve().is_relative_to(output.resolve()) or sha256(path) != metadata["sha256"]:
            raise ValueError("보완 입력 경로 또는 체크섬 오류")
        frame = pd.read_parquet(path)
        expected = pd.date_range(frame.time.min(), frame.time.max(), freq=pd.Timedelta(interval))
        missing = expected.difference(frame.time)
        if len(missing) > pd.Timedelta(days=31) / pd.Timedelta(interval):
            raise ValueError("결측이 31일을 초과합니다. 자료 범위를 먼저 검토하세요.")
        additions = []
        for date in sorted(set(missing.strftime("%Y-%m-%d"))):
            url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
            archive, record = verified_archive(url, output / "archives")
            daily = parse_klines(archive_csv(archive), interval)
            addition = daily[daily.time.isin(missing)]
            additions.append(addition)
            repairs.append({**record, "symbol": symbol, "date": date, "kind": "klines", "rows_added": len(addition)})
            print(f"일별 공식 자료 보완: {symbol} {date}, {len(addition)}봉", flush=True)
        if additions:
            frame = pd.concat([frame, *additions], ignore_index=True).sort_values("time")
            if frame.time.duplicated().any() or len(expected.difference(frame.time)):
                raise ValueError("공식 일별 자료로 결측을 모두 보완하지 못했습니다.")
            frame.to_parquet(path, index=False)
        summary["klines"] = {**metadata, "rows": len(frame), "gaps": 0, "sha256": sha256(path)}
    manifest["daily_repairs"] = repairs
    manifest["repaired_at"] = datetime.now(UTC)
    save_json(output / f"manifest-{interval}.json", manifest)
    return manifest
