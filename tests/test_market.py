import hashlib
import io
import zipfile

import pytest

from wonyotti_fr import market


def test_kline_schema_and_ohlc_validation():
    good = b"1577836800000,100,105,99,103,10,1577837699999,1000,5,6,600,0\n"
    frame = market.parse_klines(good, "15m")
    assert len(frame) == 1
    with pytest.raises(ValueError, match="OHLCV"):
        market.parse_klines(good.replace(b",105,", b",101,"), "15m")
    with pytest.raises(ValueError, match="종료 시각"):
        market.parse_klines(good, "1m")


def test_archive_rejects_path_traversal(tmp_path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("../sample.csv", "secret")
    with pytest.raises(ValueError, match="아카이브 내부"):
        market.archive_csv(path)
    assert not (tmp_path.parent / "sample.csv").exists()


def test_checksum_rejects_corruption(monkeypatch, tmp_path):
    correct = b"expected"
    checksum = (hashlib.sha256(correct).hexdigest() + "  sample.zip\n").encode()
    monkeypatch.setattr(market, "safe_get", lambda url, limit: checksum if url.endswith("CHECKSUM") else b"tampered")
    with pytest.raises(ValueError, match="체크섬 불일치"):
        market.verified_archive("https://data.binance.vision/sample.zip", tmp_path)
    assert not (tmp_path / "sample.zip").exists()


def test_cache_is_reverified(monkeypatch, tmp_path):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as handle:
        handle.writestr("sample.csv", "a,b\n1,2\n")
    data = content.getvalue()
    checksum = (hashlib.sha256(data).hexdigest() + "  sample.zip\n").encode()
    monkeypatch.setattr(market, "safe_get", lambda url, limit: checksum if url.endswith("CHECKSUM") else data)
    path, record = market.verified_archive("https://data.binance.vision/sample.zip", tmp_path)
    assert record["checksum_verified"]
    assert market.archive_csv(path) == b"a,b\n1,2\n"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="캐시 파일"):
        market.verified_archive("https://data.binance.vision/sample.zip", tmp_path)


def test_download_host_allowlist():
    with pytest.raises(ValueError, match="허용되지 않은"):
        market.safe_get("https://example.com/archive.zip", 100)
