import json

import httpx
import pandas as pd
import pytest

from wonyotti_fr.offline import demo, replay


def test_demo_runs_without_network_and_reconciles(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("오프라인 실행에서 네트워크를 호출했습니다.")
    monkeypatch.setattr(httpx, "Client", forbidden)
    output = demo(tmp_path)
    metrics = json.loads((output / "metrics.json").read_text())
    state = json.loads((output / "final_state.json").read_text())
    curve = pd.read_parquet(output / "equity.parquet")
    assert metrics["closed_trades"] > 0
    assert metrics["accounting_residual"] == pytest.approx(0, abs=1e-8)
    assert state["quantity"] == 0
    assert state["cash"] == curve.equity.iloc[-1]


def test_replay_rejects_modified_model_before_reading_prices(tmp_path):
    (tmp_path / "model.json").write_text("{}")
    (tmp_path / "frozen_selection.json").write_text(json.dumps({"model_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="체크섬"):
        replay(tmp_path, tmp_path / "missing", "BTCUSDT", "2024-01-01", "2025-01-01", tmp_path)
