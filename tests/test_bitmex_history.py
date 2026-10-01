import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from wonyotti_fr.bitmex_history import fetch_bitmex_history


def test_minute_history_retains_end_boundary_and_rejects_mixed_interval(monkeypatch, tmp_path):
    def response(url, _cache):
        query = parse_qs(urlparse(url).query)
        assert query['binSize'] == ['1m']
        assert query['startTime'] == ['2020-01-01T00:01:00+00:00']
        rows = [{'timestamp': stamp, 'open': 100, 'high': 102, 'low': 99, 'close': 101,
                 'homeNotional': 2, 'volume': 202, 'trades': 4}
                for stamp in ['2020-01-01T00:01:00Z', '2020-01-01T00:02:00Z']]
        return rows, {'url': url, 'sha256': 'synthetic'}
    monkeypatch.setattr('wonyotti_fr.bitmex_history.cached_public_json', response)
    path = fetch_bitmex_history(tmp_path, '2020-01-01', '2020-01-01 00:02', '1m')
    frame = pd.read_parquet(path)
    assert frame.time.iloc[0] == pd.Timestamp('2020-01-01T00:00Z')
    assert frame.end.iloc[-1] == pd.Timestamp('2020-01-01T00:02Z')
    assert frame.volume.tolist() == [2, 2]
    assert json.loads((tmp_path / 'manifest.json').read_text())['missing_bars'] == 0
    with pytest.raises(ValueError, match='경계'):
        fetch_bitmex_history(tmp_path, '2020-01-01', '2020-01-01 00:02', '5m')
    with pytest.raises(ValueError, match='범위·간격'):
        fetch_bitmex_history(tmp_path, '2020-01-01', '2020-01-01 00:05', '1m')
