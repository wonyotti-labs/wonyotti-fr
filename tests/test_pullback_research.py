from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.pullback_research import candidate_plan


def selection(root):
    root.mkdir()
    model = {'format': 'expansion_binary_v1', 'kind': 'logistic', 'features': MARKET_FEATURES,
             'mean': [0.] * 14, 'scale': [1.] * 14, 'coefficients': [0.] * 14, 'intercept': 2.}
    save_json(root / 'expansion_models.json', {'activity': model, 'direction': model})
    hashes = {'expansion_models.json': sha256(root / 'expansion_models.json')}
    base = {'protocol': 'expansion_v4', 'model_sha256': hashes, 'activity_threshold': .1,
            'direction_threshold': .65, 'min_hold_bars': 12, 'kind': 'logistic',
            'development_metrics': {'activity_quantile': .975}}
    save_json(root / 'base_selection.json', base)
    frozen = {'protocol': 'pullback_v7', 'model_sha256': hashes, 'offset_bps': 8, 'ttl_minutes': 5,
              'base_selection_sha256': sha256(root / 'base_selection.json'),
              'risk': asdict(EngineConfig(bar_seconds=60, max_hold_bars=3, cooldown_bars=0, max_adds=0))}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_base_selection_and_model_tampering_cannot_change_loaded_waiting_policy(tmp_path):
    root = tmp_path / 'selection'
    selection(root)
    frozen, policy = load_selection(root)
    assert len(candidate_plan()) == 6 and policy.offset_bps == frozen['offset_bps']
    path = root / 'base_selection.json'
    original = path.read_bytes()
    path.write_text('{}')
    with pytest.raises(ValueError, match='기반 선택'):
        load_selection(root)
    path.write_bytes(original)
    (root / 'expansion_models.json').write_text('{}')
    with pytest.raises(ValueError, match='모델'):
        load_selection(root)


def test_stateful_minute_policy_streamed_output_matches_reference_backend(tmp_path):
    root = tmp_path / 'selection'
    frozen = selection(root)
    time = pd.date_range('2020-01-01', periods=60, freq='1min', tz='UTC')
    close = np.where(np.arange(60) % 5 == 0, 99.8, 100.)
    frame = pd.DataFrame({'time': time, 'end': time + pd.Timedelta(minutes=1),
                           'open': close, 'high': close, 'low': close, 'close': close, 'funding_rate': 0.})
    frame[MARKET_FEATURES] = 0.
    outputs = []
    for streaming in [False, True]:
        _, policy = load_selection(root)
        outputs.append(backtest(frame, policy, EngineConfig(**frozen['risk']), tmp_path / str(streaming),
                                streaming=streaming, batch_size=7))
    assert outputs[0]['closed_trades'] > 0
    for name in ['equity', 'trades', 'fills']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'False' / f'{name}.parquet'),
                                      pd.read_parquet(tmp_path / 'True' / f'{name}.parquet'))
    assert (tmp_path / 'False' / 'final_state.json').read_bytes() == (tmp_path / 'True' / 'final_state.json').read_bytes()
    assert outputs[0]['total_return'] == outputs[1]['total_return']
