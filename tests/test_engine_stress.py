from dataclasses import asdict

import numpy as np
import pandas as pd

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_features import MARKET_FEATURES, STATE_FEATURES
from wonyotti_fr.event_model import EventModel


def test_real_process_exit_rolls_back_and_manual_control_replays(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    for name, features, intercept in [('entry', MARKET_FEATURES, [5, -5, 0]),
                                      ('management', MARKET_FEATURES + STATE_FEATURES, [-5, -5, 5])]:
        n = len(features)
        model = EventModel(features, ['enter_long', 'enter_short', 'hold'],
                           np.zeros(n), np.ones(n), np.zeros((3, n)), np.array(intercept))
        save_json(models / f'{name}_model.json', model.to_dict())
    hashes = {name: sha256(models / name) for name in ['entry_model.json', 'management_model.json']}
    save_json(models / 'files.json', hashes)
    save_json(models / 'frozen_selection.json', {'risk': asdict(EngineConfig(stop_fraction=0)),
                                               'entry_threshold': 0.5, 'management_threshold': 0.5, 'model_sha256': hashes})
    save_json(models / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(models / 'frozen_selection.json')})
    times = pd.date_range('2024-01-01', periods=8, freq='5min', tz='UTC')
    events = [{'time': t.isoformat(), 'open': 100, 'high': 101, 'low': 99, 'close': 100,
               'features': [0] * len(MARKET_FEATURES)} for t in times]
    result = verify_stress(events, models, tmp_path, {'source': 'synthetic'})
    assert result['all_passed'] and result['position_open_at_interruption']
    assert result['child_process_exit_code'] == 73
