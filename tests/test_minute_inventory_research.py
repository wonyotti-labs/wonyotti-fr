import json

import numpy as np
import pandas as pd
import pytest
from test_inventory_recent import source_frame
from test_inventory_research import inventory_selection
from test_pullback_evaluation import bars

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.inventory_research import INVENTORY_FILES, PARENT_FILES
from wonyotti_fr.minute_inputs import MINUTE_FEATURES
from wonyotti_fr.minute_inventory import (
    MinuteInventoryModels,
    MinuteInventoryPolicy,
    MinuteReductionModel,
)
from wonyotti_fr.minute_inventory_research import (
    copy_previous_inventory,
    run_minute_inventory_selection,
)
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def previous_selection(root):
    frozen = inventory_selection(root)
    frozen.update(protocol='minute_inventory_recent_v20', training_period=list(TRAINING_PERIODS[1]),
                  calibration_period=list(CALIBRATION_PERIODS[1]), development_in_sample=True)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    save_json(root / 'manifest.json', {'settings': {'files_sha256': 'synthetic_original'}})
    return frozen


def selection(root, previous):
    old = previous_selection(previous)
    copy_previous_inventory(previous, root / 'previous_inventory')
    for name in [*PARENT_FILES, 'rate_selection.json']:
        (root / name).write_bytes((previous / name).read_bytes())
    model = json.loads((previous / 'inventory_model.json').read_text())
    model.update(format=MinuteInventoryModels.format, features=MinuteInventoryModels.features)
    model['mean'] += [0.]*8
    model['scale'] += [1.]*8
    model['coef'] = [row+[0.]*8 for row in model['coef']]
    save_json(root / 'inventory_model.json', model)
    size = json.loads((previous / 'reduction_model.json').read_text())
    size.update(format=MinuteReductionModel.format, features=MinuteReductionModel.features)
    size['mean'] += [0.]*8
    size['scale'] += [1.]*8
    size['coef'] += [0.]*8
    save_json(root / 'reduction_model.json', size)
    (root / 'inventory_calibration.json').write_bytes((previous / 'inventory_calibration.json').read_bytes())
    frozen = {**old, 'protocol': 'minute_inventory_micro_v21',
        'inventory_files_sha256': {n: sha256(root / n) for n in INVENTORY_FILES},
        'previous_inventory_sha256': sha256(root / 'previous_inventory/frozen_selection.json')}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def ready_bars():
    return bars().assign(**dict.fromkeys(MINUTE_FEATURES, 0.))


@pytest.mark.parametrize('damage', ['previous', 'model', 'shared_parent', 'scale', 'risk'])
def test_minute_inventory_model_chain_rejects_tampering(tmp_path, damage):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'previous')
    assert isinstance(load_selection(root)[1], MinuteInventoryPolicy)
    if damage == 'previous':
        (root / 'previous_inventory/frozen_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'reduction_model.json').write_text('{}')
    elif damage == 'shared_parent':
        (root / 'rate_selection.json').write_text('{}')
    elif damage == 'scale':
        frozen['inventory_scales']['reduce'] *= 2
    else:
        frozen['risk']['max_adds'] += 1
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_crash_preserves_context_and_period_guard(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'previous')
    guard_replay_period(root, frozen, '2021-01-01', '2022-01-01')
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2026-09-01', '2026-11-01')
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path, {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_conditions_keep_previous_v20_results_exact(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'previous')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    def prepared(*_, **kwargs):
        assert kwargs == {'minute_inputs': True}
        return ready_bars(), {}
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', prepared)
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic_scope': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'eval', 'seen_2026', ['BTCUSDT'])
    assert len(json.loads((out / 'results.json').read_text())) == 11
    assert isinstance(load_selection(out)[1], MinuteInventoryPolicy)
    _, previous = load_selection(root / 'previous_inventory')
    backtest(bars(), previous, EngineConfig(**frozen['risk']), tmp_path / 'original')
    for file in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'original' / file),
            pd.read_parquet(out / 'BTCUSDT/previous_v20' / file), check_exact=True)
    assert json.loads((tmp_path / 'original/final_state.json').read_text()) == json.loads((out / 'BTCUSDT/previous_v20/final_state.json').read_text())


def test_full_selection_keeps_training_rows_and_new_features(tmp_path, monkeypatch):
    root, labels = tmp_path / 'previous', tmp_path / 'labels'
    previous_selection(root)
    labels.mkdir()
    save_json(labels / 'manifest.json', {'settings': {'inventory_files_sha256': 'synthetic_original'}})
    (labels / 'files.json').write_text('{}')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    frame, ledger = source_frame()
    rng = np.random.default_rng(11)
    for name in MINUTE_FEATURES:
        frame[name] = rng.normal(size=len(frame))
        frame[f'directional_{name}'] = frame[name]*frame.direction
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.verify_files', lambda *_: None)
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.load_minute_inventory_labels', lambda *_: (frame, ledger))
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_minute_inventory_selection(root, labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, policy = load_selection(out)
    assert isinstance(policy, MinuteInventoryPolicy) and len(policy.manager.features) == len(policy.size_model.features) == 44
    assert frozen['candidate'] == 0 and frozen['development_in_sample']
    assert {r['period'] for r in json.loads((out / 'development.json').read_text())} == {'2021_in_sample'}
    assert json.loads((out / 'training_support.json').read_text())['management']['export_max_error'] < 1e-12
    assert sha256(out / 'previous_inventory/frozen_selection.json') == sha256(root / 'frozen_selection.json')
