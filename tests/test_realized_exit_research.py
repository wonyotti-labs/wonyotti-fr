import json

import numpy as np
import pandas as pd
import pytest
from test_inventory_recent import source_frame
from test_minute_inventory_research import previous_selection, ready_bars
from test_minute_inventory_research import selection as minute_selection

from wonyotti_fr.addition_research import copy_minute_parent
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.minute_inputs import MINUTE_FEATURES
from wonyotti_fr.minute_inventory import MinuteInventoryPolicy
from wonyotti_fr.minute_inventory_research import run_minute_inventory_selection
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.realized_exit_research import EXIT_MODEL_FILES, run_realized_exit_selection


def selection(root, parent, previous):
    old = minute_selection(parent, previous)
    root.mkdir()
    copy_minute_parent(parent, root)
    model = json.loads((root / 'inventory_model.json').read_text())
    model['intercept'][0] -= .5
    save_json(root / 'realized_exit_model.json', model)
    calibration = json.loads((root / 'inventory_calibration.json').read_text())
    save_json(root / 'realized_exit_calibration.json', calibration)
    save_json(root / 'realized_exit_training.json', {'training_period': TRAINING_PERIODS[1],
        'calibration_period': CALIBRATION_PERIODS[1], 'non_exit_models_exact': True})
    frozen = {**old, 'protocol': 'realized_exit_v24', 'minute_selection_sha256': sha256(root / 'minute_selection.json'),
        'exit_scales': calibration['scales'], 'exit_thresholds': calibration['thresholds'],
        'exit_files_sha256': {n: sha256(root / n) for n in EXIT_MODEL_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('damage', ['parent', 'model', 'non_exit_model', 'scale', 'risk', 'period'])
def test_exit_only_chain_rejects_tampering(tmp_path, damage):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'previous')
    assert isinstance(load_selection(root)[1], MinuteInventoryPolicy)
    if damage == 'parent':
        (root / 'minute_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'realized_exit_model.json').write_text('{}')
    elif damage == 'non_exit_model':
        p = root / 'realized_exit_model.json'
        model = json.loads(p.read_text())
        model['coef'][1][0] += 1
        save_json(p, model)
        frozen['exit_files_sha256'][p.name] = sha256(p)
    elif damage == 'period':
        p = root / 'realized_exit_training.json'
        meta = json.loads(p.read_text())
        meta['training_period'][0] = '2019-01-01'
        save_json(p, meta)
        frozen['exit_files_sha256'][p.name] = sha256(p)
    elif damage == 'scale':
        frozen['exit_scales']['reduce'] *= 2
    else:
        frozen['risk']['max_adds'] = 0
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_exit_only_policy_process_crash_and_period_guard(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'previous')
    report = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2021-01-01')


def test_eleven_conditions_preserve_original_exit_model_control(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'previous')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and 'previous_v21' in strategies and 'inventory_fixed_size' not in strategies
    _, original = load_selection(tmp_path / 'parent')
    _, current = load_selection(out)
    assert original.manager.to_dict()['intercept'][0] != current.manager.to_dict()['intercept'][0]
    assert original.size_model.to_dict() == current.size_model.to_dict()
    backtest(ready_bars(), original, EngineConfig(**frozen['risk']), tmp_path / 'baseline')
    for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT/previous_v21' / n),
                                      pd.read_parquet(tmp_path / 'baseline' / n), check_exact=True)
    assert json.loads((out / 'BTCUSDT/previous_v21/final_state.json').read_text()) == json.loads((tmp_path / 'baseline/final_state.json').read_text())


def test_refit_changes_only_exit_and_preserves_training_and_baseline(tmp_path, monkeypatch):
    previous, labels = tmp_path / 'previous', tmp_path / 'labels'
    previous_selection(previous)
    labels.mkdir()
    save_json(labels / 'manifest.json', {'settings': {'inventory_files_sha256': 'synthetic_original'}})
    (labels / 'files.json').write_text('{}')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    frame, ledger = source_frame()
    rng = np.random.default_rng(24)
    for name in MINUTE_FEATURES:
        frame[name] = rng.normal(size=len(frame))
        frame[f'directional_{name}'] = frame[name] * frame.direction
    frame['exit_count'] = frame.y_exit
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.verify_files', lambda *_: None)
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.load_minute_inventory_labels', lambda *_: (frame, ledger))
    monkeypatch.setattr('wonyotti_fr.minute_inventory_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    parent = run_minute_inventory_selection(previous, labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'parent')
    new_labels = tmp_path / 'new-labels'
    new_labels.mkdir()
    save_json(new_labels / 'manifest.json', {'settings': {'minute_files_sha256': sha256(labels / 'files.json')}})
    (new_labels / 'files.json').write_text('{}')
    changed = frame.copy()
    changed['y_exit'] = np.arange(len(frame)) % 7 == 0
    changed['y_exit'] = changed.y_exit.astype(int)
    changed['exit_count'] = changed.y_exit
    monkeypatch.setattr('wonyotti_fr.realized_exit_research.load_realized_exit_labels', lambda *_: (changed, ledger))
    monkeypatch.setattr('wonyotti_fr.realized_exit_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_realized_exit_selection(parent, new_labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    _, original = load_selection(parent)
    _, current = load_selection(out)
    assert original.manager.to_dict()['coef'][1:] == current.manager.to_dict()['coef'][1:]
    assert original.manager.to_dict()['coef'][0] != current.manager.to_dict()['coef'][0]
    assert original.size_model.to_dict() == current.size_model.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    assert json.loads((out / 'realized_exit_training.json').read_text())['non_exit_frames_exact']
