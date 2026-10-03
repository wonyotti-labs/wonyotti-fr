import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_inventory_management import model_data
from test_pullback_evaluation import bars
from test_rate_policy import rate_selection

from wonyotti_fr.action_research import load_action_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.inventory_management import InventoryActionModels, InventoryRatePolicy
from wonyotti_fr.inventory_research import INVENTORY_FILES, fixed_size_control
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def inventory_selection(root):
    frozen = rate_selection(root)
    (root / 'rate_selection.json').write_bytes((root / 'frozen_selection.json').read_bytes())
    model = json.loads((root / 'action_model.json').read_text())
    model.update(format=InventoryActionModels.format, features=InventoryActionModels.features)
    model['mean'].append(0.)
    model['scale'].append(1.)
    model['coef'] = [row+[0.] for row in model['coef']]
    model['intercept'] = [float(np.log(.01/.99)), float(np.log(.25/.75)), -10.]
    save_json(root / 'inventory_model.json', model)
    save_json(root / 'reduction_model.json', model_data(.8))
    calibration = {'scales': frozen['rate_scales'], 'thresholds': frozen['thresholds']}
    save_json(root / 'inventory_calibration.json', calibration)
    frozen.update(protocol='minute_inventory_v19', candidate=0, size_mode='learned',
        rate_selection_sha256=sha256(root / 'rate_selection.json'),
        inventory_files_sha256={n: sha256(root / n) for n in INVENTORY_FILES},
        inventory_scales=copy.deepcopy(calibration['scales']), inventory_thresholds=calibration['thresholds'])
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('damage', ['parent', 'model', 'scale', 'risk', 'size_mode'])
def test_inventory_frozen_chain_and_models_reject_changes(tmp_path, damage):
    root = tmp_path / 'selection'
    frozen = inventory_selection(root)
    assert isinstance(load_selection(root)[1], InventoryRatePolicy)
    if damage == 'parent':
        (root / 'rate_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'reduction_model.json').write_text('{}')
    elif damage == 'scale':
        frozen['inventory_scales']['reduce'] *= 2
    elif damage == 'risk':
        frozen['risk']['fee_bps'] = 0
    else:
        frozen['size_mode'] = 'fixed'
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_inventory_policy_actual_process_crash_and_period_guard(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = inventory_selection(root)
    guard_replay_period(root, frozen, '2021-01-01', '2022-01-01')
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2026-09-01', '2026-11-01')
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_ten_evaluation_conditions_preserve_original_and_new_fixed_size_controls(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    frozen = inventory_selection(root)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_: (bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic_scope': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'eval', 'seen_2026', ['BTCUSDT'])
    assert isinstance(load_selection(out)[1], InventoryRatePolicy)
    assert len(json.loads((out / 'results.json').read_text())) == 10
    _, policy = load_selection(root)
    _, original = load_action_selection(root, json.loads((root / 'rate_selection.json').read_text()))
    for name, bot in [('unfiltered_v14', original), ('inventory_fixed_size', fixed_size_control(policy))]:
        dest = tmp_path / name
        backtest(bars(), bot, EngineConfig(**frozen['risk']), dest)
        for file in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(dest / file), pd.read_parquet(out / 'BTCUSDT' / name / file), check_exact=True)
        assert json.loads((dest / 'final_state.json').read_text()) == json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text())
    fills = pd.read_parquet(out / 'BTCUSDT/fixed_policy/fills.parquet')
    assert fills.reason.eq('signal_reduce').any()
