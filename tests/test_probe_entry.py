import json
from dataclasses import replace

import pandas as pd
import pytest
from test_boosted_direction import selection as boosted_selection
from test_engine import bar
from test_lifecycle import frozen_selection
from test_minute_inventory_research import ready_bars

from wonyotti_fr.action_research import load_action_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.probe_entry import copy_boosted_parent, probe_risks
from wonyotti_fr.probe_entry_research import run_probe_entry_selection
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


@pytest.fixture(autouse=True)
def fixed_source_addition_budget(monkeypatch):
    monkeypatch.setattr('test_action_replay.frozen_selection',
                        lambda root: frozen_selection(root, addition_fraction=.25))


def selection(tmp_path):
    parent, _, old = boosted_selection(tmp_path)
    root = tmp_path / 'probe'
    root.mkdir()
    copy_boosted_parent(parent, root)
    risk, _, _ = probe_risks(old)
    frozen = {**old, 'protocol': 'probe_entry_v29', 'risk': risk.__dict__,
              'boost_selection_sha256': sha256(root / 'boost_selection.json')}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, frozen


def test_first_fill_matches_scaled_control_and_addition_retains_original_budget(tmp_path):
    _, _, parent = boosted_selection(tmp_path)
    probe, original, matched = probe_risks(parent)
    configs = [replace(risk, bar_seconds=300, fee_bps=0, slippage_bps=0) for risk in [probe, original, matched]]
    def actions(event, state):
        return 'enter_long' if event['step'] == 0 else 'increase'
    outputs = []
    for config in configs:
        engine = TradingEngine(config)
        outputs.append([engine.step(bar(i), actions) for i in range(9)])
    assert outputs[0][1]['quantity'] == outputs[2][1]['quantity'] == outputs[1][1]['quantity'] / 4
    first_add = [rows[2]['fills'][0]['delta_quantity'] for rows in outputs]
    assert first_add[0] == first_add[1] == 4 * first_add[2]
    for rows, config in zip(outputs, configs, strict=True):
        assert all(row['quantity'] * 100 <= row['equity'] * config.allocation + 1e-8 for row in rows)
        assert all(abs(row['accounting_residual']) < 1e-8 for row in rows)


@pytest.mark.parametrize('damage', ['parent', 'entry_fraction', 'allocation', 'addition_fraction', 'stop_fraction'])
def test_probe_rejects_changes_outside_fixed_entry_budget(tmp_path, damage):
    root, _, frozen = selection(tmp_path)
    if damage == 'parent':
        (root / 'boost_selection.json').write_text('{}')
    else:
        frozen['risk'][damage] *= .5
        save_json(root / 'frozen_selection.json', frozen)
        save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_small_entry_actual_process_recovery(tmp_path, kind):
    root, _, _ = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_probe_eleven_conditions_preserve_original_risk_for_baselines(tmp_path, monkeypatch):
    root, parent, _ = selection(tmp_path)
    old, original = load_selection(parent)
    _, old_risk, matched_risk = probe_risks(old)
    _, v14 = load_action_selection(root, json.loads((root / 'rate_selection.json').read_text()))
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v28', 'matched_initial_risk', 'unfiltered_v14'} <= strategies
    assert not {'previous_v26', 'previous_v27'} & strategies
    for name, policy, risk in [('previous_v28', original, old_risk),
        ('matched_initial_risk', original, matched_risk), ('unfiltered_v14', v14, old_risk)]:
        backtest(ready_bars(), policy, risk, tmp_path / name)
        for filename in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT' / name / filename),
                                          pd.read_parquet(tmp_path / name / filename), check_exact=True)
        assert json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text()) == json.loads((tmp_path / name / 'final_state.json').read_text())


def test_three_development_runs_and_confirmation_without_model_changes(tmp_path, monkeypatch):
    _, parent, _ = selection(tmp_path)
    old, original = load_selection(parent)
    _, old_risk, _ = probe_risks(old)
    backtest(ready_bars(), original, old_risk, parent / 'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.probe_entry_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_probe_entry_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, policy = load_selection(out)
    assert len(json.loads((out / 'comparison.json').read_text())) == 4
    assert frozen['risk']['entry_fraction'] == .125
    assert policy.base.direction.to_dict() == original.base.direction.to_dict()
    assert policy.base.activity.to_dict() == original.base.activity.to_dict()
    assert policy.manager.to_dict() == original.manager.to_dict()
    assert policy.size_model.to_dict() == original.size_model.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
