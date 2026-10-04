import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_calibration_diagnostics import source
from test_first_state import fixed_budget  # noqa: F401
from test_first_state import selection as first_selection
from test_history_state import policy as history_policy
from test_minute_inventory_research import ready_bars

from wonyotti_fr.calibration_diagnostics import PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, PolicyDecision, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.first_state import FirstStatePolicy
from wonyotti_fr.holding_support import copy_first_parent, holding_support
from wonyotti_fr.holding_support_research import (
    prepare_holding_support,
    run_holding_support_selection,
)
from wonyotti_fr.minute_management import purged_window
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def rows():
    frame = source()
    frame.entry_time = frame.end - pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1)
    frame.log_hold_minutes = np.log1p((frame.end - frame.entry_time).dt.total_seconds()/60)
    return {name: purged_window(frame, *period) for name, period in PERIODS.items()}


def fixture(tmp_path):
    parent, history, old = first_selection(tmp_path)
    diagnosis = tmp_path / 'diagnosis'
    diagnosis.mkdir()
    for name, frame in rows().items():
        frame.to_parquet(diagnosis / f'{name}_used.parquet', index=False)
    save_json(diagnosis / 'summary.json', {'complete': True})
    for name, original in [('models.json', 'history_manager.json'), ('offsets.json', 'history_offset.json')]:
        save_json(diagnosis / name, {'histogram': json.loads((parent / original).read_text())})
    save_json(diagnosis / 'files.json', {p.name: sha256(p) for p in diagnosis.iterdir()})
    evidence = json.loads((parent / 'first_admission.json').read_text())
    evidence['settings']['diagnosis_files_sha256'] = sha256(diagnosis / 'files.json')
    save_json(parent / 'first_admission.json', evidence)
    old['first_files_sha256']['first_admission.json'] = sha256(parent / 'first_admission.json')
    save_json(parent / 'frozen_selection.json', old)
    save_json(parent / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(parent / 'frozen_selection.json')})
    root = tmp_path / 'holding'
    root.mkdir()
    copy_first_parent(parent, root)
    support = prepare_holding_support(diagnosis, parent, root)
    frozen = {**old, 'protocol': 'holding_support_v40', 'risk': {**old['risk'], 'max_hold_bars': support['max_hold_bars']},
        'first_selection_sha256': sha256(root / 'first_selection.json'), 'holding_support_sha256': sha256(root / 'holding_support.json')}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, history, diagnosis, frozen


def test_nanosecond_floor_and_no_diagnosis_or_outcome_selection():
    splits = rows()
    value = holding_support(splits)
    assert value['max_hold_bars'] == 5
    assert value['support']['training']['max_duration_ns'] == 5*60*10**9+1
    splits['diagnosis'].entry_time = splits['diagnosis'].end-pd.Timedelta(days=100)
    for name in ['training', 'calibration']:
        splits[name]['y_exit'] = 1-splits[name].y_exit
        splits[name]['net_pnl'] = -1e9
    assert holding_support(splits) == value


@pytest.mark.parametrize('damage', ['log', 'nan', 'future_entry', 'nan_time', 'crossing', 'duplicate', 'overlap', 'future_label'])
def test_holding_support_rejects_inconsistent_rows(damage):
    splits = rows()
    frame = splits['calibration']
    i = frame.index[0]
    if damage == 'log':
        frame.loc[i, 'log_hold_minutes'] += .1
    elif damage == 'nan':
        frame.loc[i, 'log_hold_minutes'] = np.nan
    elif damage == 'future_entry':
        frame.loc[i, 'entry_time'] = frame.loc[i, 'end']+pd.Timedelta(minutes=1)
    elif damage == 'nan_time':
        frame.loc[i, 'entry_time'] = pd.NaT
    elif damage == 'crossing':
        frame.loc[i, 'entry_time'] = pd.Timestamp('2020-12-31', tz='UTC')
    elif damage == 'duplicate':
        frame.loc[frame.index[1], 'end'] = frame.loc[i, 'end']
    elif damage == 'overlap':
        frame.loc[i, 'episode_id'] = splits['training'].episode_id.iloc[0]
    else:
        frame.loc[i, 'label_end'] = pd.Timestamp('2021-07-01', tz='UTC')
    with pytest.raises(ValueError):
        holding_support(splits)


@pytest.mark.parametrize('damage', ['cap', 'risk', 'parent', 'evidence', 'source'])
def test_loader_rejects_changed_cap_other_risk_parent_or_support(tmp_path, damage):
    root, _, _, _, frozen = fixture(tmp_path)
    if damage == 'cap':
        frozen['risk']['max_hold_bars'] += 1
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'parent':
        (root / 'first_selection.json').write_text('{}')
    else:
        value = json.loads((root / 'holding_support.json').read_text())
        if damage == 'evidence':
            value['max_hold_bars'] += 1
        else:
            value['source']['files_sha256'] = '0'*64
        save_json(root / 'holding_support.json', value)
        frozen['holding_support_sha256'] = sha256(root / 'holding_support.json')
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('delay,liquid', [(0, True), (1, True), (0, False), (1, False)])
def test_cap_boundary_blocks_management_until_tradable_close_and_restores(delay, liquid):
    bot = FirstStatePolicy(history_policy([0., 0., 0.]), {'reduce': .2, 'increase': .2})
    config = replace(EngineConfig(), bar_seconds=60, signal_delay_bars=delay, max_hold_bars=5,
        stop_fraction=0., cooldown_bars=15, entry_fraction=.125, addition_fraction=.25, allow_adverse_add=True)
    engine = TradingEngine(config)
    held, fills, last = [], [], None
    close_index = 1+delay+config.max_hold_bars
    for i, event in enumerate(list(iter_events(ready_bars()))[:20]):
        event = {**event, 'open': 100., 'close': 100., 'high': 100., 'low': 100.}
        if not liquid and close_index <= i < close_index+2:
            event.update(count=0, volume=0.)
        def decide(bar, state, i=i):
            result = bot(bar, state)
            if state['direction']:
                held.append(state['hold_bars'])
            return PolicyDecision('enter_long', result.state, 'synthetic') if i == 0 else result
        result = engine.step(event, decide)
        fills.extend(result['fills'])
        snapshot = engine.snapshot()
        restored = TradingEngine(config, copy.deepcopy(snapshot))
        assert restored.snapshot() == snapshot
        engine = restored
        if not liquid and close_index <= i < close_index+2:
            assert snapshot['liquidity_exit_reason'] == 'time_limit'
            assert not result['fills'] and snapshot['pending'] == 'hold'
        last = event
    assert max(held) == config.max_hold_bars
    assert [f['reason'] for f in fills] == ['entry', 'time_limit' if liquid else 'liquidity_time_limit']
    assert pd.Timestamp(fills[-1]['time'])-pd.Timestamp(fills[0]['time']) == pd.Timedelta(minutes=5 if liquid else 7)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.step(last, bot)
    assert engine.snapshot() == before


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_restores_holding_cap(tmp_path, kind):
    root, _, _, _, _ = fixture(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_variants_keep_previous_no_limit_risks_and_exact_outputs(tmp_path, monkeypatch):
    root, parent, history, _, _ = fixture(tmp_path)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v39', 'previous_v36', 'unfiltered_v14'} <= strategies
    assert load_selection(out)[0]['risk']['max_hold_bars'] == 5
    for label, reference in [('previous_v39', parent), ('previous_v36', history)]:
        frozen, bot = load_selection(reference)
        expected = tmp_path / label
        backtest(ready_bars(), bot, EngineConfig(**frozen['risk']), expected)
        assert json.loads((out / f'BTCUSDT/{label}/config.json').read_text())['max_hold_bars'] == 0
        for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / f'BTCUSDT/{label}' / n), pd.read_parquet(expected / n), check_exact=True)
        assert json.loads((out / f'BTCUSDT/{label}/final_state.json').read_text()) == json.loads((expected / 'final_state.json').read_text())
    assert json.loads((out / 'BTCUSDT/unfiltered_v14/config.json').read_text())['max_hold_bars'] == 0


def test_selection_preserves_models_thresholds_and_full_original_control(tmp_path, monkeypatch):
    _, parent, _, diagnosis, _ = fixture(tmp_path)
    old, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.holding_support_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_holding_support_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, bot = load_selection(out)
    assert frozen['risk'] == {**old['risk'], 'max_hold_bars': 5}
    assert bot.multiplier == original.multiplier and bot.scales == original.scales
    assert bot.first_thresholds == original.first_thresholds and bot.thresholds == original.thresholds
    assert bot.manager.model.to_dict() == original.manager.model.to_dict()
    assert bot.manager.offset.to_dict() == original.manager.offset.to_dict()
    assert bot.size_model.to_dict() == original.size_model.to_dict()
    assert bot.base.direction.to_dict() == original.base.direction.to_dict()
    assert bot.base.activity.to_dict() == original.base.activity.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
