import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import history_inputs
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot as net_bot
from test_net_exit_state import fixture as net_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.exit_move_research import run_exit_move_selection
from wonyotti_fr.exit_move_state import (
    EVENT_COLUMNS,
    EXIT_MOVE_FILES,
    ExitMovePolicy,
    copy_net_parent,
    prepare_exit_move,
    summarize_exit_moves,
)
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def fixture(tmp_path, monkeypatch):
    parent, _, previous, old = net_fixture(tmp_path, monkeypatch)
    from wonyotti_fr.horizon_exit_state import load_exit_horizon_inputs
    monkeypatch.setattr('wonyotti_fr.exit_move_state.load_exit_horizon_inputs', load_exit_horizon_inputs)
    diagnosis = tmp_path/'horizon-diagnosis'
    root = tmp_path/'exit-move'
    root.mkdir()
    copy_net_parent(parent, root)
    prepare_exit_move(parent, diagnosis, load_selection(parent)[1], root)
    frozen = {**old, 'protocol': 'exit_move_v54', 'net_exit_selection_sha256': sha256(root/'net_exit_selection.json'),
        'exit_move_files_sha256': {n: sha256(root/n) for n in EXIT_MOVE_FILES}}
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    return root, parent, previous, diagnosis, frozen


def events():
    times = pd.date_range('2020-02-01', periods=23, freq='min', tz='UTC')
    return pd.DataFrame({'end': times, 'entry_time': pd.Timestamp('2020-01-31', tz='UTC'),
        'episode_id': [*range(1, 21), 20, 20, 20], 'favorable_move': [*(np.arange(1, 21)/100), .8, .9, 1.]})[EVENT_COLUMNS]


def test_episode_medians_keep_equal_contribution_and_event_order():
    value, episodes = summarize_exit_moves(events())
    assert value['threshold'] == pytest.approx(.105)
    assert value['positive_events'] == 23 and value['positive_episodes'] == 20
    assert episodes.loc[episodes.episode_id.eq(20), 'median'].iloc[0] == pytest.approx(.85)
    assert value['threshold'] != events().favorable_move.median()


@pytest.mark.parametrize('damage', ['few_events', 'few_episodes', 'future', 'duplicate', 'negative', 'nan', 'identity'])
def test_invalid_price_move_source_is_rejected(damage):
    frame = events()
    if damage == 'few_events':
        frame = frame.iloc[:19]
    elif damage == 'few_episodes':
        frame['episode_id'] = 1
    elif damage == 'future':
        frame['end'] += pd.Timedelta(days=366)
    elif damage == 'duplicate':
        frame.loc[1, 'end'] = frame.end.iloc[0]
    elif damage == 'negative':
        frame.loc[0, 'favorable_move'] = -.1
    elif damage == 'nan':
        frame.loc[0, 'favorable_move'] = np.nan
    else:
        frame.loc[22, 'entry_time'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        summarize_exit_moves(frame)


@pytest.mark.parametrize('net,move,scores,intent', [(1., .02, (.03, 0., 0.), 'exit'),
    (0., .02, (.03, 0., 0.), 'exit'), (1., .0199, (.03, 0., 0.), 'hold'),
    (-.01, .03, (.03, 0., 0.), 'hold'), (-1., -.01, (.6, 0., 0.), 'exit'),
    (1., .0199, (.03, .8, .9), 'reduce'), (1., .0199, (.03, 0., .8), 'increase')])
def test_current_move_and_cost_guard_preserve_old_exits_and_action_priority(net, move, scores, intent):
    policy = ExitMovePolicy(net_bot(scores), .02)
    assert policy(*history_inputs(1, favorable_move=move, estimated_exit_net=net)).intent == intent


def test_invalid_current_move_or_floor_is_rejected():
    for bad in [None, True, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            ExitMovePolicy(net_bot(), bad)
        with pytest.raises(ValueError):
            ExitMovePolicy(net_bot(), .02)(*history_inputs(1, estimated_exit_net=1., favorable_move=bad))
    for bad in [0., -.1]:
        with pytest.raises(ValueError):
            ExitMovePolicy(net_bot(), bad)


@pytest.mark.parametrize('direction,delay', [(1, 0), (1, 1), (-1, 0), (-1, 1)])
def test_price_floor_crossing_preserves_delay_liquidity_and_accounting(direction, delay):
    risk = EngineConfig(bar_seconds=60, signal_delay_bars=delay, max_hold_bars=0, stop_fraction=0.,
        max_drawdown=1., daily_loss_limit=1., cooldown_bars=0)
    engine = TradingEngine(risk)
    engine.state['pending'] = 'enter_long' if direction == 1 else 'enter_short'
    policy = ExitMovePolicy(net_bot(), .02)
    first = history_inputs(1)[0]
    first['time'] = pd.Timestamp(first['end'])-pd.Timedelta(minutes=1)
    first['minute_features'] = first['minute_features'].tolist()
    fills = []
    for minute in range(6):
        price = 100.+direction*([1., 1.9, 3., 3., 3., 3.][minute])
        opening = 100. if minute == 0 else price
        liquid = minute not in [3, 4]
        event = {**first, 'time': (first['time']+pd.Timedelta(minutes=minute)).isoformat(),
            'end': (pd.Timestamp(first['end'])+pd.Timedelta(minutes=minute)).isoformat(),
            'open': opening, 'close': price, 'high': max(opening, price), 'low': min(opening, price),
            'count': int(liquid), 'volume': float(liquid), 'funding_rate': 0.}
        result = engine.step(event, policy)
        if minute == 1:
            assert result['next_intent'] == 'hold' and engine.view(price)['estimated_exit_net'] > 0
        if minute == 2:
            assert result['policy_event'] == 'action_exit'
        if not liquid:
            assert not result['fills']
        assert abs(result['accounting_residual']) < 1e-8
        fills.extend(result['fills'])
        engine = TradingEngine(risk, engine.snapshot())
    assert [f['reason'] for f in fills] == ['entry', 'signal_exit']
    assert pd.Timestamp(fills[-1]['time']) == first['time']+pd.Timedelta(minutes=5)
    assert result['closed_trades'][0]['net_pnl'] > 0


@pytest.mark.parametrize('damage', ['parent', 'risk', 'model', 'floor', 'source', 'events'])
def test_loader_rejects_changed_parent_or_source_summary(tmp_path, monkeypatch, damage):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'parent':
        (root/'net_exit_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'model':
        (root/'history_manager.json').write_text('{}')
    elif damage == 'events':
        name = EXIT_MOVE_FILES[1]
        rows = pd.read_parquet(root/name)
        rows.loc[0, 'favorable_move'] = -1
        rows.to_parquet(root/name, index=False)
        frozen['exit_move_files_sha256'][name] = sha256(root/name)
    else:
        name = EXIT_MOVE_FILES[0]
        value = json.loads((root/name).read_text())
        value['threshold' if damage == 'floor' else 'source_files_sha256'] = 1e-9
        save_json(root/name, value)
        frozen['exit_move_files_sha256'][name] = sha256(root/name)
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_process_exit_and_period_guard(tmp_path, monkeypatch, kind):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path/'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_eleven_variants_preserve_parent_and_original_risk(tmp_path, monkeypatch):
    root, parent, previous, _, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path/'runs', 'seen_2026', ['BTCUSDT'])
    names = {r['strategy'] for r in json.loads((out/'results.json').read_text())}
    assert len(names) == 11 and {'previous_v48', 'previous_v53', 'unfiltered_v14'} <= names
    assert 'previous_v52' not in names and isinstance(load_selection(out)[1], ExitMovePolicy)
    for label, reference in [('previous_v53', parent), ('previous_v48', previous)]:
        frozen, policy = load_selection(reference)
        expected = tmp_path/label
        backtest(ready_bars(), policy, EngineConfig(**frozen['risk']), expected)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(expected/name), pd.read_parquet(out/f'BTCUSDT/{label}'/name), check_exact=True)
        assert json.loads((expected/'final_state.json').read_text()) == json.loads((out/f'BTCUSDT/{label}/final_state.json').read_text())
        assert json.loads((out/f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']


def test_future_source_and_market_cannot_change_floor_or_models(tmp_path, monkeypatch):
    root, parent, _, diagnosis, _ = fixture(tmp_path, monkeypatch)
    old, original = load_selection(parent)
    from wonyotti_fr.exit_move_state import load_exit_horizon_inputs
    source = json.loads((diagnosis/'manifest.json').read_text())['settings']['source']
    loaded = load_exit_horizon_inputs(source)
    altered = copy.deepcopy(loaded)
    for name in ['calibration', 'diagnosis']:
        altered[0][name]['favorable_move'] = 1e6
        altered[0][name]['original_y_exit'] = 1-altered[0][name].original_y_exit
    monkeypatch.setattr('wonyotti_fr.exit_move_state.load_exit_horizon_inputs', lambda _: altered)
    changed = tmp_path/'changed'
    changed.mkdir()
    prepare_exit_move(parent, diagnosis, original, changed)
    for name in EXIT_MOVE_FILES:
        assert (root/name).read_bytes() == (changed/name).read_bytes()
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent/'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.exit_move_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_exit_move_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'runs')
    frozen, policy = load_selection(out)
    assert frozen['risk'] == old['risk']
    assert policy.manager.model.to_dict() == original.manager.model.to_dict()
    assert policy.manager.offset.to_dict() == original.manager.offset.to_dict()
    assert json.loads((out/'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    def changed_market(*args, **kwargs):
        bars = ready_bars()
        if args[3] == '2022-01-01':
            bars[['open', 'high', 'low', 'close']] *= 2.
        return bars, {}
    monkeypatch.setattr('wonyotti_fr.exit_move_research.prepare_minute_period', changed_market)
    future = run_exit_move_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'future')
    for name in [*EXIT_MOVE_FILES, 'history_manager.json', 'history_offset.json', 'history_thresholds.json',
                 'first_policy_thresholds.json', 'net_exit_selection.json']:
        assert (out/name).read_bytes() == (future/name).read_bytes()
