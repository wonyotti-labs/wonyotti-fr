import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_exit_horizon import source
from test_exit_state import fixture as exit_fixture
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import selection as history_selection
from test_minute_inventory_research import ready_bars

from wonyotti_fr.action_model import select_threshold
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.exit_horizon import (
    attach_exit_horizon,
    exit_offset_predict,
    fit_exit_offset,
    horizon_requests,
    horizon_splits,
)
from wonyotti_fr.horizon_exit_research import run_horizon_exit_selection
from wonyotti_fr.horizon_exit_state import (
    HORIZON_EXIT_FILES,
    HorizonExitPolicy,
    horizon_exit_comparison,
    prepare_horizon_exit,
    select_horizon_exit,
)
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.recent_history import copy_exit_parent


def fixture(tmp_path, monkeypatch):
    family = tmp_path/'family'
    family.mkdir()
    (family/'files.json').write_text('{}')
    def with_evidence(path):
        root, previous, frozen = history_selection(path)
        evidence = json.loads((root/'history_admission.json').read_text())
        evidence['files_sha256'] = sha256(family/'files.json')
        save_json(root/'history_admission.json', evidence)
        frozen['history_files_sha256']['history_admission.json'] = sha256(root/'history_admission.json')
        save_json(root/'frozen_selection.json', frozen)
        save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
        return root, previous, frozen
    monkeypatch.setattr('test_exit_state.history_selection', with_evidence)
    parent, activity, _, _, old = exit_fixture(tmp_path, monkeypatch)
    _, original = load_selection(parent)
    frame, reference = source(monkeypatch)
    targets = attach_exit_horizon(frame, pd.Timestamp('2020-01-13', tz='UTC'))
    rows, _, _ = horizon_splits(targets, reference)
    cp = original.manager.probabilities(rows['calibration'][original.manager.features])[:, 0]
    offset, _ = fit_exit_offset(cp, rows['calibration'].y_exit.to_numpy())
    threshold, _ = select_threshold(rows['calibration'].y_exit.to_numpy(), exit_offset_predict(offset, cp), 2.)
    vp = original.manager.probabilities(rows['diagnosis'][original.manager.features])[:, 0]
    requests, _ = horizon_requests(rows['diagnosis'], exit_offset_predict(offset, vp), threshold)
    diagnosis = tmp_path/'horizon-diagnosis'
    diagnosis.mkdir()
    save_json(diagnosis/'manifest.json', {'settings': {'source': str(family), 'source_files_sha256': sha256(family/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V51.md')), 'horizon_minutes': 15, 'trading_returns_evaluated': False}})
    save_json(diagnosis/'summary.json', {'complete': True})
    save_json(diagnosis/'offsets.json', {'original': offset.to_dict()})
    save_json(diagnosis/'thresholds.json', {'original': threshold})
    save_json(diagnosis/'requests.json', {'original': requests})
    save_json(diagnosis/'decision.json', {'exit_horizon_admitted': False})
    targets.to_parquet(diagnosis/'horizon_labels.parquet', index=False)
    for name, part in rows.items():
        part.to_parquet(diagnosis/f'{name}_used.parquet', index=False)
    save_json(diagnosis/'files.json', {p.name: sha256(p) for p in diagnosis.iterdir()})
    monkeypatch.setattr('wonyotti_fr.horizon_exit_state.load_exit_horizon_inputs', lambda _: (rows, {}, None, targets, original.manager))
    root = tmp_path/'horizon-exit'
    root.mkdir()
    copy_exit_parent(parent, root)
    prepare_horizon_exit(parent, diagnosis, original, root)
    frozen = {**old, 'protocol': 'horizon_exit_v52', 'exit_selection_sha256': sha256(root/'exit_selection.json'),
        'horizon_exit_files_sha256': {n: sha256(root/n) for n in HORIZON_EXIT_FILES}}
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    return root, parent, activity, diagnosis, frozen


def test_raw_threshold_matches_monotone_calibration_and_future_cannot_choose_it(monkeypatch):
    frame, reference = source(monkeypatch)
    frame['ret_5m'] = .1 + .4*(np.arange(len(frame)) % 60 >= 35)
    reference = {n: frame[frame.end.isin(part.end)].reset_index(drop=True) for n, part in reference.items()}
    rows, _, _ = horizon_splits(attach_exit_horizon(frame, pd.Timestamp('2020-01-13', tz='UTC')), reference)
    manager = SimpleNamespace(features=['ret_5m'], probabilities=lambda x: np.repeat(np.asarray(x)[:, :1], 3, axis=1))
    original = SimpleNamespace(manager=manager, thresholds={'exit': .9})
    cp = manager.probabilities(rows['calibration'][manager.features])[:, 0]
    offset, _ = fit_exit_offset(cp, rows['calibration'].y_exit.to_numpy())
    expected, _ = select_threshold(rows['calibration'].y_exit, exit_offset_predict(offset, cp), 2.)
    choices, checks = select_horizon_exit(rows, original, offset, expected)
    assert choices['threshold'] == .5 and all(checks['comparison'].values())
    assert checks['requests']['candidate']['detected_events'] > checks['requests']['previous_v48']['detected_events']
    bad = copy.deepcopy(rows)
    bad['diagnosis']['ret_5m'] = .1
    # 뒤쪽 진단은 허용 판정에만 쓰이며 상반기 문턱 재선택에 쓰지 않는다.
    future, _ = select_horizon_exit(bad, original, offset, expected)
    assert future == choices
    with pytest.raises(ValueError, match='단조'):
        select_horizon_exit(rows, original, offset, 1.)


@pytest.mark.parametrize('damage', ['parent', 'risk', 'model', 'plan', 'threshold', 'horizon', 'period', 'support', 'source', 'comparison', 'equal'])
def test_loader_rejects_changed_inputs_or_failed_conditions(tmp_path, monkeypatch, damage):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'parent':
        (root/'exit_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'model':
        (root/'history_manager.json').write_text('{}')
    else:
        threshold = damage in {'threshold', 'horizon', 'period', 'support'}
        name = HORIZON_EXIT_FILES[0 if threshold else 1]
        value = json.loads((root/name).read_text())
        if damage == 'threshold':
            value['threshold'] = 1.1
        elif damage == 'horizon':
            value['horizon_minutes'] = 5
        elif damage == 'period':
            value['calibration_period'] = ['2021-07-01', '2022-01-01']
        elif damage == 'support':
            value['support']['actual_positive'] = 19
        elif damage == 'plan':
            value['protocol_sha256'] = '0'*64
        elif damage == 'source':
            value['source_files_sha256'] = '0'*64
        elif damage == 'comparison':
            value['comparison']['f2_preserved'] = False
        else:
            value['all_calibration_and_diagnosis_requests_equal'] = False
        save_json(root/name, value)
        frozen['horizon_exit_files_sha256'][name] = sha256(root/name)
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


def test_eleven_conditions_keep_whole_previous_outputs_and_risk(tmp_path, monkeypatch):
    root, parent, activity, _, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path/'runs', 'seen_2026', ['BTCUSDT'])
    names = {r['strategy'] for r in json.loads((out/'results.json').read_text())}
    assert len(names) == 11 and {'previous_v48', 'previous_v46', 'unfiltered_v14'} <= names
    assert 'previous_v40' not in names and isinstance(load_selection(out)[1], HorizonExitPolicy)
    for label, previous in [('previous_v48', parent), ('previous_v46', activity)]:
        frozen, policy = load_selection(previous)
        expected = tmp_path/label
        backtest(ready_bars(), policy, EngineConfig(**frozen['risk']), expected)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(expected/name), pd.read_parquet(out/f'BTCUSDT/{label}'/name), check_exact=True)
        assert json.loads((expected/'final_state.json').read_text()) == json.loads((out/f'BTCUSDT/{label}/final_state.json').read_text())
        assert json.loads((out/f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']


def test_full_selection_keeps_all_models_and_future_market_cannot_change_threshold(tmp_path, monkeypatch):
    _, parent, _, diagnosis, _ = fixture(tmp_path, monkeypatch)
    old, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent/'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path/n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.horizon_exit_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_horizon_exit_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'runs')
    frozen, bot = load_selection(out)
    assert frozen['risk'] == old['risk']
    assert bot.manager.model.to_dict() == original.manager.model.to_dict() and bot.manager.offset.to_dict() == original.manager.offset.to_dict()
    for attr in ['size_model']:
        assert getattr(bot, attr).to_dict() == getattr(original, attr).to_dict()
    for attr in ['activity', 'direction']:
        assert getattr(bot.base, attr).to_dict() == getattr(original.base, attr).to_dict()
    for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json', 'first_policy_thresholds.json']:
        assert (out/name).read_bytes() == (parent/name).read_bytes()
    for flags in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]:
        state = {'_fill_features': np.array([flags[0], 0., 0., flags[1], 0., 0.])}
        for action in ['increase', 'reduce']:
            assert bot.action_threshold(action, state) == original.action_threshold(action, state)
        assert bot.action_threshold('exit', state) == bot.horizon_threshold
    assert json.loads((out/'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    def changed_market(*args, **kwargs):
        bars = ready_bars()
        if args[3] == '2022-01-01':
            bars[['open', 'high', 'low', 'close']] *= 2.
        return bars, {}
    monkeypatch.setattr('wonyotti_fr.horizon_exit_research.prepare_minute_period', changed_market)
    future = run_horizon_exit_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path/'future')
    for name in HORIZON_EXIT_FILES:
        assert (out/name).read_bytes() == (future/name).read_bytes()
    requests = json.loads((out/'horizon_exit_evidence.json').read_text())['requests']
    requests['candidate']['event_recall'] = -1
    with pytest.raises(ValueError):
        horizon_exit_comparison(requests)


@pytest.mark.parametrize('delay,liquid', [(0, True), (1, True), (0, False), (1, False)])
def test_changed_exit_threshold_obeys_liquidity_delay_accounting_and_reentry(delay, liquid):
    from dataclasses import replace

    from test_activity_ablation import bots

    from wonyotti_fr.engine import TradingEngine

    _, parent = bots(manager=(.03, 0., 0.))
    bot = HorizonExitPolicy(parent, .02)
    risk = replace(EngineConfig(), bar_seconds=60, signal_delay_bars=delay, max_hold_bars=50,
        stop_fraction=0., entry_fraction=.125, addition_fraction=.25, max_adds=5, cooldown_bars=0)
    engine = TradingEngine(risk)
    fills = []
    for i, bar in enumerate(list(iter_events(ready_bars()))[:40]):
        close = 100.-.2*i
        event = {**bar, 'open': 100., 'close': close, 'high': 100., 'low': close, 'count': 1, 'volume': 100.}
        if not liquid and i in [7+2*delay, 8+2*delay]:
            event.update(count=0, volume=0., open=close, high=close, low=close)
        result = engine.step(event, bot)
        fills.extend(result['fills'])
        assert abs(result['accounting_residual']) < 1e-7
        if not event['count']:
            assert not result['fills']
        snapshot = engine.snapshot()
        engine = TradingEngine(risk, copy.deepcopy(snapshot))
        assert engine.snapshot() == snapshot
    reasons = [f['reason'] for f in fills]
    assert reasons.count('entry') >= 2 and reasons.count('signal_exit') >= 2
    first, exited = fills[:2]
    assert exited['delta_quantity'] == -first['delta_quantity']
    assert abs(first['delta_quantity'])*first['price']+first['fee'] == pytest.approx(
        risk.initial_equity*risk.allocation*risk.entry_fraction)
    if not liquid:
        assert pd.Timestamp(exited['time']) == pd.Timestamp(first['time'])+pd.Timedelta(minutes=3+delay)
