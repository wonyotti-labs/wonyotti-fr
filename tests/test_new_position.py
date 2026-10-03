import json

import numpy as np
import pandas as pd
import pytest
from test_minute_inventory_research import ready_bars
from test_recent_entry import selection as recent_selection
from test_recent_entry import source_data

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.new_position import POSITION_FILES, copy_recent_parent, new_position_targets
from wonyotti_fr.new_position_research import run_new_position_selection
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.recent_entry import ENTRY_PERIOD, RecentEntryPolicy


def input_frame():
    data, episodes = source_data()
    data['expansion_episode_id'] = data.target_episode_id
    data['expansion_count'] = data.active
    data['both_directions'] = False
    return data, episodes


def target_fixture():
    data, _ = input_frame()
    data = data.iloc[:4].copy()
    times = pd.date_range(data.end.iloc[0], periods=4, freq='5min', tz='UTC').as_unit('ns')
    data['end'], data['label_end'] = times, times + pd.Timedelta(minutes=5)
    data['usable'] = [True, True, True, False]
    events = [(-10, 'open', 0, 1, 1), (10, 'increase', 1, 2, 1),
        (300, 'reverse', 2, -1, 2), (301, 'reverse', -1, 1, 3),
        (600, 'close', 1, 0, 3), (900, 'open', 0, 1, 4)]
    actions = pd.DataFrame(events, columns=['seconds', 'action', 'before_qty', 'after_qty', 'episode_id'])
    actions['time'] = times[0] + pd.to_timedelta(actions.pop('seconds'), unit='s')
    return data, actions


def selection(root, parent, minute, previous):
    old = recent_selection(parent, minute, previous)
    root.mkdir()
    copy_recent_parent(parent, root)
    model = json.loads((root / 'recent_entry_models.json').read_text())['activity']
    model['intercept'] -= .1
    save_json(root / 'new_position_activity.json', model)
    save_json(root / 'new_position_training.json', {'training_period': ENTRY_PERIOD,
        'activity_quantile': .975, 'activity_threshold': .1, 'direction_unchanged': True})
    frozen = {**old, 'protocol': 'new_position_v25', 'recent_selection_sha256': sha256(root / 'recent_selection.json'),
        'position_activity_threshold': .1, 'position_files_sha256': {n: sha256(root / n) for n in POSITION_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def short_bars():
    frame = ready_bars()
    for name in ['open', 'high', 'low', 'close']:
        frame.loc[np.arange(len(frame)) % 5 == 0, name] = 100.2
    return frame


def test_add_only_and_close_are_not_new_entries_and_boundaries_are_preserved():
    data, actions = target_fixture()
    original = data.copy(deep=True)
    targets, ledger = new_position_targets(data, actions)
    assert targets.active.tolist() == [0, 1, 0, 1]
    assert targets.new_position_count.tolist() == [0, 2, 0, 1]
    assert targets.new_position_both_directions.tolist() == [False, True, False, False]
    assert targets.target_episode_id.tolist() == [0, 2, 0, 4]
    assert targets.buy.iloc[1] == 0 and not targets.usable.iloc[3]
    assert ledger.reason.tolist() == ['outside_windows', 'first_supported', 'later_event_in_window', 'unusable_original_window']
    pd.testing.assert_frame_equal(data, original, check_exact=True)
    # 같은 시각의 여러 시작도 원본 순서의 첫 사건만 정답으로 사용한다.
    actions.loc[3, 'time'] = actions.time.iloc[2]
    tied, tied_ledger = new_position_targets(data, actions)
    assert tied.target_episode_id.iloc[1] == 2 and tied.new_position_count.iloc[1] == 2
    assert len(tied_ledger) == len(ledger)


@pytest.mark.parametrize('damage', ['quantity', 'action', 'episode'])
def test_invalid_new_position_transition_is_rejected(damage):
    data, actions = target_fixture()
    if damage == 'quantity':
        actions.loc[2, 'before_qty'] = 9
    elif damage == 'action':
        actions.loc[2, 'action'] = 'reduce'
    else:
        actions.loc[3, 'episode_id'] = 2
    with pytest.raises(ValueError):
        new_position_targets(data, actions)


@pytest.mark.parametrize('damage', ['parent', 'model', 'risk', 'threshold', 'direction'])
def test_activity_only_chain_rejects_changed_parent_or_model(tmp_path, damage):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'minute', tmp_path / 'previous')
    assert isinstance(load_selection(root)[1], RecentEntryPolicy)
    if damage == 'parent':
        (root / 'recent_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'new_position_activity.json').write_text('{}')
    elif damage == 'direction':
        p = root / 'recent_entry_models.json'
        model = json.loads(p.read_text())
        model['direction']['intercept'] += 1
        save_json(p, model)
        frozen['entry_files_sha256'][p.name] = sha256(p)
    elif damage == 'risk':
        frozen['risk']['max_adds'] = 0
    else:
        frozen['position_activity_threshold'] = .9
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_crash_keeps_new_entry_activity_state(tmp_path, kind):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'minute', tmp_path / 'previous')
    result = verify_stress(list(iter_events(short_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2021-01-01')


def test_eleven_conditions_keep_v23_entry_and_v21_original_control(tmp_path, monkeypatch):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'minute', tmp_path / 'previous')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v21', 'previous_v23'} <= strategies and 'previous_v20' not in strategies
    _, current = load_selection(out)
    _, recent = load_selection(tmp_path / 'parent')
    assert current.base.direction.data == recent.base.direction.data and current.base.activity.data != recent.base.activity.data
    assert current.manager.to_dict() == recent.manager.to_dict()
    for name, original_root in [('previous_v23', tmp_path / 'parent'), ('previous_v21', tmp_path / 'minute')]:
        _, original = load_selection(original_root)
        backtest(short_bars(), original, EngineConfig(**frozen['risk']), tmp_path / name)
        for file in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT' / name / file),
                                          pd.read_parquet(tmp_path / name / file), check_exact=True)
        assert json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text()) == json.loads((tmp_path / name / 'final_state.json').read_text())


def test_full_training_only_refits_actual_new_position_activity(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    frozen = recent_selection(parent, tmp_path / 'minute', tmp_path / 'previous')
    _, original = load_selection(parent)
    backtest(short_bars(), original, EngineConfig(**frozen['risk']), parent / 'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    data, episodes = input_frame()
    rows = []
    for row in data.iloc[::10].itertuples():
        rows.extend([(row.end + pd.Timedelta(minutes=1), 'open', 0, 1, row.episode_id),
                     (row.end + pd.Timedelta(minutes=2), 'close', 1, 0, row.episode_id)])
    actions = pd.DataFrame(rows, columns=['time', 'action', 'before_qty', 'after_qty', 'episode_id'])
    monkeypatch.setattr('wonyotti_fr.new_position_research.source_inputs', lambda *_: ({'episodes': episodes, 'actions': actions}, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.new_position_research.make_expansion_data', lambda _: data)
    monkeypatch.setattr('wonyotti_fr.new_position_research.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    out = run_new_position_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    new, current = load_selection(out)
    train = pd.read_parquet(out / 'entry_training_used.parquet')
    assert len(train) == len(data) and train.active.sum() == len(data) // 10
    pd.testing.assert_frame_equal(train[MARKET_FEATURES], data[MARKET_FEATURES], check_exact=True)
    assert current.base.direction.data == original.base.direction.data
    assert current.size_model.to_dict() == original.size_model.to_dict()
    assert new['position_activity_threshold'] == np.quantile(current.base.activity.probabilities(train[MARKET_FEATURES].to_numpy()), .975)
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
