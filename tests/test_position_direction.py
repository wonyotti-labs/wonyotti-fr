import json

import pandas as pd
import pytest
from test_minute_inventory_research import ready_bars
from test_new_position import input_frame, short_bars
from test_position_prior import selection as prior_selection
from test_recent_entry import source_data

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.position_direction import (
    DIRECTION_FILES,
    DIRECTION_PERIOD,
    copy_prior_parent,
    position_direction_training,
)
from wonyotti_fr.position_direction_research import run_position_direction_selection
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def selection(root, parent, position, recent, minute, previous):
    old = prior_selection(parent, position, recent, minute, previous)
    _, original = load_selection(parent)
    root.mkdir()
    copy_prior_parent(parent, root)
    model = original.base.direction.to_dict()
    model['intercept'] = 2.
    save_json(root / 'new_position_direction.json', model)
    save_json(root / 'new_direction_training.json', {'training_period': DIRECTION_PERIOD,
        'direction_threshold': .65, 'prior_offset_applied': False,
        'model': {'rows': 2000, 'positive': 1000, 'negative': 1000}})
    frozen = {**old, 'protocol': 'position_direction_v27', 'prior_selection_sha256': sha256(root / 'prior_selection.json'),
        'direction_files_sha256': {n: sha256(root / n) for n in DIRECTION_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_direction_training_keeps_only_supported_starts_and_both_episode_boundaries():
    data, episodes = source_data()
    episodes.loc[episodes.episode_id.eq(1), 'entry_time'] = pd.Timestamp('2017-12-31', tz='UTC')
    episodes.loc[episodes.episode_id.eq(2), 'exit_time'] = pd.Timestamp('2022-01-02', tz='UTC')
    data.loc[20, 'target_episode_id'] = 1
    data.loc[21, 'target_episode_id'] = 2
    data.loc[22, 'end'] = pd.Timestamp('2018-01-01', tz='UTC')
    data.loc[23, 'label_end'] = pd.Timestamp('2021-12-31', tz='UTC')
    data.loc[24, 'usable'] = False
    train, ledger = position_direction_training(data, episodes)
    assert ledger.loc[:9, 'reason'].eq('left_episode_boundary').all()
    assert ledger.loc[10:19, 'reason'].eq('right_episode_boundary').all()
    assert ledger.loc[20:24, 'reason'].tolist() == ['left_episode_boundary', 'right_episode_boundary',
        'before_training_or_left_embargo', 'after_training_or_right_embargo', 'unusable_original_event']
    assert ledger.loc[26, 'reason'] == 'no_new_position'
    assert train.active.eq(1).all()
    pd.testing.assert_frame_equal(train, data.loc[ledger.reason.eq('included')], check_exact=True)


@pytest.mark.parametrize('damage', ['parent', 'model', 'period', 'support', 'prior', 'risk'])
def test_new_direction_chain_rejects_tampering_and_insufficient_support(tmp_path, damage):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'position', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    if damage == 'parent':
        (root / 'prior_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'new_position_direction.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['max_adds'] = 0
    else:
        p = root / 'new_direction_training.json'
        meta = json.loads(p.read_text())
        if damage == 'period':
            meta['training_period'][0] = '2020-01-01'
        elif damage == 'support':
            meta['model']['rows'] = 999
        else:
            meta['prior_offset_applied'] = True
        save_json(p, meta)
        frozen['direction_files_sha256'][p.name] = sha256(p)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_direct_direction_actual_crash_preserves_policy_state(tmp_path, kind):
    root = tmp_path / 'new'
    selection(root, tmp_path / 'parent', tmp_path / 'position', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_conditions_preserve_v25_and_v26_with_no_extra_prior(tmp_path, monkeypatch):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'position', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v25', 'previous_v26'} <= strategies and 'previous_v21' not in strategies
    _, current = load_selection(out)
    assert current.base.direction.data['intercept'] == 2.
    for name, parent in [('previous_v25', tmp_path / 'position'), ('previous_v26', tmp_path / 'parent')]:
        _, original = load_selection(parent)
        assert current.base.activity.data == original.base.activity.data
        assert current.manager.to_dict() == original.manager.to_dict()
        backtest(short_bars(), original, EngineConfig(**frozen['risk']), tmp_path / name)
        for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT' / name / n),
                                          pd.read_parquet(tmp_path / name / n), check_exact=True)
        assert json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text()) == json.loads((tmp_path / name / 'final_state.json').read_text())


def test_full_new_direction_fit_keeps_activity_and_uses_direct_model(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    frozen = prior_selection(parent, tmp_path / 'position', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    _, original = load_selection(parent)
    backtest(short_bars(), original, EngineConfig(**frozen['risk']), parent / 'candidate-00')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    data, episodes = input_frame()
    rows = []
    for i, row in enumerate(data.iloc[::2].itertuples()):
        side = 1 if i % 2 else -1
        rows.extend([(row.end + pd.Timedelta(minutes=1), 'open', 0, side, i + 1),
                     (row.end + pd.Timedelta(minutes=2), 'close', side, 0, i + 1)])
    actions = pd.DataFrame(rows, columns=['time', 'action', 'before_qty', 'after_qty', 'episode_id'])
    episodes = pd.DataFrame({'episode_id': actions.iloc[::2].episode_id.to_numpy(),
        'entry_time': actions.iloc[::2].time.to_numpy(), 'exit_time': actions.iloc[1::2].time.to_numpy()})
    monkeypatch.setattr('wonyotti_fr.position_direction_research.source_inputs', lambda *_: ({'episodes': episodes, 'actions': actions}, {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.position_direction_research.make_expansion_data', lambda _: data)
    monkeypatch.setattr('wonyotti_fr.position_direction_research.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    out = run_position_direction_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    _, current = load_selection(out)
    meta = json.loads((out / 'new_direction_training.json').read_text())
    assert meta['model']['rows'] == 1400 and meta['model']['positive'] == meta['model']['negative'] == 700
    assert current.base.direction.to_dict() == json.loads((out / 'new_position_direction.json').read_text())
    assert current.base.activity.data == original.base.activity.data
    assert current.size_model.to_dict() == original.size_model.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
