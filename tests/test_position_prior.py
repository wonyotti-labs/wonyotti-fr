import json

import numpy as np
import pandas as pd
import pytest
from test_minute_inventory_research import selection as minute_selection
from test_new_position import input_frame, short_bars
from test_new_position import selection as position_selection

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.new_position_research import run_new_position_selection
from wonyotti_fr.position_prior import (
    PRIOR_FILES,
    adjust_direction,
    copy_position_parent,
    prior_offset,
)
from wonyotti_fr.position_prior_research import run_position_prior_selection
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.recent_entry import ENTRY_PERIOD, run_recent_entry_selection


def selection(root, parent, recent, minute, previous):
    old = position_selection(parent, recent, minute, previous)
    _, original = load_selection(parent)
    root.mkdir()
    copy_position_parent(parent, root)
    old_counts, new_counts = {'buy': 600, 'sell': 1400}, {'buy': 150, 'sell': 150}
    offset = prior_offset(old_counts, new_counts)
    save_json(root / 'position_direction_model.json', adjust_direction(original.base.direction, old_counts, new_counts).to_dict())
    save_json(root / 'direction_prior.json', {'training_period': ENTRY_PERIOD, 'old_counts': old_counts, 'new_counts': new_counts, 'offset': offset})
    frozen = {**old, 'protocol': 'position_prior_v26', 'position_selection_sha256': sha256(root / 'position_selection.json'),
        'direction_prior_offset': offset, 'prior_files_sha256': {n: sha256(root / n) for n in PRIOR_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_prior_shift_is_exact_odds_ratio_with_other_coefficients_fixed(tmp_path):
    root = tmp_path / 'new'
    selection(root, tmp_path / 'parent', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    _, before = load_selection(tmp_path / 'parent')
    _, after = load_selection(root)
    x = np.random.default_rng(26).normal(size=(100, 14))
    one, two = before.base.direction.probabilities(x), after.base.direction.probabilities(x)
    np.testing.assert_allclose(two / (1 - two), one / (1 - one) * 1400 / 600, rtol=1e-14)
    for key in ['mean', 'scale', 'coefficients']:
        assert before.base.direction.data[key] == after.base.direction.data[key]
    assert before.base.activity.data == after.base.activity.data
    assert before.manager.to_dict() == after.manager.to_dict()


@pytest.mark.parametrize('old,new', [({'buy': 49, 'sell': 1000}, {'buy': 100, 'sell': 100}),
    ({'buy': 400, 'sell': 400}, {'buy': 100, 'sell': 100}), ({'buy': 600, 'sell': 1400}, {'buy': 90, 'sell': 90}),
    ({'buy': 600., 'sell': 1400}, {'buy': 100, 'sell': 100})])
def test_prior_support_is_not_relaxed(old, new):
    with pytest.raises(ValueError):
        prior_offset(old, new)


@pytest.mark.parametrize('damage', ['parent', 'model', 'coefficient', 'offset', 'risk'])
def test_prior_chain_rejects_other_model_changes(tmp_path, damage):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    if damage == 'parent':
        (root / 'position_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'position_direction_model.json').write_text('{}')
    elif damage == 'coefficient':
        p = root / 'position_direction_model.json'
        model = json.loads(p.read_text())
        model['coefficients'][0] += 1
        save_json(p, model)
        frozen['prior_files_sha256'][p.name] = sha256(p)
    elif damage == 'offset':
        frozen['direction_prior_offset'] += .1
    else:
        frozen['risk']['max_adds'] = 0
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_prior_policy_actual_crash_keeps_frozen_state(tmp_path, kind):
    root = tmp_path / 'new'
    selection(root, tmp_path / 'parent', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    result = verify_stress(list(iter_events(short_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_conditions_include_exact_original_v25(tmp_path, monkeypatch):
    root = tmp_path / 'new'
    frozen = selection(root, tmp_path / 'parent', tmp_path / 'recent', tmp_path / 'minute', tmp_path / 'previous')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v21', 'previous_v25'} <= strategies and 'previous_v23' not in strategies
    _, current = load_selection(out)
    _, original = load_selection(tmp_path / 'parent')
    assert current.base.direction.data != original.base.direction.data
    backtest(short_bars(), original, EngineConfig(**frozen['risk']), tmp_path / 'baseline')
    for n in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT/previous_v25' / n),
                                      pd.read_parquet(tmp_path / 'baseline' / n), check_exact=True)
    assert json.loads((out / 'BTCUSDT/previous_v25/final_state.json').read_text()) == json.loads((tmp_path / 'baseline/final_state.json').read_text())


def test_full_pipeline_validates_original_fit_and_only_adjusts_prior(tmp_path, monkeypatch):
    minute = tmp_path / 'minute'
    frozen = minute_selection(minute, tmp_path / 'previous')
    _, original = load_selection(minute)
    bars = short_bars()
    backtest(bars, original, EngineConfig(**frozen['risk']), minute / 'candidate-00')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    data, episodes = input_frame()
    data['buy'] = (np.arange(len(data)) // 2 % 3 != 0).astype(int)
    rows = []
    for row in data.iloc[::10].itertuples():
        side = 1 if row.episode_id % 2 else -1
        rows.extend([(row.end + pd.Timedelta(minutes=1), 'open', 0, side, row.episode_id),
                     (row.end + pd.Timedelta(minutes=2), 'close', side, 0, row.episode_id)])
    actions = pd.DataFrame(rows, columns=['time', 'action', 'before_qty', 'after_qty', 'episode_id'])
    for module in ['recent_entry', 'new_position_research']:
        monkeypatch.setattr(f'wonyotti_fr.{module}.source_inputs', lambda *_: ({'episodes': episodes, 'actions': actions}, {'synthetic': True}))
        monkeypatch.setattr(f'wonyotti_fr.{module}.make_expansion_data', lambda _: data)
        monkeypatch.setattr(f'wonyotti_fr.{module}.prepare_minute_period', lambda *_, **__: (bars, {}))
    recent = run_recent_entry_selection(minute, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'recent')
    parent = run_new_position_selection(recent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'position')
    monkeypatch.setattr('wonyotti_fr.position_prior_research.prepare_minute_period', lambda *_, **__: (bars, {}))
    out = run_position_prior_selection(parent, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    meta = json.loads((out / 'direction_prior.json').read_text())
    assert meta['new_counts'] == {'buy': 140, 'sell': 140}
    assert sum(meta['old_counts'].values()) == 1400
    assert meta['offset'] == prior_offset(meta['old_counts'], meta['new_counts'])
    _, current = load_selection(out)
    _, parent_policy = load_selection(parent)
    assert current.base.direction.data['coefficients'] == parent_policy.base.direction.data['coefficients']
    assert current.base.direction.data['intercept'] != parent_policy.base.direction.data['intercept']
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
