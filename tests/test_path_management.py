import copy

import numpy as np
import pandas as pd
import pytest
from test_action_model import FixedScores, data
from test_action_replay import selection
from test_minute_management import inputs
from test_pullback_evaluation import ConstantBase, bars

from wonyotti_fr.action_model import ActionModels
from wonyotti_fr.action_research import action_diagnostics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.minute_management import FEATURES, management_events
from wonyotti_fr.path_management import (
    PATH_FEATURES,
    PATH_STATE,
    PathActionModels,
    PathActionPolicy,
    add_price_path,
)
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.pullback_policy import PullbackPolicy


def path_selection(root):
    frozen = selection(root, recent=True)
    payload = {'format': PathActionModels.format, 'features': PathActionModels.features,
               'actions': ['exit', 'reduce', 'increase'], 'kind': 'logistic',
               'mean': [0.]*35, 'scale': [1.]*35, 'coef': [[0.]*35 for _ in range(3)], 'intercept': [-10., -10., 10.]}
    save_json(root / 'action_model.json', payload)
    frozen.update(protocol='minute_path_v12', model_sha256={'action_model.json': sha256(root / 'action_model.json')})
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('direction', [1, -1])
def test_batch_online_path_parity_with_rebased_entry_and_future_invariance(direction):
    minute, five, states, orders = inputs()
    frame, _, _ = management_events(minute, five, states, orders, minute.end.iloc[-1])
    start = frame[frame.direction.eq(1)].index[0]
    frame.loc[start:start+4, 'close'] = [100, 110, 104, 98, 105]
    frame.loc[start:start+4, 'direction'] = direction
    frame.loc[start+2:start+4, 'average_entry'] = 105
    batch = add_price_path(frame)
    changed = frame.copy()
    changed.loc[start+4:, 'close'] *= 100
    pd.testing.assert_frame_equal(batch.iloc[:start+4], add_price_path(changed).iloc[:start+4])
    policy = PathActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([0, 0, 0]),
                              {'exit': .5, 'reduce': .5, 'increase': .5})
    stored = {}
    for n, (_, row) in enumerate(batch.iloc[start:start+5].iterrows()):
        state = {'direction': direction, 'halted': False, 'policy_state': stored, 'pending': 'hold',
                 'bar_seconds': 60, 'hold_bars': n+1, 'adds': int(n >= 2),
                 'favorable_move': direction*(row.close/row.average_entry-1),
                 'position_entry_time': row.entry_time.isoformat(), 'average_entry': row.average_entry}
        bar = {'features': np.zeros(14), 'close': row.close, 'end': row.end}
        decision = policy(bar, state)
        stored = decision.state
        values = policy.feature_values(bar, {**state, '_path_bounds': (stored['path_low'], stored['path_high'])})
        np.testing.assert_allclose(values[-3:], row[PATH_FEATURES].to_numpy(dtype=float), atol=1e-15)
    row = batch.iloc[start+2]
    expected = [(110/105-1), (100/105-1), 6/105] if direction == 1 else [1-100/105, 1-110/105, 4/105]
    np.testing.assert_allclose(row[PATH_FEATURES].to_numpy(dtype=float), expected)


def test_path_reset_on_new_episode_cooldown_tracking_and_corrupt_state_rejection():
    minute, five, states, orders = inputs()
    frame, _, _ = management_events(minute, five, states, orders, minute.end.iloc[-1])
    batch = add_price_path(frame)
    first_short = batch[batch.direction.eq(-1)].iloc[0]
    assert first_short.best_close_move == first_short.worst_close_move and first_short.giveback_from_best == 0
    policy = PathActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([0, .8, 0]),
                              {'exit': .5, 'reduce': .5, 'increase': .5})
    bar = {'features': np.zeros(14), 'close': 100., 'end': '2021-01-01T00:01:00+00:00'}
    state = {'direction': 1, 'halted': False, 'policy_state': {}, 'pending': 'hold', 'bar_seconds': 60,
             'hold_bars': 1, 'adds': 0, 'favorable_move': 0, 'average_entry': 100.,
             'position_entry_time': '2021-01-01T00:00:00+00:00'}
    first = policy(bar, state)
    second = policy({**bar, 'close': 110., 'end': '2021-01-01T00:02:00+00:00'},
                    {**state, 'hold_bars': 2, 'policy_state': first.state})
    assert second.event == 'action_cooldown' and second.state['path_high'] == 110
    flat = policy({**bar, 'end': '2021-01-01T00:03:00+00:00'},
                  {**state, 'direction': 0, 'policy_state': second.state})
    assert flat.state == {}
    broken = copy.deepcopy(first.state)
    broken['path_high'] = 99
    with pytest.raises(ValueError, match='숫자'):
        policy({**bar, 'end': '2021-01-01T00:02:00+00:00'}, {**state, 'policy_state': broken})
    with pytest.raises(ValueError, match='누락'):
        policy(bar, {**state, 'hold_bars': 2})


@pytest.mark.parametrize('kind', ['logistic', 'tree'])
def test_path_model_export_and_legacy_format_separation(kind):
    train, calibration = data()
    train, calibration = train.copy(), calibration.copy()
    for frame in [train, calibration]:
        for i, name in enumerate(PATH_FEATURES):
            frame[name] = frame[FEATURES[i]].abs()
    model, _, report = PathActionModels.fit(train, calibration, kind)
    assert report['export_max_error'] < 1e-12
    values = calibration[model.features].to_numpy()[:10]
    np.testing.assert_allclose(model.probabilities(values), np.vstack([model.probabilities(v[None, :])[0] for v in values]))
    with pytest.raises(ValueError, match='형식'):
        ActionModels.from_dict(model.to_dict())


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_path_state_actual_crash_recovery_and_frozen_loading(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = path_selection(root)
    assert load_selection(root)[0] == frozen
    report = verify_stress(list(iter_events(bars())), root, tmp_path, {'kind': kind}, kind, True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73


@pytest.mark.parametrize('delay', [0, 1])
def test_path_state_accounting_and_waiting_diagnostics_with_delayed_entry(tmp_path, delay):
    policy = PathActionPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([0., .3, .2]),
                              {'exit': .1, 'reduce': .1, 'increase': .1})
    frame = bars()
    frame.loc[:7, ['open', 'high', 'low', 'close']] = [100., 100.1, 99.9, 100.]
    frame.loc[8:, ['open', 'high', 'low', 'close']] = [99., 99.1, 98.9, 99.]
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, signal_delay_bars=delay)
    backtest(frame, policy, config, tmp_path / 'run')
    report = action_diagnostics(tmp_path / 'run', frame, config)
    assert report['waiting']['entry_fills'] > 0
    import json
    curve = pd.read_parquet(tmp_path / 'run/equity.parquet')
    held = curve[curve.quantity.ne(0)]
    assert held.policy_state.map(lambda s: PATH_STATE <= set(json.loads(s))).all()


def test_diagnostic_scope_requires_failed_confirmation_and_records_omitted_conditions(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    path_selection(root)
    confirmation = root / 'confirmation-2022' / 'metrics.json'
    save_json(confirmation, {'total_return': .1, 'closed_trades': 30, 'permanent_halt': False})
    args = (root, tmp_path, tmp_path, tmp_path / 'out', 'seen_2026', ['BTCUSDT'])
    with pytest.raises(ValueError, match='전체 평가'):
        run_pullback_evaluation(*args, diagnostic_only=True)
    with pytest.raises(ValueError, match='BTC'):
        run_pullback_evaluation(*args[:-1], ['ETHUSDT'], diagnostic_only=True)
    save_json(confirmation, {'total_return': -.1, 'closed_trades': 30, 'permanent_halt': False})
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_: (bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic_scope_test': True})
    out = run_pullback_evaluation(*args, diagnostic_only=True)
    import json
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['diagnostic_only'] and not summary['all_variant_conditions_completed']
    assert summary['condition_runs'] == 1 and summary['annual_runs'] == 0
    assert '평가하지 않았다' in (out / 'REPORT.md').read_text()
