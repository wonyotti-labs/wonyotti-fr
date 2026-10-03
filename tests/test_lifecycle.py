import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from test_pullback_evaluation import ConstantBase, bars
from test_pullback_research import selection

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_replay import run_event_replay
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.lifecycle_model import (
    ACTIONS,
    MANAGEMENT_FEATURES,
    LifecyclePolicy,
    ManagementModel,
)
from wonyotti_fr.lifecycle_research import (
    candidate_plan,
    lifecycle_diagnostics,
    risk_config,
    sizing_from_training,
)
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import evaluation_period
from wonyotti_fr.pullback_policy import PullbackPolicy


def manager(intent='increase'):
    intercept = [0., 0., 0.]
    intercept[ACTIONS.index(intent)] = 10.
    return ManagementModel.from_dict({
        'format': 'lifecycle_management_v1', 'features': MANAGEMENT_FEATURES, 'actions': ACTIONS,
        'mean': [0.]*18, 'scale': [1.]*18, 'activity_coef': [0.]*18, 'activity_intercept': 30.,
        'action_coef': [[0.]*18 for _ in range(3)], 'action_intercept': intercept,
    })


def frozen_selection(root):
    previous = selection(root)
    previous.update(offset_bps=16)
    sizing = {'addition_fraction': .1, 'reduction_fraction': .5}
    save_json(root / 'pullback_selection.json', previous)
    save_json(root / 'management_model.json', manager().to_dict())
    frozen = {
        'protocol': 'lifecycle_v9', 'frequency_factor': .5, 'stop_fraction': .04, 'activity_threshold': .1,
        'risk': asdict(risk_config(previous, sizing, .04)), 'sizing': sizing,
        'pullback_selection_sha256': sha256(root / 'pullback_selection.json'),
        'model_sha256': {'management_model.json': sha256(root / 'management_model.json')},
        'training_period': ['2018-03-01', '2021-01-01'], 'selection_period': ['2021-01-01', '2022-01-01'],
        'confirmation_period': ['2022-01-01', '2023-01-01'],
        'observed_evaluation_period': ['2023-01-01', '2026-01-01'],
        'seen_2026_period': ['2026-01-01', '2026-10-01'], 'evaluation_end_exclusive': '2026-10-01',
        'unseen_evaluation_available': False,
    }
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


def test_train_export_invalid_input_and_support():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(1600, 18)), columns=MANAGEMENT_FEATURES)
    frame['direction'] = rng.choice([-1, 1], len(frame))
    frame['target'] = np.tile(['hold', *ACTIONS], 400)
    frame['end'] = pd.date_range('2018-01-01', periods=len(frame), freq='h', tz='UTC')
    frame['label_end'] = frame.end + pd.Timedelta(minutes=5)
    fitted, checks = ManagementModel.fit(frame)
    assert checks['export_max_error'] < 1e-12 and len(checks['thresholds']) == 2
    assert len(candidate_plan()) == 4
    payload = fitted.to_dict()
    payload['scale'][0] = 0
    with pytest.raises(ValueError, match='척도'):
        ManagementModel.from_dict(payload)
    with pytest.raises(ValueError, match='지원'):
        ManagementModel.fit(frame.iloc[:999])
    values = frame[MANAGEMENT_FEATURES].to_numpy()
    values[0, 0] = np.nan
    scores, actions = fitted.probabilities(values)
    assert np.isnan(scores[0]) and np.isnan(actions[0]).all()
    np.testing.assert_allclose(actions[1:].sum(axis=1), 1)


def test_management_uses_minutes_and_only_confirmed_boundaries():
    policy = LifecyclePolicy(PullbackPolicy(ConstantBase(), 16, 5), manager('reduce'), .1)
    state = {'direction': 1, 'halted': False, 'policy_state': {}, 'bar_seconds': 60,
             'pending': 'hold', 'favorable_move': -.01, 'hold_bars': 61, 'adds': 2}
    captured = []
    original = policy.manager.probabilities
    def predict(values):
        captured.append(values.copy())
        return original(values)
    policy.manager.probabilities = predict
    bar = {'end': '2021-01-01T01:00:00+00:00', 'close': 100., 'features': [0.]*14}
    assert policy(bar, state).intent == 'reduce'
    assert captured[-1][0, 16] == pytest.approx(np.log1p(61))
    assert policy({**bar, 'end': '2021-01-01T01:01:00+00:00'}, state).intent == 'hold'
    assert len(captured) == 1


def test_sizing_only_uses_supported_training_orders():
    times = pd.date_range('2020-01-01', periods=41, freq='h', tz='UTC')
    orders = pd.DataFrame({'target_time': times, 'target': ['increase']*20 + ['reduce']*21,
                           'before_qty': 100, 'orderqty': [20]*20 + [40]*20 + [1000000]})
    result = sizing_from_training(orders, pd.DataFrame({'target_time': times[:-1]}))
    assert result['addition_fraction'] == .1 and result['reduction_fraction'] == .4
    assert result['counts'] == {'increase': 20, 'reduce': 20}


def test_additions_without_time_cap_are_bounded_and_accounted(tmp_path):
    frame = bars().assign(volume=1., count=1)
    policy = LifecyclePolicy(PullbackPolicy(ConstantBase(), 16, 5), manager(), .1)
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, max_adds=5, allow_adverse_add=True,
                          addition_fraction=.1, stop_fraction=.04)
    target = tmp_path / 'managed'
    backtest(frame, policy, config, target)
    report = lifecycle_diagnostics(target, frame, config)
    trades = pd.read_parquet(target / 'trades.parquet')
    curve = pd.read_parquet(target / 'equity.parquet')
    assert report['trades_over_30_minutes'] > 0 and report['trades_with_additions'] > 0
    assert trades['adds'].max() <= 5 and curve.exposure.max() < .51
    assert report['decomposition']['fills_by_reason'].get('time_limit', 0) == 0


def test_disabled_management_reproduces_v7_and_future_change_preserves_prefix(tmp_path):
    frame = bars()
    entry = PullbackPolicy(ConstantBase(), 16, 5)
    disabled = LifecyclePolicy(entry, manager(), .1, enabled=False)
    config = EngineConfig(bar_seconds=60, max_hold_bars=30, max_adds=0)
    for name, policy in [('v7', entry), ('disabled', disabled)]:
        backtest(frame, policy, config, tmp_path / name)
    for name in ['equity', 'fills', 'trades']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'v7' / f'{name}.parquet'),
                                      pd.read_parquet(tmp_path / 'disabled' / f'{name}.parquet'), check_exact=True)
    policy = LifecyclePolicy(entry, manager(), .1)
    changed = frame.copy()
    changed.loc[30:, ['open', 'high', 'low', 'close', *MARKET_FEATURES]] *= 2
    engines = [TradingEngine(config), TradingEngine(config)]
    results = []
    for engine, data in zip(engines, [frame, changed], strict=True):
        policy.prepare(data)
        results.append([engine.step(event, policy) for event in list(iter_events(data))[:30]])
    assert results[0] == results[1]


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_crash_and_resume_for_lifecycle(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen_selection(root)
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {'kind': kind}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_period_model_tamper_and_resume_identity(monkeypatch, tmp_path):
    root = tmp_path / 'selection'
    frozen = frozen_selection(root)
    assert evaluation_period(frozen, 'observed') == ('2023-01-01', '2026-01-01')
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2026-10-01', '2026-11-01')
    with pytest.raises(ValueError):
        evaluation_period(frozen, 'verified_2022_2024')
    feature = tmp_path / 'feature'
    feature.mkdir()
    (feature / 'manifest-5m.json').write_text('{}')
    (tmp_path / 'manifest-1m.json').write_text('{}')
    monkeypatch.setattr('wonyotti_fr.event_replay.prepare_minute_period', lambda *_: (bars(), {}))
    args = (root, tmp_path, 'BTCUSDT', '2021-01-01', '2021-01-02', tmp_path / 'journal.sqlite', tmp_path / 'runs')
    first = run_event_replay(*args, max_bars=5, feature_market=feature)
    assert json.loads((first / 'replay.json').read_text())['final_state']['policy_state']
    last = run_event_replay(*args, feature_market=feature, verify_memory=True)
    assert json.loads((last / 'replay.json').read_text())['memory_parity']
    (root / 'management_model.json').write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        load_selection(root)
