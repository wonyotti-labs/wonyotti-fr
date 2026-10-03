import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from test_pullback_evaluation import ConstantBase, bars
from test_pullback_research import selection

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_replay import run_event_replay
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.net_edge_labels import isolated_outcome, label_opportunities, potential_entries
from wonyotti_fr.net_edge_model import NET_FEATURES, NetEdgeModel, NetEdgePolicy, net_values
from wonyotti_fr.net_edge_research import candidate_plan, net_diagnostics
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import evaluation_period
from wonyotti_fr.pullback_policy import PullbackPolicy


def constant_model(value=20.):
    return NetEdgeModel.from_dict({'format': 'net_edge_ridge_v1', 'features': NET_FEATURES, 'alpha': 10,
                                  'mean': [0.]*31, 'scale': [1.]*31, 'coefficients': [0.]*31, 'intercept': value})


def config():
    return EngineConfig(bar_seconds=60, max_hold_bars=30, cooldown_bars=15, max_adds=0, stop_fraction=.04)


def fixed_selection(root):
    base = selection(root)
    base.update(offset_bps=16, risk=asdict(config()))
    save_json(root / 'pullback_selection.json', base)
    save_json(root / 'net_model.json', constant_model().to_dict())
    frozen = {'protocol': 'net_edge_v8', 'alpha': 10, 'margin_bps': 0, 'risk': asdict(config()),
              'pullback_selection_sha256': sha256(root / 'pullback_selection.json'),
              'model_sha256': {'net_model.json': sha256(root / 'net_model.json')},
              'training_period': ['2020-01-01', '2021-01-01'], 'selection_period': ['2021-01-01', '2022-01-01'],
              'confirmation_period': ['2022-01-01', '2023-01-01'],
              'observed_evaluation_period': ['2023-01-01', '2026-01-01'],
              'seen_2026_period': ['2026-01-01', '2026-10-01'], 'evaluation_end_exclusive': '2026-10-01',
              'unseen_evaluation_available': False}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('direction', [-1, 1])
def test_target_matches_next_open_cost_and_funding_cashflows(direction):
    frame = bars().iloc[:31].copy()
    frame[['open', 'high', 'low', 'close']] = 100.
    frame.loc[0, 'funding_rate'] = .01
    frame.loc[5, 'funding_rate'] = .001
    outcome = isolated_outcome(frame, direction, config())
    entry, exit_price = 100*(1+direction*.0003), 100*(1-direction*.0003)
    expected_bps = (direction*(exit_price-entry) - (entry+exit_price)*.0005 - direction*100*.001)/entry*10000
    assert outcome['net_bps'] == pytest.approx(expected_bps)
    assert outcome['entry_time'] == frame.time.iloc[0]
    assert outcome['exit_time'] == frame.time.iloc[30]
    assert outcome['exit_reason'] == 'time_limit'
    assert outcome['hold_minutes'] == 30


@pytest.mark.parametrize('gap', [True, False])
def test_target_preserves_gap_and_intrabar_losses(gap):
    frame = bars().iloc[:31].copy()
    frame[['open', 'high', 'low', 'close']] = 100.
    if gap:
        frame.loc[3, ['open', 'high', 'low', 'close']] = 94.
    else:
        frame.loc[3, 'low'] = 94.
    outcome = isolated_outcome(frame, 1, config())
    assert outcome['exit_reason'] == ('gap_stop' if gap else 'intrabar_stop')
    assert outcome['exit_price'] == pytest.approx((94 if gap else 100.03*.96)*.9997)
    assert outcome['net_bps'] < -400


def test_future_prices_do_not_change_past_opportunities_and_labels_are_purged():
    frame = bars().assign(count=1, volume=1.)
    policy = PullbackPolicy(ConstantBase(), 16, 5)
    original, counts = potential_entries(frame, policy)
    changed = frame.copy()
    changed.loc[30:, ['open', 'high', 'low', 'close']] *= 1.2
    altered, _ = potential_entries(changed, policy)
    boundary = frame.end.iloc[29]
    pd.testing.assert_frame_equal(original[original.decision_time <= boundary], altered[altered.decision_time <= boundary])
    labels, summary = label_opportunities(frame, original, config(), '2020-01-03')
    assert counts['triggered'] == len(original) and summary['boundary_excluded'] > 0
    assert (labels.signal_time < labels.decision_time).all()
    assert (labels.decision_time == labels.entry_time).all()
    assert (labels.decision_time < labels.label_end).all()
    assert (labels.label_end < pd.Timestamp('2020-01-02T00:00Z')).all()
    changed_labels, _ = label_opportunities(changed, original.iloc[:1], config(), '2020-01-03')
    pd.testing.assert_frame_equal(labels.iloc[:1][MARKET_FEATURES], changed_labels[MARKET_FEATURES])
    assert labels.iloc[0].net_bps != changed_labels.iloc[0].net_bps
    boundary_labels, _ = label_opportunities(frame, original, config(), '2020-01-02')
    assert boundary_labels.empty
    with pytest.raises(ValueError, match='연속'):
        potential_entries(frame.drop(index=10), policy)


def test_regression_export_and_training_support_are_checked():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(240, 14)), columns=MARKET_FEATURES)
    frame['order_direction'] = rng.choice([-1, 1], len(frame))
    frame['favorable_bps'], frame['wait_minutes'] = 20., 2
    frame['decision_time'] = pd.date_range('2020-01-01', periods=len(frame), freq='1D', tz='UTC')
    frame['label_end'] = frame.decision_time + pd.Timedelta(minutes=31)
    frame['net_bps'] = frame.ret_1h * frame.order_direction * 3 - 16
    model, checks = NetEdgeModel.fit(frame, 10)
    assert checks['export_max_error'] < 1e-10
    assert checks['training_mse'] < checks['constant_mean_mse']
    values = net_values(frame[MARKET_FEATURES], frame.order_direction, frame.favorable_bps, frame.wait_minutes)
    values[0, 0] = np.nan
    assert np.isnan(model.predict(values)[0])
    for bad in [frame.iloc[:199], frame.assign(order_direction=1), frame.assign(decision_time=frame.decision_time.iloc[0])]:
        with pytest.raises(ValueError, match='부족'):
            NetEdgeModel.fit(bad, 10)
    payload = model.to_dict()
    payload['scale'][0] = 0
    with pytest.raises(ValueError, match='계수'):
        NetEdgeModel.from_dict(payload)


@pytest.mark.parametrize('margin', [0, 8])
def test_gate_rejects_and_ends_wait_without_retrying_future_price(tmp_path, margin):
    frame = bars().assign(count=1, volume=1.)
    policy = NetEdgePolicy(PullbackPolicy(ConstantBase(), 16, 5), constant_model(-10.), margin)
    metrics = backtest(frame, policy, config(), tmp_path / 'filtered')
    diagnosis = net_diagnostics(tmp_path / 'filtered', frame, policy, config())
    assert metrics['closed_trades'] == 0 and diagnosis['gate']['rejected'] > 0
    assert diagnosis['gate']['accepted'] == 0
    curve = pd.read_parquet(tmp_path / 'filtered/equity.parquet')
    assert curve.loc[curve.policy_event.eq('filtered'), 'policy_state'].eq('{}').all()
    assert pd.read_parquet(tmp_path / 'filtered/net_gate_decisions.parquet').predicted_net_bps.eq(-10).all()


def test_filter_disabled_reproduces_v7_complete_outputs(tmp_path):
    frame = bars()
    base = PullbackPolicy(ConstantBase(), 16, 5)
    filtered = NetEdgePolicy(base, constant_model(-100.), 8, False)
    for name, policy in [('v7', base), ('disabled', filtered)]:
        backtest(frame, policy, config(), tmp_path / name)
    for file in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'v7' / file), pd.read_parquet(tmp_path / 'disabled' / file))
    assert (tmp_path / 'v7/final_state.json').read_bytes() == (tmp_path / 'disabled/final_state.json').read_bytes()


def test_frozen_model_tampering_and_confirmation_overlap_rejected(tmp_path):
    root = tmp_path / 'selection'
    frozen = fixed_selection(root)
    _, policy = load_selection(root)
    assert isinstance(policy, NetEdgePolicy) and len(candidate_plan()) == 4
    assert evaluation_period(frozen, 'observed') == ('2023-01-01', '2026-01-01')
    with pytest.raises(ValueError):
        evaluation_period({**frozen, 'observed_evaluation_period': ['2022-01-01', '2026-01-01']}, 'observed')
    with pytest.raises(ValueError):
        evaluation_period(frozen, 'verified_2022_2024')
    with pytest.raises(ValueError, match='관찰한'):
        guard_replay_period(root, frozen, '2026-10-01', '2026-10-02')
    (root / 'net_model.json').write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_v8_actual_process_crash_restores_pending_and_position(tmp_path, kind):
    root = tmp_path / 'selection'
    fixed_selection(root)
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {'synthetic': True}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_v8_replay_uses_minute_data_and_persists_wait(monkeypatch, tmp_path):
    root = tmp_path / 'selection'
    fixed_selection(root)
    features = tmp_path / 'features'
    features.mkdir()
    (features / 'manifest-5m.json').write_text('{}')
    (tmp_path / 'manifest-1m.json').write_text('{}')
    monkeypatch.setattr('wonyotti_fr.event_replay.prepare_minute_period', lambda *_: (bars(), {}))
    args = (root, tmp_path, 'BTCUSDT', '2020-01-01', '2020-01-02', tmp_path / 'journal.sqlite', tmp_path / 'runs')
    first = run_event_replay(*args, max_bars=5, feature_market=features)
    assert json.loads((first / 'replay.json').read_text())['final_state']['policy_state']
    final = run_event_replay(*args, verify_memory=True, feature_market=features)
    assert json.loads((final / 'replay.json').read_text())['memory_parity']
