import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_pullback_evaluation import bars
from test_rate_policy import policy, rate_selection

from wonyotti_fr.action_research import action_diagnostics
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.lifecycle_edge import LifecycleNetPolicy, lifecycle_labels, lifecycle_outcome
from wonyotti_fr.lifecycle_edge_research import run_lifecycle_edge_labels
from wonyotti_fr.net_edge_model import NET_FEATURES, NetEdgeModel


def model(score):
    return NetEdgeModel.from_dict({'format': 'net_edge_ridge_v1', 'features': NET_FEATURES, 'alpha': 100,
        'mean': [0.]*len(NET_FEATURES), 'scale': [1.]*len(NET_FEATURES),
        'coefficients': [0.]*len(NET_FEATURES), 'intercept': float(score)})


def flat_bars():
    frame = bars()
    frame[['open', 'high', 'low', 'close']] = 100.
    frame['count'], frame['volume'] = 10, 100.
    return frame


def test_whole_lifecycle_label_keeps_loss_fees_and_funding_until_natural_exit():
    frame = flat_bars()
    frame.loc[1, 'funding_rate'] = .001
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0)
    result, fills = lifecycle_outcome(list(iter_events(frame)), 0, 1, config, policy([.2, 0, 0]),
                                      pd.Timestamp('2020-01-02', tz='UTC'))
    quantity = config.initial_equity * config.allocation * config.entry_fraction / (100 * 1.0005)
    expected_fees, expected_funding = quantity * 100 * .001, quantity * 100 * .001
    assert [f['reason'] for f in fills] == ['entry', 'signal_exit']
    assert result['label_status'] == 'closed' and result['hold_minutes'] == 5
    assert result['fees'] == pytest.approx(expected_fees)
    assert result['funding_cost'] == pytest.approx(expected_funding)
    assert result['net_pnl'] == pytest.approx(-expected_fees - expected_funding)
    assert result['label_end'] == pd.Timestamp('2020-01-01T00:06Z')


def test_partial_actions_are_included_and_prices_after_exit_do_not_change_label():
    frame = flat_bars()
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0, max_adds=5, allow_adverse_add=True)
    bot = policy([.1, .3, .2])
    cutoff = pd.Timestamp('2020-01-02', tz='UTC')
    result, fills = lifecycle_outcome(list(iter_events(frame)), 0, -1, config, bot, cutoff)
    assert {'entry', 'increase', 'signal_reduce', 'signal_exit'} <= {f['reason'] for f in fills}
    assert result['adds'] > 0 and result['max_accounting_residual'] < 1e-7
    changed = frame.copy()
    changed.loc[changed.time.ge(result['label_end']), ['open', 'high', 'low', 'close']] *= 3
    again, again_fills = lifecycle_outcome(list(iter_events(changed)), 0, -1, config, bot, cutoff)
    assert result == again and fills == again_fills


def test_unclosed_position_is_right_censored_without_forced_close():
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, slippage_bps=0)
    result, fills = lifecycle_outcome(list(iter_events(flat_bars())), 0, 1, config, policy([0, 0, 0]),
                                      pd.Timestamp('2020-01-01T00:10Z'))
    assert result['label_status'] == 'right_censored' and result['net_bps'] is None
    assert result['remaining_quantity'] > 0 and result['observed_minutes'] == 9
    assert [f['reason'] for f in fills] == ['entry']


def test_label_ledger_preserves_censored_and_boundary_opportunities():
    frame = flat_bars()
    rows = pd.DataFrame({'entry_index': [0, 59], 'decision_time': [frame.time.iloc[0], frame.time.iloc[59]], 'order_direction': [1, -1]})
    config = EngineConfig(bar_seconds=60, max_hold_bars=0)
    training, ledger, summary = lifecycle_labels(frame, rows, config, policy([0, 0, 0]), '2020-01-03')
    assert training.empty and ledger.label_status.eq('right_censored').all()
    assert summary['forced_boundary_closes'] == 0 and not summary['losing_labels_removed']
    with pytest.raises(ValueError, match='중복'):
        lifecycle_labels(frame, pd.concat([rows, rows]), config, policy([0, 0, 0]), '2020-01-03')
    with pytest.raises(ValueError, match='설정'):
        lifecycle_outcome(list(iter_events(frame)), 0, 1, replace(config, max_hold_bars=30),
                          policy([0, 0, 0]), pd.Timestamp('2020-01-02', tz='UTC'))


def test_entry_filter_rejects_whole_wait_and_does_not_change_existing_management(tmp_path):
    frame, config = bars(), EngineConfig(bar_seconds=60, max_hold_bars=0)
    base = policy([.1, .1, .1])
    for name, strategy in [('base', base), ('disabled', LifecycleNetPolicy(base, model(7), enabled=False)),
                           ('accepted', LifecycleNetPolicy(base, model(8))), ('rejected', LifecycleNetPolicy(base, model(7)))]:
        backtest(frame, strategy, config, tmp_path / name)
        action_diagnostics(tmp_path / name, frame, config)
    for file in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        expected = pd.read_parquet(tmp_path / 'base' / file)
        for name in ['disabled', 'accepted']:
            pd.testing.assert_frame_equal(expected, pd.read_parquet(tmp_path / name / file), check_exact=True)
    assert json.loads((tmp_path / 'rejected/metrics.json').read_text())['closed_trades'] == 0
    events = pd.read_parquet(tmp_path / 'rejected/equity.parquet')
    assert events.policy_event.eq('filtered').any()
    assert events[events.policy_event.eq('filtered')].policy_state.eq('{}').all()
    assert np.isfinite(events.equity).all()


def test_lifecycle_label_run_keeps_censored_ledger_and_hashes(tmp_path, monkeypatch):
    root = tmp_path / 'selection'
    rate_selection(root)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.lifecycle_edge_research.prepare_minute_period',
                        lambda *_: (bars().assign(volume=100., count=10), {}))
    out = run_lifecycle_edge_labels(root, tmp_path, tmp_path, tmp_path / 'runs')
    summary = json.loads((out / 'summary.json').read_text())
    ledger = pd.read_parquet(out / 'opportunity_ledger.parquet')
    assert summary['complete'] and len(ledger) == summary['labels']['opportunities'] > 0
    assert ledger.label_status.eq('right_censored').any()
    assert summary['labels']['forced_boundary_closes'] == 0
    from wonyotti_fr.common import sha256
    for name, digest in json.loads((out / 'files.json').read_text()).items():
        assert sha256(out / name) == digest


def edge_selection(root, score=8, weighted=False):
    from wonyotti_fr.common import save_json, sha256
    frozen = rate_selection(root)
    (root / 'rate_selection.json').write_bytes((root / 'frozen_selection.json').read_bytes())
    save_json(root / 'net_model.json', model(score).to_dict())
    frozen.update(protocol='lifecycle_edge_v15', candidate=0, alpha=100, margin_bps=8,
                  net_training_period=['2021-01-01', '2022-01-01'],
                  rate_selection_sha256=sha256(root / 'rate_selection.json'),
                  model_sha256={**frozen['model_sha256'], 'net_model.json': sha256(root / 'net_model.json')})
    if weighted:
        from test_label_weighting import intervals

        from wonyotti_fr.label_weighting import WEIGHTING, lifecycle_weights
        _, ledger, report = lifecycle_weights(intervals())
        ledger.to_parquet(root / 'training_weights.parquet', index=False)
        save_json(root / 'weighting.json', report)
        frozen.update(protocol='lifecycle_edge_v16', sample_weighting=WEIGHTING,
                      training_weights_sha256=sha256(root / 'training_weights.parquet'),
                      weighting_sha256=sha256(root / 'weighting.json'))
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('damage', ['parent', 'margin', 'model', 'management', 'risk'])
def test_lifecycle_selection_rejects_changed_model_parent_and_fixed_settings(tmp_path, damage):
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.event_research import load_selection
    root = tmp_path / 'selection'
    frozen = edge_selection(root)
    assert isinstance(load_selection(root)[1], LifecycleNetPolicy)
    if damage in {'parent', 'model', 'management'}:
        name = {'parent': 'rate_selection.json', 'model': 'net_model.json', 'management': 'action_model.json'}[damage]
        (root / name).write_text('{}')
    elif damage == 'margin':
        frozen['margin_bps'] = 0
    else:
        frozen['risk']['fee_bps'] = 0
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_lifecycle_filter_actual_crash_recovery_and_replay_period(tmp_path, kind, weighted):
    from wonyotti_fr.engine_stress import verify_stress
    from wonyotti_fr.period_guard import guard_replay_period
    root = tmp_path / 'selection'
    frozen = edge_selection(root, weighted=weighted)
    with pytest.raises(ValueError, match='관찰한 기간'):
        guard_replay_period(root, frozen, '2020-01-01', '2021-01-01')
    guard_replay_period(root, frozen, '2021-01-01', '2022-01-01')
    result = verify_stress(list(iter_events(bars())), root, tmp_path, {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


@pytest.mark.parametrize('score', [7, 8])
def test_lifecycle_gate_diagnostics_recomputes_and_rejects_changed_predictions(tmp_path, score):
    from wonyotti_fr.lifecycle_edge_research import lifecycle_edge_diagnostics
    frame, config = bars(), EngineConfig(bar_seconds=60, max_hold_bars=0)
    bot = LifecycleNetPolicy(policy([.1, .1, .1]), model(score))
    backtest(frame, bot, config, tmp_path / 'run')
    report = lifecycle_edge_diagnostics(tmp_path / 'run', frame, bot, config)
    assert report['gate']['opportunities'] > 0
    assert report['gate']['accepted' if score == 7 else 'rejected'] == 0
    wrong = LifecycleNetPolicy(bot.manager, model(15-score))
    with pytest.raises(ValueError, match='불일치'):
        lifecycle_edge_diagnostics(tmp_path / 'run', frame, wrong, config)


@pytest.mark.parametrize('weighted', [False, True])
def test_lifecycle_evaluation_copies_frozen_chain_and_unfiltered_control(tmp_path, monkeypatch, weighted):
    from wonyotti_fr.event_research import load_selection
    from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
    root = tmp_path / 'selection'
    edge_selection(root, 7, weighted=weighted)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_: (bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic_scope_test': True})
    output = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    assert isinstance(load_selection(output)[1], LifecycleNetPolicy)
    rows = json.loads((output / 'results.json').read_text())
    assert len(rows) == 9
    unfiltered = next(row for row in rows if row['strategy'] == 'unfiltered_v14')
    fixed = next(row for row in rows if row['strategy'] == 'fixed_policy')
    assert unfiltered['closed_trades'] > fixed['closed_trades'] == 0
    gate = json.loads((output / 'BTCUSDT/fixed_policy/lifecycle_edge_diagnostics.json').read_text())['gate']
    assert gate['rejected'] > 0 and gate['accepted'] == 0


@pytest.mark.parametrize('weighted', [False, True])
def test_lifecycle_training_uses_all_closed_labels_and_marks_in_sample(tmp_path, monkeypatch, weighted):
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.event_features import MARKET_FEATURES
    from wonyotti_fr.event_research import load_selection
    from wonyotti_fr.lifecycle_edge_research import run_lifecycle_edge_selection
    root, labels = tmp_path / 'selection', tmp_path / 'labels'
    frozen = rate_selection(root)
    metrics = {'total_return': -.01, 'max_drawdown': -.02, 'closed_trades': 100, 'permanent_halt': False}
    frozen['development_metrics'] = metrics
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    save_json(root / 'confirmation-2022/metrics.json', metrics)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    labels.mkdir()
    frame = pd.DataFrame(np.zeros((200, len(MARKET_FEATURES))), columns=MARKET_FEATURES)
    frame['decision_time'] = pd.date_range('2021-01-02', periods=200, freq='1D', tz='UTC')
    frame['label_end'] = frame.decision_time + pd.Timedelta(hours=1)
    frame['order_direction'], frame['net_bps'] = np.tile([-1, 1], 100), np.tile([-10., 30.], 100)
    frame['favorable_bps'], frame['wait_minutes'] = 20., 2
    frame['net_pnl'], frame['entry_notional'] = frame.net_bps * .1, 1000.
    frame['hold_minutes'], frame['label_status'] = 60, 'closed'
    frame.to_parquet(labels / 'training_labels.parquet', index=False)
    censored = frame.tail(1).assign(label_status='right_censored', net_bps=np.nan, net_pnl=np.nan)
    pd.concat([frame, censored], ignore_index=True).to_parquet(labels / 'opportunity_ledger.parquet', index=False)
    save_json(labels / 'summary.json', {'complete': True})
    save_json(labels / 'files.json', {name: sha256(labels / name) for name in ['training_labels.parquet', 'opportunity_ledger.parquet', 'summary.json']})
    save_json(labels / 'manifest.json', {'settings': {'reference_sha256': sha256(root / 'frozen_selection.json'),
        'training_period': ['2021-01-01', '2022-01-01'], 'market_manifest_sha256': sha256(tmp_path / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(tmp_path / 'manifest-5m.json')}})
    monkeypatch.setattr('wonyotti_fr.lifecycle_edge_research.prepare_minute_period', lambda *_: (bars(), {}))
    output = run_lifecycle_edge_selection(root, labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs', overlap_weighted=weighted)
    assert isinstance(load_selection(output)[1], LifecycleNetPolicy)
    support = json.loads((output / 'training_diagnostics.json').read_text())
    assert support['rows'] == 200 and support['negative_labels'] == support['positive_labels'] == 100
    assert json.loads((output / 'development.json').read_text())[0]['in_sample']
    assert not json.loads((output / 'development.json').read_text())[0]['selected_by_pnl']
    assert len(json.loads((output / 'comparison.json').read_text())) == 4
    if weighted:
        weights = pd.read_parquet(output / 'training_weights.parquet')
        assert len(weights) == 200 and weights.sample_weight.eq(1).all()
        (output / 'weighting.json').write_text('{}')
        with pytest.raises(ValueError, match='가중치'):
            load_selection(output)
