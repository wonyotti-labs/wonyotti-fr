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
