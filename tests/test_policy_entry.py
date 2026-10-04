import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_exit_move_state import fixture as move_fixture
from test_first_state import fixed_budget  # noqa: F401
from test_history_state import history_inputs
from test_label_weighting import training
from test_lifecycle_edge import model
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.exit_move_state import ExitMovePolicy
from wonyotti_fr.label_weighting import WEIGHTING, lifecycle_weights
from wonyotti_fr.lifecycle_edge import LifecycleNetPolicy
from wonyotti_fr.net_edge_model import net_values
from wonyotti_fr.outcome_journal import OutcomeJournal
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.policy_entry import (
    ENTRY_FILES,
    LABEL_FILES,
    copy_exit_move_parent,
    load_policy_outcome_training,
    prepare_policy_entry,
)
from wonyotti_fr.policy_entry_research import run_policy_entry_selection
from wonyotti_fr.policy_outcomes import OUTCOME_PERIOD, outcome_frame
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def source_labels(root, reference):
    root.mkdir()
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (root/n).write_text('{}')
    import wonyotti_fr.policy_outcomes as implementation
    settings = {'reference': str(reference), 'reference_sha256': sha256(reference/'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V55.md')), 'period': OUTCOME_PERIOD,
        'market': str(root), 'features': str(root), 'market_manifest_sha256': sha256(root/'manifest-1m.json'),
        'feature_manifest_sha256': sha256(root/'manifest-5m.json'),
        'implementation_sha256': {n: sha256(Path(implementation.__file__).parent/n)
                                 for n in ['engine.py', 'lifecycle_edge.py', 'policy_outcomes.py']},
        'all_opportunities': True, 'forced_boundary_closes': False, 'new_models_fitted': False}
    save_json(root/'manifest.json', {'settings': settings})
    save_json(root/'input_verification.json', {})
    frame = training()
    opportunities = frame.drop(columns=['label_end', 'net_bps']).copy()
    opportunities['signal_time'] = opportunities.decision_time-pd.to_timedelta(opportunities.wait_minutes, unit='min')
    opportunities['entry_index'] = np.arange(len(frame))
    opportunities['reference_price'], opportunities['decision_close'] = 1000., 998.
    tail = opportunities.iloc[-2:].copy()
    tail['decision_time'] = pd.to_datetime(['2021-12-30T23:58Z', '2021-12-31T00:01Z'])
    tail['signal_time'] = tail.decision_time-pd.Timedelta(minutes=1)
    opportunities = pd.concat([opportunities, tail], ignore_index=True)
    opportunities.to_parquet(root/'potential_entries.parquet', index=False)
    records = []
    with OutcomeJournal(root/'outcomes.sqlite', {**settings, 'opportunities_sha256': sha256(root/'potential_entries.parquet')}) as journal:
        for i, opportunity in enumerate(opportunities.to_dict('records')):
            if i < len(frame):
                direction = opportunity['order_direction']
                net = float(frame.net_bps.iloc[i])*.1
                fills = [{'time': opportunity['decision_time'], 'reason': 'entry', 'delta_quantity': direction,
                          'price': 1000., 'fee': .5},
                         {'time': frame.label_end.iloc[i]-pd.Timedelta(minutes=1), 'reason': 'signal_exit',
                          'delta_quantity': -direction, 'price': 1000.+direction*(net+1.), 'fee': .5}]
                outcome = {'label_status': 'closed', 'net_bps': float(frame.net_bps.iloc[i]), 'net_pnl': net,
                    'entry_notional': 1000., 'entry_time': fills[0]['time'], 'exit_time': fills[1]['time'],
                    'label_end': frame.label_end.iloc[i], 'exit_reason': 'signal_exit', 'hold_minutes': 59,
                    'adds': 0, 'fees': 1., 'funding_cost': 0., 'fills': 2, 'max_accounting_residual': 0.}
            elif i == len(frame):
                fills = []
                outcome = {'label_status': 'right_censored', 'net_bps': None,
                    'label_end': pd.Timestamp('2021-12-31', tz='UTC'), 'remaining_quantity': 0.,
                    'observed_minutes': 1, 'max_accounting_residual': 0.}
            else:
                fills, outcome = [], {'label_status': 'outside_training_boundary'}
            journal.append(i, opportunity, {'outcome': outcome, 'fills': fills})
            records.append({**opportunity, **outcome})
    ledger = outcome_frame(records)
    train = ledger[ledger.label_status.eq('closed')].reset_index(drop=True)
    ledger.to_parquet(root/'opportunity_ledger.parquet', index=False)
    train.to_parquet(root/'training_labels.parquet', index=False)
    _, intervals, weighting = lifecycle_weights(train)
    intervals.to_parquet(root/'label_intervals.parquet', index=False)
    save_json(root/'support.json', {'overlap': weighting, 'weights_used_for_fitting': False})
    save_json(root/'summary.json', {'complete': True, 'opportunities': len(ledger), 'processed': len(ledger),
        'closed': len(train), 'statuses': ledger.label_status.value_counts().to_dict(),
        'losing_labels': int(train.net_bps.lt(0).sum()), 'positive_labels': int(train.net_bps.gt(0).sum()),
        'losing_labels_removed': False, 'forced_boundary_closes': 0, 'profitability_accepted': False})
    save_json(root/'files.json', {n: sha256(root/n) for n in LABEL_FILES})
    return train


def fixture(tmp_path, monkeypatch, *, constant=False):
    parent, _, _, _, previous = move_fixture(tmp_path, monkeypatch)
    labels = tmp_path/'policy-labels'
    train = source_labels(labels, parent)
    root = tmp_path/'policy-entry'
    root.mkdir()
    copy_exit_move_parent(parent, root)
    prepare_policy_entry(parent, labels, labels, labels, root)
    if constant:
        save_json(root/ENTRY_FILES[0], model(8).to_dict())
    frozen = {**previous, 'protocol': 'policy_entry_v56', 'exit_move_selection_sha256': sha256(root/'exit_move_selection.json'),
        'policy_entry_alpha': 100, 'policy_entry_margin_bps': 8, 'policy_entry_training_period': OUTCOME_PERIOD,
        'policy_entry_sample_weighting': WEIGHTING, 'policy_entry_files_sha256': {n: sha256(root/n) for n in ENTRY_FILES}}
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    return root, parent, labels, train, frozen


def test_complete_losses_censoring_and_model_match_independent_normal_equations(tmp_path, monkeypatch):
    root, parent, labels, train, _ = fixture(tmp_path, monkeypatch)
    before = sha256(labels/'outcomes.sqlite')
    restored, weight, _, _, evidence = load_policy_outcome_training(parent, labels, labels, labels)
    pd.testing.assert_frame_equal(restored, train, check_exact=True)
    assert before == sha256(labels/'outcomes.sqlite')
    assert evidence['statuses'] == {'closed': 200, 'right_censored': 1, 'outside_training_boundary': 1}
    assert evidence['losing_labels'] > 0 and len(restored) == 200
    _, policy = load_selection(root)
    from wonyotti_fr.event_features import MARKET_FEATURES
    values = net_values(train[MARKET_FEATURES], train.order_direction, train.favorable_bps, train.wait_minutes)
    mean = np.average(values, axis=0, weights=weight)
    scale = np.sqrt(np.average((values-mean)**2, axis=0, weights=weight))
    scale[scale == 0] = 1.
    x = np.column_stack([(values-mean)/scale, np.ones(len(values))])
    penalty = np.eye(x.shape[1])*100
    penalty[-1, -1] = 0.
    coef = np.linalg.solve(x.T@(weight[:, None]*x)+penalty, x.T@(weight*train.net_bps.to_numpy()))
    np.testing.assert_allclose(policy.model.predict(values), x@coef, atol=1e-10)
    assert policy.manager.manager.model.to_dict() == load_selection(parent)[1].manager.model.to_dict()


@pytest.mark.parametrize('damage', ['partial', 'lost_loss', 'label_net', 'interval', 'source', 'market', 'journal'])
def test_input_rejects_incomplete_changed_labels_and_false_refingerprints(tmp_path, monkeypatch, damage):
    _, parent, labels, _, _ = fixture(tmp_path, monkeypatch)
    if damage == 'partial':
        p = labels/'summary.json'
        data = json.loads(p.read_text())
        data['complete'] = False
        save_json(p, data)
    elif damage in {'lost_loss', 'label_net'}:
        p = labels/'training_labels.parquet'
        data = pd.read_parquet(p)
        if damage == 'lost_loss':
            data = data[data.net_bps.ge(0)]
        else:
            data.loc[0, 'net_bps'] += 100
        data.to_parquet(p, index=False)
    elif damage == 'interval':
        p = labels/'label_intervals.parquet'
        data = pd.read_parquet(p)
        data.loc[0, 'sample_weight'] = 0.
        data.to_parquet(p, index=False)
    elif damage == 'market':
        (labels/'manifest-1m.json').write_text('{"changed":true}')
    elif damage == 'source':
        p = labels/'manifest.json'
        data = json.loads(p.read_text())
        data['settings']['reference_sha256'] = '0'*64
        save_json(p, data)
    else:
        import sqlite3
        with sqlite3.connect(labels/'outcomes.sqlite') as connection:
            connection.execute('DELETE FROM outcomes WHERE sequence=0')
    save_json(labels/'files.json', {n: sha256(labels/n) for n in LABEL_FILES})
    with pytest.raises((ValueError, AssertionError)):
        load_policy_outcome_training(parent, labels, labels, labels)


@pytest.mark.parametrize('buy,close', [(.8, 99.8), (.2, 100.2)])
@pytest.mark.parametrize('score', [7.99, 8., 8.01, float('nan')])
def test_only_flat_trigger_is_gated_and_rejected_wait_is_cleared(buy, close, score):
    from test_activity_ablation import bots

    from wonyotti_fr.horizon_exit_state import HorizonExitPolicy
    from wonyotti_fr.net_exit_state import NetExitPolicy
    _, manager = bots(buy)
    manager = ExitMovePolicy(NetExitPolicy(HorizonExitPolicy(manager, .02)), .02)
    gate = model(8.)
    gate.predict = lambda values: np.full(len(values), score)
    policy = LifecycleNetPolicy(manager, gate)
    bar, state = history_inputs(5, direction=0, hold_bars=0, position_entry_time=None)
    waiting = policy(bar, state)
    bar, state = history_inputs(6, waiting.state, direction=0, hold_bars=0, position_entry_time=None)
    bar['close'] = close
    result = policy(bar, state)
    assert result.event == ('triggered' if score >= 8 else 'filtered') and not result.state
    if result.event == 'filtered':
        bar, state = history_inputs(7, result.state, direction=0, hold_bars=0, position_entry_time=None)
        bar['close'] = close
        assert policy(bar, state).event != 'triggered'


@pytest.mark.parametrize('scores', [(0., 0., 0.), (.8, .8, .8), (0., .8, .8), (0., 0., .8)])
def test_rejecting_entry_filter_preserves_held_management_and_own_history(scores):
    original = ExitMovePolicy(bot(scores), .02)
    candidate = LifecycleNetPolicy(original, model(-1e6))
    before, after = {}, {}
    for minute in range(1, 25):
        values = {'adds': int(minute >= 3), 'fraction': .7 if minute >= 5 else 1., 'estimated_exit_net': 5.}
        a, b = original(*history_inputs(minute, before, **values)), candidate(*history_inputs(minute, after, **values))
        assert a == b
        before, after = a.state, b.state


@pytest.mark.parametrize('delay', [0, 1])
def test_accepted_filter_keeps_execution_delay_and_nontradable_wait(tmp_path, monkeypatch, delay):
    _, parent, _, _, old = fixture(tmp_path, monkeypatch)
    _, original = load_selection(parent)
    frame = ready_bars().assign(count=1, volume=100.)
    frame.loc[5:9, ['count', 'volume']] = 0
    config = replace(EngineConfig(**old['risk']), signal_delay_bars=delay)
    for name, policy in [('parent', original), ('accepted', LifecycleNetPolicy(original, model(8.)))]:
        backtest(frame, policy, config, tmp_path/name)
    for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'parent'/name),
                                      pd.read_parquet(tmp_path/'accepted'/name), check_exact=True)
    fills = pd.read_parquet(tmp_path/'accepted/fills.parquet')
    assert len(fills) > 0 and not pd.to_datetime(fills.time, utc=True).isin(frame.time.iloc[5:10]).any()


@pytest.mark.parametrize('damage', ['margin', 'risk', 'parent', 'model', 'weight', 'loss', 'plan'])
def test_loaded_filter_rejects_changed_parent_model_risk_and_weights(tmp_path, monkeypatch, damage):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'margin':
        frozen['policy_entry_margin_bps'] = 0
    elif damage == 'risk':
        frozen['risk']['fee_bps'] = 0
    elif damage == 'parent':
        (root/'exit_move_selection.json').write_text('{}')
    elif damage == 'model':
        (root/ENTRY_FILES[0]).write_text('{}')
    elif damage == 'weight':
        p = root/ENTRY_FILES[2]
        frame = pd.read_parquet(p)
        frame.loc[0, 'sample_weight'] = 0
        frame.to_parquet(p, index=False)
        frozen['policy_entry_files_sha256'][p.name] = sha256(p)
    else:
        p = root/ENTRY_FILES[5]
        value = json.loads(p.read_text())
        value['losing_labels' if damage == 'loss' else 'protocol_sha256'] = 0
        save_json(p, value)
        frozen['policy_entry_files_sha256'][p.name] = sha256(p)
    save_json(root/'frozen_selection.json', frozen)
    save_json(root/'frozen_integrity.json', {'frozen_selection_sha256': sha256(root/'frozen_selection.json')})
    with pytest.raises((ValueError, AssertionError)):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_actual_crash_and_replay_keep_current_management_and_filter(tmp_path, monkeypatch, kind):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch, constant=True)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path/'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_full_selection_future_market_invariance_and_eleven_controls(tmp_path, monkeypatch):
    _, parent, labels, _, _ = fixture(tmp_path, monkeypatch)
    old, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent/'candidate-00')
    monkeypatch.setattr('wonyotti_fr.policy_entry_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_policy_entry_selection(parent, labels, labels, labels, labels, labels, tmp_path/'runs')
    assert json.loads((out/'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    assert load_selection(out)[0]['risk'] == old['risk']
    def future_prices(*args, **kwargs):
        frame = ready_bars()
        if args[3] == '2022-01-01':
            frame[['open', 'high', 'low', 'close']] *= 2.
        return frame, {}
    monkeypatch.setattr('wonyotti_fr.policy_entry_research.prepare_minute_period', future_prices)
    altered = run_policy_entry_selection(parent, labels, labels, labels, labels, labels, tmp_path/'changed')
    for name in ENTRY_FILES:
        assert (out/name).read_bytes() == (altered/name).read_bytes()
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    evaluation = run_pullback_evaluation(out, labels, labels, tmp_path/'evaluation', 'seen_2026', ['BTCUSDT'])
    names = {row['strategy'] for row in json.loads((evaluation/'results.json').read_text())}
    assert len(names) == 11 and {'previous_v48', 'previous_v54', 'unfiltered_v14'} <= names and 'previous_v53' not in names
    assert isinstance(load_selection(evaluation)[1], LifecycleNetPolicy)
    for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(parent/'candidate-00'/name),
            pd.read_parquet(evaluation/'BTCUSDT/previous_v54'/name), check_exact=True)
    assert json.loads((parent/'candidate-00/final_state.json').read_text()) == json.loads((evaluation/'BTCUSDT/previous_v54/final_state.json').read_text())
