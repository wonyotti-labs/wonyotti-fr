import copy
import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_addition_effect import event
from test_engine import config
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot

from wonyotti_fr.close_effect import (
    CLOSE_FEATURES,
    IndexedEvents,
    close_effect_label,
    collect_close_effects,
    immediate_close,
    run_close_effect_labels,
    validate_close_record,
)
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import PolicyDecision, TradingEngine
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.exit_move_state import ExitMovePolicy
from wonyotti_fr.outcome_journal import OutcomeJournal
from wonyotti_fr.streaming_backtest import streaming_backtest


def held(direction=1, **extra):
    cfg = config(bar_seconds=60, **extra)
    engine = TradingEngine(cfg)
    engine.state['pending'] = 'enter_long' if direction == 1 else 'enter_short'
    engine.step(event(0), lambda *_: 'hold')
    return cfg, engine.snapshot()


@pytest.mark.parametrize('direction', [1, -1])
def test_immediate_close_exact_fee_slippage_funding_and_unchanged_start(direction):
    cfg, state = held(direction, fee_bps=5, slippage_bps=3)
    before = copy.deepcopy(state)
    future = [event(1, 105., funding_rate=.001), event(2, 90.)]
    trace = immediate_close(future, 0, state, cfg, pd.Timestamp('2021-01-02', tz='UTC'))
    q, execution = state['quantity'], 105*(1-direction*.0003)
    expected = state['cash']+q*execution-abs(q)*execution*.0005-q*105*.001
    assert trace['final_cash'] == pytest.approx(expected, abs=1e-9)
    assert trace['status'] == 'closed' and trace['observed_minutes'] == 1
    assert trace['remaining_quantity'] == 0 and len(trace['fills']) == 1 and state == before


@pytest.mark.parametrize('price,sign', [(90., 1), (110., -1), (100., 0)])
def test_close_advantage_keeps_loss_avoidance_lost_profit_and_zero(price, sign):
    cfg, state = held()
    future = [event(1), event(2, price)]
    cutoff = pd.Timestamp('2021-01-02', tz='UTC')
    trace = immediate_close(future, 0, state, cfg, cutoff)
    engine = TradingEngine(cfg, state)
    engine.step(future[0], lambda *_: 'exit')
    continued = engine.step(future[1], lambda *_: 'hold')
    trade = continued['closed_trades'][0]
    opportunity = {'state': state, 'decision_time': state['last_end'], 'start': 0,
                   **dict.fromkeys(CLOSE_FEATURES, 0.)}
    outcome = close_effect_label(opportunity, trace, trade, cfg, cutoff)
    assert np.sign(outcome['close_advantage_pnl']) == sign
    assert outcome['continue_cash'] == pytest.approx(engine.state['cash'])
    assert outcome['close_advantage_bps'] == pytest.approx((trace['final_cash']-engine.state['cash'])/10000*10000)
    record = {'opportunity': opportunity, 'close': trace, 'outcome': outcome}
    validate_close_record(opportunity, record, trade, cfg, cutoff, future)
    damaged = copy.deepcopy(record)
    damaged['outcome']['close_advantage_bps'] += 1
    with pytest.raises(ValueError, match='정답'):
        validate_close_record(opportunity, damaged, trade, cfg, cutoff, future)
    damaged = copy.deepcopy(record)
    damaged['close']['final_cash'] += 1
    with pytest.raises(ValueError, match='현금'):
        validate_close_record(opportunity, damaged, trade, cfg, cutoff, future)


def test_no_trade_wait_gap_risk_priority_and_boundary_censoring():
    cfg, state = held(stop_fraction=.04)
    future = [event(1, 100., count=0, volume=0, funding_rate=.001), event(2, 90., funding_rate=.001)]
    cutoff = pd.Timestamp('2021-01-02', tz='UTC')
    trace = immediate_close(future, 0, state, cfg, cutoff)
    assert trace['observed_minutes'] == 2 and trace['fills'][0]['reason'] == 'gap_stop'
    assert trace['funding_cost'] == pytest.approx(state['quantity']*(100+90)*.001)
    censored = immediate_close(future, 0, state, cfg, pd.Timestamp(future[1]['end']))
    assert censored['status'] == 'right_censored' and not censored['fills']
    assert censored['remaining_quantity'] == state['quantity']
    with pytest.raises(ValueError, match='지연'):
        immediate_close(future, 0, state, replace(cfg, signal_delay_bars=1), cutoff)


def test_original_exit_is_zero_and_forced_end_never_becomes_target():
    cfg, state = held(fee_bps=5, slippage_bps=3)
    state['pending'] = 'exit'
    future = [event(1, funding_rate=.001)]
    cutoff = pd.Timestamp('2021-01-02', tz='UTC')
    trace = immediate_close(future, 0, state, cfg, cutoff)
    original = TradingEngine(cfg, state).step(future[0], lambda *_: 'hold')['closed_trades'][0]
    opportunity = {'state': state, 'decision_time': state['last_end']}
    result = close_effect_label(opportunity, trace, original, cfg, cutoff)
    assert result['close_advantage_pnl'] == pytest.approx(0., abs=1e-9)
    forced = close_effect_label(opportunity, trace, {**original, 'exit_reason': 'end_of_test', 'exit_time': future[0]['end']}, cfg, cutoff)
    assert forced['label_status'] == 'right_censored' and forced['close_advantage_pnl'] is None
    boundary = close_effect_label(opportunity, trace, original, cfg, pd.Timestamp(future[0]['end']))
    assert boundary['label_status'] == 'right_censored' and boundary['close_advantage_bps'] is None


def test_indexed_market_events_match_stream_without_future_values():
    frame = ready_bars().assign(count=1, volume=1.)
    indexed = IndexedEvents(frame)
    assert [indexed[i] for i in range(len(indexed))] == list(iter_events(frame))
    changed = frame.copy()
    changed.loc[changed.index >= 20, ['open', 'high', 'low', 'close']] *= 3
    later = IndexedEvents(changed)
    assert [indexed[i] for i in range(20)] == [later[i] for i in range(20)]


class CollectionPolicy:
    manager = SimpleNamespace(features=CLOSE_FEATURES)

    def prepare(self, frame):
        pass

    def feature_values(self, bar, view):
        values = np.zeros(len(CLOSE_FEATURES))
        values[CLOSE_FEATURES.index('direction')] = view['direction']
        values[CLOSE_FEATURES.index('favorable_move')] = view['favorable_move']
        return values

    def __call__(self, bar, view):
        if not view['direction']:
            return PolicyDecision('enter_long', {}, 'test_enter')
        self.feature_values(bar, view)
        return PolicyDecision('exit' if view['hold_bars'] >= 17 else 'hold', {}, 'test_management')


def collection_fixture(tmp_path):
    frame = ready_bars().assign(count=1, volume=1.)
    frame[['time', 'end']] += pd.Timedelta(days=366)
    cfg = config(bar_seconds=60, fee_bps=5, slippage_bps=3)
    root = tmp_path/'parent'
    root.mkdir()
    streaming_backtest(frame, CollectionPolicy(), cfg, root/'candidate-00', 8192)
    save_json(root/'frozen_selection.json', {'protocol': 'exit_move_v54', 'risk': asdict(cfg)})
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        save_json(tmp_path/name, {})
    return frame, cfg, root


def test_collection_resume_matches_full_and_all_original_outputs(tmp_path):
    frame, cfg, root = collection_fixture(tmp_path)
    cutoff = frame.end.iloc[30]
    with OutcomeJournal(tmp_path/'resume.sqlite', {'fixed': True}) as journal:
        rows, _, complete = collect_close_effects(frame, CollectionPolicy(), cfg, root/'candidate-00', tmp_path/'partial', journal, cutoff, 2)
        assert not complete and len(rows) == 2
        resumed, counts, complete = collect_close_effects(frame, CollectionPolicy(), cfg, root/'candidate-00', tmp_path/'resumed', journal, cutoff)
        assert complete and counts['boundaries'] == len(frame)//5
        assert counts['boundaries'] == counts['opportunities']+sum(v for k, v in counts.items() if k.startswith('excluded_'))
    with OutcomeJournal(tmp_path/'fresh.sqlite', {'fixed': True}) as journal:
        fresh, _, complete = collect_close_effects(frame, CollectionPolicy(), cfg, root/'candidate-00', tmp_path/'fresh', journal, cutoff)
    assert resumed == fresh and complete
    assert any(r['label_status'] == 'closed' for r in fresh)
    assert any(r['label_status'] == 'right_censored' for r in fresh)
    assert all(r['close_advantage_pnl'] is not None for r in fresh if r['label_status'] == 'closed')


def test_real_history_policy_features_and_future_price_invariance(tmp_path):
    frame = ready_bars().assign(count=1, volume=1.)
    frame[['time', 'end']] += pd.Timedelta(days=366)
    cfg = config(bar_seconds=60, max_hold_bars=20, fee_bps=5, slippage_bps=3)
    outputs = []
    for name, changed in [('original', False), ('future', True)]:
        data = frame.copy()
        if changed:
            data.loc[data.index >= 40, ['open', 'high', 'low', 'close']] *= 1.2
        policy = ExitMovePolicy(bot((0., .05, .8)), .02)
        policy.manager.features = CLOSE_FEATURES
        reference = tmp_path/(name+'-reference')
        streaming_backtest(data, policy, cfg, reference, 8192)
        with OutcomeJournal(tmp_path/(name+'.sqlite'), {'case': name}) as journal:
            rows, _, complete = collect_close_effects(data, policy, cfg, reference, tmp_path/(name+'-replay'),
                journal, pd.Timestamp('2021-12-31', tz='UTC'))
            assert complete and rows
            records = [json.loads(r[0]) for r in journal.connection.execute('SELECT payload FROM outcomes ORDER BY sequence')]
        outputs.append([r['opportunity'] for r in records if pd.Timestamp(r['opportunity']['decision_time']) <= data.time.iloc[40]])
    assert outputs[0] and outputs[0] == outputs[1]


def test_pipeline_and_changed_input_rejection(tmp_path, monkeypatch):
    frame, cfg, root = collection_fixture(tmp_path)
    frozen = json.loads((root/'frozen_selection.json').read_text())
    monkeypatch.setattr('wonyotti_fr.close_effect.load_selection', lambda _: (frozen, CollectionPolicy()))
    monkeypatch.setattr('wonyotti_fr.close_effect.prepare_minute_period', lambda *_, **__: (frame.copy(), {}))
    out = run_close_effect_labels(root, tmp_path, tmp_path, tmp_path/'runs', max_opportunities=2)
    assert not json.loads((out/'summary.json').read_text())['complete']
    run_close_effect_labels(root, tmp_path, tmp_path, tmp_path/'unused', resume=out)
    result = json.loads((out/'summary.json').read_text())
    assert result['complete'] and not result['new_models_fitted'] and not result['profitability_accepted']
    for name, checksum in json.loads((out/'files.json').read_text()).items():
        assert sha256(out/name) == checksum
    with pytest.raises(ValueError, match='완료'):
        run_close_effect_labels(root, tmp_path, tmp_path, tmp_path/'unused', resume=out)
    partial = run_close_effect_labels(root, tmp_path, tmp_path, tmp_path/'more', max_opportunities=1)
    (tmp_path/'manifest-1m.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='변경'):
        run_close_effect_labels(root, tmp_path, tmp_path, tmp_path/'unused', resume=partial)


@pytest.mark.parametrize('before', [True, False])
def test_actual_process_interruption_preserves_complete_close_record(tmp_path, before):
    cfg, state = held()
    future, cutoff = [event(1)], pd.Timestamp('2021-01-02', tz='UTC')
    trace = immediate_close(future, 0, state, cfg, cutoff)
    payload = {'opportunity': {'state': state}, 'close': trace, 'outcome': {'close_advantage_bps': -1.}}
    input_path = tmp_path/'case.json'
    save_json(input_path, payload)
    db = tmp_path/'outcomes.sqlite'
    code = '''import json,os,sys
from pathlib import Path
from wonyotti_fr.outcome_journal import OutcomeJournal
record=json.loads(Path(sys.argv[2]).read_text())
with OutcomeJournal(Path(sys.argv[1]), {'purpose':'close'}) as journal:
 journal.append(0, record['opportunity'], record, before_commit=(lambda:os._exit(73)) if sys.argv[3]=='True' else None)
 os._exit(73)
'''
    assert subprocess.run([sys.executable, '-c', code, str(db), str(input_path), str(before)], check=False).returncode == 73
    record = json.loads(input_path.read_text())
    with OutcomeJournal(db, {'purpose': 'close'}) as journal:
        assert journal.count() == (0 if before else 1)
        journal.append(0, record['opportunity'], record)
        assert journal.read(0, record['opportunity']) == record
