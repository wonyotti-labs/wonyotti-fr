import json

import pandas as pd
import pytest
from test_action_model import FixedScores
from test_path_management import path_selection
from test_pullback_evaluation import ConstantBase, bars

from wonyotti_fr.action_research import action_diagnostics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.path_management import ReversalPathPolicy
from wonyotti_fr.pullback_policy import PullbackPolicy


def policy():
    return ReversalPathPolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores([.9, 0, 0]),
                              {'exit': .5, 'reduce': .5, 'increase': .5})


def reversal_selection(root):
    frozen = path_selection(root)
    model = json.loads((root / 'action_model.json').read_text())
    model['intercept'] = [10., -10., -10.]
    save_json(root / 'action_model.json', model)
    frozen.update(thresholds={'exit': .1, 'reduce': .9, 'increase': .9},
                  model_sha256={'action_model.json': sha256(root / 'action_model.json')})
    save_json(root / 'frozen_selection.json', frozen)
    (root / 'path_selection.json').write_bytes((root / 'frozen_selection.json').read_bytes())
    frozen.update(protocol='minute_reverse_v13', candidate=0,
                  path_selection_sha256=sha256(root / 'path_selection.json'))
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return frozen


@pytest.mark.parametrize('delay', [0, 1])
def test_reversal_two_legs_costs_cooldown_and_path_reset(tmp_path, delay):
    frame = bars()
    config = EngineConfig(bar_seconds=60, max_hold_bars=0, signal_delay_bars=delay)
    metrics = backtest(frame, policy(), config, tmp_path / 'run')
    report = action_diagnostics(tmp_path / 'run', frame, config)
    fills = pd.read_parquet(tmp_path / 'run/fills.parquet')
    reversals = fills[fills.reason.eq('signal_reverse')]
    assert len(reversals) >= 2 and metrics['max_accounting_residual'] < 1e-7
    assert pd.to_datetime(reversals.time, utc=True).diff().dropna().ge(pd.Timedelta(minutes=3)).all()
    curve = pd.read_parquet(tmp_path / 'run/equity.parquet')
    for leg in reversals.itertuples():
        pair = fills[fills.time.eq(leg.time)]
        assert pair.reason.tolist() == ['signal_reverse', 'entry']
        assert pair.fee.gt(0).all()
        assert pair.delta_quantity.prod() > 0
        first_end = pd.Timestamp(leg.time) + pd.Timedelta(minutes=1)
        row = curve[pd.to_datetime(curve.time, utc=True).eq(first_end)].iloc[0]
        if row.completed:
            assert row.quantity == 0 and row.policy_state == '{}'
            continue
        state = json.loads(row.policy_state)
        assert state['path_entry_time'] == leg.time
        assert state['path_low'] == state['path_high']
        assert state['management_direction'] == state['path_direction']
        assert row.policy_event == 'action_cooldown'
    assert report['waiting']['reversal_entries'] == len(reversals)
    assert report['waiting']['matched_entry_fills'] == metrics['closed_trades']


def test_gap_stop_prevents_a_pending_model_reversal():
    config = EngineConfig(bar_seconds=60, stop_fraction=.04, max_hold_bars=0)
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long'
    events = list(iter_events(bars()))
    first = engine.step({**events[0], 'open': 100., 'high': 100., 'low': 99., 'close': 100.}, policy())
    assert first['next_intent'] == 'hold'
    second = engine.step({**events[1], 'open': 100., 'high': 100., 'low': 99., 'close': 100.}, policy())
    assert second['next_intent'] == 'enter_short'
    next_event = {**events[2], 'open': 94., 'high': 95., 'low': 93., 'close': 94.}
    result = engine.step(next_event, policy())
    assert [f['reason'] for f in result['fills']] == ['gap_stop']
    assert result['quantity'] == 0 and result['next_intent'] == 'hold'


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_reversal_frozen_model_and_actual_recovery(tmp_path, kind):
    root = tmp_path / 'selection'
    frozen = reversal_selection(root)
    assert isinstance(load_selection(root)[1], ReversalPathPolicy)
    report = verify_stress(list(iter_events(bars())), root, tmp_path, {}, kind, True)
    assert report['all_passed'] and report['child_process_exit_code'] == 73
    frozen['multiplier'] = 1.5
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError, match='고정 기반'):
        load_selection(root)
