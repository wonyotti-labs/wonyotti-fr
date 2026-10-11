import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_activity_ablation import bots
from test_history_state import history_inputs
from test_minute_inventory_research import ready_bars

from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.exit_state import ExitStatePolicy
from wonyotti_fr.first_exit_verification import verify_execution
from wonyotti_fr.management_support import (
    FEATURES,
    ManagementInputTrace,
    SupportCounts,
    joint_codes,
    training_profile,
)
from wonyotti_fr.management_support_research import (
    reference_counts,
    summarize_parquet,
    verify_profile,
)
from wonyotti_fr.streaming_backtest import ParquetRows


def values(rows=101):
    x = np.tile(np.linspace(0, 1, rows).reshape(-1, 1), (1, 50))
    x[:, 28], x[:, 31], x[:, 35], x[:, 44], x[:, 47] = 1, 0, 1, 0, 0
    return x


def parent(scores=(0., 0., 0.)):
    _, previous = bots(manager=scores)
    result = ExitStatePolicy(previous)
    result.manager.features = FEATURES
    return result


def test_profile_fixed_training_quantiles_and_closed_boundaries():
    x = values()
    profile = training_profile(x)
    verify_profile(pd.DataFrame(x, columns=FEATURES), profile)
    np.testing.assert_allclose(np.asarray(profile['bounds'])[:, 0],
                               [0., .01, .05, .25, .5, .75, .95, .99, 1.], rtol=0, atol=1e-15)
    counter = SupportCounts(profile)
    for part in np.array_split(x, 7):
        counter.update(part)
    result = counter.result()
    assert result == reference_counts(pd.DataFrame(x, columns=FEATURES), profile)
    assert result['feature_counts']['ret_5m']['outside_training_range'] == 0
    assert result['feature_counts']['ret_5m']['outside_training_central98'] == 2
    assert result['unseen_joint_rows'] == 0


def test_nonfinite_and_unseen_joint_are_separate_from_range_counts():
    x = values()
    profile = training_profile(x)
    probe = x[:4].copy()
    probe[0, 0] = -1
    probe[1, 0] = np.nan
    probe[2, 0] = np.inf
    probe[3, 31] = 2
    result = SupportCounts(profile)
    result.update(probe)
    actual = result.result()
    assert actual == reference_counts(pd.DataFrame(probe, columns=FEATURES), profile)
    assert actual['finite_rows'] == 2 and actual['nonfinite_rows'] == 2
    assert actual['unseen_joint_rows'] == 1
    assert actual['feature_counts']['ret_5m']['outside_training_range'] == 1
    assert actual['feature_counts']['ret_5m']['nonfinite'] == 2


def test_duplicate_quantiles_and_empty_comparison():
    x = values(1)
    profile = training_profile(x)
    verify_profile(pd.DataFrame(x, columns=FEATURES), profile)
    counter = SupportCounts(profile)
    counter.update(np.empty((0, 50)))
    assert counter.result()['rows'] == 0
    assert counter.result() == reference_counts(pd.DataFrame(columns=FEATURES, dtype=float), profile)
    counter.update(x)
    assert counter.result()['group_outside_training_central98']['all'] == 0


@pytest.mark.parametrize('column,value', [(28, 0), (31, .5), (31, 6), (35, 0), (35, 1.1), (44, 2), (47, -1)])
def test_invalid_joint_states_rejected(column, value):
    x = values(1)
    x[0, column] = value
    with pytest.raises(ValueError):
        joint_codes(x)


@pytest.mark.parametrize('key,value', [('features', ['wrong']), ('rows', 0),
    ('fitted_from', 'diagnosis'), ('joint_counts', [0]*96), ('bounds', [[0]*50]*8)])
def test_tampered_training_profile_rejected(key, value):
    profile = training_profile(values())
    profile[key] = value
    with pytest.raises(ValueError):
        SupportCounts(profile)


@pytest.mark.parametrize('scores', [(0., 0., 0.), (.9, .9, .9), (0., .9, .9), (0., 0., .9)])
@pytest.mark.parametrize('direction', [-1, 1])
def test_tracing_preserves_decision_and_input_state(scores, direction):
    policy, records = parent(scores), []
    bar, state = history_inputs(1, direction=direction)
    before = copy.deepcopy(state)
    expected = policy(bar, state)
    assert 'feature_values' not in vars(policy)
    trace = ManagementInputTrace(policy, records.append)
    assert trace(bar, state) == expected
    assert state == before and 'feature_values' not in vars(policy)
    assert len(records) == 1 and records[0]['direction'] == direction
    assert records[0]['intent'] == expected.intent
    assert trace.coverage()['all_captured_inputs_formula_verified']


def test_failure_restores_class_and_instance_method_and_detects_wrong_formula():
    policy = parent()
    bar, state = history_inputs(1)
    original = policy.feature_values
    def invalid(*args):
        value = original(*args).copy()
        value[0] += 1
        return value
    policy.feature_values = invalid
    with pytest.raises(ValueError, match='별도 수식'):
        ManagementInputTrace(policy, lambda row: None)(bar, state)
    assert policy.feature_values is invalid
    def failure(*_):
        raise RuntimeError('주입 오류')
    policy.feature_values = failure
    with pytest.raises(RuntimeError):
        ManagementInputTrace(policy, lambda row: None)(bar, state)
    assert policy.feature_values is failure
    del policy.feature_values
    with pytest.raises(RuntimeError):
        ManagementInputTrace(policy, failure)(bar, state)
    assert 'feature_values' not in vars(policy)
    with pytest.raises(ValueError):
        ManagementInputTrace(SimpleNamespace(manager=SimpleNamespace(features=[])), lambda row: None)


def test_flat_or_waiting_rows_are_counted_without_manufactured_features():
    policy, records = parent(), []
    trace = ManagementInputTrace(policy, records.append)
    bar, state = history_inputs(1, direction=0, hold_bars=0, position_entry_time=None)
    trace(bar, state)
    assert not records and trace.coverage()['management_input_calls'] == 0
    assert sum(trace.coverage()['without_management_input'].values()) == 1


def test_actual_engine_replay_and_parquet_summary(tmp_path):
    bars = ready_bars().assign(count=1, volume=1.)
    config = EngineConfig(bar_seconds=60, max_hold_bars=20, fee_bps=5, slippage_bps=3)
    backtest(bars, parent(), config, tmp_path/'baseline')
    writer = ParquetRows(tmp_path/'trace.parquet', 7)
    trace = ManagementInputTrace(parent(), writer.append)
    try:
        proof = verify_execution(tmp_path/'baseline', bars, trace)
    finally:
        writer.close()
    assert proof['all_policy_decisions_fills_trades_equity_and_state_exact']
    frame = pd.read_parquet(tmp_path/'trace.parquet')
    assert len(frame) > 5
    profile = training_profile(frame[FEATURES])
    summary = summarize_parquet(tmp_path/'trace.parquet', profile)
    assert summary['all']['rows'] == len(frame)
    assert sum(row['rows'] for key, row in summary.items() if key != 'all') == len(frame)


def test_future_change_preserves_trace_prefix_and_restart_state():
    bars = ready_bars().assign(count=1, volume=1.)
    def run(data, resume):
        records, policy = [], parent((0., .9, 0.))
        trace = ManagementInputTrace(policy, records.append)
        trace.prepare(data)
        engine = TradingEngine(EngineConfig(bar_seconds=60, max_hold_bars=20))
        for i, bar in enumerate(iter_events(data)):
            engine.step(bar, trace, final=i == len(data)-1)
            if resume and i % 5 == 0:
                engine = TradingEngine(engine.config, engine.snapshot())
        return records
    original = run(bars, False)
    assert original == run(bars, True)
    altered = bars.copy()
    altered.loc[altered.index >= 40, ['open', 'high', 'low', 'close']] *= 1.2
    cutoff = bars.iloc[40].end
    assert [r for r in original if pd.Timestamp(r['end']) < cutoff] == [
        r for r in run(altered, False) if pd.Timestamp(r['end']) < cutoff]
