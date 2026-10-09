import numpy as np
import pandas as pd
import pytest
from test_close_context import dataset
from test_close_learning import real_source

from wonyotti_fr.close_context import attach_close_context
from wonyotti_fr.close_economics import ECONOMIC_FEATURES, load_economic_inputs
from wonyotti_fr.close_flow import FLOW_FEATURES, FLOW_WINDOWS, attach_close_flow
from wonyotti_fr.close_learning_inputs import load_close_training
from wonyotti_fr.context_position import CONTEXT_FEATURES
from wonyotti_fr.continuation_inputs import PENDING_FEATURES, load_pending_inputs
from wonyotti_fr.minute_close_effects import run_minute_close_effect_labels
from wonyotti_fr.minute_close_inputs import attach_minute_close_inputs


def minute_dataset(start=19000):
    market, frame = dataset(start=start, size=20)
    market['taker_buy_volume'] = market.volume*(np.arange(len(market))%3)/2
    minute = frame.loc[frame.index.repeat(5)].reset_index(drop=True)
    minute['decision_time'] += pd.to_timedelta(np.tile(np.arange(5), len(frame)), unit='min')
    minute['direction'] = np.where(np.arange(len(minute))%2, -1., 1.)
    return market, minute


def test_confirmed_minute_attachment_preserves_every_original_and_legacy_grid_value():
    market, frame = minute_dataset()
    result, linkage = attach_minute_close_inputs(frame, market)
    pd.testing.assert_frame_equal(result.drop(columns=CONTEXT_FEATURES+FLOW_FEATURES), frame, check_exact=True)
    original = frame.iloc[::5].reset_index(drop=True)
    expected = attach_close_flow(attach_close_context({'legacy': original}, market), market)['legacy']
    pd.testing.assert_frame_equal(result.iloc[::5].reset_index(drop=True), expected, check_exact=True)
    np.testing.assert_array_equal(linkage.elapsed_seconds, np.tile(np.arange(5)*60, 20))
    pd.testing.assert_series_equal(linkage.confirmed_feature_end, frame.decision_time.dt.floor('5min'), check_names=False)
    for name in FLOW_WINDOWS:
        np.testing.assert_array_equal(result['directional_'+name], result[name]*frame.direction)
    future = market.copy()
    changed = future.end.gt(frame.decision_time.max())
    future.loc[changed, ['open', 'high', 'low', 'close']] *= 10
    future.loc[changed, 'taker_buy_volume'] = -1e10
    other, links = attach_minute_close_inputs(frame, future)
    pd.testing.assert_frame_equal(other, result, check_exact=True)
    pd.testing.assert_frame_equal(links, linkage, check_exact=True)


@pytest.mark.parametrize('damage', ['stale', 'before_first', 'warmup', 'gap', 'past_value', 'duplicate', 'subminute', 'timezone', 'direction', 'nonfinite', 'extra'])
def test_minute_attachment_rejects_time_and_context_damage_without_removing_original_rows(damage):
    market, frame = minute_dataset(start=1000 if damage == 'warmup' else 19000)
    if damage == 'stale':
        market = market[market.end.ne(frame.decision_time.iloc[0])].reset_index(drop=True)
    elif damage == 'before_first':
        frame = frame.iloc[:1].copy()
        frame['decision_time'] = market.end.iloc[0]-pd.Timedelta(minutes=1)
    elif damage == 'gap':
        market = market.drop(index=10000).reset_index(drop=True)
    elif damage == 'past_value':
        frame.loc[0, 'ret_5m'] += 1
    elif damage == 'duplicate':
        frame.loc[1, 'decision_time'] = frame.decision_time.iloc[0]
    elif damage == 'subminute':
        frame.loc[0, 'decision_time'] += pd.Timedelta(seconds=1)
    elif damage == 'timezone':
        frame['decision_time'] = frame.decision_time.dt.tz_convert('Asia/Seoul')
    elif damage == 'direction':
        frame.loc[0, 'direction'] = 0.
    elif damage == 'nonfinite':
        frame.loc[0, 'current_exit_net_bps'] = np.inf
    elif damage == 'extra':
        frame['ret_4d'] = 0.
    preserved = frame.copy(deep=True)
    with pytest.raises((ValueError, AssertionError)):
        attach_minute_close_inputs(frame, market)
    pd.testing.assert_frame_equal(frame, preserved, check_exact=True)


def test_minute_atomic_economics_and_pending_inputs_preserve_every_five_minute_value(tmp_path, monkeypatch):
    legacy = real_source(tmp_path, monkeypatch)
    minute = run_minute_close_effect_labels(legacy, tmp_path/'minute')
    original, _ = load_close_training(legacy)
    full, _ = load_close_training(minute, decision_seconds=60)
    old_economic, _ = load_economic_inputs(legacy, original)
    old_all, _ = load_pending_inputs(legacy, old_economic)
    new_economic, economic_proof = load_economic_inputs(minute, full, decision_seconds=60)
    new_all, pending_proof = load_pending_inputs(minute, new_economic, decision_seconds=60)
    pd.testing.assert_frame_equal(new_all.drop(columns=ECONOMIC_FEATURES+PENDING_FEATURES), full, check_exact=True)
    pd.testing.assert_frame_equal(new_all[new_all.decision_time.isin(original.decision_time)].reset_index(drop=True), old_all, check_exact=True)
    assert economic_proof['rows'] == pending_proof['rows'] == len(full)
    assert economic_proof['future_fill_or_outcome_used'] is False and pending_proof['future_execution_or_target_used'] is False
    with pytest.raises(ValueError, match='파일'):
        load_economic_inputs(minute, full)
    with pytest.raises(ValueError, match='파일'):
        load_pending_inputs(minute, new_economic)
    with pytest.raises(ValueError, match='파일'):
        load_economic_inputs(legacy, original, decision_seconds=60)


@pytest.mark.parametrize('seconds', [True, 60., 120, 0])
def test_atomic_feature_grid_must_be_explicit_and_supported(tmp_path, seconds):
    for function in [load_economic_inputs, load_pending_inputs]:
        with pytest.raises(ValueError, match='간격'):
            function(tmp_path/'missing', pd.DataFrame(), decision_seconds=seconds)
