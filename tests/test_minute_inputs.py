import numpy as np
import pandas as pd
import pytest
from test_inventory_management import bot
from test_rate_policy import inputs

from wonyotti_fr.engine import EngineConfig, TradingEngine, validate_bar
from wonyotti_fr.event_backtest import iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.minute_inputs import MINUTE_FEATURES, minute_features
from wonyotti_fr.minute_inventory import MinuteInventoryPolicy, attach_minute_inputs


def minutes(n=120):
    time = pd.date_range('2021-01-01', periods=n, freq='min', tz='UTC')
    close = 100 + np.arange(n)*.01
    frame = pd.DataFrame({'time': time, 'end': time+pd.Timedelta(minutes=1),
        'open': close, 'close': close, 'high': close+.02, 'low': close-.01,
        'volume': np.arange(n, dtype=float)+1, 'count': 1, 'funding_rate': 0.})
    for key in MARKET_FEATURES:
        frame[key] = 0.
    return frame


def test_closed_minute_values_and_previous_volume_mean():
    frame = minutes()
    values = minute_features(frame)
    assert values.minute_volume_ratio_1h.iloc[:60].isna().all()
    row = values.iloc[60]
    assert row.minute_ret_1m == pytest.approx(frame.close.iloc[60]/frame.close.iloc[59]-1)
    assert row.minute_ret_3m == pytest.approx(frame.close.iloc[60]/frame.close.iloc[57]-1)
    assert row.minute_range_fraction == pytest.approx(.03/frame.close.iloc[60])
    assert row.minute_volume_ratio_1h == pytest.approx(61/30.5)
    # 현재 거래량 급등이 분모를 바꾸지 않는지 확인한다.
    frame.loc[60, 'volume'] *= 100
    assert minute_features(frame).minute_volume_ratio_1h.iloc[60] == pytest.approx(row.minute_volume_ratio_1h*100)


def test_future_price_volume_changes_and_prefix_calculation_are_invariant():
    frame = minutes()
    full = minute_features(frame)
    pd.testing.assert_frame_equal(full.iloc[:80], minute_features(frame.iloc[:80]), check_exact=True)
    changed = frame.copy()
    changed.loc[80:, ['open', 'close', 'high', 'low', 'volume']] *= 5
    pd.testing.assert_frame_equal(minute_features(changed).iloc[:80], full.iloc[:80], check_exact=True)


@pytest.mark.parametrize('damage', ['gap', 'duplicate', 'not_closed', 'bad_high', 'negative_volume'])
def test_invalid_minute_input_is_rejected(damage):
    frame = minutes()
    if damage == 'gap':
        frame = frame.drop(index=10)
    elif damage == 'duplicate':
        frame.loc[10, ['time', 'end']] = frame.loc[9, ['time', 'end']]
    elif damage == 'not_closed':
        frame.loc[10, 'end'] += pd.Timedelta(minutes=1)
    elif damage == 'bad_high':
        frame.loc[10, 'high'] = 1
    else:
        frame.loc[10, 'volume'] = -1
    with pytest.raises(ValueError):
        minute_features(frame)


def test_zero_volume_history_is_unavailable_without_future_fill():
    frame = minutes()
    frame.loc[:59, 'volume'] = 0
    values = minute_features(frame)
    assert np.isnan(values.minute_volume_ratio_1h.iloc[60])
    assert np.isfinite(values.minute_volume_ratio_1h.iloc[61])


def test_event_serialization_keeps_legacy_features_and_optional_context():
    frame = minutes()
    values = minute_features(frame)
    original = list(iter_events(frame))
    extended = list(iter_events(frame.merge(values, on='end', validate='one_to_one')))
    for a, b in zip(original, extended, strict=True):
        assert len(b['minute_features']) == 4
        assert a == {k: v for k, v in b.items() if k != 'minute_features'}
        validate_bar(b, 60)
    assert extended[0]['minute_features'][0] is None
    np.testing.assert_array_equal(extended[70]['minute_features'], values.loc[70, MINUTE_FEATURES])
    incomplete = frame.assign(minute_ret_1m=0.)
    with pytest.raises(ValueError, match='일부 누락'):
        list(iter_events(incomplete))


@pytest.mark.parametrize('values', [[0.]*3, [True]*4, [float('inf')]*4, 'bad'])
def test_invalid_event_context_rolls_back_engine(values):
    engine = TradingEngine(EngineConfig(bar_seconds=60))
    before = engine.snapshot()
    event = next(iter_events(minutes(1)))
    event['minute_features'] = values
    with pytest.raises(ValueError):
        engine.step(event, lambda *_: 'hold')
    assert engine.snapshot() == before


def test_directional_context_matches_runtime_and_preserves_source_state():
    frame = minutes().assign(direction=-1, usable=False)
    frame.loc[60:, 'usable'] = True
    values = minute_features(frame)
    attached = attach_minute_inputs(frame, values)
    pd.testing.assert_frame_equal(attached[frame.columns], frame, check_exact=True)
    base = bot(intercept=None)
    policy = MinuteInventoryPolicy(base, base.manager, base.thresholds, base.multiplier, base.scales)
    bar, state = inputs(1)
    state.update(direction=-1, remaining_fraction=.5, _path_bounds=(100., 100.))
    bar['minute_features'] = values.loc[70, MINUTE_FEATURES].to_list()
    got = policy.feature_values(bar, state)
    assert len(got) == 44
    np.testing.assert_array_equal(got[-8:], attached.loc[70, MINUTE_FEATURES+[f'directional_{k}' for k in MINUTE_FEATURES]].to_numpy(dtype=float))
    with pytest.raises(ValueError, match='누락'):
        attach_minute_inputs(frame.assign(usable=True), values)


def test_prepared_period_uses_only_available_past_warmup_and_preserves_legacy_rows(monkeypatch, tmp_path):
    from test_minute_data import source

    from wonyotti_fr.minute_data import prepare_minute_period
    minute, five = source()
    funding = pd.DataFrame({'time': pd.to_datetime([], utc=True), 'rate': pd.Series(dtype=float)})
    monkeypatch.setattr('wonyotti_fr.minute_data.load_market', lambda _p, _s, interval: (minute if interval == '1m' else five, funding))
    monkeypatch.setattr('wonyotti_fr.minute_data.check_funding_coverage', lambda *_: None)
    legacy, _ = prepare_minute_period(tmp_path, tmp_path, 'BTCUSDT', '2020-01-02', '2020-01-03')
    enriched, checks = prepare_minute_period(tmp_path, tmp_path, 'BTCUSDT', '2020-01-02', '2020-01-03', minute_inputs=True)
    pd.testing.assert_frame_equal(enriched[legacy.columns], legacy, check_exact=True)
    assert checks['minute_input_warmup_rows'] == 60 and checks['minute_input_available_rows'] == len(legacy)
    expected = minute_features(minute).set_index('end').loc[enriched.end, MINUTE_FEATURES].to_numpy()
    np.testing.assert_allclose(enriched[MINUTE_FEATURES], expected, rtol=0, atol=1e-12)
    first, checks = prepare_minute_period(tmp_path, tmp_path, 'BTCUSDT', '2020-01-01', '2020-01-02', minute_inputs=True)
    assert checks['minute_input_warmup_rows'] == 0
    assert first.minute_volume_ratio_1h.iloc[:60].isna().all()
    assert np.isfinite(first[MINUTE_FEATURES].iloc[60:]).all().all()


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_utc_timestamp_resolution_does_not_change_confirmed_inputs(unit):
    frame = minutes()
    expected = minute_features(frame)
    frame['time'] = frame.time.astype(f'datetime64[{unit}, UTC]')
    frame['end'] = frame.end.astype(f'datetime64[{unit}, UTC]')
    pd.testing.assert_frame_equal(minute_features(frame), expected, check_exact=True)
    frame['end'] = frame.end.dt.tz_localize(None)
    with pytest.raises(ValueError):
        minute_features(frame)
