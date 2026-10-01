import json

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_diagnostics import decompose_run
from wonyotti_fr.event_features import MARKET_FEATURES


def bars():
    times = pd.date_range('2020-01-01T23:00Z', periods=500, freq='5min')
    price = 100 + 3 * np.sin(np.arange(len(times)) / 9)
    frame = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5),
                          'open': price, 'high': price + 2, 'low': price - 2, 'close': price + .5,
                          'funding_rate': np.where(np.arange(len(times)) % 96 == 0, .001, 0)})
    frame[MARKET_FEATURES] = 0.
    return frame


def policy(event, state):
    index = pd.Timestamp(event['time']).value // (300 * 10**9)
    if state['direction'] == 0:
        return 'enter_long' if index % 2 else 'enter_short'
    return ['hold', 'increase', 'reduce', 'enter_short', 'enter_long', 'exit'][index % 6]


@pytest.mark.parametrize('batch_size', [1, 7, 8192])
@pytest.mark.parametrize('cash', [False, True])
def test_chunked_output_matches_every_legacy_row_and_state(tmp_path, batch_size, cash):
    strategy = (lambda _event, _state: 'hold') if cash else policy
    config = EngineConfig(stop_fraction=.04, max_hold_bars=12, cooldown_bars=1, signal_delay_bars=1)
    expected = backtest(bars(), strategy, config, tmp_path / 'legacy', streaming=False)
    actual = backtest(bars(), strategy, config, tmp_path / 'streamed', batch_size=batch_size)
    for name in ['equity', 'fills', 'trades']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'legacy' / f'{name}.parquet'),
                                      pd.read_parquet(tmp_path / 'streamed' / f'{name}.parquet'))
    assert json.loads((tmp_path / 'legacy/final_state.json').read_text()) == json.loads((tmp_path / 'streamed/final_state.json').read_text())
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
        else:
            assert actual[key] == expected[key]
    metadata = json.loads((tmp_path / 'streamed/storage.json').read_text())
    assert all(value['peak_buffer_rows'] <= batch_size for value in metadata['outputs'].values())
    assert metadata['outputs']['equity']['rows'] == len(bars())


def test_failure_preserves_partial_output_without_success_metrics(tmp_path):
    data = bars()
    data.loc[20, 'high'] = 0
    output = tmp_path / 'failure'
    with pytest.raises(ValueError, match='OHLC'):
        backtest(data, policy, EngineConfig(), output, batch_size=7)
    assert len(pd.read_parquet(output / 'equity.parquet')) == 20
    assert json.loads((output / 'failure.json').read_text())['processed_bars'] == 20
    assert not (output / 'metrics.json').exists()
    with pytest.raises(FileExistsError):
        backtest(bars(), policy, EngineConfig(), output)


def test_minute_engine_records_actual_holding_duration(tmp_path):
    frame = bars()
    frame['time'] = pd.date_range('2020-01-01', periods=len(frame), freq='1min', tz='UTC')
    frame['end'] = frame.time + pd.Timedelta(minutes=1)
    config = EngineConfig(bar_seconds=60, max_hold_bars=6)
    directory = tmp_path / 'minute'
    backtest(frame, policy, config, directory, batch_size=17)
    trades = pd.read_parquet(directory / 'trades.parquet')
    assert len(trades) > 0
    decomposition = decompose_run(directory, config.initial_equity)
    assert decomposition['median_hold_minutes'] == trades.hold_bars.median()
    with pytest.raises(ValueError, match='초기 자본'):
        decompose_run(directory, config.initial_equity + 1)
