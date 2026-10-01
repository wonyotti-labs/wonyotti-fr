import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.pullback_diagnostics import waiting_diagnostics
from wonyotti_fr.pullback_evaluation import evaluation_period
from wonyotti_fr.pullback_policy import PullbackPolicy


class ConstantBase:
    def prepare(self, _frame):
        pass

    def __call__(self, _bar, _state):
        return 'enter_long'


def bars():
    time = pd.date_range('2020-01-01', periods=60, freq='1min', tz='UTC')
    close = np.where(np.arange(60) % 5 == 0, 99.8, 100.)
    frame = pd.DataFrame({'time': time, 'end': time + pd.Timedelta(minutes=1),
                          'open': close, 'high': close, 'low': close, 'close': close, 'funding_rate': 0.})
    frame[MARKET_FEATURES] = 0.
    return frame


@pytest.mark.parametrize('baseline,delay', [(None, 0), (None, 1), ('immediate', 0), ('cash', 0)])
def test_waiting_diagnostics_connects_actual_entry_prices_and_handles_cash(tmp_path, baseline, delay):
    frame = bars()
    policy = PullbackPolicy(ConstantBase(), 8, 5, baseline)
    config = EngineConfig(bar_seconds=60, max_hold_bars=3, cooldown_bars=0, max_adds=0, signal_delay_bars=delay)
    output = tmp_path / 'run'
    metrics = backtest(frame, policy, config, output)
    checks = waiting_diagnostics(output, frame, delay)
    assert checks['matched_entry_fills'] == metrics['closed_trades']
    if baseline == 'cash':
        assert checks['entry_fills'] == 0 and checks['mean_favorable_at_fill_bps'] is None
    elif baseline is None:
        episodes = pd.read_parquet(output / 'waiting_episodes.parquet')
        triggered = episodes[episodes.status.eq('triggered')]
        assert triggered.favorable_at_decision_bps.ge(8).all()
        actual = triggered[triggered.executed]
        assert (actual.expected_entry_time - actual.decision_time).eq(pd.Timedelta(minutes=delay)).all()
        if delay == 0:
            assert actual.favorable_at_fill_bps.lt(0).all()


def test_missing_wait_start_cannot_be_hidden_by_diagnostics(tmp_path):
    frame = bars()
    output = tmp_path / 'run'
    backtest(frame, PullbackPolicy(ConstantBase(), 8, 5), EngineConfig(bar_seconds=60, max_hold_bars=3), output)
    path = output / 'equity.parquet'
    curve = pd.read_parquet(path)
    curve.loc[curve.policy_event.eq('armed'), 'policy_event'] = 'idle'
    curve.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='시작 없이'):
        waiting_diagnostics(output, frame, 0)


def test_observed_period_guard_rejects_future_or_wrong_resolution():
    frozen = {'protocol': 'pullback_v7', 'observed_evaluation_period': ['2022-01-01', '2026-01-01'],
              'seen_2026_period': ['2026-01-01', '2026-10-01'], 'evaluation_end_exclusive': '2026-10-01',
              'unseen_evaluation_available': False, 'risk': {'bar_seconds': 60}}
    assert evaluation_period(frozen, 'seen_2026') == ('2026-01-01', '2026-10-01')
    with pytest.raises(ValueError, match='이미 관찰한'):
        evaluation_period(frozen, 'new')
    with pytest.raises(ValueError, match='범위'):
        evaluation_period({**frozen, 'evaluation_end_exclusive': '2027-01-01'}, 'seen_2026')
    with pytest.raises(ValueError, match='간격'):
        evaluation_period({**frozen, 'risk': {'bar_seconds': 300}}, 'observed')
