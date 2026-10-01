import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.execution_study import markouts
from wonyotti_fr.expansion_data import expansion_targets


def test_expansion_uses_first_increase_including_flat_and_excludes_end_boundary():
    times = pd.date_range('2020-01-01', periods=3, freq='5min', tz='UTC').as_unit('ns')
    events = pd.DataFrame({'end': times, 'label_end': times + pd.Timedelta(minutes=5),
                           'target_time': pd.NaT, 'target': 'hold', 'target_episode_id': 0})
    orders = pd.DataFrame({'target_time': [times[0], times[0] + pd.Timedelta(minutes=1), times[1]],
                           'target': ['reduce', 'increase', 'enter_short'], 'target_episode_id': [1, 1, 2],
                           'side': ['Sell', 'Buy', 'Sell']})
    result = expansion_targets(events, orders)
    assert result.active.tolist() == [1, 1, 0]
    assert result.buy.iloc[:2].tolist() == [1, 0]
    assert result.expansion_count.tolist() == [1, 1, 0]
    assert result.target_episode_id.tolist() == [1, 2, 0]


def test_markout_reference_is_strictly_past_and_outcome_is_not_before_horizon():
    times = pd.date_range('2020-01-01', periods=5, freq='5min', tz='UTC').as_unit('ns')
    bars = pd.DataFrame({'end': times, 'close': [100., 200., 210., 220., 230.]})
    orders = pd.DataFrame({'order_key': ['a'], 'target_time': [times[1]], 'side': ['Buy'], 'target': ['increase']})
    fills = pd.DataFrame({'order_key': ['a'], 'lastpx': [190.], 'execcost': [-100000],
                          'execcomm': [-25], 'lastliquidityind': ['AddedLiquidity']})
    result = markouts(orders, fills, bars, (5,)).iloc[0]
    assert result.reference_price == 100
    assert result.outcome_price == 210
    assert result.first_fill_fee_bps == -2.5
    assert result.markout_bps == pytest.approx(np.log(210 / 190) * 10000)
    assert result.markout_bps == pytest.approx(result.reference_to_fill_bps + result.subsequent_reference_move_bps)
    orders['target_time'] += pd.Timedelta(seconds=1)
    result = markouts(orders, fills, bars, (5,)).iloc[0]
    assert result.reference_price == 200
    assert result.outcome_price == 220
    assert pd.isna(markouts(orders, fills, bars, (60,)).outcome_price.iloc[0])
