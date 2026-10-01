import pandas as pd
import pytest

from wonyotti_fr.context_study import execution_costs, order_context
from wonyotti_fr.event_features import MARKET_FEATURES


def test_order_context_requires_strictly_prior_completed_bar():
    times = pd.date_range('2020-01-01', periods=2, freq='5min', tz='UTC')
    features = pd.DataFrame({'time': times, 'end': times + pd.Timedelta(minutes=5)})
    features[MARKET_FEATURES] = 0.01
    features.loc[1, MARKET_FEATURES] = 9
    orders = pd.DataFrame({'target_time': [times[1] + pd.Timedelta(minutes=5)], 'side': ['Sell']})
    result = order_context(orders, features)
    assert result.ret_1h.iloc[0] == 0.01
    assert result.signed_prior_1h.iloc[0] == -0.01
    assert result.available.iloc[0]


def test_limit_order_type_does_not_imply_maker_fee():
    executions = pd.DataFrame({'exectype': ['Trade', 'Trade'], 'settlcurrency': ['XBt', 'XBt'],
                               'time': pd.to_datetime(['2020-01-01', '2020-01-01'], utc=True),
                               'execcost': [-200000000, 200000000], 'execcomm': [-50000, 150000],
                               'execid': ['a', 'b'], 'ordtype': ['Limit', 'Limit'],
                               'lastliquidityind': ['AddedLiquidity', 'RemovedLiquidity']})
    result = execution_costs(executions).set_index('lastliquidityind')
    assert result.loc['AddedLiquidity', 'notional_weighted_fee_bps'] == pytest.approx(-2.5)
    assert result.loc['RemovedLiquidity', 'notional_weighted_fee_bps'] == pytest.approx(7.5)
    executions.loc[1, 'settlcurrency'] = 'USDt'
    with pytest.raises(ValueError, match='정산 통화'):
        execution_costs(executions)
