import json

import pandas as pd
import pytest

from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.event_backtest import summarize_observations


@pytest.mark.parametrize('final,bars,expected', [(12000., 2, None), (-100., 2, None), (0., 2, -1.), (11000., 525960, .1)])
def test_undefined_annualization_preserves_actual_return_and_serializable_summary(final, bars, expected):
    engine = TradingEngine(EngineConfig(bar_seconds=60))
    daily = pd.Series([10000., final], index=pd.date_range('2021-01-01', periods=2, tz='UTC'))
    result = summarize_observations(engine, bars, final, .5, 0., daily, {})
    assert result['total_return'] == final/10000-1
    assert result['final_equity'] == final
    if expected is None:
        assert result['annualized_return'] is None
    else:
        assert result['annualized_return'] == pytest.approx(expected)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
