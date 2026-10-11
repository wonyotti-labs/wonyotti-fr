import pandas as pd
import pytest

from wonyotti_fr.portfolio_price_activity import activity_record, boundary_activity


@pytest.mark.parametrize('trades,volume,reason', [(1, 1, 'observed_trade'), (0, 0, 'no_observed_trade'),
    (1, 0, 'no_observed_trade'), (None, 100, 'missing_activity'), (10, None, 'missing_activity')])
def test_activity_requires_actual_trade_and_volume(trades, volume, reason):
    result = activity_record('XBTUSD', '2021-01-01T00:00:00+00:00', trades, volume)
    assert result['activity'] == reason and result['observed_trade'] == (reason == 'observed_trade')


@pytest.mark.parametrize('number', [-1, .5, True, float('nan'), float('inf'), '3'])
def test_invalid_activity_rejected(number):
    with pytest.raises(ValueError):
        activity_record('XBTUSD', '2021-01-01T00:00:00+00:00', number, 10)


def test_any_required_untraded_contract_preserves_boundary_as_unknown():
    rows = pd.DataFrame([activity_record(s, t, n, n) for s, t, n in [
        ('XBTUSD', '2021-01-01T00:00:00+00:00', 4), ('ETHUSD', '2021-01-01T00:00:00+00:00', 0),
        ('XBTUSD', '2021-01-01T00:05:00+00:00', 2)]])
    details, grouped = boundary_activity(rows[['symbol', 'end']], rows)
    assert len(details) == 3
    assert grouped.all_prices_have_observed_trade.tolist() == [False, True]
    assert grouped.unobserved_prices.tolist() == [1, 0]
    with pytest.raises(ValueError):
        boundary_activity(rows[['symbol', 'end']], rows.iloc[:2])
    with pytest.raises(ValueError):
        boundary_activity(rows[['symbol', 'end']], pd.concat([rows, rows.iloc[:1]]))
