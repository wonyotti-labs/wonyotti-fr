import pandas as pd
import pytest
from test_minute_repair import source

from wonyotti_fr.aggregate_verification import selected_aggregate_trades
from wonyotti_fr.minute_repair import reconcile_window


def with_gap():
    date, trades, prior, truth, five = source()
    trades.loc[3:, 'id'] += 1
    aggregate = pd.DataFrame({'aggregate_id': range(len(trades)), 'price': trades.price, 'qty': trades.qty,
                              'first_id': trades.id, 'last_id': trades.id, 'time': trades.time,
                              'is_buyer_maker': trades.is_buyer_maker})
    return date, trades, prior, five, aggregate


def test_gap_requires_all_market_trades_corroborated_and_duplicates_are_rejected():
    _, trades, prior, five, aggregate = with_gap()
    replacements, _, checks = reconcile_window(prior, five, trades, aggregates=aggregate)
    assert len(replacements) == 1 and checks['changed_minute_id_gaps'] == 1
    assert checks['aggregate_corroboration']['target_trades'] == 2
    with pytest.raises(ValueError, match='한 번씩'):
        reconcile_window(prior, five, trades, aggregates=aggregate.drop(index=3))
    with pytest.raises(ValueError, match='중복'):
        reconcile_window(prior, five, trades, aggregates=pd.concat([aggregate, aggregate.iloc[[3]]], ignore_index=True))


@pytest.mark.parametrize('column,value', [('qty', 1.1), ('price', 999.), ('time', 1577836800000), ('is_buyer_maker', True)])
def test_aggregate_match_requires_quantity_price_side_and_timestamp(column, value):
    _, trades, prior, five, aggregate = with_gap()
    aggregate.loc[3, column] = value
    with pytest.raises(ValueError, match='불일치'):
        reconcile_window(prior, five, trades, aggregates=aggregate)


@pytest.mark.parametrize('header', [False, True])
def test_aggregate_csv_retains_first_row_and_rejects_wrong_archive_day(header):
    date, trades, _, _, aggregate = with_gap()
    selected, checks = selected_aggregate_trades(aggregate.to_csv(index=False, header=header).encode(), date, trades)
    assert selected.aggregate_id.iloc[0] == 0 and checks['selected_rows'] == 10
    with pytest.raises(ValueError, match='날짜'):
        selected_aggregate_trades(aggregate.to_csv(index=False, header=header).encode(), date + pd.Timedelta(days=1), trades)
