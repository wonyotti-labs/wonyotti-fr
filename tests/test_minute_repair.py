import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.minute_repair import rebuilt_minutes, reconcile_window, selected_archive_trades


def source():
    date = pd.Timestamp('2021-01-01T00:00Z')
    trades = pd.DataFrame({'id': np.arange(1, 11), 'price': np.arange(100., 110.), 'qty': 1.,
                           'quote_qty': np.arange(100., 110.),
                           'time': date.value // 10**6 + np.arange(10) * 30_000, 'is_buyer_maker': False})
    truth = rebuilt_minutes(trades)
    prior = truth.copy()
    prior.loc[1, ['high', 'close', 'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote']] = [102, 102, 1, 102, 1, 1, 102]
    five = pd.DataFrame({'time': [date], 'end': [date + pd.Timedelta(minutes=5)],
                         'open': [100.], 'high': [109.], 'low': [100.], 'close': [109.], 'volume': [10.], 'count': [10]})
    return date, trades, prior, truth, five


@pytest.mark.parametrize('header', [False, True])
def test_official_header_variants_retain_first_trade_and_select_exact_window(header):
    date, trades, _, _, five = source()
    selected, check = selected_archive_trades(trades.to_csv(index=False, header=header).encode(), date, pd.DatetimeIndex(five.end))
    assert len(selected) == 10 and selected.id.iloc[0] == 1
    assert check['archive_rows'] == 10 and check['archive_id_gaps'] == 0


def test_only_incomplete_minute_is_rebuilt_after_independent_five_minute_match():
    _, trades, prior, truth, five = source()
    original = prior.copy(deep=True)
    replacements, changes, check = reconcile_window(prior, five, trades)
    assert check['changed_minutes'] == 1 and len(changes) == 1
    assert replacements.close.iloc[0] == truth.close.iloc[1] == 103
    assert replacements.volume.iloc[0] == 2 and replacements['count'].iloc[0] == 2
    pd.testing.assert_frame_equal(prior, original)
    with pytest.raises(ValueError, match='공식 5분'):
        reconcile_window(prior, five.assign(count=11), trades)


def test_id_gap_in_rebuilt_minute_rejects_repair_and_unrelated_gap_is_recorded():
    _, trades, prior, _, five = source()
    broken = trades.copy()
    broken.loc[3:, 'id'] += 1
    with pytest.raises(ValueError, match='연속적'):
        reconcile_window(prior, five, broken)
    unrelated = trades.copy()
    unrelated.loc[1:, 'id'] += 1
    replacements, _, check = reconcile_window(prior, five, unrelated)
    assert len(replacements) == 1 and check['unchanged_minute_id_gaps'] == 1


def test_duplicate_ids_or_wrong_day_in_archive_cannot_enter_reconstruction():
    date, trades, _, _, five = source()
    for invalid in [trades.assign(id=1), trades.assign(time=trades.time + 86_400_000)]:
        with pytest.raises(ValueError, match='원체결'):
            selected_archive_trades(invalid.to_csv(index=False, header=False).encode(), date, pd.DatetimeIndex(five.end))
