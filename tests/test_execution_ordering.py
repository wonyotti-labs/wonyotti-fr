import pandas as pd
import pytest

from wonyotti_fr.execution_ordering import inspect_cumulative_order, verify_cumulative_order


def example():
    first = pd.Timestamp('2021-01-01', tz='UTC')
    records = [(0, 'a', 3, 10, 10, 'Filled'), (0, 'a', 7, 7, 10, 'PartiallyFilled'),
        (0, 'b', 2, 2, 5, 'PartiallyFilled'), (1, 'b', 3, 5, 5, 'Filled'),
        (2, 'special:x', 1, 1, 1, 'Filled')]
    return pd.DataFrame([dict(time=first+pd.Timedelta(seconds=s), symbol='XBTUSD', exectype='Trade',
        order_key=key, execid=str(i), lastqty=qty, cumqty=cum, orderqty=orderqty, ordstatus=status, side='Buy')
        for i,(s,key,qty,cum,orderqty,status) in enumerate(records)])


def test_same_timestamp_cumulative_order_explains_source_mismatch_without_rewriting():
    raw = example()
    original = raw.copy(deep=True)
    rows, result = inspect_cumulative_order(raw)
    assert result['source_prefix_mismatches'] == 2
    assert result['sorted_cumqty_within_timestamp_mismatches'] == 0
    assert result['group_end_max_cumqty_mismatches'] == 0
    assert result['last_observed_status_changes'] == 1
    assert result['first_observed_quantity_changes'] == 0
    assert result['first_observed_time_changes'] == 0
    assert result['cumulative_order_last_status'] == {'Filled': 2}
    assert verify_cumulative_order(raw, rows, result)['raw_rows_verified'] == 4
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


def test_never_reorders_different_timestamps_to_manufacture_consistency():
    raw = example()
    raw.loc[1, 'time'] += pd.Timedelta(microseconds=1)
    raw = raw.sort_values('time', kind='stable')
    rows, result = inspect_cumulative_order(raw)
    assert result['sorted_cumqty_within_timestamp_mismatches'] == 2
    assert result['group_end_max_cumqty_mismatches'] > 0
    verify_cumulative_order(raw, rows, result)


def test_missing_quantity_remains_mismatch():
    raw = example().drop(index=1).reset_index(drop=True)
    rows, result = inspect_cumulative_order(raw)
    assert result['sorted_cumqty_within_timestamp_mismatches'] == 1
    assert result['last_order_max_cumqty_mismatches'] == 1
    assert not result['missing_executions_proven']
    verify_cumulative_order(raw, rows, result)


def test_first_quantity_change_reported_without_altering_request():
    raw = example()
    raw.loc[1, 'orderqty'] = 11
    _, result = inspect_cumulative_order(raw)
    assert result['first_observed_quantity_changes'] == 1


@pytest.mark.parametrize('bad', ['duplicate', 'negative', 'fractional', 'na_time', 'unsorted', 'side_change', 'missing_column'])
def test_bad_input_rejected(bad):
    raw = example()
    if bad == 'duplicate':
        raw.loc[1, 'execid'] = raw.execid.iloc[0]
    elif bad == 'negative':
        raw.loc[1, 'lastqty'] = -1
    elif bad == 'fractional':
        raw['cumqty'] = raw.cumqty.astype(float)
        raw.loc[1, 'cumqty'] = 7.5
    elif bad == 'na_time':
        raw.loc[1, 'time'] = pd.NaT
    elif bad == 'unsorted':
        raw = raw.iloc[::-1]
    elif bad == 'side_change':
        raw.loc[1, 'side'] = 'Sell'
    else:
        raw = raw.drop(columns='cumqty')
    with pytest.raises(ValueError):
        inspect_cumulative_order(raw)


def test_verifier_detects_changed_prefix():
    raw=example()
    rows,result=inspect_cumulative_order(raw)
    rows.loc[0,'cumulative_order_prefix'] += 1
    with pytest.raises(AssertionError):
        verify_cumulative_order(raw,rows,result)
