import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.management_account_reference import verify_account_state, verify_account_summary
from wonyotti_fr.management_account_state import (
    ACCOUNT_COLUMNS,
    account_state_frame,
    summarize_account_state,
)


def source():
    times = pd.date_range('2020-01-01', periods=8, freq='5min', tz='UTC')
    quantity = [0, 50, -100, 200, 500, 501, 49, -51]
    frame = pd.DataFrame({'end': times, 'state_quantity': quantity, 'paired_usable': [True]*8,
        'paired_part': ['training']*8, 'state_episode_id': [0, 1, 2, 3, 4, 4, 4, 5],
        'all_y_exit': [True, False]*4, 'all_y_reduce': [False, True]*4, 'all_y_increase': [False]*8})
    ledger = pd.DataFrame({'time': times, 'xbt_quantity': quantity, 'external_uncertain_count': 0,
        'external_min_satoshi': 0, 'external_max_satoshi': 0, 'unsupported_reasons': '',
        'equity_min_usd': 100., 'equity_max_usd': 100.,
        'equity_min_satoshi': '1000000', 'equity_max_satoshi': '1000000'})
    activity = pd.DataFrame({'end': times, 'all_prices_have_observed_trade': True})
    prices = pd.DataFrame({'end': times, 'symbol': 'XBTUSD', 'close': 10000., 'support': 'available'})
    return frame, ledger, activity, prices


@pytest.mark.parametrize('unit', ['ns', 'us'])
def test_signed_quantity_absolute_notional_decimal_and_exact_bands(unit):
    frames = source()
    for f in frames:
        col = 'time' if 'time' in f else 'end'
        f[col] = f[col].astype(f'datetime64[{unit}, UTC]')
    result = account_state_frame(*frames)
    verify_account_state(*frames, result)
    verify_account_summary(result, summarize_account_state(result))
    assert result.account_band.tolist() == ['zero', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,5]', '(5,inf)', '(0,0.5]', '(0.5,1]']
    assert result.futures_leverage.tolist() == [0, .5, 1, 2, 5, 5.01, .49, .51]


def test_missing_boundary_is_not_forward_filled():
    f, ledger, a, p = source()
    ledger = ledger.drop(index=3)
    result = account_state_frame(f, ledger, a, p)
    verify_account_state(f, ledger, a, p, result)
    assert result.account_reason.iloc[3] == 'missing_boundary'
    assert pd.isna(result.futures_leverage.iloc[3])
    assert len(result) == len(f)


@pytest.mark.parametrize('reason', ['missing_activity', 'missing_price', 'unsupported_valuation',
    'unobserved_price', 'uncertain_external', 'nonpositive_equity'])
def test_unsupported_states_retained_with_reasons(reason):
    f, ledger, a, p = source()
    if reason == 'missing_activity':
        a = a.drop(index=3)
    elif reason == 'missing_price':
        p = p.drop(index=3)
    elif reason == 'unsupported_valuation':
        ledger.loc[3, 'unsupported_reasons'] = 'ALT:price'
        ledger.loc[3, ['equity_min_usd', 'equity_max_usd']] = np.nan
    elif reason == 'unobserved_price':
        a.loc[3, 'all_prices_have_observed_trade'] = False
    elif reason == 'uncertain_external':
        ledger.loc[3, 'external_uncertain_count'] = 1
        ledger.loc[3, 'external_max_satoshi'] = 100
        ledger.loc[3, 'equity_max_usd'] = 100.01
    else:
        ledger.loc[3, ['equity_min_usd', 'equity_max_usd']] = 0.
    result = account_state_frame(f, ledger, a, p)
    verify_account_state(f, ledger, a, p, result)
    verify_account_summary(result, summarize_account_state(result))
    assert result.account_reason.iloc[3] == reason
    assert not result.account_usable.iloc[3] and pd.isna(result.futures_leverage.iloc[3])


def test_future_account_and_labels_do_not_change_current_features():
    f, ledger, a, p = source()
    before = account_state_frame(f, ledger, a, p)
    f.loc[4:, 'state_quantity'] = ledger.loc[4:, 'xbt_quantity'] = 999
    f.loc[:, ['all_y_exit', 'all_y_reduce', 'all_y_increase']] = True
    after = account_state_frame(f, ledger, a, p)
    pd.testing.assert_frame_equal(before.loc[:3, ACCOUNT_COLUMNS], after.loc[:3, ACCOUNT_COLUMNS], check_exact=True)
    f.paired_usable = False
    result = account_state_frame(f, ledger, a, p)
    assert result.account_supported.all() and not result.account_usable.any()
    np.testing.assert_array_equal(result.futures_leverage, after.futures_leverage)


@pytest.mark.parametrize('problem', ['quantity', 'duplicate_ledger', 'duplicate_price', 'unsorted', 'naive',
    'bad_flag', 'bad_cash', 'unknown_cash', 'nonfinite', 'price', 'equity_interval', 'fractional_quantity', 'off_boundary'])
def test_bad_input_fails_closed(problem):
    f, ledger, a, p = source()
    if problem == 'quantity':
        ledger.loc[0, 'xbt_quantity'] = 1
    elif problem == 'duplicate_ledger':
        ledger.loc[1, 'time'] = ledger.time.iloc[0]
    elif problem == 'duplicate_price':
        p.loc[1, 'end'] = p.end.iloc[0]
    elif problem == 'unsorted':
        a = a.iloc[::-1]
    elif problem == 'naive':
        f.end = f.end.dt.tz_localize(None)
    elif problem == 'bad_flag':
        a.all_prices_have_observed_trade = 1
    elif problem == 'bad_cash':
        ledger.loc[0, 'external_uncertain_count'] = -1
    elif problem == 'unknown_cash':
        ledger.loc[0, 'external_max_satoshi'] = 10
    elif problem == 'nonfinite':
        ledger.loc[0, 'equity_min_usd'] = np.nan
    elif problem == 'price':
        p.loc[0, 'close'] = 0.
    elif problem == 'equity_interval':
        ledger.loc[0, 'equity_max_usd'] = 101.
    elif problem == 'fractional_quantity':
        ledger.xbt_quantity = ledger.xbt_quantity.astype(float)
        ledger.loc[0, 'xbt_quantity'] = .5
    else:
        f.loc[0, 'end'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        account_state_frame(f, ledger, a, p)


@pytest.mark.parametrize('column,value', [('futures_leverage', 90.), ('account_usable', False),
    ('account_band', 'zero'), ('account_reason', 'missing_price'), ('account_equity_usd', 101.)])
def test_independent_reference_rejects_state_mutation(column, value):
    frames = source()
    result = account_state_frame(*frames)
    result.loc[1, column] = value
    with pytest.raises(AssertionError):
        verify_account_state(*frames, result)


def test_independent_reference_rejects_inconsistent_satoshi_conversion():
    f, ledger, a, p = source()
    ledger.loc[1, ['equity_min_satoshi', 'equity_max_satoshi']] = '1000100'
    result = account_state_frame(f, ledger, a, p)
    with pytest.raises(AssertionError):
        verify_account_state(f, ledger, a, p, result)


def test_pre_position_unknown_episode_retained_in_denominator():
    frame, ledger, activity, prices = source()
    frame.state_episode_id = frame.state_episode_id.astype(object)
    frame.loc[0, 'state_episode_id'] = None
    frame.loc[0, 'paired_usable'] = False
    frame.state_quantity = frame.state_quantity.astype(float)
    frame.loc[0, 'state_quantity'] = np.nan
    ledger = ledger.iloc[1:]
    result = account_state_frame(frame, ledger, activity, prices)
    verify_account_state(frame, ledger, activity, prices, result)
    summary = summarize_account_state(result)
    verify_account_summary(result, summary)
    assert summary['all_rows'] == 8
    assert summary['groups'][0]['all']['episodes'] == 5
    assert result.account_reason.iloc[0] == 'missing_boundary'


def test_account_features_survive_exact_parquet_roundtrip(tmp_path):
    frame, ledger, activity, prices = source()
    result = account_state_frame(frame, ledger.iloc[1:], activity, prices)
    path = tmp_path/'account.parquet'
    result.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(result, restored, check_exact=True)
    verify_account_state(frame, ledger.iloc[1:], activity, prices, restored)
