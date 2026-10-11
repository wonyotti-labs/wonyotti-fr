import numpy as np
import pandas as pd
import pytest
from test_recorded_change import source

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.market_alignment import align_market, aligned_matrix, verify_alignment

FIRST, LAST = '2021-01-01T00:00Z', '2021-01-02T00:00Z'


def features():
    times = pd.date_range(FIRST, periods=302, freq='5min')
    frame = pd.DataFrame({'end': times})
    for j, name in enumerate(MARKET_FEATURES):
        frame[name] = np.arange(len(times), dtype=float)+j/100.
    return frame


def test_exact_start_and_end_matching_preserves_every_source_column():
    original, market = source(), features()
    aligned = align_market(original, market, FIRST, LAST)
    proof = verify_alignment(original, market, aligned, FIRST, LAST)
    assert proof['complete'] and len(aligned) == len(original)
    assert not aligned.loc[0, 'alignment_supported']
    assert aligned.loc[1, 'start_market_ret_5m'] == 2
    assert aligned.loc[1, 'end_market_ret_5m'] == 3
    assert not aligned.loc[286, 'alignment_supported']
    assert 'end:outside_period' in aligned.loc[286, 'alignment_reason']
    assert aligned_matrix(aligned, 'window_start').shape == aligned_matrix(aligned, 'window_end').shape
    np.testing.assert_allclose(aligned_matrix(aligned, 'window_end')-aligned_matrix(aligned, 'window_start'), 1.)
    pd.testing.assert_frame_equal(aligned[original.columns], original)


def test_missing_nonfinite_and_source_support_have_separate_reasons():
    market = features().drop(index=3).reset_index(drop=True)
    market.loc[market.end == pd.Timestamp('2021-01-01T00:25Z'), 'ret_1h'] = np.nan
    aligned = align_market(source(), market, FIRST, LAST)
    verify_alignment(source(), market, aligned, FIRST, LAST)
    assert 'end:missing_market' in aligned.loc[1, 'alignment_reason']
    assert 'start:missing_market' in aligned.loc[2, 'alignment_reason']
    assert 'end:nonfinite_market' in aligned.loc[3, 'alignment_reason']
    assert not aligned.loc[1:4, 'alignment_supported'].any()
    assert len(aligned) == 300


@pytest.mark.parametrize('bad', ['duplicate', 'unsorted', 'naive', 'off_boundary', 'string_feature', 'extra_collision'])
def test_ambiguous_market_and_overwritten_fields_rejected(bad):
    original, market = source(), features()
    if bad == 'duplicate':
        market = pd.concat([market, market.tail(1)])
    elif bad == 'unsorted':
        market = market.iloc[::-1]
    elif bad == 'naive':
        market['end'] = market.end.dt.tz_localize(None)
    elif bad == 'off_boundary':
        market['end'] += pd.Timedelta(seconds=1)
    elif bad == 'string_feature':
        market['ret_5m'] = market.ret_5m.astype(str)
    else:
        original['start_market_ret_5m'] = 0.
    with pytest.raises(ValueError):
        align_market(original, market, FIRST, LAST)


@pytest.mark.parametrize('field', ['end_market_ret_5m', 'alignment_supported', 'alignment_reason', 'execution_change', 'extra'])
def test_independent_full_alignment_rejects_tampering(field):
    original, market = source(), features()
    aligned = align_market(original, market, FIRST, LAST)
    if field == 'alignment_supported':
        aligned.loc[1, field] = False
    elif field == 'alignment_reason':
        aligned.loc[1, field] = 'missing'
    else:
        aligned.loc[1, field] = 100.
    with pytest.raises(ValueError):
        verify_alignment(original, market, aligned, FIRST, LAST)


def test_future_changes_and_original_state_do_not_change_earlier_market_inputs():
    original, market = source(), features()
    before = align_market(original, market, FIRST, LAST)
    original['current_net_exposure'] = 1e9
    original['original_quantity'] = -1e9
    original['execution_change'] *= -100
    original['xbtusd_execution_change'] *= -100
    market.loc[market.end > pd.Timestamp('2021-01-01T01:00Z'), MARKET_FEATURES] = -1e9
    after = align_market(original, market, FIRST, LAST)
    columns = [f'{side}_market_{name}' for side in ['start', 'end'] for name in MARKET_FEATURES]
    pd.testing.assert_frame_equal(before.loc[before.label_end <= pd.Timestamp('2021-01-01T01:00Z'), columns],
        after.loc[after.label_end <= pd.Timestamp('2021-01-01T01:00Z'), columns])
    assert after.alignment_supported.equals(before.alignment_supported)


@pytest.mark.parametrize('first,last', [('2021-01-01', LAST), (LAST, FIRST), ('2021-01-01T00:00:01Z', LAST)])
def test_period_boundary_validation(first, last):
    with pytest.raises(ValueError):
        align_market(source(), features(), first, last)
