import numpy as np
import pytest
from test_paired_repair import inputs

from wonyotti_fr.aggregate_verification import UnmatchedAggregateTrades
from wonyotti_fr.minute_repair import BAR_COLUMNS
from wonyotti_fr.paired_repair import canonical_window


def test_missing_trade_link_stays_false_when_unchanged_bar_values_are_proven():
    import pandas as pd
    _, trades, prior, truth, five, aggregate = inputs()
    aggregate = aggregate.iloc[1:]
    ends = pd.DatetimeIndex(five.end)
    with pytest.raises(UnmatchedAggregateTrades) as caught:
        canonical_window(prior, five, trades, aggregate, ends)
    assert caught.value.ids.tolist() == [1] and caught.value.report['unmatched_trades'] == 1
    updated, proof = canonical_window(prior, five, trades, aggregate, ends,
                                      candle_sources={'daily': truth, 'monthly': truth})
    assert not proof['corroboration']['target_exactly_once']
    assert proof['unchanged_candle_evidence']['exact_equal_raw_original_daily_monthly']
    assert not proof['unchanged_candle_evidence']['individual_trade_corroboration_completed']
    assert proof['unchanged_candle_evidence']['remaining_minutes_aggregate_check']['target_exactly_once']
    assert proof['bar_values_verified']
    assert len(updated['1m']) == 1 and updated['1m'].time.iloc[0] == truth.time.iloc[1]


@pytest.mark.parametrize('column', BAR_COLUMNS)
@pytest.mark.parametrize('source_name', ['daily', 'monthly'])
def test_every_candle_value_in_each_official_frequency_must_match_exactly(column, source_name):
    import pandas as pd
    _, trades, prior, truth, five, aggregate = inputs()
    sources = {'daily': truth.copy(), 'monthly': truth.copy()}
    sources[source_name][column] = sources[source_name][column].astype(float)
    sources[source_name].loc[0, column] = np.nextafter(float(truth.loc[0, column]), np.inf)
    with pytest.raises(ValueError):
        canonical_window(prior, five, trades, aggregate.iloc[1:], pd.DatetimeIndex(five.end), candle_sources=sources)


def test_candle_proof_cannot_replace_unverified_changes_or_contradictory_trade_groups():
    import pandas as pd
    _, trades, prior, truth, five, aggregate = inputs()
    ends = pd.DatetimeIndex(five.end)
    with pytest.raises(ValueError, match='아홉 값'):
        canonical_window(prior, five, trades, aggregate.drop(index=3), ends,
                         candle_sources={'daily': truth, 'monthly': truth})
    wrong = aggregate.copy()
    wrong.loc[0, 'qty'] = 2.
    with pytest.raises(ValueError, match='불일치'):
        canonical_window(prior, five, trades, wrong, ends, candle_sources={'daily': truth, 'monthly': truth})
    with pytest.raises(ValueError, match='일별·월별'):
        canonical_window(prior, five, trades, aggregate.iloc[1:], ends, candle_sources={'daily': truth})
    with pytest.raises(ValueError):
        canonical_window(prior, five, trades, aggregate.iloc[1:], ends,
                         candle_sources={'daily': pd.concat([truth, truth.iloc[[0]]]), 'monthly': truth})
