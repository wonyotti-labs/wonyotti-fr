import copy

import numpy as np
import pandas as pd
import pytest
from test_collateral_direction import events, points, wallet

from wonyotti_fr.collateral_diagnostics import (
    comparison_rows,
    grouped_summary,
    independent_summary_check,
)
from wonyotti_fr.collateral_direction import direction_ledger, summarize_rows


def comparison_fixture():
    e = events()
    ledger = direction_ledger(e, wallet(), points(e))
    part = pd.DataFrame({'end': ledger.time.iloc[:3].to_list(), 'label_end': ledger.time.iloc[1:4].to_list(),
        'direction': [0, 1, 1], 'position_target': [2, 2, 1], 'episode_id': [0, 1, 1],
        'position_target_episode_id': [1, 1, 2]})
    return {'training': part}, ledger


def test_all_rows_preserved_with_both_boundaries_and_independent_counts():
    parts, ledger = comparison_fixture()
    result = comparison_rows(parts, ledger)
    assert len(result) == 6
    assert result[result.boundary.eq('target')].futures_direction.tolist() == ['long', 'long', 'short']
    assert result[result.boundary.eq('target')].account_direction.tolist() == ['long']*3
    for group in grouped_summary(result):
        mask = np.ones(len(result), dtype=bool)
        for key, value in group['group'].items():
            mask &= result[key].eq(value)
        independent_summary_check(result.loc[mask], {k: v for k, v in group.items() if k != 'group'})


def test_changed_prior_direction_rejected():
    parts, ledger = comparison_fixture()
    parts['training'].loc[2, 'position_target'] = 2
    with pytest.raises(ValueError, match='기존 선물'):
        comparison_rows(parts, ledger)


@pytest.mark.parametrize('field', ['rows', 'different_rows', 'comparable_rows', 'cross_table'])
def test_independent_summary_rejects_modified_count(field):
    _, ledger = comparison_fixture()
    expected = copy.deepcopy(summarize_rows(ledger))
    if field == 'cross_table':
        expected['cross_table_futures_rows_account_columns']['short']['long'] += 1
    else:
        expected[field] += 1
    with pytest.raises(ValueError, match='교차표'):
        independent_summary_check(ledger, expected)


def test_missing_support_is_retained_in_denominator():
    _, ledger = comparison_fixture()
    ledger.loc[1, 'account_direction'] = 'unsupported_other_contract'
    ledger.loc[2, 'account_direction'] = 'ambiguous'
    result = summarize_rows(ledger)
    assert result['rows'] == 7 and result['comparable_rows'] == 5
    assert result['direction_counts']['ambiguous'] == 1
    independent_summary_check(ledger, result)
