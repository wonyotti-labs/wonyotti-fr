import copy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from test_close_effect import CollectionPolicy
from test_engine import config
from test_minute_inventory_research import ready_bars

from wonyotti_fr.close_effect import CLOSE_FEATURES
from wonyotti_fr.common import sha256
from wonyotti_fr.context_position import CONTEXT_FEATURES
from wonyotti_fr.engine import PolicyDecision
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.first_opportunity_collection import (
    EXPANDED_FIRST_FEATURES,
    collect_first_opportunities,
    compare_replayed_prefix,
    expanded_first_values,
)
from wonyotti_fr.outcome_journal import OutcomeJournal
from wonyotti_fr.streaming_backtest import streaming_backtest


class ExitFirstPolicy(CollectionPolicy):
    def __call__(self, bar, view):
        if not view['direction']:
            return PolicyDecision('enter_long', {}, 'test_enter')
        self.feature_values(bar, view)
        return PolicyDecision('exit', {}, 'test_management')


def fixture(path, *, policy=CollectionPolicy, future=False):
    bars = ready_bars().assign(count=1, volume=1.)
    if future:
        bars.loc[bars.index >= 10, ['open', 'high', 'low', 'close']] *= 1.03
    cfg = config(bar_seconds=60, fee_bps=5, slippage_bps=3)
    source = path/'baseline'
    streaming_backtest(bars, policy(), cfg, source, 8192)
    width = len(MARKET_FEATURES)+len(CONTEXT_FEATURES)+5
    keys = {pd.Timestamp(value).value//(300*10**9)*(300*10**9) for value in bars.end}
    context = {key: np.zeros(width) for key in keys}
    return bars, cfg, source, context


def execute(path, data, *, policy=CollectionPolicy, max_new=None, journal_path=None, cutoff=None, output_name='collected'):
    bars, cfg, source, context = data
    journal_path = journal_path or path/'outcomes.sqlite'
    cutoff = cutoff or pd.Timestamp('2020-01-02', tz='UTC')
    with OutcomeJournal(journal_path, {'synthetic_collection': True}) as journal:
        result = collect_first_opportunities(bars, policy(), cfg, source, path/output_name, journal, cutoff, context, max_new=max_new)
        journal.verify()
    return result


def payloads(path):
    with sqlite3.connect(path) as connection:
        return [json.loads(row[0]) for row in connection.execute('SELECT payload FROM outcomes ORDER BY sequence')]


def test_every_first_choice_full_membership_cash_equity_cost_and_natural_end(tmp_path):
    data = fixture(tmp_path)
    rows, positions, counts, complete = execute(tmp_path, data)
    assert complete and rows and len({row['position_entry_time'] for row in rows}) == len(rows)
    membership = pd.read_parquet(tmp_path/'collected/management_membership.parquet')
    selected = membership[membership.selected_first]
    expected = membership[membership.eligible].drop_duplicates('position_entry_time')
    pd.testing.assert_frame_equal(selected, expected, check_exact=True)
    assert len(membership) == counts['management_rows'] > len(rows)
    assert sum(item['management_rows'] for item in positions) == len(membership)
    records = payloads(tmp_path/'outcomes.sqlite')
    for row, record in zip(rows, records, strict=True):
        state, trace = record['opportunity']['state'], record['close']
        assert set(EXPANDED_FIRST_FEATURES) <= set(row) and len(EXPANDED_FIRST_FEATURES) == 74
        assert row['reference_equity'] == state['cash']+state['quantity']*state['last_close']
        if row['label_status'] == 'closed':
            assert row['first_target_common_bps'] == (trace['final_cash']-row['continue_cash'])/row['reference_equity']*10000
            fill = trace['fills'][0]
            expected_cash = state['cash']-fill['delta_quantity']*fill['price']-fill['fee']-trace['funding_cost']
            assert trace['final_cash'] == pytest.approx(expected_cash, abs=1e-9)
            assert pd.Timestamp(row['label_end']) > pd.Timestamp(row['decision_time'])
    for name in ['equity', 'trades', 'fills']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/f'collected/{name}.parquet'), pd.read_parquet(data[2]/f'{name}.parquet'), check_exact=True)


def test_resume_preserves_first_journal_and_full_uninterrupted_result(tmp_path):
    data = fixture(tmp_path)
    partial = execute(tmp_path, data, max_new=1, output_name='partial')
    assert not partial[-1] and len(partial[0]) == 1
    first = payloads(tmp_path/'outcomes.sqlite')[0]
    resumed = execute(tmp_path, data, output_name='resumed')
    full = execute(tmp_path, data, journal_path=tmp_path/'separate.sqlite', output_name='full')
    assert resumed == full and resumed[-1]
    assert payloads(tmp_path/'outcomes.sqlite')[0] == first
    assert payloads(tmp_path/'outcomes.sqlite') == payloads(tmp_path/'separate.sqlite')
    with OutcomeJournal(tmp_path/'outcomes.sqlite', {'synthetic_collection': True}) as journal:
        modified = copy.deepcopy(first['opportunity'])
        modified['reference_equity'] += 1
        with pytest.raises(ValueError):
            journal.read(0, modified)


@pytest.mark.parametrize('unavailable', [False, True])
def test_no_eligible_and_missing_features_keep_all_positions_and_management_rows(tmp_path, unavailable):
    data = fixture(tmp_path, policy=ExitFirstPolicy)
    if unavailable:
        data = (*data[:3], {})
    rows, positions, counts, complete = execute(tmp_path, data, policy=ExitFirstPolicy)
    assert complete and not rows and positions and counts['management_rows'] > 0
    assert all(not row['has_eligible_opportunity'] for row in positions)
    assert any(not row['management_rows'] for row in positions)
    assert all(row['reason'] == (('unavailable_features_only' if unavailable else 'no_eligible_opportunity')
        if row['management_rows'] else 'no_management_opportunity') for row in positions)
    membership = pd.read_parquet(tmp_path/'collected/management_membership.parquet')
    assert not membership.selected_first.any() and not membership.eligible.any()
    assert membership.available.eq(not unavailable).all()


def test_boundary_censored_and_outside_first_choices_are_retained(tmp_path):
    data = fixture(tmp_path)
    rows, positions, _, complete = execute(tmp_path, data, cutoff=data[0].end.iloc[8])
    assert complete and rows and all(row['label_status'] in {'right_censored', 'outside_training_boundary'} for row in rows)
    assert all(row['first_target_common_bps'] is None for row in rows)
    assert sum(row['has_eligible_opportunity'] for row in positions) == len(rows)


def test_future_prices_change_targets_but_not_earlier_first_inputs(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    original = fixture(a)
    future = fixture(b, future=True)
    execute(a, original)
    execute(b, future)
    first, other = payloads(a/'outcomes.sqlite')[0], payloads(b/'outcomes.sqlite')[0]
    assert first['opportunity'] == other['opportunity']
    assert first['outcome']['close_advantage_pnl'] != other['outcome']['close_advantage_pnl']


def test_context_is_past_confirmed_and_state_economics_are_current(tmp_path):
    data = fixture(tmp_path)
    execute(tmp_path, data)
    record = payloads(tmp_path/'outcomes.sqlite')[0]
    op = record['opportunity']
    state = op['state']
    values = np.array([op[name] for name in CLOSE_FEATURES])
    context = copy.deepcopy(data[3])
    original = expanded_first_values(values, state, data[1], context)
    future = pd.Timestamp(state['last_end']).value//(300*10**9)*(300*10**9)+300*10**9
    context[future] = np.ones_like(next(iter(context.values())))
    np.testing.assert_array_equal(expanded_first_values(values, state, data[1], context), original)
    chosen = pd.Timestamp(state['last_end']).value//(300*10**9)*(300*10**9)
    context[chosen][0] = .1
    with pytest.raises(ValueError, match='시장 특징'):
        expanded_first_values(values, state, data[1], context)
    assert expanded_first_values(values, state, data[1], {}) is None


def test_streaming_parity_multiple_batches_prefix_and_value_damage(tmp_path):
    frame = pd.DataFrame({'number': np.arange(17000), 'value': np.r_[np.nan, np.arange(16999, dtype=float)]})
    original, same, prefix = [tmp_path/name for name in ['original.parquet', 'same.parquet', 'prefix.parquet']]
    frame.to_parquet(original, index=False)
    frame.to_parquet(same, index=False)
    frame.iloc[:9000].to_parquet(prefix, index=False)
    compare_replayed_prefix(same, original, True)
    compare_replayed_prefix(prefix, original, False)
    with pytest.raises(ValueError):
        compare_replayed_prefix(prefix, original, True)
    frame.loc[8300, 'value'] += 1
    frame.to_parquet(same, index=False)
    with pytest.raises(AssertionError):
        compare_replayed_prefix(same, original, True)


def test_original_sources_unchanged_and_replayed_source_damage_rejected(tmp_path):
    data = fixture(tmp_path)
    before = {p.name: sha256(p) for p in data[2].iterdir() if p.is_file()}
    execute(tmp_path, data)
    assert before == {p.name: sha256(p) for p in data[2].iterdir() if p.is_file()}
    path = data[2]/'fills.parquet'
    values = pd.read_parquet(path)
    values.loc[0, 'fee'] += .01
    values.to_parquet(path, index=False)
    with pytest.raises(AssertionError):
        execute(tmp_path, data, journal_path=tmp_path/'damaged.sqlite', output_name='damaged')
