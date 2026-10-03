import json

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.inventory_study import (
    attach_inventory,
    inventory_states,
    order_sizes,
    run_inventory_study,
)


def sample():
    frame = pd.DataFrame({'time': pd.date_range('2021-01-01T00:00Z', periods=5, freq='2min'),
        'order_key': ['a', 'b', 'c', 'b', 'd'], 'action': ['open', 'reduce', 'increase', 'reduce', 'close'],
        'before_qty': [0, 100, 50, 51, 2], 'after_qty': [100, 50, 51, 2, 0],
        'episode_id': [1]*5, 'realized_btc': [0., 2., 0., -1., .1], 'fee_btc': [.01]*5})
    executions = pd.DataFrame({'time': frame.time, 'symbol': 'XBTUSD', 'exectype': 'Trade',
        'order_key': frame.order_key, 'orderqty': [100, 99, 1, 99, 2],
        'side': ['Buy', 'Sell', 'Buy', 'Sell', 'Sell'], 'lastqty': [100, 50, 1, 49, 2]})
    return frame, executions


def test_inventory_uses_only_past_max_and_retains_small_residuals():
    actions, _ = sample()
    result = inventory_states(actions)
    np.testing.assert_array_equal(result.remaining_fraction, [1., .5, .51, .02, 0.])
    np.testing.assert_array_equal(result.before_fraction, [0., 1., .5, .51, .02])
    assert len(result) == len(actions) and result.realized_btc.sum() == actions.realized_btc.sum()
    future = actions.iloc[[-1]].assign(time=actions.time.iloc[-1]+pd.Timedelta(minutes=1), action='open',
        before_qty=0, after_qty=10000, episode_id=2, order_key='future')
    changed = inventory_states(pd.concat([actions, future], ignore_index=True))
    pd.testing.assert_frame_equal(changed.iloc[:len(actions)], result, check_exact=True)


def test_before_fraction_on_reversal_uses_previous_episode_maximum():
    actions, _ = sample()
    actions.loc[4, ['action', 'after_qty', 'episode_id']] = ['reverse', -200, 2]
    result = inventory_states(actions)
    assert result.before_fraction.iloc[-1] == .02 and result.remaining_fraction.iloc[-1] == 1
    assert result.max_quantity_so_far.iloc[-1] == 200


def test_same_timestamp_last_state_is_visible_only_after_boundary():
    actions, _ = sample()
    actions.loc[2, 'time'] = actions.time.iloc[1]
    frame = pd.DataFrame({'end': pd.to_datetime(['2021-01-01T00:02Z', '2021-01-01T00:03Z']),
                          'episode_id': [1, 1], 'usable': [True, True]})
    joined = attach_inventory(frame, inventory_states(actions))
    np.testing.assert_array_equal(joined.remaining_fraction, [1., .51])
    frame.loc[1, 'episode_id'] = 2
    with pytest.raises(ValueError, match='에피소드'):
        attach_inventory(frame, inventory_states(actions))


@pytest.mark.parametrize('damage', ['quantity_chain', 'nan', 'noninteger', 'initial_position'])
def test_inventory_rejects_invalid_raw_position_chain(damage):
    actions, _ = sample()
    if damage == 'quantity_chain':
        actions.loc[2, 'before_qty'] = 2
    elif damage == 'nan':
        actions['after_qty'] = actions.after_qty.astype(float)
        actions.loc[2, 'after_qty'] = np.nan
    elif damage == 'noninteger':
        actions['after_qty'] = actions.after_qty.astype(float)
        actions.loc[2, 'after_qty'] = 1.5
    else:
        actions.loc[0, 'before_qty'] = 1
    with pytest.raises(ValueError):
        inventory_states(actions)


def test_requested_near_close_keeps_partial_fills_and_interleaved_orders():
    actions, executions = sample()
    result = order_sizes(executions, actions, inventory_states(actions)).set_index('order_key')
    assert len(result) == 4 and result.loc['b', 'target'] == 'reduce'
    assert result.loc['b', 'requested_fraction'] == .99
    assert result.loc['b', 'requested_residual_quantity'] == 1
    assert result.loc['b', 'executed_quantity'] == 99 and result.loc['b', 'filled_fraction'] == 1
    assert result.loc['b', 'interleaved'] and result.loc['b', 'fill_span_seconds'] == 240
    assert result.loc['b', 'realized_btc'] == 1
    damaged = executions.copy()
    damaged.loc[1, 'lastqty'] += 1
    with pytest.raises(ValueError, match='체결량'):
        order_sizes(damaged, actions, inventory_states(actions))


def test_inventory_report_reconciles_costs_without_fitting_or_changing_rows(tmp_path):
    audit, labels = tmp_path / 'audit', tmp_path / 'labels'
    audit.mkdir()
    labels.mkdir()
    actions, executions = sample()
    actions.to_parquet(audit / 'actions.parquet', index=False)
    executions.to_parquet(audit / 'executions.parquet', index=False)
    pd.DataFrame({'closed': [True]}).to_parquet(audit / 'episodes.parquet', index=False)
    save_json(audit / 'reconstruction.json', {'realized_gross_btc': 1.1, 'trade_fees_btc': .05, 'funding_cost_btc': 0.})
    frame = pd.DataFrame({'end': pd.date_range('2021-01-01T00:01Z', periods=8, freq='min'),
        'episode_id': 1, 'usable': True, 'adds_capped': 1, 'y_exit': 0, 'y_reduce': 0, 'y_increase': 0})
    frame.to_parquet(labels / 'events.parquet', index=False)
    save_json(labels / 'summary.json', {'complete': True})
    save_json(labels / 'files.json', {'events.parquet': sha256(labels / 'events.parquet')})
    fingerprints = {n: sha256(audit / n) for n in ['actions.parquet', 'executions.parquet', 'episodes.parquet']}
    save_json(labels / 'manifest.json', {'settings': {'audit_sha256': fingerprints}})
    out = run_inventory_study(audit, labels, tmp_path / 'runs', [])
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['complete'] and summary['raw_accounting_exact'] and summary['source_rows_preserved']
    assert not summary['new_model_fit'] and not summary['threshold_selected_by_profit']
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'source_states.parquet')[actions.columns], actions)
    assert fingerprints == {n: sha256(audit / n) for n in fingerprints}
    (audit / 'episodes.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='지문'):
        run_inventory_study(audit, labels, tmp_path / 'runs', [])
