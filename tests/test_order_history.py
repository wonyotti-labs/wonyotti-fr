import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_calibration_diagnostics import source

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from wonyotti_fr.management_diagnostics import management_metrics
from wonyotti_fr.minute_inventory import MinuteInventoryModels
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES, OrderHistoryModels, attach_order_history
from wonyotti_fr.order_history_diagnostics import history_admission, run_order_history_diagnostics


def example():
    first = pd.Timestamp('2021-01-01', tz='UTC')
    entries = [(0, 'open', 'open', 0, 100, 1), (.5, 'a', 'increase', 100, 120, 1),
        (.75, 'a', 'increase', 120, 130, 1), (1, 'b', 'reduce', 130, 110, 1),
        (5, None, 'funding', 110, 110, 1), (15.5, 'c', 'reduce', 110, 100, 1),
        (16, 'd', 'close', 100, 0, 1), (17, 'e', 'open', 0, 10, 2), (17+1/6, 'f', 'increase', 10, 20, 2)]
    actions = pd.DataFrame(entries, columns=['minutes', 'order_key', 'action', 'before_qty', 'after_qty', 'episode_id'])
    actions['time'] = (first + pd.to_timedelta(actions.pop('minutes'), unit='min')).dt.round('s')
    frame = pd.DataFrame({'end': first + pd.to_timedelta([.5, 1, 15.5, 16, 17, 18], unit='min'),
        'episode_id': [1, 1, 1, 1, 2, 2]}, index=[10, 20, 30, 40, 50, 60])
    return frame, actions


def test_strict_boundary_partial_fills_recent_left_edge_and_episode_reset():
    frame, actions = example()
    got, ledger = attach_order_history(frame, actions)
    assert ledger.order_key.tolist() == ['a', 'b', 'c', 'f']
    assert got.loc[10, HISTORY_FEATURES].eq(0).all()
    assert got.loc[20, 'past_increase_exists'] == 1
    assert got.loc[20, 'past_increase_log_minutes_since'] == np.log1p(.5)
    assert got.loc[20, 'past_increase_log_count_15m'] == np.log(2)
    assert got.loc[20, 'past_reduce_exists'] == 0
    assert got.loc[30, 'past_increase_log_count_15m'] == np.log(2)
    assert got.loc[40, 'past_increase_log_count_15m'] == 0
    assert got.loc[40, 'past_reduce_log_count_15m'] == np.log(3)
    assert got.loc[50, HISTORY_FEATURES].eq(0).all()
    assert got.loc[60, 'past_reduce_exists'] == 0
    assert got.loc[60, 'past_increase_log_count_15m'] == np.log(2)
    pd.testing.assert_frame_equal(got[frame.columns], frame, check_exact=True)
    microseconds = actions.copy()
    microseconds['time'] = microseconds.time.astype('datetime64[us, UTC]')
    pd.testing.assert_frame_equal(got, attach_order_history(frame, microseconds)[0], check_exact=True)


def test_later_partial_action_cannot_relabel_first_order_history():
    frame, actions = example()
    expected, _ = attach_order_history(frame, actions)
    changed = actions.copy()
    changed.loc[2, 'action'] = 'reduce'
    changed.loc[2:6, 'before_qty'] = [120, 115, 95, 95, 85]
    changed.loc[2:6, 'after_qty'] = [115, 95, 95, 85, 0]
    actual, ledger = attach_order_history(frame, changed)
    assert ledger.loc[ledger.order_key.eq('a'), 'action'].tolist() == ['increase']
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_future_order_time_and_quantity_changes_do_not_change_past_features():
    frame, actions = example()
    expected, _ = attach_order_history(frame, actions)
    actions.loc[5, 'time'] += pd.Timedelta(seconds=15)
    actions.loc[5, 'after_qty'] = 90
    actions.loc[6, 'before_qty'] = 90
    actual, _ = attach_order_history(frame, actions)
    pd.testing.assert_frame_equal(actual.iloc[:3], expected.iloc[:3], check_exact=True)
    assert actual.loc[40, 'past_reduce_log_minutes_since'] != expected.loc[40, 'past_reduce_log_minutes_since']


@pytest.mark.parametrize('damage', ['order', 'time', 'quantity', 'action', 'direction', 'episode'])
def test_invalid_order_history_is_rejected(damage):
    frame, actions = example()
    if damage == 'order':
        actions.loc[1, 'order_key'] = None
    elif damage == 'time':
        actions.loc[1, 'time'] = pd.NaT
    elif damage == 'quantity':
        actions.loc[1, 'before_qty'] += 1
    elif damage == 'action':
        actions.loc[1, 'action'] = 'other'
    elif damage == 'direction':
        actions.loc[1, 'action'] = 'reduce'
    else:
        actions.loc[1, 'episode_id'] = -1
    with pytest.raises(ValueError):
        attach_order_history(frame, actions)


def test_full_history_diagnosis_preserves_inputs_and_matches_independent_fit(tmp_path, monkeypatch):
    selection, labels, inventory, audit = [tmp_path / name for name in ['selection', 'labels', 'inventory', 'audit']]
    for path in [selection, labels, inventory, audit]:
        path.mkdir()
    (labels / 'files.json').write_text('{}')
    (selection / 'frozen_selection.json').write_text('{}')
    save_json(selection / 'manifest.json', {'settings': {'files_sha256': sha256(labels / 'files.json')}})
    save_json(labels / 'manifest.json', {'settings': {'inventory_labels': str(inventory)}})
    frame = source()
    records = []
    for i, row in enumerate(frame.itertuples()):
        grow = i % 2 == 0
        quantity = 120 if grow else 80
        records.extend([(row.entry_time, f'{i}-open', 'open', 0, 100, row.episode_id),
            (row.end-pd.Timedelta(minutes=2), f'{i}-manage', 'increase' if grow else 'reduce', 100, quantity, row.episode_id),
            (row.end+pd.Timedelta(minutes=2), f'{i}-close', 'close', quantity, 0, row.episode_id)])
    actions = pd.DataFrame(records, columns=['time', 'order_key', 'action', 'before_qty', 'after_qty', 'episode_id'])
    actions.to_parquet(audit / 'actions.parquet', index=False)
    save_json(inventory / 'manifest.json', {'settings': {'audit_sha256': {'actions.parquet': sha256(audit / 'actions.parquet')}}})
    train = purged_window(frame, *TRAINING_PERIODS[1]).reset_index(drop=True)
    validation = purged_window(frame, *CALIBRATION_PERIODS[1]).reset_index(drop=True)
    train.to_parquet(selection / 'training_used.parquet', index=False)
    validation.to_parquet(selection / 'calibration_used.parquet', index=False)
    old, _, _ = MinuteInventoryModels.fit(train, validation, 'logistic')
    monkeypatch.setattr('wonyotti_fr.order_history_diagnostics.load_selection',
        lambda _: ({'protocol': 'minute_inventory_micro_v21'}, SimpleNamespace(manager=old)))
    monkeypatch.setattr('wonyotti_fr.order_history_diagnostics.load_minute_inventory_labels', lambda _: (frame, None))
    out = run_order_history_diagnostics(selection, labels, audit, tmp_path / 'runs')
    used, diagnostic = [pd.read_parquet(out / name) for name in ['training_used.parquet', 'diagnosis_used.parquet']]
    pd.testing.assert_frame_equal(used[train.columns], train, check_exact=True)
    pd.testing.assert_frame_equal(diagnostic[validation.columns], validation, check_exact=True)
    predictions = pd.read_parquet(out / 'predictions.parquet')
    model = OrderHistoryModels.from_dict(json.loads((out / 'model.json').read_text()))
    x, vx = used[model.features].to_numpy(), diagnostic[model.features].to_numpy()
    scaler = StandardScaler().fit(x)
    for a in ACTIONS:
        learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), used[f'y_{a}'])
        np.testing.assert_allclose(learner.predict_proba(scaler.transform(vx))[:, 1], predictions[f'{a}_history'], atol=1e-12, rtol=0)
    metrics = json.loads((out / 'metrics.json').read_text())
    for a in ACTIONS:
        for name in metrics[a]:
            assert metrics[a][name] == management_metrics(predictions['y_' + a], predictions[a + '_' + name])
    assert json.loads((out / 'decision.json').read_text()) == history_admission(metrics)
    assert not list(out.rglob('trades.parquet'))
    assert all(sha256(out / name) == digest for name, digest in json.loads((out / 'files.json').read_text()).items())
    frame.loc[0, 'ret_5m'] += 1
    with pytest.raises(AssertionError):
        run_order_history_diagnostics(selection, labels, audit, tmp_path / 'failed')
    failed, = (tmp_path / 'failed').iterdir()
    assert (failed / 'failure.json').exists() and not (failed / 'model.json').exists()
