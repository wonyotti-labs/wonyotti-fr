import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.realized_exit import realized_exit_targets


def fixture():
    end = pd.date_range('2021-01-01', periods=5, freq='min', tz='UTC')
    frame = pd.DataFrame({'end': end, 'label_end': end + pd.Timedelta(minutes=1),
        'episode_id': [1, 1, 2, 0, 4], 'usable': [True, True, True, False, False],
        'y_exit': [1, 0, 1, 0, 0], 'exit_count': [1, 0, 1, 0, 0]})
    actions = pd.DataFrame([
        (-10, 'enter', 0, 10, 1), (5, 'reduce', 10, 5, 1), (60, 'reverse', 5, -4, 2),
        (130, 'close', -4, 0, 2), (135, 'enter', 0, 3, 3), (140, 'close', 3, 0, 3),
        (239, 'enter', 0, 2, 4), (250, 'close', 2, 0, 4), (350, 'enter', 0, 1, 5),
        (360, 'close', 1, 0, 5),
    ], columns=['seconds', 'action', 'before_qty', 'after_qty', 'episode_id'])
    actions['time'] = end[0] + pd.to_timedelta(actions.pop('seconds'), unit='s')
    return frame, actions


def test_actual_ending_replaces_first_partial_order_without_touching_inputs():
    frame, actions = fixture()
    original, source = frame.copy(deep=True), actions.copy(deep=True)
    targets, ledger, summary = realized_exit_targets(frame, actions)
    assert targets.y_exit.tolist() == [0, 1, 1, 0, 0]
    assert targets.requested_y_exit.tolist() == [1, 0, 1, 0, 0]
    assert ledger.reason.tolist() == ['linked', 'linked', 'different_episode',
                                      'unusable_features_or_range', 'outside_minutes']
    assert ledger.before_episode_id.tolist() == [1, 2, 3, 4, 5]
    assert summary['linked_actual_events'] == 2 and summary['actual_events'] == 5
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    pd.testing.assert_frame_equal(actions, source, check_exact=True)
    assert targets.exit_count.sum() == ledger.reason.eq('linked').sum()


def test_future_ending_only_changes_target_window():
    frame, actions = fixture()
    old, _, _ = realized_exit_targets(frame, actions)
    actions.loc[2, 'time'] += pd.Timedelta(seconds=1)
    new, _, _ = realized_exit_targets(frame, actions)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    actions.loc[2, 'time'] -= pd.Timedelta(seconds=2)
    new, _, _ = realized_exit_targets(frame, actions)
    assert new.y_exit.tolist() == [1, 0, 1, 0, 0]


@pytest.mark.parametrize('damage', ['quantity', 'action', 'order', 'nonfinite', 'window'])
def test_invalid_actual_ending_or_boundary_is_rejected(damage):
    frame, actions = fixture()
    if damage == 'quantity':
        actions.loc[2, 'before_qty'] = 6
    elif damage == 'action':
        actions.loc[2, 'action'] = 'reduce'
    elif damage == 'order':
        actions.loc[2, 'time'] = actions.time.iloc[0]
    elif damage == 'nonfinite':
        actions.loc[2, 'before_qty'] = np.nan
    else:
        frame.loc[2, 'label_end'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        realized_exit_targets(frame, actions)
