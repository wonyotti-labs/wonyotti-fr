import copy

import numpy as np
import pandas as pd
import pytest
from test_state_action_windows import dataset

from wonyotti_fr.minute_management import FEATURES
from wonyotti_fr.paired_management import (
    PairedManagementLinear,
    PairedManagementModel,
    direct_requests,
    management_frame,
    split_support,
)
from wonyotti_fr.paired_management_evaluation import admission, choose_thresholds, evaluate
from wonyotti_fr.paired_management_reference import (
    independent_models,
    manual_admission,
    manual_metrics,
    manual_threshold,
    numeric_linear_scores,
    verify_frame,
)
from wonyotti_fr.position_boost_verification import compare_statistics
from wonyotti_fr.state_action_windows import action_windows, requested_orders


def source():
    raw, events, actions, episodes, ledger, features = dataset()
    raw.loc[raw.order_key.eq('h'), 'orderqty'] = 2
    orders = requested_orders(raw, ledger)
    windows, filled, requested = action_windows(features, ledger, orders, episodes)
    bars = features[['end']].copy().assign(close=np.linspace(100, 110, len(features)))
    return windows, filled, requested, events, actions, episodes, bars


def test_full_state_features_labels_and_independent_cost_replay():
    windows, filled, requested, events, actions, episodes, bars = source()
    frame = management_frame(windows, filled, requested, events, actions, episodes, bars)
    proof = verify_frame(windows, filled, requested, events, episodes, bars, frame)
    assert proof['all_windows_verified'] == len(windows)
    held = frame.iloc[3]
    assert held.basis_average_entry == 100 and held.basis_adds == 0
    assert held.log_hold_minutes == np.log1p(5)
    assert held.all_y_exit and not held.first_y_exit
    assert held.all_y_increase and not held.first_y_increase
    assert held.first_y_reduce and held.all_y_reduce
    assert held.paired_usable
    _, support = split_support(frame)
    assert not support['supported']


def test_current_and_future_features_or_labels_never_enter_past_inputs():
    args = source()
    before = management_frame(*args)
    windows, filled, requested, events, actions, episodes, bars = [x.copy() for x in args]
    end_columns = [k for k in windows if k.startswith('end_market_')]
    windows[end_columns] += 10000
    windows.loc[windows.index >= 4, 'y_requested_exit'] = True
    bars.loc[bars.index >= 4, 'close'] *= 10
    after = management_frame(windows, filled, requested, events, actions, episodes, bars)
    pd.testing.assert_frame_equal(before[FEATURES].iloc[:4], after[FEATURES].iloc[:4], check_exact=True)
    assert after.log_hold_minutes.iloc[3] == np.log1p(5)


def test_exact_boundary_fill_cost_cannot_change_predecision_basis():
    args = [x.copy() for x in source()]
    before = management_frame(*args)
    args[3].loc[args[3].time.ge(args[0].end.iloc[3]), 'cost_satoshi'] *= 2
    after = management_frame(*args)
    pd.testing.assert_frame_equal(before[FEATURES].iloc[:4], after[FEATURES].iloc[:4], check_exact=True)
    assert before.basis_average_entry.iloc[4] != after.basis_average_entry.iloc[4]


@pytest.mark.parametrize('bad', ['feature', 'basis', 'label', 'period', 'time'])
def test_independent_frame_tampering_rejected(bad):
    windows, filled, requested, events, actions, episodes, bars = source()
    frame = management_frame(windows, filled, requested, events, actions, episodes, bars)
    key = {'feature': FEATURES[0], 'basis': 'basis_average_entry', 'label': 'first_y_exit',
        'period': 'paired_reason', 'time': 'basis_state_time'}[bad]
    frame.loc[3, key] = {'feature': 999., 'basis': 999., 'label': True, 'period': 'wrong',
        'time': frame.end.iloc[3]}[bad]
    with pytest.raises(AssertionError):
        verify_frame(windows, filled, requested, events, episodes, bars, frame)


def test_missing_close_preserves_row_with_exclusion_reason():
    args = list(source())
    args[-1] = args[-1].drop(index=3)
    frame = management_frame(*args)
    assert len(frame) == len(args[0]) and frame.paired_reason.iloc[3] == 'nonfinite_state_features'


@pytest.mark.parametrize('at', ['2021-01-01', '2021-07-01'])
def test_period_embargo_and_related_unclosed_episode_boundary(at):
    args = [x.copy() for x in source()]
    offset = pd.Timestamp(at, tz='UTC')-pd.Timestamp('2020-01-03', tz='UTC')
    for frame in args:
        for key in frame:
            if isinstance(frame[key].dtype, pd.DatetimeTZDtype):
                frame[key] += offset
    frame = management_frame(*args)
    assert frame.paired_reason.iloc[3] == 'left_embargo'
    for source_frame in args:
        for key in source_frame:
            if isinstance(source_frame[key].dtype, pd.DatetimeTZDtype):
                source_frame[key] += pd.Timedelta(days=2)
    episodes = args[-2]
    episodes.loc[episodes.episode_id.eq(3), 'closed'] = False
    frame = management_frame(*args)
    assert frame.paired_reason.iloc[3] == 'right_episode_boundary'
    verify_frame(args[0], args[1], args[2], args[3], episodes, args[-1], frame)


def test_direct_exit_needs_no_common_activity_and_has_priority():
    masks, chosen = direct_requests(np.array([[.2, .9, .9], [.1, .8, .8], [.1, .1, .8], [.1, .1, .1]]),
        {'exit': .2, 'reduce': .5, 'increase': .5})
    assert chosen.tolist() == ['exit', 'reduce', 'increase', 'hold']
    assert masks[0].tolist() == [True, True, True]


@pytest.mark.parametrize('bad', ['nan', 'shape', 'bounds', 'missing_threshold', 'threshold_nan'])
def test_invalid_direct_request_rejected(bad):
    x, t = np.zeros((2, 3)), dict.fromkeys(('exit', 'reduce', 'increase'), .5)
    if bad == 'nan':
        x[0, 0] = np.nan
    elif bad == 'shape':
        x = x[:, :2]
    elif bad == 'bounds':
        x[0, 0] = 1.1
    elif bad == 'missing_threshold':
        t.pop('exit')
    else:
        t['exit'] = np.nan
    with pytest.raises(ValueError):
        direct_requests(x, t)


@pytest.fixture(scope='module')
def models():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1200, 32))
    y = np.column_stack([x[:, i]+rng.normal(size=len(x)) > 0 for i in range(3)]).astype(int)
    fy = y*(rng.random(y.shape) > .3)
    vx = rng.normal(size=(150, 32))
    vy = np.column_stack([vx[:, i]+rng.normal(size=len(vx)) > 0 for i in range(3)]).astype(int)
    a, _ = PairedManagementModel.fit(x, fy, vx)
    b, _ = PairedManagementModel.fit(x, y, vx)
    train, cal = pd.DataFrame(x, columns=FEATURES), pd.DataFrame(vx, columns=FEATURES)
    train['label_end'] = pd.date_range('2020-01-03', periods=len(x), freq='5min', tz='UTC')
    cal['end'] = pd.date_range('2021-01-03', periods=len(vx), freq='5min', tz='UTC')
    for i, action in enumerate(('exit', 'reduce', 'increase')):
        train['y_'+action], cal['y_'+action] = y[:, i], vy[:, i]
    linear, _, _ = PairedManagementLinear.fit(train, cal, 'logistic')
    return x, fy, y, vx, vy, a, b, linear


def test_all_trees_linear_parameters_scalar_predictions_and_metrics(models):
    x, fy, y, vx, vy, a, b, linear = models
    p = {'first': a.probabilities(vx), 'all': b.probabilities(vx), 'linear': linear.probabilities(vx),
        'constant': np.tile(y.mean(axis=0), (len(vx), 1))}
    raw = {'first': a.to_dict(), 'all': b.to_dict()}
    reference = independent_models(raw, linear.to_dict(), x, fy, y, vx)
    for name in p:
        np.testing.assert_allclose(p[name], reference[name], rtol=0, atol=1e-12)
    for model in (a, b, linear):
        scalar = np.vstack([model.probabilities(row[None, :]) for row in vx])
        np.testing.assert_allclose(scalar, model.probabilities(vx), rtol=0, atol=1e-12)
    thresholds, details = choose_thresholds(vy, p)
    metrics, _ = evaluate(vy, p, thresholds)
    for name in p:
        for i, action in enumerate(('exit', 'reduce', 'increase')):
            beta = (2., 1., .5)[i]
            expected_t, detail = manual_threshold(vy[:, i], p[name][:, i], beta)
            assert thresholds[name][action] == expected_t
            compare_statistics(details[name][action], detail)
            compare_statistics(metrics[action][name], manual_metrics(vy[:, i], p[name][:, i], expected_t, beta))
    assert admission(metrics) == manual_admission(metrics)


@pytest.mark.parametrize('bad', ['format', 'feature', 'setting', 'cycle'])
def test_model_serialization_tampering_rejected(models, bad):
    raw = models[5].to_dict()
    if bad == 'format':
        raw['format'] = 'position_target_histogram_v1'
    elif bad == 'feature':
        raw['features'] = raw['features'][::-1]
    elif bad == 'setting':
        raw['settings']['max_iter'] = 65
    else:
        raw['models'][0]['trees'][0]['left'][0] = 0
    with pytest.raises(ValueError):
        PairedManagementModel.from_dict(raw)


def test_admission_all_conditions_and_strict_gain_boundary():
    item = {'log_loss': .1, 'average_precision': .2, 'recall': .2, 'f_beta': .2}
    metrics = {a: {name: copy.deepcopy(item) for name in ('first', 'all', 'linear', 'constant')}
        for a in ('exit', 'reduce', 'increase')}
    for a in metrics:
        metrics[a]['all'].update(log_loss=.09, average_precision=.3, recall=.3, f_beta=.3)
    assert admission(metrics)['management_admitted']
    assert admission(metrics) == manual_admission(metrics)
    metrics['reduce']['all']['log_loss'] = .099
    assert not admission(metrics)['management_admitted']
    assert admission(metrics) == manual_admission(metrics)


def test_future_validation_values_do_not_change_fitted_trees(models):
    x, fy, _, vx, _, a, _, _ = models
    changed, _ = PairedManagementModel.fit(x, fy, vx*100)
    assert changed.to_dict() == a.to_dict()


def test_linear_export_score_at_exact_threshold_preserves_request(models):
    *_, vx, _, _, _, model = models
    actual = model.probabilities(vx)
    expected = numeric_linear_scores(model.to_dict(), vx)
    np.testing.assert_array_equal(actual, expected)
    thresholds = {action: float(actual[12, i]) for i, action in enumerate(('exit', 'reduce', 'increase'))}
    masks, chosen = direct_requests(actual, thresholds)
    ref_masks, ref_chosen = direct_requests(expected, thresholds)
    assert masks[12].all()
    np.testing.assert_array_equal(masks, ref_masks)
    np.testing.assert_array_equal(chosen, ref_chosen)
