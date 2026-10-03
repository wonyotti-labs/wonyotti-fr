import copy

import numpy as np
import pandas as pd
import pytest
from test_action_model import FixedScores
from test_pullback_evaluation import ConstantBase
from test_rate_policy import inputs

from wonyotti_fr.inventory_labels import sizing_labels, sizing_training
from wonyotti_fr.inventory_management import (
    InventoryActionModels,
    InventoryRatePolicy,
    ReductionModel,
)
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.pullback_policy import PullbackPolicy


def model_data(intercept=.8):
    n = len(InventoryActionModels.features)
    return {'format': ReductionModel.format, 'features': ReductionModel.features, 'alpha': 100.,
            'mean': [0.]*n, 'scale': [1.]*n, 'coef': [0.]*n, 'intercept': intercept}


def bot(scores=(0, 1, 0), intercept=.8):
    return InventoryRatePolicy(PullbackPolicy(ConstantBase(), 16, 5), FixedScores(scores),
        dict.fromkeys(ACTIONS, .5), 1., dict.fromkeys(ACTIONS, 1.),
        ReductionModel.from_dict(model_data(intercept)) if intercept is not None else None)


@pytest.mark.parametrize('intercept,intent,fraction', [(.8, 'reduce', .8), (1.5, 'reduce', 1.),
                                                     (-.1, 'hold', None), (None, 'reduce', None)])
def test_actual_inventory_feature_and_sized_decision(intercept, intent, fraction):
    bar, state = inputs(1)
    state['remaining_fraction'] = .25
    policy = bot(intercept=intercept)
    context, _ = policy.path_context(bar, state)
    assert policy.feature_values(bar, context)[-1] == .25
    result = policy(bar, state)
    assert result.intent == intent and result.reduction_fraction == fraction
    assert result.state['rate_reduce'] == 0
    bar, state = inputs(2, result.state)
    state['remaining_fraction'] = .1
    assert policy(bar, state).event == 'action_rate_cooldown'


def test_flat_exit_preserves_original_behavior_and_no_size_prediction():
    policy = bot((1, 0, 0))
    bar, state = inputs(1)
    state['remaining_fraction'] = .001
    result = policy(bar, state)
    assert result.intent == 'exit' and result.reduction_fraction is None


@pytest.mark.parametrize('field,value', [('alpha', 1.), ('coef', [0.]), ('scale', [0.]*36),
                                        ('intercept', float('nan')), ('features', ['future_quantity'])])
def test_numeric_size_model_rejects_corrupt_export(field, value):
    data = model_data()
    data[field] = value
    with pytest.raises(ValueError):
        ReductionModel.from_dict(data)


def test_size_fit_is_train_only_export_exact_and_has_minimum_support():
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(rng.normal(size=(200, 36)), columns=ReductionModel.features)
    frame['end'] = pd.date_range('2019-01-03', periods=200, freq='D', tz='UTC')
    frame['label_end'] = frame.end + pd.Timedelta(hours=2)
    frame['reduction_target'] = np.clip(.5 + .1*frame.iloc[:, 0], .01, 1)
    model, support = ReductionModel.fit(frame)
    restored = ReductionModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.predict(frame[model.features]), restored.predict(frame[model.features]))
    assert support['rows'] == 200 and support['export_max_error'] < 1e-12
    for bad in [frame.iloc[:99], frame.assign(label_end=pd.Timestamp('2020-07-01', tz='UTC')),
                frame.assign(reduction_target=0)]:
        with pytest.raises(ValueError):
            ReductionModel.fit(bad)
    exported = model.to_dict()
    exported['mean'][0] += 1
    assert model.to_dict() != exported


def label_example():
    ends = pd.date_range('2019-01-03', periods=5, freq='D', tz='UTC')
    frame = pd.DataFrame({'end': ends, 'inventory_quantity': 100})
    orders = pd.DataFrame({'order_key': list('abcde'), 'window_end': ends, 'end': ends, 'reason': 'linked',
        'action': ['reduce']*4+['increase'], 'before_qty': [100, 100, 50, 100, 100]})
    sizes = pd.DataFrame({'order_key': list('abcde'), 'last_time': ends + pd.Timedelta(hours=3, seconds=2),
        'executed_quantity': [60, 20, 20, 40, 10], 'interleaved': [False, False, False, True, False]})
    orders = pd.concat([orders, orders.iloc[[1]].assign(order_key='f', action='exit')], ignore_index=True)
    sizes = pd.concat([sizes, sizes.iloc[[1]].assign(order_key='f')], ignore_index=True)
    return frame, orders, sizes


def test_sizing_target_uses_actual_partial_fill_and_preserves_unsupported_rows():
    frame, orders, sizes = label_example()
    original = copy.deepcopy(orders)
    ledger = sizing_labels(frame, orders, sizes).set_index('order_key')
    assert len(ledger) == len(orders)
    assert ledger.loc['a', 'reduction_target'] == .6
    assert ledger.loc['a', 'size_label_end'] == frame.end.iloc[0] + pd.Timedelta(hours=3, minutes=1)
    assert ledger.loc['a', 'size_reason'] == 'supported'
    assert ledger.loc['b', 'size_reason'] == 'multiple_management_orders'
    assert ledger.loc['c', 'size_reason'] == 'boundary_quantity_changed'
    assert ledger.loc['d', 'size_reason'] == 'interleaved_fills'
    assert ledger.loc['e', 'size_reason'] == 'not_reduce'
    pd.testing.assert_frame_equal(orders, original)


def test_size_training_purges_late_fills_and_future_episode_crossing():
    ends = pd.to_datetime(['2019-02-01', '2020-06-29T00:00Z', '2020-06-29T01:00Z', '2020-07-01T01:00Z'], utc=True, format='mixed')
    frame = pd.DataFrame(np.zeros((4, 36)), columns=ReductionModel.features)
    frame = frame.assign(end=ends, entry_time=pd.Timestamp('2019-01-03', tz='UTC'),
        episode_id=[1, 2, 3, 3], label_end=ends+pd.Timedelta(minutes=1), usable=True)
    ledger = pd.DataFrame({'order_key': list('abc'), 'window_end': ends[:3], 'size_reason': 'supported',
        'reduction_target': .5, 'size_label_end': [ends[0]+pd.Timedelta(minutes=1), pd.Timestamp('2020-06-30', tz='UTC'), ends[2]+pd.Timedelta(minutes=1)]})
    result = sizing_training(frame, ledger)
    assert result.order_key.tolist() == ['a']
