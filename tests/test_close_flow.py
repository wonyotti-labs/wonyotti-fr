import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_close_context import dataset
from test_close_utility import examples
from test_context_position import bars
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_context import ContextCloseModel, attach_close_context
from wonyotti_fr.close_flow import (
    FLOW_FEATURES,
    FLOW_WINDOWS,
    FlowCloseModel,
    attach_close_flow,
    flow_close_admission,
    flow_features,
)
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS


def flow_bars(count=400):
    frame = bars(count).assign(count=1.)
    frame['volume'] = np.arange(count, dtype=float)+1
    frame['taker_buy_volume'] = frame.volume*(np.arange(count)%3)/2
    return frame


def test_flow_uses_volume_weighted_window_sums_and_only_current_confirmed_bars():
    frame = flow_bars()
    result = flow_features(frame)
    for name, window in FLOW_WINDOWS.items():
        assert result[name].iloc[:window-1].isna().all()
        for i in range(window-1, len(frame)):
            part = frame.iloc[i-window+1:i+1]
            expected = 2*part.taker_buy_volume.sum()/part.volume.sum()-1
            assert result.loc[i, name] == pytest.approx(expected, abs=1e-12)
    assert result.loc[2, 'taker_imbalance_15m'] != pytest.approx(0.)
    changed = frame.copy()
    changed.loc[301:, 'taker_buy_volume'] = changed.loc[301:, 'volume']
    pd.testing.assert_frame_equal(flow_features(changed).iloc[:301], result.iloc[:301], check_exact=True)
    assert result.iloc[-1][list(FLOW_WINDOWS)].between(-1, 1).all()


def test_empty_volume_has_no_direction_and_gaps_require_each_window_to_restart():
    frame = flow_bars()
    zero = frame.assign(open=100., high=100., low=100., close=100., volume=0., count=0., taker_buy_volume=0.)
    result = flow_features(zero)
    for name, window in FLOW_WINDOWS.items():
        assert result[name].iloc[:window-1].isna().all()
        assert result[name].iloc[window-1:].eq(0).all()
    gap = frame.drop(index=100).reset_index(drop=True)
    result = flow_features(gap)
    for name, window in FLOW_WINDOWS.items():
        assert result[name].iloc[100:100+window-1].isna().all()
        assert result.loc[100+window-1, name] == pytest.approx(
            2*gap.taker_buy_volume.iloc[100:100+window].sum()/gap.volume.iloc[100:100+window].sum()-1, abs=1e-12)


@pytest.mark.parametrize('damage', ['negative', 'excess', 'nan', 'bad_volume', 'bad_time', 'empty'])
def test_invalid_buy_volume_or_market_is_rejected(damage):
    frame = flow_bars()
    if damage == 'negative':
        frame.loc[0, 'taker_buy_volume'] = -.01
    elif damage == 'excess':
        frame.loc[0, 'taker_buy_volume'] = frame.loc[0, 'volume']+.01
    elif damage == 'nan':
        frame.loc[0, 'taker_buy_volume'] = np.nan
    elif damage == 'bad_volume':
        frame.loc[0, 'volume'] = np.inf
    elif damage == 'bad_time':
        frame.loc[0, 'end'] += pd.Timedelta(minutes=1)
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        flow_features(frame)


def test_attachment_preserves_original64_and_uses_position_direction_without_future_changes():
    market, original = dataset()
    market['taker_buy_volume'] = market.volume*(np.arange(len(market))%3)/2
    original['direction'] = np.where(np.arange(len(original))%2, 1., -1.)
    context = attach_close_context({'training': original}, market)['training']
    attached = attach_close_flow({'training': context}, market)['training']
    pd.testing.assert_frame_equal(attached.drop(columns=FLOW_FEATURES), context, check_exact=True)
    assert len(FlowCloseModel.features) == 74 and FlowCloseModel.features[:64] == ContextCloseModel.features
    for name in FLOW_WINDOWS:
        np.testing.assert_array_equal(attached['directional_'+name], attached[name]*attached.direction)
    future = market.copy()
    future.loc[future.end.gt(context.decision_time.max()), 'taker_buy_volume'] = -1e8
    pd.testing.assert_frame_equal(attach_close_flow({'training': context}, future)['training'], attached, check_exact=True)
    changed = context.copy()
    changed.loc[0, 'ret_16d'] += 1
    with pytest.raises(AssertionError):
        attach_close_flow({'training': changed}, market)
    changed = context.copy()
    changed.loc[0, 'direction'] = 0.
    with pytest.raises(ValueError, match='방향'):
        attach_close_flow({'training': changed}, market)
    with pytest.raises(ValueError, match='중복'):
        attach_close_flow({'training': attached}, market)


def test_model_preserves_costs_and_matches_independent_fit_with_flow_direction_validation():
    x, y, w = examples()
    rng = np.random.default_rng(71)
    x[:, 28] = np.where(np.arange(len(x))%2, 1., -1.)
    context = np.c_[x, rng.normal(size=(len(x), 8))]
    raw = rng.uniform(-1, 1, size=(len(x), 5))
    full = np.c_[context, raw, raw*x[:, 28, None]]
    model, support, ledger = FlowCloseModel.fit(full, y, w, full[:100])
    _, prior_support, prior_ledger = ContextCloseModel.fit(context, y, w, context[:100])
    assert {k: v for k, v in support.items() if k != 'export'} == {k: v for k, v in prior_support.items() if k != 'export'}
    pd.testing.assert_frame_equal(ledger, prior_ledger, check_exact=True)
    used = ledger.fit_used.to_numpy()
    with threadpool_limits(limits=1):
        independent = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(full[used], y[used] > 0, sample_weight=ledger.fit_weight[used])
        np.testing.assert_allclose(model.probabilities(full)[:, 0], independent.predict_proba(full)[:, 1], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(FlowCloseModel.from_dict(model.to_dict()).probabilities(full), model.probabilities(full))
    for column, value in [(28, 0.), (52, 2.), (64, 1.1), (69, 5.), (70, np.nan)]:
        changed = full.copy()
        changed[0, column] = value
        with pytest.raises(ValueError):
            model.probabilities(changed)
    future = full[:100].copy()
    future[:, 64:] *= -1
    other, _, other_ledger = FlowCloseModel.fit(full, y, w, future)
    assert model.to_dict() == other.to_dict()
    pd.testing.assert_frame_equal(ledger, other_ledger, check_exact=True)
    with pytest.raises(ValueError):
        ContextCloseModel.from_dict(model.to_dict())


def test_twenty_four_gates_preserve_prior_requirements_and_add_all_context_comparisons():
    common = {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.}
    metrics = {name: {**common, 'weighted_regret_bps': regret} for name, regret in
        [('flow', 1.), ('context', 1.4), ('utility', 1.5), ('training_constant', 2.), ('continuation', 2.), ('weekly', 2.)]}
    probability = {name: {'rows': 200, 'cost_log_loss': loss, 'cost_brier': brier} for name, loss, brier in
        [('flow', .45, .18), ('context', .5, .2), ('utility', .55, .25), ('training_constant', .6, .3)]}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': mean} for name, mean in
        [('flow', 3.), ('context', 2.), ('utility', 1.5), ('continuation', 1.), ('weekly', 1.)]}
    interval = {'intervals': {'flow': {'lower': .1}, 'paired_difference': {'lower': .1}}}
    decision = flow_close_admission(metrics, probability, first, interval, interval, interval)
    assert decision['flow_admitted'] and len(decision['checks']) == 24
    for field, value in [('cost_log_loss', .44), ('cost_brier', .17)]:
        changed = copy.deepcopy(probability)
        changed['context'][field] = value
        assert not flow_close_admission(metrics, changed, first, interval, interval, interval)['flow_admitted']
    changed = copy.deepcopy(metrics)
    changed['context']['weighted_regret_bps'] = 1.
    assert not flow_close_admission(changed, probability, first, interval, interval, interval)['flow_admitted']
    changed = copy.deepcopy(first)
    changed['context']['all_position_mean_common_bps'] = 3.
    assert not flow_close_admission(metrics, probability, changed, interval, interval, interval)['flow_admitted']
    changed = copy.deepcopy(interval)
    changed['intervals']['paired_difference']['lower'] = 0.
    assert not flow_close_admission(metrics, probability, first, interval, interval, changed)['flow_admitted']
    changed = copy.deepcopy(metrics)
    changed['flow']['selected_weighted_mean_bps'] = -1.
    assert not flow_close_admission(changed, probability, first, interval, interval, interval)['flow_admitted']
