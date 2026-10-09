import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_close_capacity import internal_frames
from test_close_context_diagnostics import synthetic_context
from test_close_flow_diagnostics import synthetic_flow
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_flow import FlowCloseModel
from wonyotti_fr.close_learning_inputs import close_learning_splits
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.stopping_close import (
    StoppingCloseModel,
    fit_stopping_close,
    fit_stopping_teachers,
    position_folds,
    stopping_admission,
    stopping_targets,
)


def setup():
    ledger = synthetic_flow(synthetic_context({'all': internal_frames()}, None), None)['all']
    ledger.loc[ledger.index % 17 == 0, 'close_advantage_bps'] = 0.
    ledger['close_advantage_pnl'] = ledger.close_advantage_bps*ledger.decision_equity/10000
    ledger['close_cash'] = ledger.continue_cash+ledger.close_advantage_pnl
    rows, _ = close_learning_splits(ledger)
    weights = rows['training'][['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(rows['training']))
    return rows, weights


def small_rows():
    rows, _ = setup()
    frame = rows['training'].iloc[:20].copy()
    frame['decision_equity'] = np.arange(20)+100.
    frame['continue_cash'] = 100.
    frame['close_cash'] = np.tile([100., 90., 120., 70., 110., 120., 90., 100., 101., 100.], 2)
    frame['close_advantage_pnl'] = frame.close_cash-frame.continue_cash
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    frame['original_intent'] = 'hold'
    frame.loc[[2, 7], 'original_intent'] = 'exit'
    return frame


def independent_targets(frame, score):
    choices, values = [], []
    for i, row in frame.iterrows():
        later = [j for j in range(i+1, len(frame)) if frame.position_entry_time.iloc[j] == row.position_entry_time
            and score[j] > .5 and frame.original_intent.iloc[j] != 'exit']
        future = later[0] if later else -1
        cash = frame.close_cash.iloc[future] if future >= 0 else row.continue_cash
        choices.append(future)
        values.append((row.close_cash-cash)/row.decision_equity*10000)
    return np.array(choices), np.array(values)


def test_future_first_choice_excludes_self_exit_ties_and_other_positions_without_maximum_selection():
    frame = small_rows()
    score = np.r_[.8, .8, .9, .5, np.zeros(6), np.zeros(10)]
    indices = np.arange(len(frame))*3+100
    result = stopping_targets(frame, score, indices)
    chosen, values = independent_targets(frame, score)
    np.testing.assert_array_equal(result.future_opportunity_index, np.where(chosen >= 0, indices[np.maximum(chosen, 0)], -1))
    np.testing.assert_array_equal(result.stopping_advantage_bps, values)
    assert result.future_opportunity_index.iloc[0] == indices[1]
    assert result.stopping_advantage_bps.iloc[0] > 0 and frame.close_advantage_bps.iloc[0] == 0
    assert result.stopping_advantage_bps.iloc[1] < 0
    assert result.natural_close_fallback.iloc[1:].all()
    assert result.future_decision_time.iloc[1:].isna().all()
    assert result.stopping_advantage_bps.eq(0).any()
    pd.testing.assert_series_equal(result.target_available_time, frame.label_end, check_names=False)
    pd.testing.assert_frame_equal(result[frame.columns.intersection(result.columns)], frame[frame.columns.intersection(result.columns)], check_exact=True)


def test_position_hash_folds_are_stable_across_rows_order_and_timestamp_precision():
    frame = small_rows()
    expected = [int(hashlib.sha256(str(t.value).encode()).hexdigest(), 16) % 5 for t in frame.position_entry_time]
    np.testing.assert_array_equal(position_folds(frame), expected)
    changed = frame.iloc[::-1].copy()
    changed['position_entry_time'] = changed.position_entry_time.astype('datetime64[ms, UTC]')
    np.testing.assert_array_equal(position_folds(changed), expected[::-1])


@pytest.mark.parametrize('damage', ['fraction', 'duplicate', 'negative', 'unsigned_descending', 'unsigned_overflow'])
def test_original_indices_reject_lossy_types_duplicates_and_integer_overflow(damage):
    frame = small_rows()
    indices = np.arange(len(frame))
    if damage == 'fraction':
        indices = indices.astype(float)+.5
    elif damage == 'duplicate':
        indices[1] = indices[0]
    elif damage == 'negative':
        indices[0] = -1
    elif damage == 'unsigned_descending':
        indices = indices[::-1].astype(np.uint64)
    else:
        indices = indices.astype(np.uint64)+(1 << 63)
    with pytest.raises(ValueError, match='행 번호'):
        stopping_targets(frame, np.full(len(frame), .6), indices)


@pytest.mark.parametrize('damage', ['cutoff', 'cash', 'equity', 'entry', 'duplicate', 'naive', 'unfinished', 'direction'])
def test_unavailable_or_inconsistent_cash_and_position_rows_are_rejected(damage):
    frame = small_rows()
    if damage == 'cutoff':
        frame.loc[0, 'label_end'] = pd.Timestamp('2021-09-30', tz='UTC')
    elif damage == 'cash':
        frame.loc[0, 'continue_cash'] += 1
    elif damage == 'equity':
        frame.loc[0, 'decision_equity'] = 0
    elif damage == 'entry':
        frame.loc[0, 'position_entry_time'] = frame.decision_time.iloc[0]
    elif damage == 'duplicate':
        frame.loc[1, 'decision_time'] = frame.decision_time.iloc[0]
    elif damage == 'naive':
        frame['decision_time'] = frame.decision_time.dt.tz_localize(None)
    elif damage == 'unfinished':
        frame.loc[0, 'label_status'] = 'right_censored'
    else:
        frame.loc[0, 'direction'] *= -1
    with pytest.raises(ValueError):
        stopping_targets(frame, np.full(len(frame), .6), np.arange(len(frame)))


def assert_classifier(data, x, y, w, validation):
    cost = w*np.abs(y)
    fit = cost > 0
    normalized = cost/cost[fit].mean()
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x[fit], y[fit] > 0, sample_weight=normalized[fit])
        binary = data['models'][0]
        assert binary['baseline'] == learner._baseline_prediction[0, 0]
        for exported, trees in zip(binary['trees'], learner._predictors, strict=True):
            nodes = trees[0].nodes
            leaf = nodes['is_leaf'].astype(bool)
            for name, expected in {'left': np.where(leaf, -1, nodes['left'].astype(int)), 'right': np.where(leaf, -1, nodes['right'].astype(int)),
                'value': np.where(leaf, nodes['value'], 0.), 'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)),
                'threshold': np.where(leaf, 0., nodes['num_threshold'])}.items():
                np.testing.assert_array_equal(exported[name], expected)
        return learner.predict_proba(validation)[:, 1]


def test_all_five_excluded_position_teachers_targets_and_final_model_match_independent_fits(tmp_path):
    rows, weights = setup()
    train = rows['training']
    indices = np.arange(len(train))*2
    model, support = fit_stopping_close(train, weights, rows['diagnosis'], indices, tmp_path)
    targets = pd.read_parquet(tmp_path/'stopping_targets.parquet')
    membership = pd.read_parquet(tmp_path/'teacher_training_membership.parquet')
    models = json.loads((tmp_path/'teacher_models.json').read_text())
    supports = json.loads((tmp_path/'teacher_support.json').read_text())
    folds = position_folds(train)
    x = train[model.features].to_numpy()
    assert len(models) == membership.model_key.nunique() == 5 and len(membership) == 4*len(train)
    assert membership.fit_used.eq(False).any()
    for fold in range(5):
        selected = folds == fold
        fit = train.loc[~selected]
        saved = membership[membership.model_key.eq(f'fold-{fold:02}')].reset_index(drop=True)
        np.testing.assert_array_equal(saved.opportunity_index, indices[~selected])
        assert not set(saved.position_entry_time) & set(train.loc[selected, 'position_entry_time'])
        counts = fit.position_entry_time.value_counts().to_dict()
        raw = np.array([1./counts[t] for t in fit.position_entry_time])
        w = raw/raw.mean()
        np.testing.assert_array_equal(saved.sample_weight, w)
        np.testing.assert_array_equal(saved.original_weight, w)
        cost = w*np.abs(fit.close_advantage_bps.to_numpy())
        used = cost > 0
        normalized = cost/cost[used].mean()
        np.testing.assert_array_equal(saved.cost_weight, cost)
        np.testing.assert_array_equal(saved.fit_weight, normalized)
        np.testing.assert_array_equal(saved.fit_used, used)
        assert saved.loc[~used, 'reason'].eq('zero_effect').all()
        details = supports[f'fold-{fold:02}']
        assert details['normalizer'] == cost[used].mean()
        assert details['training_constant_score'] == np.average(fit.close_advantage_bps.to_numpy()[used] > 0, weights=normalized[used])
        expected = assert_classifier(models[f'fold-{fold:02}'], x[~selected], fit.close_advantage_bps.to_numpy(), w, x[selected])
        np.testing.assert_allclose(targets.teacher_score[selected], expected, rtol=0, atol=1e-12)
    chosen, target = independent_targets(train, targets.teacher_score.to_numpy())
    np.testing.assert_array_equal(targets.future_opportunity_index, np.where(chosen >= 0, indices[np.maximum(chosen, 0)], -1))
    np.testing.assert_array_equal(targets.stopping_advantage_bps, target)
    diagnosis = rows['diagnosis'][model.features].to_numpy()
    expected = assert_classifier(model.to_dict(), x, target, weights.sample_weight.to_numpy(), diagnosis)
    np.testing.assert_allclose(model.probabilities(diagnosis)[:, 0], expected, rtol=0, atol=1e-12)
    assert model.format == 'stopping_cost_weighted_close_v1' and support['rows'] == len(train)
    costs = pd.read_parquet(tmp_path/'stopping_training_cost_ledger.parquet')
    np.testing.assert_array_equal(costs.cost_weight, weights.sample_weight.to_numpy()*abs(target))
    assert costs.loc[~costs.fit_used, 'reason'].eq('zero_effect').all()
    with pytest.raises(ValueError):
        FlowCloseModel.from_dict(model.to_dict())
    with pytest.raises(ValueError):
        StoppingCloseModel.from_dict(models['fold-00'])


def test_excluded_position_targets_do_not_change_its_teacher_and_future_diagnosis_does_not_change_training(tmp_path):
    rows, weights = setup()
    train = rows['training']
    indices = np.arange(len(train))
    for name in ['original', 'changed', 'future']:
        (tmp_path/name).mkdir()
    model, _ = fit_stopping_close(train, weights, rows['diagnosis'], indices, tmp_path/'original')
    selected = position_folds(train) == 0
    changed = train.copy()
    changed.loc[selected, 'close_advantage_bps'] *= -7
    changed['close_advantage_pnl'] = changed.close_advantage_bps*changed.decision_equity/10000
    changed['close_cash'] = changed.continue_cash+changed.close_advantage_pnl
    fit_stopping_teachers(changed, indices, tmp_path/'changed')
    assert json.loads((tmp_path/'original'/'teacher_models.json').read_text())['fold-00'] == json.loads((tmp_path/'changed'/'teacher_models.json').read_text())['fold-00']
    a, b = (pd.read_parquet(tmp_path/name/'stopping_targets.parquet') for name in ['original', 'changed'])
    np.testing.assert_array_equal(a.teacher_score[selected], b.teacher_score[selected])
    diagnosis = rows['diagnosis'].copy()
    diagnosis['close_advantage_bps'] *= -100
    diagnosis['favorable_move'] += 1
    future, _ = fit_stopping_close(train, weights, diagnosis, indices, tmp_path/'future')
    assert future.to_dict() == model.to_dict()
    assert json.loads((tmp_path/'original'/'teacher_models.json').read_text()) == json.loads((tmp_path/'future'/'teacher_models.json').read_text())
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'original'/'stopping_targets.parquet'), pd.read_parquet(tmp_path/'future'/'stopping_targets.parquet'), check_exact=True)


def test_all_thirty_seven_gates_preserve_old_requirements_and_add_weekly_model_comparisons():
    common = {'rows': 200, 'positions': 40, 'selected': 120, 'selected_positions': 30, 'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.}
    names = ['stopping_flow', 'weekly_flow', 'flow', 'context', 'utility', 'training_constant', 'continuation', 'weekly', 'weekly_constant']
    metrics = {name: {**common, 'weighted_regret_bps': 1.+i} for i, name in enumerate(names)}
    probability = {name: {'rows': 200, 'cost_log_loss': .4+i*.03, 'cost_brier': .1+i*.02} for i, name in enumerate(names)}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 10.-i} for i, name in enumerate(names)}
    interval = {'intervals': {'stopping_flow': {'lower': .1}, 'paired_difference': {'lower': .1}}}
    decision = stopping_admission(metrics, probability, first, *[interval]*5)
    assert decision['stopping_flow_admitted'] and len(decision['checks']) == 37
    for field in ['cost_log_loss', 'cost_brier']:
        changed = copy.deepcopy(probability)
        changed['weekly_flow'][field] = probability['stopping_flow'][field]-.01
        assert not stopping_admission(metrics, changed, first, *[interval]*5)['stopping_flow_admitted']
    changed = copy.deepcopy(metrics)
    changed['weekly_flow']['weighted_regret_bps'] = 1.
    assert not stopping_admission(changed, probability, first, *[interval]*5)['stopping_flow_admitted']
    changed = copy.deepcopy(first)
    changed['weekly_flow']['all_position_mean_common_bps'] = 10.
    assert not stopping_admission(metrics, probability, changed, *[interval]*5)['stopping_flow_admitted']
    changed = copy.deepcopy(interval)
    changed['intervals']['paired_difference']['lower'] = 0.
    assert not stopping_admission(metrics, probability, first, *[interval]*4, changed)['stopping_flow_admitted']
