import json

import numpy as np
import pandas as pd
import pytest
from test_stopping_close import assert_classifier, setup, small_rows

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.minute_close_learning import MinuteCloseModel
from wonyotti_fr.stopping_close import position_folds
from wonyotti_fr.visited_close import (
    VisitedCloseModel,
    fit_visitation_teachers,
    fit_visited_close,
    visitation_rows,
)


def training_setup():
    original, _ = setup()
    rows = {}
    for name, part in original.items():
        frame = part.copy()
        step = frame.groupby('position_entry_time').cumcount().to_numpy()
        frame['favorable_move'] = step/100
        frame['close_advantage_bps'] = np.where(step >= 5, 12., -10.)
        frame.loc[frame.index % 17 == 0, 'close_advantage_bps'] = 0.
        frame['close_advantage_pnl'] = frame.close_advantage_bps*frame.decision_equity/10000
        frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
        repeated = frame.loc[frame.index.repeat(5)].reset_index(drop=True)
        repeated['decision_time'] += pd.to_timedelta(np.tile(np.arange(-4, 1), len(frame)), unit='min')
        rows[name] = repeated
    train = rows['training']
    weights = train[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(train))
    return rows, weights


def independent_prefix(frame, score):
    stopped = set()
    included = []
    for row, value in zip(frame.itertuples(index=False), score, strict=True):
        included.append(row.position_entry_time not in stopped)
        if value > .5 and row.original_intent != 'exit':
            stopped.add(row.position_entry_time)
    return np.array(included)


def test_first_teacher_choice_is_inclusive_with_exit_ties_no_choice_and_all_positions_preserved():
    frame = small_rows()
    frame.loc[0, 'original_intent'] = 'exit'
    score = np.r_[.9, .5, .9, .6, np.full(6, .8), np.full(10, .2)]
    indices = np.arange(len(frame))*3+100
    result = visitation_rows(frame, score, indices)
    expected = np.r_[np.ones(4, bool), np.zeros(6, bool), np.ones(10, bool)]
    np.testing.assert_array_equal(result.visited_prefix, expected)
    np.testing.assert_array_equal(result.visited_prefix, independent_prefix(frame, score))
    np.testing.assert_array_equal(result.first_teacher_opportunity_index, np.r_[np.full(10, indices[3]), np.full(10, -1)])
    assert result.first_teacher_close_time.iloc[10:].isna().all()
    np.testing.assert_allclose(result.visited_position_weight, np.r_[np.full(4, 1.75), np.zeros(6), np.full(10, .7)], rtol=0, atol=1e-12)
    sums = result.groupby('position_entry_time').visited_position_weight.sum()
    np.testing.assert_allclose(sums, [7., 7.], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.original_position_weight, position_weights(frame))
    pd.testing.assert_frame_equal(result[frame.columns.intersection(result.columns)], frame[frame.columns.intersection(result.columns)], check_exact=True)
    later = score.copy()
    later[4:10] = 0.
    stable = ['visited_prefix', 'first_teacher_opportunity_index', 'first_teacher_close_time', 'visited_position_weight']
    pd.testing.assert_frame_equal(visitation_rows(frame, later, indices)[stable], result[stable], check_exact=True)
    changed = frame.copy()
    changed['close_advantage_pnl'] *= -100
    changed['close_advantage_bps'] *= -100
    changed['close_cash'] = changed.continue_cash+changed.close_advantage_pnl
    np.testing.assert_array_equal(visitation_rows(changed, score, indices).visited_prefix, expected)


@pytest.mark.parametrize('damage', ['score_nan', 'score_range', 'off_minute', 'indices', 'maturity'])
def test_invalid_score_clock_indices_and_unconfirmed_targets_are_rejected(damage):
    frame = small_rows()
    scores, indices = np.full(len(frame), .6), np.arange(len(frame))
    if damage == 'score_nan':
        scores[0] = np.nan
    elif damage == 'score_range':
        scores[0] = 1.1
    elif damage == 'off_minute':
        frame.loc[0, 'decision_time'] += pd.Timedelta(seconds=1)
    elif damage == 'indices':
        indices[1] = indices[0]
    else:
        frame.loc[0, 'label_end'] = pd.Timestamp('2021-09-30', tz='UTC')
    with pytest.raises(ValueError):
        visitation_rows(frame, scores, indices)


def test_excluded_teachers_visited_weights_costs_and_final_model_match_independent_fits(tmp_path):
    rows, weights = training_setup()
    train = rows['training']
    indices = np.arange(len(train))*2+10
    model, support = fit_visited_close(train, weights, rows['diagnosis'], indices, tmp_path)
    visit = pd.read_parquet(tmp_path/'visitation_ledger.parquet')
    member = pd.read_parquet(tmp_path/'teacher_training_membership.parquet')
    teachers = json.loads((tmp_path/'teacher_models.json').read_text())
    folds, x = position_folds(train), train[model.features].to_numpy()
    assert len(teachers) == 5 and len(member) == 4*len(train)
    for fold in range(5):
        selected = folds == fold
        fit = train.loc[~selected]
        saved = member[member.model_key.eq(f'fold-{fold:02}')].reset_index(drop=True)
        assert not set(saved.position_entry_time) & set(train.loc[selected, 'position_entry_time'])
        np.testing.assert_array_equal(saved.opportunity_index, indices[~selected])
        counts = fit.position_entry_time.value_counts().to_dict()
        raw = np.array([1./counts[t] for t in fit.position_entry_time])
        w = raw/raw.mean()
        np.testing.assert_array_equal(saved.sample_weight, w)
        cost = w*np.abs(fit.close_advantage_bps.to_numpy())
        normalized = cost/cost[cost > 0].mean()
        np.testing.assert_array_equal(saved.cost_weight, cost)
        np.testing.assert_array_equal(saved.fit_weight, normalized)
        np.testing.assert_array_equal(saved.fit_used, cost > 0)
        assert teachers[f'fold-{fold:02}']['format'] == MinuteCloseModel.format
        expected = assert_classifier(teachers[f'fold-{fold:02}'], x[~selected], fit.close_advantage_bps.to_numpy(), w, x[selected])
        np.testing.assert_allclose(visit.teacher_score[selected], expected, rtol=0, atol=1e-12)
    included = independent_prefix(train, visit.teacher_score.to_numpy())
    np.testing.assert_array_equal(visit.visited_prefix, included)
    retained = train.loc[included].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'visited_training_used.parquet'), retained, check_exact=True)
    new_weights = position_weights(retained)
    np.testing.assert_array_equal(visit.visited_position_weight[included], new_weights)
    assert (visit.visited_position_weight[~included] == 0).all()
    assert support['visited_rows'] == len(retained) < len(train) == support['original_rows']
    assert support['original_positions'] == support['visited_positions'] == train.position_entry_time.nunique()
    costs = pd.read_parquet(tmp_path/'visited_training_cost_ledger.parquet')
    np.testing.assert_array_equal(costs.cost_weight, new_weights*np.abs(retained.close_advantage_bps.to_numpy()))
    assert costs.loc[~costs.fit_used, 'reason'].eq('zero_effect').all()
    audit = pd.read_parquet(tmp_path/'full_training_contribution.parquet')
    assert audit.loc[~included, 'reason'].eq('after_first_teacher_close').all()
    assert not audit.loc[~included, 'fit_used'].any() and audit.loc[~included, ['cost_weight', 'fit_weight']].eq(0).all().all()
    pd.testing.assert_series_equal(audit.close_advantage_bps, train.close_advantage_bps, check_exact=True)
    diagnosis = rows['diagnosis'][model.features].to_numpy()
    expected = assert_classifier(model.to_dict(), x[included], retained.close_advantage_bps.to_numpy(), new_weights, diagnosis)
    np.testing.assert_allclose(model.probabilities(diagnosis)[:, 0], expected, rtol=0, atol=1e-12)
    assert model.format == 'visited_minute_cost_weighted_close_v1'
    with pytest.raises(ValueError):
        MinuteCloseModel.from_dict(model.to_dict())
    with pytest.raises(ValueError):
        VisitedCloseModel.from_dict(teachers['fold-00'])


def test_own_fold_target_change_and_future_diagnosis_do_not_leak_into_teacher_or_training(tmp_path):
    rows, weights = training_setup()
    train = rows['training']
    indices = np.arange(len(train))
    for name in ['original', 'changed', 'future']:
        (tmp_path/name).mkdir()
    model, _ = fit_visited_close(train, weights, rows['diagnosis'], indices, tmp_path/'original')
    own_fold = position_folds(train) == 0
    changed = train.copy()
    changed.loc[own_fold, 'close_advantage_bps'] *= -7
    changed['close_advantage_pnl'] = changed.close_advantage_bps*changed.decision_equity/10000
    changed['close_cash'] = changed.continue_cash+changed.close_advantage_pnl
    fit_visitation_teachers(changed, indices, tmp_path/'changed')
    original_teachers = json.loads((tmp_path/'original/teacher_models.json').read_text())
    assert original_teachers['fold-00'] == json.loads((tmp_path/'changed/teacher_models.json').read_text())['fold-00']
    a, b = (pd.read_parquet(tmp_path/name/'visitation_ledger.parquet') for name in ['original', 'changed'])
    np.testing.assert_array_equal(a.teacher_score[own_fold], b.teacher_score[own_fold])
    np.testing.assert_array_equal(a.visited_prefix[own_fold], b.visited_prefix[own_fold])
    diagnosis = rows['diagnosis'].copy()
    diagnosis['close_advantage_bps'] *= -100
    diagnosis['favorable_move'] += 1
    future, _ = fit_visited_close(train, weights, diagnosis, indices, tmp_path/'future')
    assert future.to_dict() == model.to_dict()
    assert original_teachers == json.loads((tmp_path/'future/teacher_models.json').read_text())
    for name in ['visitation_ledger', 'teacher_training_membership', 'visited_training_weights', 'visited_training_cost_ledger', 'full_training_contribution']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'original'/f'{name}.parquet'), pd.read_parquet(tmp_path/'future'/f'{name}.parquet'), check_exact=True)


def test_insufficient_visited_support_is_rejected_before_final_fit(tmp_path, monkeypatch):
    rows, weights = training_setup()
    train = rows['training']
    indices = np.arange(len(train))
    membership = visitation_rows(train, np.full(len(train), .9), indices)
    monkeypatch.setattr('wonyotti_fr.visited_close.fit_visitation_teachers', lambda *_args: membership)
    with pytest.raises(ValueError, match='지원 부족'):
        fit_visited_close(train, weights, rows['diagnosis'], indices, tmp_path)
    assert not (tmp_path/'model.json').exists()
