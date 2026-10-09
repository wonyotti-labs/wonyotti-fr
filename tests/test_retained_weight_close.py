import json

import numpy as np
import pandas as pd
import pytest
from test_stopping_close import assert_classifier
from test_visited_close import training_setup

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.retained_weight_close import (
    RetainedWeightCloseModel,
    fit_retained_weight_close,
    retained_training_rows,
)
from wonyotti_fr.visited_close import visitation_rows


def setup():
    rows, _ = training_setup()
    train = rows['training']
    group = train.groupby('position_entry_time').ngroup().to_numpy()
    keep = train.groupby('position_entry_time').cumcount().to_numpy() < np.where(group % 2 == 0, 35, 50)
    train = train.loc[keep].reset_index(drop=True)
    group = train.groupby('position_entry_time').ngroup().to_numpy()
    threshold = np.array([.01, .05, .08])[group % 3]
    score = np.where(train.favorable_move.to_numpy() >= threshold, .8, .2)
    indices = np.arange(len(train))*3+100
    visit = visitation_rows(train, score, indices)
    weights = train[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(train))
    return train, weights, visit, rows['diagnosis']


def test_retained_original_weight_ratios_costs_and_independent_model_match(tmp_path):
    train, original, visit, diagnosis = setup()
    model, support = fit_retained_weight_close(train, original, visit, diagnosis, tmp_path)
    selected = visit.visited_prefix.to_numpy()
    kept = train.loc[selected].reset_index(drop=True)
    counts = train.position_entry_time.value_counts().to_dict()
    unscaled = np.array([1./counts[t] for t in kept.position_entry_time])
    manual = unscaled/unscaled.mean()
    weight = pd.read_parquet(tmp_path/'retained_training_weights.parquet')
    np.testing.assert_allclose(weight.sample_weight, manual, rtol=0, atol=1e-12)
    ratios = weight.sample_weight.to_numpy()/original.loc[selected, 'sample_weight'].to_numpy()
    np.testing.assert_allclose(ratios, np.full(len(ratios), ratios[0]), rtol=0, atol=1e-12)
    assert np.isclose(weight.sample_weight.mean(), 1.)
    assert not np.allclose(weight.sample_weight, position_weights(kept))
    assert weight.groupby('position_entry_time').sample_weight.sum().nunique() > 1
    pd.testing.assert_frame_equal(kept, pd.read_parquet(tmp_path/'retained_training_used.parquet'), check_exact=True)
    assert support['original_positions'] == support['retained_positions'] == train.position_entry_time.nunique()
    assert support['retained_rows'] == len(kept) < len(train) == support['original_rows']
    assert support['after_first_teacher_close_rows'] == (~selected).sum()
    assert support['original_prefix_weight_mean'] == original.loc[selected, 'sample_weight'].mean()
    costs = pd.read_parquet(tmp_path/'retained_training_cost_ledger.parquet')
    y = kept.close_advantage_bps.to_numpy()
    expected = weight.sample_weight.to_numpy()*np.abs(y)
    np.testing.assert_array_equal(costs.cost_weight, expected)
    np.testing.assert_array_equal(costs.fit_weight, expected/expected[expected > 0].mean())
    np.testing.assert_array_equal(costs.fit_used, y != 0)
    assert costs.loc[y == 0, 'reason'].eq('zero_effect').all()
    audit = pd.read_parquet(tmp_path/'full_training_contribution.parquet')
    pd.testing.assert_frame_equal(audit[visit.columns.drop('reason')], visit.drop(columns='reason'), check_exact=True)
    np.testing.assert_array_equal(audit.retained_original_weight[selected], weight.sample_weight)
    assert audit.loc[~selected, ['retained_original_weight', 'cost_weight', 'fit_weight']].eq(0).all().all()
    assert not audit.loc[~selected, 'fit_used'].any()
    assert audit.loc[~selected, 'reason'].eq('after_first_teacher_close').all()
    numeric = assert_classifier(model.to_dict(), kept[model.features].to_numpy(), y, weight.sample_weight.to_numpy(), diagnosis[model.features].to_numpy())
    np.testing.assert_allclose(model.probabilities(diagnosis[model.features].to_numpy())[:, 0], numeric, rtol=0, atol=1e-12)
    assert json.loads((tmp_path/'model.json').read_text()) == model.to_dict()


@pytest.mark.parametrize('damage', ['original_weights', 'prefix', 'visit_weight', 'score', 'indices', 'support'])
def test_invalid_original_weights_or_changed_prefix_are_rejected_before_fit(tmp_path, monkeypatch, damage):
    train, weights, visit, diagnosis = setup()
    if damage == 'original_weights':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'prefix':
        visit.loc[0, 'visited_prefix'] = False
    elif damage == 'visit_weight':
        visit.loc[0, 'visited_position_weight'] *= 2
    elif damage == 'score':
        visit.loc[0, 'teacher_score'] = np.nan
    elif damage == 'indices':
        visit.loc[1, 'opportunity_index'] = visit.opportunity_index.iloc[0]
    else:
        visit = visitation_rows(train, np.ones(len(train)), visit.opportunity_index)
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError('잘못된 입력 뒤 학습 호출')

    monkeypatch.setattr(RetainedWeightCloseModel, 'fit', forbidden)
    with pytest.raises((ValueError, AssertionError)):
        fit_retained_weight_close(train, weights, visit, diagnosis, tmp_path)
    assert calls == [] and not (tmp_path/'model.json').exists()


def test_time_storage_units_and_future_diagnosis_do_not_change_training_or_models(tmp_path):
    train, weights, visit, diagnosis = setup()
    time_columns = ['decision_time', 'position_entry_time', 'label_end', 'continue_end']
    outputs = []
    for unit in ['us', 'ns']:
        frame = train.copy()
        for name in time_columns:
            frame[name] = frame[name].astype(f'datetime64[{unit}, UTC]')
        original = frame[['decision_time', 'position_entry_time']].assign(sample_weight=weights.sample_weight)
        fixed = visitation_rows(frame, visit.teacher_score, visit.opportunity_index)
        future = diagnosis.copy()
        if unit == 'ns':
            future['close_advantage_bps'] *= -100
            future['favorable_move'] += 1
        out = tmp_path/unit
        out.mkdir()
        model, _ = fit_retained_weight_close(frame, original, fixed, future, out)
        outputs.append((out, model))
    assert outputs[0][1].to_dict() == outputs[1][1].to_dict()
    for filename in ['retained_training_weights', 'retained_training_cost_ledger', 'full_training_contribution']:
        left, right = [pd.read_parquet(out/f'{filename}.parquet') for out, _model in outputs]
        for value in [left, right]:
            for name in value.select_dtypes('datetimetz').columns:
                value[name] = value[name].astype('datetime64[ns, UTC]')
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert retained_training_rows(train, weights, visit)[2].sum() == visit.visited_prefix.sum()
