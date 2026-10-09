import json

import numpy as np
import pandas as pd
import pytest
from test_close_threshold import ledger_examples
from test_stopping_close import assert_classifier, independent_targets, small_rows

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_threshold import threshold_splits
from wonyotti_fr.early_stopping_close import (
    EarlyStoppingCloseModel,
    fit_early_stopping_close,
    validate_early_stopping_training,
)
from wonyotti_fr.stopping_close import position_folds, stopping_targets


def early_setup():
    original = ledger_examples()
    frame = original.loc[original.index.repeat(5)].reset_index(drop=True)
    frame['decision_time'] += pd.to_timedelta(np.tile(np.arange(-4, 1), len(original)), unit='min')
    rows, assignment = threshold_splits(frame)
    training = rows['early_training']
    weights = training[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(training))
    indices = np.flatnonzero(assignment.split.eq('early_training'))
    return rows, weights, indices


def test_all_384_trees_excluded_positions_weights_original_rows_costs_and_targets_match_independent_fits(tmp_path):
    rows, weights, indices = early_setup()
    training, calibration = rows['early_training'], rows['calibration']
    model, support = fit_early_stopping_close(training, weights, calibration, indices, tmp_path)
    targets = pd.read_parquet(tmp_path/'early_stopping_targets.parquet')
    teachers = json.loads((tmp_path/'early_teacher_models.json').read_text())
    membership = pd.read_parquet(tmp_path/'early_teacher_training_membership.parquet')
    folds = position_folds(training)
    assert len(teachers) == 5 and len(membership) == 4*len(training)
    for fold in range(5):
        heldout = folds == fold
        selected = training.loc[~heldout].reset_index(drop=True)
        record = membership[membership.model_key.eq(f'fold-{fold:02}')].reset_index(drop=True)
        assert not set(record.position_entry_time) & set(training.loc[heldout, 'position_entry_time'])
        pd.testing.assert_frame_equal(record[['decision_time', 'position_entry_time']], selected[['decision_time', 'position_entry_time']], check_exact=True)
        np.testing.assert_array_equal(record.opportunity_index, indices[~heldout])
        counts = selected.position_entry_time.value_counts().to_dict()
        raw = np.array([1/counts[time] for time in selected.position_entry_time])
        np.testing.assert_array_equal(record.sample_weight, raw/raw.mean())
        np.testing.assert_array_equal(record.cost_weight, record.sample_weight*np.abs(selected.close_advantage_bps))
        prediction = assert_classifier(teachers[f'fold-{fold:02}'], selected[model.features].to_numpy(),
            selected.close_advantage_bps.to_numpy(), record.sample_weight.to_numpy(), training.loc[heldout, model.features].to_numpy())
        np.testing.assert_allclose(prediction, targets.loc[heldout, 'teacher_score'], rtol=0, atol=1e-12)
    expected_indices = np.full(len(training), -1, dtype=int)
    expected_targets = np.empty(len(training))
    for group in training.groupby('position_entry_time', sort=False).indices.values():
        subset = training.iloc[group].reset_index(drop=True)
        future, effects = independent_targets(subset, targets.teacher_score.iloc[group].to_numpy())
        expected_indices[group] = np.where(future >= 0, indices[group[np.maximum(future, 0)]], -1)
        expected_targets[group] = effects
    np.testing.assert_array_equal(targets.future_opportunity_index, expected_indices)
    np.testing.assert_array_equal(targets.stopping_advantage_bps, expected_targets)
    np.testing.assert_array_equal(targets.opportunity_index, indices)
    assert (targets.target_available_time < pd.Timestamp('2021-07-31', tz='UTC')).all()
    assert targets.stopping_advantage_bps.lt(0).any() and targets.stopping_advantage_bps.eq(0).any()
    pd.testing.assert_series_equal(targets.close_advantage_bps, training.close_advantage_bps, check_exact=True)
    costs = pd.read_parquet(tmp_path/'stopping_training_cost_ledger.parquet')
    np.testing.assert_array_equal(costs.original_weight, weights.sample_weight)
    np.testing.assert_array_equal(costs.cost_weight, weights.sample_weight*np.abs(expected_targets))
    assert costs.loc[expected_targets == 0, 'fit_weight'].eq(0).all()
    prediction = assert_classifier(model.to_dict(), training[model.features].to_numpy(), expected_targets,
        weights.sample_weight.to_numpy(), calibration[model.features].to_numpy())
    np.testing.assert_allclose(prediction, model.probabilities(calibration[model.features].to_numpy())[:, 0], rtol=0, atol=1e-12)
    assert support['diagnosis_used_for_export'] is False and model.format == EarlyStoppingCloseModel.format


@pytest.mark.parametrize('damage', ['cutoff', 'minute', 'features', 'weights', 'indices', 'short_support'])
def test_early_boundary_and_invalid_inputs_are_rejected_before_any_teacher_fit(tmp_path, monkeypatch, damage):
    rows, weights, indices = early_setup()
    training = rows['early_training'].copy()
    if damage == 'cutoff':
        training.loc[0, 'label_end'] = pd.Timestamp('2021-07-31', tz='UTC')
    elif damage == 'minute':
        training.loc[0, 'decision_time'] += pd.Timedelta(seconds=1)
    elif damage == 'features':
        training.loc[0, 'favorable_move'] = np.nan
    elif damage == 'weights':
        weights.loc[0, 'sample_weight'] *= 2
    elif damage == 'indices':
        indices[1] = indices[0]
    else:
        selected = training.position_entry_time.ge(pd.Timestamp('2021-02-01', tz='UTC'))
        training = training.loc[selected].reset_index(drop=True)
        indices = indices[selected]
        weights = training[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(training))
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError('잘못된 앞 학습 뒤 보조 적합 호출')

    monkeypatch.setattr('wonyotti_fr.early_stopping_close.MinuteCloseModel.fit', forbidden)
    with pytest.raises((ValueError, AssertionError)):
        fit_early_stopping_close(training, weights, rows['calibration'], indices, tmp_path)
    assert not calls and not (tmp_path/'model.json').exists()


def test_minute_future_search_excludes_self_exit_ties_other_positions_and_maximum_cash():
    frame = small_rows()
    frame['decision_time'] = frame.position_entry_time+pd.to_timedelta(np.tile(np.arange(1, 11), 2), unit='min')
    scores = np.r_[.9, .9, .9, .5, np.zeros(6), np.zeros(10)]
    indices = np.arange(len(frame))*4+10
    targets = stopping_targets(frame, scores, indices)
    future, values = independent_targets(frame, scores)
    np.testing.assert_array_equal(targets.future_opportunity_index, np.where(future >= 0, indices[np.maximum(future, 0)], -1))
    np.testing.assert_array_equal(targets.stopping_advantage_bps, values)
    assert targets.future_opportunity_index.iloc[0] == indices[1]
    assert targets.natural_close_fallback.iloc[1:].all()
    assert targets.stopping_advantage_bps.iloc[0] > 0 and targets.stopping_advantage_bps.iloc[1] < 0


def test_calibration_future_labels_and_input_units_do_not_change_teachers_targets_or_final_model(tmp_path):
    rows, weights, indices = early_setup()
    output = []
    for unit in ['us', 'ns']:
        train, calibration = rows['early_training'].copy(), rows['calibration'].copy()
        for frame in [train, calibration]:
            for name in frame.select_dtypes('datetimetz').columns:
                frame[name] = frame[name].astype(f'datetime64[{unit}, UTC]')
        if unit == 'ns':
            calibration['close_advantage_bps'] *= -100
        weight = train[['decision_time', 'position_entry_time']].assign(sample_weight=weights.sample_weight)
        destination = tmp_path/unit
        destination.mkdir()
        validate_early_stopping_training(train)
        model, _ = fit_early_stopping_close(train, weight, calibration, indices, destination)
        output.append((destination, model))
    assert output[0][1].to_dict() == output[1][1].to_dict()
    assert json.loads((output[0][0]/'early_teacher_models.json').read_text()) == json.loads((output[1][0]/'early_teacher_models.json').read_text())
    for name in ['early_stopping_targets', 'early_teacher_training_membership', 'stopping_training_cost_ledger']:
        values = [pd.read_parquet(folder/f'{name}.parquet') for folder, _ in output]
        for value in values:
            for column in value.select_dtypes('datetimetz').columns:
                value[column] = value[column].astype('datetime64[ns, UTC]')
        pd.testing.assert_frame_equal(*values, check_exact=True)
