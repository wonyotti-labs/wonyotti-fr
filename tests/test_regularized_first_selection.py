import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_first_opportunity_close import first_examples
from test_managed_first_model import fixed_manager
from test_minute_first_inputs import first_setup
from threadpoolctl import threadpool_limits

from wonyotti_fr.first_linear_close import FIRST_LINEAR_SETTINGS
from wonyotti_fr.minute_first_fit import fit_minute_first
from wonyotti_fr.regularized_first_model import (
    REGULARIZATION_STRENGTHS,
    RegularizedFirstModel,
    regularized_settings,
)
from wonyotti_fr.regularized_first_selection import (
    REGULARIZATION_WINDOWS,
    choose_regularization,
    fit_regularization_selection,
    fit_regularized_first,
    regularization_membership,
    score_column,
)


def regularized_source(path):
    with patch('test_expanded_first_linear.first_examples', lambda *_a, **_kw: first_examples('2020-04-01', 600, 8)):
        source, _, _, bars = first_setup(path)
    fit_minute_first(source, fixed_manager(), bars, source)
    return source


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    return regularized_source(tmp_path_factory.mktemp('regularized')/'source')


def independent_fit(frame, strength):
    x, y = frame[RegularizedFirstModel.features].to_numpy(), frame.first_target_common_bps.to_numpy()
    active = y != 0
    scaler = StandardScaler().fit(x)
    learner = LogisticRegression(**{**FIRST_LINEAR_SETTINGS, 'C': strength}).fit(scaler.transform(x)[active], y[active] > 0,
        sample_weight=abs(y[active])/abs(y[active]).mean())
    return scaler, learner


def verify_model(data, scaler, learner):
    for name, array in [('mean', scaler.mean_), ('scale', scaler.scale_), ('coefficient', learner.coef_[0])]:
        np.testing.assert_array_equal(data[name], array)
    assert data['intercept'] == learner.intercept_[0]


def test_all13_models_time_membership_pooling_and_final_scores(reference, tmp_path, monkeypatch):
    original_read = pd.read_parquet

    def after_choice(path, *args, **kwargs):
        if str(path).endswith('/minute_first_calibration.parquet'):
            assert (tmp_path/'selection/selection.json').exists()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', after_choice)
    model, support = fit_regularized_first(reference, tmp_path)
    training = original_read(reference/'minute_first_training.parquet')
    calibration = original_read(reference/'minute_first_calibration.parquet')
    assert len(training) == 1079 and support['new_models_fitted'] == 13
    members = original_read(tmp_path/'selection/selection_membership.parquet')
    pooled = original_read(tmp_path/'selection/selection_predictions.parquet')
    assert len(members) == 3*len(training) and not pooled.input_row.duplicated().any()
    with threadpool_limits(limits=1):
        for number, (left, right) in enumerate(REGULARIZATION_WINDOWS):
            start, end = pd.Timestamp(left, tz='UTC'), pd.Timestamp(right, tz='UTC')
            cutoff = start-pd.Timedelta(days=2)
            train_mask = training.position_entry_time.lt(cutoff)&training.decision_time.lt(cutoff)&training.label_end.lt(cutoff)
            valid_mask = training.position_entry_time.ge(start)&training.decision_time.ge(start)&training.decision_time.lt(end)&training.label_end.lt(end)
            train, valid = training[train_mask], training[valid_mask]
            member = members[members.fold.eq(f'fold-{number:02}')]
            np.testing.assert_array_equal(member.role.eq('training'), train_mask)
            np.testing.assert_array_equal(member.role.eq('validation'), valid_mask)
            target = train.first_target_common_bps.to_numpy()
            np.testing.assert_array_equal(member.loc[member.role.eq('training'), 'fit_weight'], abs(target)/abs(target[target != 0]).mean())
            saved = json.loads((tmp_path/f'selection/fold-{number:02}/models.json').read_text())
            pred = pooled[pooled.fold.eq(f'fold-{number:02}')]
            np.testing.assert_array_equal(pred.input_row, np.flatnonzero(valid_mask))
            for strength in REGULARIZATION_STRENGTHS:
                scaler, learner = independent_fit(train, strength)
                verify_model(saved[str(strength)], scaler, learner)
                np.testing.assert_allclose(pred[score_column(strength)], learner.predict_proba(scaler.transform(valid[model.features].to_numpy()))[:, 1], rtol=0, atol=1e-12)
        target, costs = pooled.first_target_common_bps.to_numpy(), abs(pooled.first_target_common_bps.to_numpy())
        losses = {}
        for strength in REGULARIZATION_STRENGTHS:
            safe = np.clip(pooled[score_column(strength)].to_numpy(), 1e-15, 1-1e-15)
            losses[strength] = float(np.dot(costs/costs.sum(), -((target > 0)*np.log(safe)+(target <= 0)*np.log1p(-safe))))
        selected = min(losses, key=lambda value: (losses[value], value))
        decision = json.loads((tmp_path/'selection/selection.json').read_text())
        assert decision['selected_strength'] == support['selected_strength'] == selected
        assert not decision['external_calibration_used_for_selection']
        for strength, loss in losses.items():
            assert decision['candidates'][str(strength)]['cost_log_loss'] == loss
        scaler, learner = independent_fit(training, selected)
        verify_model(model.to_dict(), scaler, learner)
        np.testing.assert_allclose(model.probabilities(calibration[model.features].to_numpy())[:, 0],
            learner.predict_proba(scaler.transform(calibration[model.features].to_numpy()))[:, 1], rtol=0, atol=1e-12)
    assert FIRST_LINEAR_SETTINGS['C'] == .1


def test_strength_point_one_reproduces_prior79_model(reference):
    training, calibration = [pd.read_parquet(reference/('minute_first_'+phase+'.parquet')) for phase in ['training', 'calibration']]
    model, _, _ = RegularizedFirstModel.fit(training[RegularizedFirstModel.features].to_numpy(), training.first_target_common_bps,
        calibration[RegularizedFirstModel.features].to_numpy(), strength=.1)
    prior = json.loads((reference/'model.json').read_text())
    assert all(prior[name] == value for name, value in model.to_dict().items() if name != 'format')


def test_future_training_rows_do_not_change_earlier_fold_models(reference, tmp_path):
    training = pd.read_parquet(reference/'minute_first_training.parquet')
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    fit_regularization_selection(training, a)
    future = training.copy()
    mask = future.decision_time.ge(pd.Timestamp('2021-03-01', tz='UTC'))
    future.loc[mask, 'favorable_move'] += .5
    future.loc[mask, 'close_advantage_pnl'] *= -1
    future['close_cash'] = future.continue_cash+future.close_advantage_pnl
    future['first_target_common_bps'] = future.close_advantage_pnl/future.reference_equity*10000
    fit_regularization_selection(future, b)
    assert json.loads((a/'fold-00/models.json').read_text()) == json.loads((b/'fold-00/models.json').read_text())


def test_exact_tie_uses_smallest_strength_and_keeps_zero_cost_rows():
    rows = 90
    predictions = pd.DataFrame({'input_row': np.arange(rows), 'position_entry_time': pd.date_range('2021-03-01', periods=rows, freq='D', tz='UTC'),
        'fold': np.repeat(['fold-00', 'fold-01', 'fold-02'], 30), 'first_target_common_bps': np.where(np.arange(rows) % 3, -2., 0.),
        'training_constant_score': .55, **{score_column(value): .55 for value in REGULARIZATION_STRENGTHS}})
    decision = choose_regularization(predictions)
    assert decision['selected_strength'] == .001 and decision['validation_rows'] == 90
    assert all(item['nonzero_cost_rows'] == 60 for item in decision['candidates'].values())


def test_cutoff_labels_purge_gap_and_validation_crossing_are_preserved(reference):
    frame = pd.read_parquet(reference/'minute_first_training.parquet').copy()
    train_index = frame.index[frame.position_entry_time.ge(pd.Timestamp('2021-02-26', tz='UTC'))][0]
    frame.loc[train_index, 'label_end'] = pd.Timestamp('2021-02-27', tz='UTC')
    validation_index = frame.index[frame.position_entry_time.ge(pd.Timestamp('2021-03-01', tz='UTC'))][0]
    frame.loc[validation_index, 'label_end'] = pd.Timestamp('2021-05-01', tz='UTC')
    members, _ = regularization_membership(frame)
    first = members[members.fold.eq('fold-00')].set_index('input_row')
    assert first.loc[train_index, 'reason'] == 'training_label_not_available'
    assert first.loc[validation_index, 'reason'] == 'validation_label_crosses_end'
    assert first.reason.eq('purge_gap').any() and len(first) == len(frame)
    assert first.loc[first.role.ne('training'), 'fit_weight'].eq(0).all()


@pytest.mark.parametrize('strength', [None, True, 0., -.1, .0001, 2., np.nan])
def test_unplanned_regularization_strength_rejected(strength):
    with pytest.raises(ValueError):
        regularized_settings(strength)


def test_external_calibration_changes_leave_all_choices_and_final_model_unchanged(reference, tmp_path):
    from wonyotti_fr.source_snapshot import copy_snapshot

    changed = tmp_path/'changed'
    changed.mkdir()
    for name in ['minute_first_training.parquet', 'minute_first_calibration.parquet', 'combined_training_costs.parquet', 'training_support.json', 'model.json']:
        copy_snapshot(reference/name, changed/name)
    calibration = pd.read_parquet(changed/'minute_first_calibration.parquet')
    calibration['favorable_move'] += .5
    calibration['close_advantage_pnl'] *= -1
    calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
    calibration['first_target_common_bps'] = calibration.close_advantage_pnl/calibration.reference_equity*10000
    calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    calibration.to_parquet(changed/'minute_first_calibration.parquet', index=False)
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    first, _ = fit_regularized_first(reference, a)
    second, _ = fit_regularized_first(changed, b)
    assert first.to_dict() == second.to_dict()
    assert json.loads((a/'selection/selection.json').read_text()) == json.loads((b/'selection/selection.json').read_text())
    for file in (a/'selection').rglob('*'):
        if file.is_file():
            assert file.read_bytes() == (b/'selection'/file.relative_to(a/'selection')).read_bytes()


def test_insufficient_early_fold_support_blocks_all_candidate_fits(reference, tmp_path, monkeypatch):
    training = pd.read_parquet(reference/'minute_first_training.parquet')
    training = training[training.source_phase.eq('original_2021')].reset_index(drop=True)
    calls = []
    monkeypatch.setattr(RegularizedFirstModel, 'fit', lambda *_a, **_kw: calls.append(True))
    with pytest.raises(ValueError):
        fit_regularization_selection(training, tmp_path)
    assert not calls
