import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_expanded_first_tree import tree_source
from test_managed_first_model import fixed_manager
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_effect import CLOSE_FEATURES
from wonyotti_fr.first_linear_close import FIRST_LINEAR_SETTINGS
from wonyotti_fr.first_opportunity_close import first_opportunity_rows
from wonyotti_fr.managed_first_fit import fit_managed_first
from wonyotti_fr.managed_first_model import (
    MANAGER_FEATURES,
    augment_manager_inputs,
)
from wonyotti_fr.probability_first_fit import fit_probability_first
from wonyotti_fr.probability_first_model import ProbabilityFirstLinearModel


def probability_source(path):
    source = tree_source(path)
    fit_managed_first(source, fixed_manager(), source)
    return source


def test_three_coefficients_preserve77_inputs_and_numeric_scores(tmp_path):
    source = probability_source(tmp_path/'source')
    output = tmp_path/'fit'
    output.mkdir()
    manager = fixed_manager()
    model, support = fit_probability_first(source, manager, output)
    base = pd.read_parquet(source/'combined_first_training.parquet')
    frame = pd.read_parquet(source/'managed_first_training.parquet')
    pd.testing.assert_frame_equal(frame[base.columns], base, check_exact=True)
    values = base[CLOSE_FEATURES].to_numpy()
    for i, key in enumerate(MANAGER_FEATURES):
        logits = .2*i+64*np.where(values[:, i] <= 0, -.02, .03)+manager.offset.offsets[i]
        np.testing.assert_allclose(frame[key], 1/(1+np.exp(-logits)), rtol=0, atol=1e-14)
    assert (frame[MANAGER_FEATURES].sum(axis=1) > 1).any()
    x, y = frame[model.features].to_numpy(), frame.first_target_common_bps.to_numpy()
    fit = y != 0
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(scaler.transform(x)[fit], y[fit] > 0,
            sample_weight=abs(y[fit])/abs(y[fit]).mean())
        data = model.to_dict()
        assert len(data['features']) == 3
        for key, expected in [('mean', scaler.mean_), ('scale', scaler.scale_), ('coefficient', learner.coef_[0])]:
            np.testing.assert_array_equal(data[key], expected)
        assert data['intercept'] == learner.intercept_[0]
        for item in [frame, pd.read_parquet(source/'managed_first_calibration.parquet')]:
            xx = item[model.features].to_numpy()
            np.testing.assert_allclose(model.probabilities(xx)[:, 0], learner.predict_proba(scaler.transform(xx))[:, 1], rtol=0, atol=1e-12)
    assert support['training_constant_score'] == json.loads((source/'training_support.json').read_text())['training_constant_score']


def test_future_cash_and_calibration_inputs_do_not_change_earlier_model_or_manager(tmp_path):
    source = probability_source(tmp_path/'source')
    manager = fixed_manager()
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    model, _ = fit_probability_first(source, manager, a)
    path = source/'calibration_used.parquet'
    frame = pd.read_parquet(path)
    frame['favorable_move'] += .3
    frame['close_advantage_pnl'] *= -1
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    frame.to_parquet(path, index=False)
    first, _ = first_opportunity_rows(frame)
    augment_manager_inputs(first, manager).to_parquet(source/'managed_first_calibration.parquet', index=False)
    other, _ = fit_probability_first(source, manager, b)
    assert model.to_dict() == other.to_dict()
    original = pd.read_parquet(source/'combined_first_training.parquet')
    changed = original.copy()
    for name in ['first_target_common_bps', 'close_cash', 'continue_cash', 'close_advantage_pnl']:
        changed[name] *= -17
    pd.testing.assert_frame_equal(augment_manager_inputs(original, manager)[MANAGER_FEATURES],
        augment_manager_inputs(changed, manager)[MANAGER_FEATURES], check_exact=True)


@pytest.mark.parametrize('value', [-.1, 1.1, np.nan, np.inf])
def test_invalid_management_probabilities_rejected(tmp_path, value):
    source = probability_source(tmp_path/'source')
    frame = pd.read_parquet(source/'combined_first_training.parquet')
    manager = fixed_manager()
    manager.probabilities = lambda values: np.full((len(values), 3), value)
    with pytest.raises(ValueError):
        augment_manager_inputs(frame, manager)
    x = np.full((len(frame), 3), value)
    with pytest.raises(ValueError):
        ProbabilityFirstLinearModel.matrix(x)
