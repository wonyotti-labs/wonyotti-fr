import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_expanded_first_linear import expanded_setup
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.expanded_first_linear import fit_expanded_first_linear
from wonyotti_fr.expanded_first_tree import fit_expanded_first_tree
from wonyotti_fr.first_opportunity_close import FirstOpportunityCloseModel, first_opportunity_rows
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS


def tree_source(path):
    path.mkdir()
    inputs = expanded_setup()
    training, weights, calibration, previous, added, positions = inputs
    fit_expanded_first_linear(*inputs, path)
    tables = {'early_training_used': training, 'early_training_weights': weights, 'calibration_used': calibration,
        'calibration_weights': calibration[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(calibration)),
        'first_training_ledger': previous, 'added_first_opportunity_ledger': added, 'added_all_positions': positions}
    assignment = pd.concat([frame[['decision_time', 'position_entry_time']].assign(split=name)
        for name, frame in [('early_training', training), ('calibration', calibration)]], ignore_index=True)
    tables['threshold_exclusion_ledger'] = assignment
    for name, frame in tables.items():
        frame.to_parquet(path/(name+'.parquet'), index=False)
    return path


def test_all64_trees_equal_costs_first_rows_and_numeric_scores(tmp_path):
    source = tree_source(tmp_path/'source')
    output = tmp_path/'tree'
    output.mkdir()
    model, support = fit_expanded_first_tree(source, output)
    frame = pd.read_parquet(source/'combined_first_training.parquet')
    x, y = frame[model.features].to_numpy(), frame.first_target_common_bps.to_numpy()
    fit = y != 0
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x[fit], y[fit] > 0, sample_weight=abs(y[fit])/abs(y[fit]).mean())
        saved = model.to_dict()['models'][0]
        assert len(saved['trees']) == learner.n_iter_ == 64 and saved['baseline'] == learner._baseline_prediction[0, 0]
        for tree, part in zip(saved['trees'], learner._predictors, strict=True):
            nodes = part[0].nodes
            leaf = nodes['is_leaf'].astype(bool)
            for name, values in {'left': np.where(leaf, -1, nodes['left'].astype(int)), 'right': np.where(leaf, -1, nodes['right'].astype(int)),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)), 'threshold': np.where(leaf, 0., nodes['num_threshold']),
                'value': np.where(leaf, nodes['value'], 0.)}.items():
                np.testing.assert_array_equal(tree[name], values)
        validation, _ = first_opportunity_rows(pd.read_parquet(source/'calibration_used.parquet'))
        for values in [x, validation[model.features].to_numpy()]:
            np.testing.assert_allclose(model.probabilities(values)[:, 0], learner.predict_proba(values)[:, 1], rtol=0, atol=1e-12)
    prior = json.loads((source/'training_support.json').read_text())
    for name in ['eligible_positions', 'fit_positions', 'zero_effect_positions', 'normalizer', 'training_constant_score']:
        assert support[name] == prior[name]
    assert support['old_combined_rows_and_costs_exact'] and not support['diagnosis_used_for_export']


def test_future_calibration_changes_do_not_change_tree(tmp_path):
    source = tree_source(tmp_path/'source')
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir()
    b.mkdir()
    model, _ = fit_expanded_first_tree(source, a)
    path = source/'calibration_used.parquet'
    frame = pd.read_parquet(path)
    frame['favorable_move'] += .3
    frame['close_advantage_pnl'] *= -1
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    frame.to_parquet(path, index=False)
    other, _ = fit_expanded_first_tree(source, b)
    assert other.to_dict() == model.to_dict()


@pytest.mark.parametrize('damage', ['feature', 'source', 'cost', 'weight', 'target', 'population', 'constant'])
def test_changed_combined_or_original_records_rejected_before_tree_fit(tmp_path, monkeypatch, damage):
    source = tree_source(tmp_path/'source')
    if damage == 'constant':
        path = source/'training_support.json'
        values = json.loads(path.read_text())
        values['training_constant_score'] += .01
        path.write_text(json.dumps(values))
    else:
        file, key = {'feature': ('combined_first_training', 'favorable_move'), 'source': ('combined_first_training', 'source_opportunity_index'),
            'cost': ('combined_training_costs', 'fit_weight'), 'weight': ('early_training_weights', 'sample_weight'),
            'target': ('added_first_opportunity_ledger', 'first_target_common_bps'), 'population': ('original_training_positions', 'reference_equity')}[damage]
        path = source/(file+'.parquet')
        values = pd.read_parquet(path)
        values.loc[0, key] += 1
        values.to_parquet(path, index=False)
    calls = []
    monkeypatch.setattr(FirstOpportunityCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        fit_expanded_first_tree(source, tmp_path)
    assert not calls
