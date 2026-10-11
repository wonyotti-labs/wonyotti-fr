from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_causal_management import prepared as prepared

from wonyotti_fr.management_score_fusion import (
    SETTINGS,
    fit_score_fusion,
    fusion_admission,
    fusion_matrix,
    fusion_predictions,
    fusion_probabilities,
)
from wonyotti_fr.quiet_management import evaluate_quiet
from wonyotti_fr.quiet_management_reference import verify_quiet_evaluation
from wonyotti_fr.score_fusion_reference import (
    verify_fusion_admission,
    verify_fusion_predictions,
    verify_fusion_refits,
)
from wonyotti_fr.score_fusion_research import checked_fusion_reference


@pytest.fixture(scope='module')
def fused(prepared):
    original, _, ledgers, _ = prepared
    return original, ledgers, fit_score_fusion(original['calibration'], ledgers['calibration'])


def test_sixteen_refits_and_all_predictions(fused):
    original, ledgers, fitted = fused
    assert verify_fusion_refits(original['calibration'], ledgers['calibration'], fitted)['independent_calibrator_refits'] == 16
    for phase, frame in original.items():
        metrics = {}
        for kind in ['absolute', 'joint']:
            saved = fusion_predictions(frame, ledgers[phase], fitted, kind)
            unchanged = verify_fusion_predictions(frame, ledgers[phase], saved, fitted, kind, calibration=phase == 'calibration')
            metrics[kind] = evaluate_quiet(saved)
            verify_quiet_evaluation(saved, metrics[kind])
        if phase == 'diagnosis':
            old = evaluate_quiet(frame)
            decision = fusion_admission(metrics['joint'], metrics['absolute'], old, unchanged)
            verify_fusion_admission(metrics['joint'], metrics['absolute'], old, unchanged, decision)
            assert len(decision['checks']) == 39
            broken = deepcopy(decision)
            key = 'increase_absolute_ap_preserved'
            broken['checks'][key] = not broken['checks'][key]
            with pytest.raises(AssertionError):
                verify_fusion_admission(metrics['joint'], metrics['absolute'], old, unchanged, broken)


def test_future_labels_leave_fits_unchanged(fused):
    original, ledgers, fitted = fused
    changed = {p: f.copy() for p, f in original.items()}
    for action in ['reduce', 'increase']:
        changed['diagnosis']['y_' + action] = 1 - changed['diagnosis']['y_' + action]
    assert fit_score_fusion(changed['calibration'], ledgers['calibration']) == fitted


@pytest.mark.parametrize('kind', ['action', 'state', 'source', 'kind', 'features', 'settings', 'scale', 'coef', 'intercept', 'extra'])
def test_corrupt_model_binding_or_numbers_rejected(fused, kind):
    raw = deepcopy(fused[2]['joint']['models']['reduce']['first']['tree'])
    if kind in ['action', 'state', 'source', 'kind']:
        raw[kind] = 'wrong'
    elif kind == 'features':
        raw['features'] = raw['features'][::-1]
    elif kind == 'settings':
        raw['settings']['C'] = 1.
    elif kind == 'scale':
        raw['scale'][0] = 0.
    elif kind == 'coef':
        raw['coef'] = [0.]
    elif kind == 'intercept':
        raw['intercept'] = np.nan
    else:
        raw['extra'] = True
    with pytest.raises(ValueError):
        fusion_probabilities(raw, np.array([.1]), np.array([.2]), 'reduce', 'first', 'tree', 'joint')
    assert SETTINGS['C'] == .1


def test_absolute_control_does_not_use_relative_value(fused):
    fitted = fused[2]
    raw = fitted['absolute']['models']['increase']['restart']['tree']
    a = np.array([0., .05, .5, 1.])
    first = fusion_probabilities(raw, a, a, 'increase', 'restart', 'tree', 'absolute')
    second = fusion_probabilities(raw, a, 1 - a, 'increase', 'restart', 'tree', 'absolute')
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(fusion_matrix(a, a, 'joint')).all()


@pytest.mark.parametrize('kind', ['nan', 'length', 'range', 'unsupported'])
def test_invalid_inputs_rejected(kind):
    a, b, name = np.array([.1, .2]), np.array([.2, .3]), 'joint'
    if kind == 'nan':
        a[0] = np.nan
    elif kind == 'length':
        b = b[:1]
    elif kind == 'range':
        b[0] = -.1
    else:
        name = 'other'
    with pytest.raises(ValueError):
        fusion_matrix(a, b, name)


@pytest.mark.parametrize('column,index', [('reduce_tree_raw_score', 5001), ('reduce_tree_score', 0),
    ('reduce_tree_request', 5001), ('increase_linear_score', 5002), ('increase_baseline_score', 7000), ('exit_score', 6000)])
def test_predictions_fallback_and_recent_damage_rejected(fused, column, index):
    original, ledgers, fitted = fused
    frame, ledger = original['diagnosis'], ledgers['diagnosis']
    saved = fusion_predictions(frame, ledger, fitted, 'joint')
    saved.loc[index, column] = not saved.loc[index, column] if column.endswith('request') else saved.loc[index, column] + .01
    with pytest.raises(AssertionError):
        verify_fusion_predictions(frame, ledger, saved, fitted, 'joint')


def test_empty_support_keeps_original(fused):
    original, ledgers, fitted = fused
    frame, ledger = original['diagnosis'].iloc[:60], ledgers['diagnosis'].iloc[:60]
    saved = fusion_predictions(frame, ledger, fitted, 'joint')
    pd.testing.assert_frame_equal(saved, frame, check_exact=True)
    verify_fusion_predictions(frame, ledger, saved, fitted, 'joint')


def test_insufficient_support_and_invalid_source(fused, tmp_path):
    original, ledgers, _ = fused
    changed = original['calibration'].copy()
    changed['y_increase'] = 0
    with pytest.raises(ValueError, match='지원'):
        fit_score_fusion(changed, ledgers['calibration'])
    path = tmp_path / 'proof.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        checked_fusion_reference(path, '0' * 64)
