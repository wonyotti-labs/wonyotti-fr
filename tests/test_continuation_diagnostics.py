import copy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_close_calibration import source_fixture
from test_close_learning import real_source, reseal
from test_first_close_diagnostics import cases, financial_examples
from test_first_state import fixed_budget  # noqa: F401
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_economics import (
    ECONOMIC_FEATURES,
    load_economic_inputs,
    run_close_economic_diagnosis,
)
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import (
    continuation_admission,
    reproduce_first_close,
    run_continuation_diagnosis,
)
from wonyotti_fr.continuation_inputs import (
    PENDING_ACTIONS,
    PENDING_FEATURES,
    ContinuationCloseModel,
    load_pending_inputs,
    pending_values,
    validate_pending_matrix,
)
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.first_close_diagnostics import (
    first_close_positions,
    paired_week_blocks,
    run_first_close_diagnosis,
)
from wonyotti_fr.journal import canonical, digest


def test_current_intent_encoding_is_exact_and_future_execution_or_targets_do_not_change_it():
    frame = pd.DataFrame({'original_intent': PENDING_ACTIONS, 'future_execution': ['exit']*4, 'future_pnl': [1.]*4})
    expected = pending_values(frame.original_intent)
    np.testing.assert_array_equal(expected, np.eye(4))
    frame['future_execution'], frame['future_pnl'] = 'hold', -1e8
    np.testing.assert_array_equal(pending_values(frame.original_intent), expected)
    for bad in [['enter_long'], [None], [np.nan], [pd.NA], [["hold"]]]:
        with pytest.raises(ValueError):
            pending_values(bad)
    for bad in [np.zeros((2, 4)), np.ones((2, 4)), np.eye(3), np.array([[.5, .5, 0., 0.]])]:
        with pytest.raises(ValueError):
            validate_pending_matrix(bad)


def test_pending_inputs_match_every_actual_next_request_and_reject_resigned_state_mismatch(tmp_path, monkeypatch):
    root = real_source(tmp_path, monkeypatch)
    original = pd.read_parquet(root/'opportunity_ledger.parquet')
    economic, _ = load_economic_inputs(root, original)
    before = sha256(root/'outcomes.sqlite')
    augmented, evidence = load_pending_inputs(root, economic)
    pd.testing.assert_frame_equal(augmented.drop(columns=PENDING_FEATURES), economic, check_exact=True)
    np.testing.assert_array_equal(augmented[PENDING_FEATURES].to_numpy(), pending_values(original.original_intent))
    assert before == sha256(root/'outcomes.sqlite') and evidence['all_current_pending_and_original_requests_exact']
    connection = sqlite3.connect(root/'outcomes.sqlite')
    number, _source, payload, prior, _chain = connection.execute('SELECT * FROM outcomes ORDER BY sequence LIMIT 1').fetchone()
    record = json.loads(payload)
    state = record['opportunity']['state']
    state['pending'] = 'hold' if state['pending'] == 'exit' else 'exit'
    source, payload = digest(canonical(record['opportunity'])), canonical(record)
    connection.execute('UPDATE outcomes SET input_hash=?,payload=?,chain_hash=? WHERE sequence=?',
        (source, payload, digest(canonical([number, source, payload, prior])), number))
    connection.commit()
    connection.close()
    reseal(root)
    with pytest.raises(ValueError, match='실제 요청'):
        load_pending_inputs(root, economic)


def test_numeric_model_matches_independent_refit_and_rejects_non_one_hot_predictors():
    rng = np.random.default_rng(64)
    x = np.c_[rng.normal(size=(500, 52)), np.tile(np.eye(4), (125, 1))]
    y, w = x[:, 0]*10+x[:, -1]*5, np.linspace(.5, 1.5, 500)
    model, _ = ContinuationCloseModel.fit(x, y, w, x[:100])
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(x, y, sample_weight=w)
        np.testing.assert_allclose(model.predict(x), learner.predict(x), rtol=0, atol=1e-10)
    future = x[:100].copy()
    future[:, :52] *= -100
    future[:, -4:] = np.roll(future[:, -4:], 1, axis=1)
    changed, _ = ContinuationCloseModel.fit(x, y, w, future)
    assert model.to_dict() == changed.to_dict()
    bad = x.copy()
    bad[0, -4:] = [1., 1., 0., 0.]
    with pytest.raises(ValueError):
        ContinuationCloseModel.fit(bad, y, w, x[:100])
    with pytest.raises(ValueError):
        model.predict(bad)
    np.testing.assert_array_equal(ContinuationCloseModel.from_dict(model.to_dict()).predict(x), model.predict(x))


def test_named_paired_blocks_preserve_original_draws_and_results():
    f = cases()
    old = {name: first_close_positions(f, 'predicted_'+name) for name in ['economic', 'boosted']}
    original = paired_week_blocks(old)
    renamed = paired_week_blocks({'continuation': old['economic'], 'economic': old['boosted']}, model_names=['continuation', 'economic'])
    mapping = {'effect_sum_continuation': 'effect_sum_economic', 'effect_sum_economic': 'effect_sum_boosted',
        'continuation': 'economic', 'economic': 'boosted'}
    for old_frame, new_frame in zip(original[:3], renamed[:3], strict=True):
        pd.testing.assert_frame_equal(old_frame, new_frame.rename(columns=mapping), check_exact=True)
    assert original[3]['intervals']['economic'] == renamed[3]['intervals']['continuation']
    assert original[3]['intervals']['paired_difference'] == renamed[3]['intervals']['paired_difference']
    for names in [['economic'], ['economic', 'economic'], ['', 'boosted'], ['economic', None]]:
        with pytest.raises(ValueError):
            paired_week_blocks(old, model_names=names)


def test_all_twelve_checks_include_first_choice_effect_without_discarding_old_failures():
    candidate = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    refs = ['economic', 'boosted', 'ridge', 'constant']
    metrics = {k: {**candidate, 'weighted_mse': 100., 'mse': 100.} for k in refs}
    metrics['continuation'] = candidate
    first = {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': 1.}
    result = continuation_admission(metrics, first)
    assert result['continuation_inputs_admitted'] and len(result['checks']) == 12
    cases = [(k, 'weighted_mse', 50.) for k in refs]+[(k, 'mse', 49.) for k in refs[:-1]]
    cases += [('continuation', 'selected', 99), ('continuation', 'selected_positions', 29),
        ('continuation', 'selected_weighted_mean_bps', 0.), ('continuation', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        altered, first_altered = copy.deepcopy(metrics), dict(first)
        altered[name][field] = value
        if field == 'selected_positions':
            first_altered[field] = value
        assert not continuation_admission(altered, first_altered)['continuation_inputs_admitted']
    assert not continuation_admission(metrics, {**first, 'all_position_mean_common_bps': 0.})['continuation_inputs_admitted']
    with pytest.raises(ValueError):
        continuation_admission(metrics, {**first, 'selected_positions': 29})


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr('test_close_calibration.calibration_examples', financial_examples)
    previous = source_fixture(tmp_path, monkeypatch)
    def economic(_labels, ledger):
        augmented = ledger.copy()
        augmented[ECONOMIC_FEATURES[0]] = .25
        augmented[ECONOMIC_FEATURES[1]] = augmented.favorable_move*10000
        return augmented, {'rows': len(ledger), 'future_fill_or_outcome_used': False}
    def pending(_labels, ledger):
        augmented = ledger.copy()
        augmented[PENDING_FEATURES] = pending_values(ledger.original_intent)
        return augmented, {'rows': len(ledger), 'future_execution_or_target_used': False}
    monkeypatch.setattr('wonyotti_fr.close_economics.load_economic_inputs', economic)
    monkeypatch.setattr('wonyotti_fr.continuation_diagnostics.load_pending_inputs', pending)
    economic = run_close_economic_diagnosis(previous, tmp_path/'economic')
    return run_first_close_diagnosis(economic, tmp_path/'first'), economic


def test_full_pipeline_reproduces_first_choices_blocks_and_all_old_models_rows_weights(tmp_path, monkeypatch):
    reference, economic = fixture(tmp_path, monkeypatch)
    out = run_continuation_diagnosis(reference, tmp_path/'new')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and not summary['profitability_accepted']
    assert summary['all_previous_outputs_reproduced'] and summary['all_original_rows_and_weights_preserved']
    assert len(summary['checks']) == 12
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='predicted_continuation'), pd.read_parquet(economic/'predictions.parquet'), check_exact=True)
    for name in ['training', 'diagnosis']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_used.parquet').drop(columns=PENDING_FEATURES), pd.read_parquet(economic/f'{name}_used.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_weights.parquet'), pd.read_parquet(economic/f'{name}_weights.parquet'), check_exact=True)
    positions = pd.read_parquet(reference/'positions_economic.parquet')
    positions.loc[0, 'first_effect_common_bps'] += 1
    positions.to_parquet(reference/'positions_economic.parquet', index=False)
    hashes = json.loads((reference/'files.json').read_text())
    hashes['positions_economic.parquet'] = sha256(reference/'positions_economic.parquet')
    save_json(reference/'files.json', hashes)
    with pytest.raises(AssertionError):
        reproduce_first_close(reference, tmp_path/'rejected')
