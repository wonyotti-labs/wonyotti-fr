import json

import numpy as np
import pandas as pd
import pytest
from test_continuation_diagnostics import fixture
from test_first_close_margin import examples
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_capacity import capacity_class
from wonyotti_fr.close_capacity_diagnostics import run_close_capacity_diagnosis
from wonyotti_fr.close_learning import close_metrics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_diagnostics import run_continuation_diagnosis
from wonyotti_fr.continuation_inputs import ContinuationCloseModel
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.exposure_close_diagnostics import run_exposure_close_diagnosis
from wonyotti_fr.first_close_diagnostics import first_close_positions
from wonyotti_fr.first_close_margin_diagnostics import (
    calibrate_first_margin,
    evaluate_first_margin,
    reproduce_weekly,
    run_first_close_margin_diagnosis,
)
from wonyotti_fr.weekly_close_diagnostics import run_weekly_close_diagnosis


def margin_fixture(root):
    root.mkdir()
    cal = examples()
    for name in ContinuationCloseModel.features:
        if name not in cal:
            cal[name] = 0.
    cal['ret_5m'], cal['pending_hold'] = cal.predicted_half, 1.
    cal = cal.drop(columns=['predicted_half', 'sample_weight'])
    train = cal.copy()
    for name in ['position_entry_time', 'decision_time', 'label_end', 'continue_end']:
        train[name] -= pd.Timedelta(days=180)
    cls = capacity_class('depth2_iter64')
    tree = {'left': [1, -1, 3, -1, -1], 'right': [2, -1, 4, -1, -1],
        'feature': [0, -2, 0, -2, -2], 'threshold': [1., 0., 3., 0., 0.], 'value': [0., .5, 0., 2., 5.]}
    leaf = {'left': [-1], 'right': [-1], 'feature': [-2], 'threshold': [0.], 'value': [0.]}
    model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': REGRESSION_SETTINGS,
        'baseline': 0., 'trees': [tree]+[leaf]*63})
    save_json(root/'selection_models.json', {'depth2_iter64': model.to_dict()})
    for name, frame in [('training', train), ('selection', cal)]:
        frame.to_parquet(root/f'selection_{name}_used.parquet', index=False)
        frame[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(frame)).to_parquet(root/f'selection_{name}_weights.parquet', index=False)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']
    cal[keys].assign(sample_weight=position_weights(cal), predicted_depth2_iter64=model.predict(cal[model.features])).to_parquet(root/'selection_selection_predictions.parquet', index=False)
    return cal, model


def test_calibration_preserves_fixed_model_and_is_independent_of_future_diagnosis(tmp_path):
    capacity = tmp_path/'capacity'
    frame, original = margin_fixture(capacity)
    first = tmp_path/'first'
    first.mkdir()
    model, selected = calibrate_first_margin(capacity, first)
    assert selected['selection_passed'] and selected['chosen_margin_bps'] == 1.
    assert model.to_dict() == original.to_dict()
    pd.testing.assert_frame_equal(pd.read_parquet(first/'calibration_used.parquet'), frame, check_exact=True)
    changed = frame.copy()
    changed['close_advantage_bps'] *= -1e8
    changed['ret_5m'] *= -100
    changed.to_parquet(capacity/'diagnosis_used.parquet', index=False)
    second = tmp_path/'second'
    second.mkdir()
    model2, selected2 = calibrate_first_margin(capacity, second)
    assert model2.to_dict() == model.to_dict() and selected2 == selected
    score = pd.read_parquet(capacity/'selection_selection_predictions.parquet')
    score.loc[0, 'predicted_depth2_iter64'] += 1
    score.to_parquet(capacity/'selection_selection_predictions.parquet', index=False)
    rejected = tmp_path/'rejected'
    rejected.mkdir()
    with pytest.raises(AssertionError):
        calibrate_first_margin(capacity, rejected)


def test_successful_calibration_evaluates_only_selected_margin_with_unchanged_forecast_errors(tmp_path):
    capacity = tmp_path/'capacity'
    cal, _ = margin_fixture(capacity)
    out = tmp_path/'out'
    out.mkdir()
    model, selected = calibrate_first_margin(capacity, out)
    source = tmp_path/'weekly'
    source.mkdir()
    diagnosis = cal.copy()
    for name in ['position_entry_time', 'decision_time', 'label_end', 'continue_end']:
        diagnosis[name] += pd.Timedelta(days=92)
    diagnosis.to_parquet(source/'diagnosis_used.parquet', index=False)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']
    predictions = diagnosis[keys].assign(sample_weight=1.)
    metrics = {}
    for name in ['ridge', 'boosted', 'constant', 'economic', 'continuation', 'exposure', 'capacity', 'weekly']:
        predictions['predicted_'+name] = .1
        metrics[name] = close_metrics(predictions, np.full(len(predictions), .1))
    predictions.to_parquet(source/'predictions.parquet', index=False)
    save_json(source/'metrics.json', metrics)
    for name in ['continuation', 'weekly']:
        first_close_positions(diagnosis.assign(prediction=.1), 'prediction').to_parquet(source/f'positions_{name}.parquet', index=False)
    decision = evaluate_first_margin(source, model, selected, out)
    assert decision['final_diagnosis_evaluated'] and len(decision['checks']) == 10
    saved = pd.read_parquet(out/'predictions.parquet')
    pd.testing.assert_frame_equal(saved.drop(columns='predicted_half'), predictions, check_exact=True)
    metrics = json.loads((out/'metrics.json').read_text())
    for field in ['weighted_mse', 'mse', 'mean_predicted_bps', 'mean_actual_bps']:
        assert metrics['margin'][field] == metrics['half_zero'][field]
    assert metrics['margin']['selected_positions'] == 40
    assert metrics['margin']['selected'] == 80 and not decision['checks']['at_least_100_selected']
    assert json.loads((out/'block_intervals.json').read_text())['seed'] == 63


def test_unsupported_selection_never_calls_new_final_evaluation(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    capacity = tmp_path/'capacity'
    cal, model = margin_fixture(capacity)
    save_json(source/'files.json', {})
    save_json(source/'reference_parity.json', {'reproduction': str(capacity)})
    for name in ['previous_models', 'weekly_models', 'metrics']:
        save_json(source/f'{name}.json', {})
    for name in ['predictions', 'exclusion_ledger', 'diagnosis_used', 'diagnosis_weights']:
        cal.to_parquet(source/f'{name}.parquet', index=False)
    # 원본 전체 재현은 별도 연결 검사에서 검증하고 실패 분기의 새 예측 호출만 차단한다.
    monkeypatch.setattr('wonyotti_fr.first_close_margin_diagnostics.reproduce_weekly', lambda *_: source)
    def failed(_source, output):
        save_json(output/'selection.json', {'selection_passed': False})
        return model, {'selection_passed': False}
    monkeypatch.setattr('wonyotti_fr.first_close_margin_diagnostics.calibrate_first_margin', failed)
    monkeypatch.setattr('wonyotti_fr.first_close_margin_diagnostics.evaluate_first_margin',
        lambda *_: pytest.fail('실패한 내부 선택으로 새 마지막 진단 실행'))
    out = run_first_close_margin_diagnosis(source, tmp_path/'out')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and not summary['final_diagnosis_evaluated'] and not summary['margin_admitted']
    assert not (out/'predictions.parquet').exists()
    assert (out/'previous_predictions.parquet').exists()


def test_full_pipeline_reproduces_weekly_models_and_rejects_resigned_tampering(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, monkeypatch)
    continuation = run_continuation_diagnosis(first, tmp_path/'continuation')
    exposure = run_exposure_close_diagnosis(continuation, tmp_path/'exposure')
    capacity = run_close_capacity_diagnosis(exposure, tmp_path/'capacity')
    reference = run_weekly_close_diagnosis(capacity, tmp_path/'weekly')
    out = run_first_close_margin_diagnosis(reference, tmp_path/'margin')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced'] and not summary['new_models_fitted']
    assert not summary['profitability_accepted']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'previous_predictions.parquet'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    assert json.loads((out/'previous_models.json').read_text()) == json.loads((reference/'previous_models.json').read_text())
    assert json.loads((out/'previous_weekly_models.json').read_text()) == json.loads((reference/'weekly_models.json').read_text())
    assert json.loads((out/'frozen_half_model.json').read_text()) == json.loads((capacity/'selection_models.json').read_text())['depth2_iter64']
    models = json.loads((reference/'weekly_models.json').read_text())
    models['week-00']['baseline'] += 1
    save_json(reference/'weekly_models.json', models)
    hashes = json.loads((reference/'files.json').read_text())
    hashes['weekly_models.json'] = sha256(reference/'weekly_models.json')
    save_json(reference/'files.json', hashes)
    with pytest.raises(ValueError, match='재현 불일치'):
        reproduce_weekly(reference, tmp_path/'tampered')
