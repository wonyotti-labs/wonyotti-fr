import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_close_threshold import threshold_examples
from test_minute_close_diagnostics import fixture, seal
from test_visited_close_diagnostics import signal_examples

from wonyotti_fr.close_threshold import ThresholdCloseModel
from wonyotti_fr.close_threshold_diagnostics import (
    THRESHOLD_INPUT_FILES,
    run_close_threshold_diagnosis,
)
from wonyotti_fr.close_threshold_reference import RETAINED_DIAGNOSIS_FILES
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_close_diagnostics import run_minute_close_diagnosis
from wonyotti_fr.retained_weight_close_diagnostics import run_retained_weight_close_diagnosis
from wonyotti_fr.retained_weight_close_evaluation import RETAINED_POLICIES
from wonyotti_fr.visited_close_diagnostics import run_visited_close_diagnosis


def retained_reference(tmp_path, monkeypatch, successful=True):
    monkeypatch.setattr('test_minute_close_diagnostics.financial_examples', threshold_examples if successful else signal_examples)
    labels, legacy, raw, _ = fixture(tmp_path, monkeypatch)
    minute = run_minute_close_diagnosis(labels, legacy, tmp_path/'minute')
    visited = run_visited_close_diagnosis(minute, tmp_path/'visited')
    reference = run_retained_weight_close_diagnosis(visited, tmp_path/'retained')
    assert set(json.loads((reference/'files.json').read_text())) == RETAINED_DIAGNOSIS_FILES
    return reference, labels, legacy, raw


def test_complete_threshold_pipeline_preserves_old_results_and_calibration_then_evaluates_once(tmp_path, monkeypatch):
    reference, _, _, _ = retained_reference(tmp_path, monkeypatch)
    out = run_close_threshold_diagnosis(reference, tmp_path/'threshold')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced'] and summary['whole_positions_disjoint']
    assert summary['diagnosis_evaluated'] and summary['selected_threshold'] > .5
    assert summary['new_models_fitted'] == 1 and summary['refit_after_selection'] is False
    assert summary['profitability_accepted'] is False and len(summary['checks']) == 12
    assert summary['positions']['diagnosis'] == 62
    files = json.loads((reference/'files.json').read_text())
    for name, value in files.items():
        assert sha256(reference/name) == value == sha256(out/'baseline_source'/name)
    for name in THRESHOLD_INPUT_FILES:
        assert sha256(out/name) == sha256(reference/name)
    old = pd.read_parquet(reference/'predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet')[old.columns], old, check_exact=True)
    for name in RETAINED_POLICIES:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'positions_{name}.parquet'), pd.read_parquet(reference/f'positions_{name}.parquet'), check_exact=True)
    for name in ['metrics', 'first_metrics', 'probability_metrics']:
        before, after = [json.loads((folder/f'{name}.json').read_text()) for folder in [reference, out]]
        assert {key: after[key] for key in before} == before
    assignment = pd.read_parquet(out/'threshold_exclusion_ledger.parquet')
    original = pd.read_parquet(out/'minute_ledger.parquet')
    pd.testing.assert_frame_equal(assignment[original.columns], original, check_exact=True)
    np.testing.assert_array_equal(assignment.opportunity_index, np.arange(len(original)))
    assert len(json.loads((out/'calibration/first_metrics.json').read_text())) == 9
    assert sha256(out/'calibration/files.json') == summary['calibration_files_sha256']


def test_no_eligible_threshold_blocks_new_diagnosis_predictions_and_keeps_all_candidates(tmp_path, monkeypatch):
    reference, _, _, _ = retained_reference(tmp_path, monkeypatch, successful=False)
    original = ThresholdCloseModel.probabilities
    rows = pd.read_parquet(reference/'diagnosis_used.parquet')
    calls = []

    def guarded(self, values):
        calls.append(len(values))
        if len(values) == len(rows):
            raise RuntimeError('선택 실패 뒤 마지막 진단 예측 호출')
        return original(self, values)

    monkeypatch.setattr(ThresholdCloseModel, 'probabilities', guarded)
    out = run_close_threshold_diagnosis(reference, tmp_path/'blocked')
    selection = json.loads((out/'selection.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert calls and not selection['selection_passed'] and selection['selected_threshold'] is None
    assert not summary['diagnosis_evaluated'] and not summary['threshold_admitted']
    assert not (out/'predictions.parquet').exists() and not (out/'intervals.json').exists()
    assert (out/'model.json').is_file() and (out/'baseline_source/predictions.parquet').is_file()
    assert len(selection['candidates']) == 7 and all(not item['eligible'] for item in selection['candidates'].values())
    assert len(json.loads((out/'calibration/first_metrics.json').read_text())) == 9


def test_future_diagnosis_features_and_labels_do_not_change_early_model_or_selected_threshold(tmp_path, monkeypatch):
    reference, labels, legacy, raw = retained_reference(tmp_path, monkeypatch)
    first = run_close_threshold_diagnosis(reference, tmp_path/'first')
    changed = raw.copy()
    mask = changed.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC')) & changed.decision_time.dt.minute.mod(5).ne(0)
    changed.loc[mask, ['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed.loc[mask, 'close_cash'] = changed.loc[mask, 'continue_cash']+changed.loc[mask, 'close_advantage_pnl']
    changed.loc[mask, 'favorable_move'] += .001
    changed.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    minute = run_minute_close_diagnosis(labels, legacy, tmp_path/'changed-minute')
    visited = run_visited_close_diagnosis(minute, tmp_path/'changed-visited')
    retained = run_retained_weight_close_diagnosis(visited, tmp_path/'changed-retained')
    future = run_close_threshold_diagnosis(retained, tmp_path/'changed-threshold')
    for name in ['model', 'training_support', 'selection']:
        assert json.loads((first/f'{name}.json').read_text()) == json.loads((future/f'{name}.json').read_text())
    for name in ['early_training_used', 'early_training_weights', 'early_training_cost_ledger', 'calibration_used', 'calibration/predictions']:
        pd.testing.assert_frame_equal(pd.read_parquet(first/f'{name}.parquet'), pd.read_parquet(future/f'{name}.parquet'), check_exact=True)


@pytest.mark.parametrize('damage', ['score', 'model', 'weight', 'snapshot', 'nested_reproduction'])
def test_resigned_prior_or_nested_damage_is_rejected_before_new_early_fit(tmp_path, monkeypatch, damage):
    reference, _, _, _ = retained_reference(tmp_path, monkeypatch)
    if damage in ['score', 'weight']:
        path = reference/('predictions.parquet' if damage == 'score' else 'retained_training_weights.parquet')
        frame = pd.read_parquet(path)
        if damage == 'score':
            frame.loc[0, 'retained_score'] = 1-frame.loc[0, 'retained_score']
        else:
            frame.loc[0, 'sample_weight'] *= 2
        frame.to_parquet(path, index=False)
    elif damage == 'model':
        path = reference/'model.json'
        value = json.loads(path.read_text())
        value['models'][0]['baseline'] += 1
        save_json(path, value)
    elif damage == 'snapshot':
        path = reference/'code_snapshot/retained_weight_close.py'
        path.write_text(path.read_text()+'\n# 합성 변조\n')
    else:
        proof = json.loads((reference/'reference_parity.json').read_text())
        copied = tmp_path/'modified-nested'
        shutil.copytree(Path(proof['reproduction']), copied)
        value = json.loads((copied/'model.json').read_text())
        value['models'][0]['baseline'] += 1
        save_json(copied/'model.json', value)
        seal(copied)
        proof['reproduction'], proof['reproduction_files_sha256'] = str(copied), sha256(copied/'files.json')
        save_json(reference/'reference_parity.json', proof)
    seal(reference)
    calls = []

    def forbidden(*_args):
        calls.append(True)
        raise RuntimeError('변조된 이전 결과 뒤 새 학습 호출')

    monkeypatch.setattr('wonyotti_fr.close_threshold_diagnostics.fit_threshold_close', forbidden)
    destination = tmp_path/'rejected'
    with pytest.raises((ValueError, AssertionError)):
        run_close_threshold_diagnosis(reference, destination)
    assert calls == []
    out = next(destination.iterdir())
    assert (out/'failure.json').is_file() and not (out/'model.json').exists() and not (out/'summary.json').exists()
