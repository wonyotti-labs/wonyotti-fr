import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_close_threshold_diagnostics import retained_reference
from test_minute_close_diagnostics import seal

from wonyotti_fr.close_threshold_diagnostics import run_close_threshold_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.early_stopping_close import EarlyStoppingCloseModel
from wonyotti_fr.early_stopping_diagnostics import (
    EARLY_STOPPING_INPUT_FILES,
    run_early_stopping_diagnosis,
)
from wonyotti_fr.early_stopping_reference import THRESHOLD_FAILURE_FILES
from wonyotti_fr.minute_close_diagnostics import run_minute_close_diagnosis
from wonyotti_fr.retained_weight_close_diagnostics import run_retained_weight_close_diagnosis
from wonyotti_fr.source_snapshot import copy_snapshot
from wonyotti_fr.visited_close_diagnostics import run_visited_close_diagnosis


def threshold_reference(tmp_path, monkeypatch):
    retained, labels, legacy, raw = retained_reference(tmp_path, monkeypatch, successful=False)
    reference = run_close_threshold_diagnosis(retained, tmp_path/'threshold')
    assert set(json.loads((reference/'files.json').read_text())) == THRESHOLD_FAILURE_FILES
    assert json.loads((reference/'summary.json').read_text())['diagnosis_evaluated'] is False
    return reference, labels, legacy, raw


@pytest.fixture(scope='module')
def shared_threshold_reference(tmp_path_factory):
    # 같은 완료 기준을 공유하되 변조는 독립 사본에서만 수행한다.
    with pytest.MonkeyPatch.context() as monkeypatch:
        root = tmp_path_factory.mktemp('early-stopping-reference')
        reference, _, _, _ = threshold_reference(root, monkeypatch)
        yield reference


def clone_reference(source, target):
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            copy_snapshot(path, target/path.name)
    for directory in ['code_snapshot', 'baseline_source', 'calibration']:
        shutil.copytree(source/directory, target/directory)
    return target


def test_complete_pipeline_preserves_all_prior_rows_failed_selection_and_blocks_new_diagnosis(tmp_path, monkeypatch, shared_threshold_reference):
    reference = shared_threshold_reference
    original = EarlyStoppingCloseModel.probabilities
    diagnosis_rows = len(pd.read_parquet(reference/'diagnosis_used.parquet'))
    calls = []

    def guarded(self, values):
        calls.append(len(values))
        if len(values) == diagnosis_rows:
            raise RuntimeError('내부 보정 실패 뒤 마지막 진단 예측 호출')
        return original(self, values)

    monkeypatch.setattr(EarlyStoppingCloseModel, 'probabilities', guarded)
    out = run_early_stopping_diagnosis(reference, tmp_path/'run')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced'] and summary['original_rows_weights_and_targets_preserved']
    assert summary['whole_positions_disjoint'] and summary['teacher_models'] == 5 and summary['candidate_models'] == 1
    assert summary['policy_iterations'] == 1 and summary['threshold_search'] is False and summary['refit_after_calibration'] is False
    assert calls and summary['diagnosis_evaluated'] is False and summary['profitability_accepted'] is False
    assert not (out/'predictions.parquet').exists() and not (out/'intervals.json').exists()
    assert not json.loads((out/'calibration_decision.json').read_text())['calibration_passed']
    assert len(json.loads((out/'calibration/metrics.json').read_text())) == 5
    assert json.loads((out/'baseline_source/selection.json').read_text())['selection_passed'] is False
    files = json.loads((reference/'files.json').read_text())
    for name, value in files.items():
        assert sha256(reference/name) == value == sha256(out/'baseline_source'/name)
    for name in EARLY_STOPPING_INPUT_FILES:
        assert sha256(out/name) == sha256(reference/name)
    old_calibration = json.loads((reference/'calibration/files.json').read_text())
    for name, value in old_calibration.items():
        assert sha256(out/'baseline_calibration'/name) == value
    assert sha256(out/'baseline_calibration/files.json') == sha256(reference/'calibration/files.json')
    targets = pd.read_parquet(out/'early_stopping_targets.parquet')
    assignment = pd.read_parquet(out/'threshold_exclusion_ledger.parquet')
    np.testing.assert_array_equal(targets.opportunity_index, np.flatnonzero(assignment.split.eq('early_training')))
    assert len(targets) == summary['rows']['early_training']


@pytest.mark.parametrize('damage', ['model', 'weight', 'calibration', 'selection', 'snapshot', 'nested_reproduction'])
def test_resigned_source_failure_calibration_and_nested_damage_are_rejected_before_any_new_teacher(tmp_path, monkeypatch, shared_threshold_reference, damage):
    reference = clone_reference(shared_threshold_reference, tmp_path/'source')
    if damage == 'model':
        value = json.loads((reference/'model.json').read_text())
        value['models'][0]['baseline'] += 1
        save_json(reference/'model.json', value)
    elif damage == 'weight':
        path = reference/'early_training_weights.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'sample_weight'] *= 2
        frame.to_parquet(path, index=False)
    elif damage == 'calibration':
        path = reference/'calibration/predictions.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'early_score'] = 1-frame.loc[0, 'early_score']
        frame.to_parquet(path, index=False)
        seal(reference/'calibration')
        summary = json.loads((reference/'summary.json').read_text())
        summary['calibration_files_sha256'] = sha256(reference/'calibration/files.json')
        save_json(reference/'summary.json', summary)
    elif damage == 'selection':
        value = json.loads((reference/'selection.json').read_text())
        value['selection_passed'], value['diagnosis_allowed'], value['selected_threshold'] = True, True, .8
        save_json(reference/'selection.json', value)
    elif damage == 'snapshot':
        path = reference/'code_snapshot/close_threshold.py'
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
        raise RuntimeError('변조된 이전 결과 뒤 새 보조 학습 호출')

    monkeypatch.setattr('wonyotti_fr.early_stopping_diagnostics.fit_early_stopping_close', forbidden)
    destination = tmp_path/'rejected'
    with pytest.raises((ValueError, AssertionError)):
        run_early_stopping_diagnosis(reference, destination)
    assert not calls
    out = next(destination.iterdir())
    assert (out/'failure.json').is_file() and not (out/'early_teacher_models.json').exists() and not (out/'model.json').exists()


def test_future_diagnosis_features_and_labels_preserve_all_new_teachers_targets_model_and_calibration(tmp_path, monkeypatch):
    reference, labels, legacy, raw = threshold_reference(tmp_path, monkeypatch)
    first = run_early_stopping_diagnosis(reference, tmp_path/'first')
    changed = raw.copy()
    selected = changed.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC')) & changed.decision_time.dt.minute.mod(5).ne(0)
    changed.loc[selected, ['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed.loc[selected, 'close_cash'] = changed.loc[selected, 'continue_cash']+changed.loc[selected, 'close_advantage_pnl']
    changed.loc[selected, 'favorable_move'] += .001
    changed.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    minute = run_minute_close_diagnosis(labels, legacy, tmp_path/'changed-minute')
    visited = run_visited_close_diagnosis(minute, tmp_path/'changed-visited')
    retained = run_retained_weight_close_diagnosis(visited, tmp_path/'changed-retained')
    threshold = run_close_threshold_diagnosis(retained, tmp_path/'changed-threshold')
    future = run_early_stopping_diagnosis(threshold, tmp_path/'changed-stopping')
    for name in ['early_teacher_models', 'early_teacher_support', 'model', 'training_support', 'calibration_decision']:
        assert json.loads((first/f'{name}.json').read_text()) == json.loads((future/f'{name}.json').read_text())
    for name in ['early_teacher_training_membership', 'early_stopping_targets', 'stopping_training_cost_ledger', 'calibration/predictions']:
        pd.testing.assert_frame_equal(pd.read_parquet(first/f'{name}.parquet'), pd.read_parquet(future/f'{name}.parquet'), check_exact=True)
