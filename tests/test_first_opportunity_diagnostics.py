import json

import numpy as np
import pandas as pd
import pytest
from test_first_opportunity_close import setup

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_threshold import THRESHOLD_SPLITS
from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.early_stopping_diagnostics import save_policy_results
from wonyotti_fr.first_opportunity_close import FirstOpportunityCloseModel
from wonyotti_fr.first_opportunity_diagnostics import run_first_opportunity_diagnosis
from wonyotti_fr.first_opportunity_reference import REGRESSION_FAILURE_FILES, REQUIRED_REVIEW_CHECKS
from wonyotti_fr.stopping_regression_evaluation import evaluate_regression_policies


def seal(root):
    save_json(root/'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'files.json'})


def source_fixture(path, *, future_changed=False):
    # 이전 적합을 흉내 내지 않고 검산 기록의 신뢰·지문 재사용 계약만 구성한다.
    root = new_run(path, 'stopping-regression-diagnosis', {'candidate': 'stopping_regression', 'periods': THRESHOLD_SPLITS})
    training, weights, calibration = setup()
    if future_changed:
        calibration['favorable_move'] += 1.
        calibration['close_advantage_pnl'] *= -3
        calibration['close_cash'] = calibration.continue_cash+calibration.close_advantage_pnl
        calibration['close_advantage_bps'] = calibration.close_advantage_pnl/calibration.decision_equity*10000
    for name in REGRESSION_FAILURE_FILES-{'manifest.json'}:
        (root/name).write_text('opaque synthetic prior evidence\n')
    for name, frame in [('early_training', training), ('calibration', calibration)]:
        frame.to_parquet(root/f'{name}_used.parquet', index=False)
        frame[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(frame)).to_parquet(root/f'{name}_weights.parquet', index=False)
    assignment = pd.concat([training.assign(split='early_training'), calibration.assign(split='calibration')], ignore_index=True)
    assignment.to_parquet(root/'threshold_exclusion_ledger.parquet', index=False)
    calibrated = calibration.assign(sample_weight=position_weights(calibration))
    old = evaluate_regression_policies(calibrated, np.ones(len(calibration)), np.full(len(calibration), .8), np.full(len(calibration), .7), -1.)
    folder = root/'calibration'
    folder.mkdir()
    save_policy_results(folder, old)
    seal(folder)
    save_json(root/'calibration_decision.json', old['calibration_decision'])
    save_json(root/'summary.json', {'complete': True, 'profitability_accepted': False, 'diagnosis_evaluated': False,
        'calibration_files_sha256': sha256(folder/'files.json')})
    seal(root)
    review = path/'synthetic-verification.json'
    save_json(review, {**dict.fromkeys(REQUIRED_REVIEW_CHECKS, True), 'source_files_sha256': sha256(root/'files.json'),
        'calibration_passed': False, 'diagnosis_evaluated': False, 'profitability_accepted': False})
    return root, review, sha256(review)


def test_complete_single_fit_pipeline_preserves_original_inputs_and_blocks_diagnosis(tmp_path, monkeypatch):
    reference, review, digest = source_fixture(tmp_path/'source')
    before = json.loads((reference/'files.json').read_text())
    original_read = pd.read_parquet

    def no_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_diagnosis)
    result = run_first_opportunity_diagnosis(reference, review, digest, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1
    assert not summary['diagnosis_evaluated'] and not summary['previous_models_refitted']
    assert not summary['external_nested_runs_reverified'] and not summary['profitability_accepted']
    assert len(summary['checks']) == 6
    assert all(sha256(reference/name) == value for name, value in before.items())
    assert (result/'previous_verification.json').read_bytes() == review.read_bytes()
    assert (result/'previous_calibration_decision.json').read_bytes() == (reference/'calibration_decision.json').read_bytes()
    assert not any((result/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])
    original = original_read(reference/'calibration/predictions.parquet')
    new = original_read(result/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(new[original.columns], original, check_exact=True)
    all_positions = original_read(result/'all_training_positions.parquet')
    assert len(all_positions) == 600 and (~all_positions.has_eligible_opportunity).sum() == 60
    metadata = json.loads((result/'files.json').read_text())
    assert all(sha256(result/name) == value for name, value in metadata.items())


@pytest.mark.parametrize('damage', ['verification', 'wrong_digest', 'raw_training', 'weights_resealed', 'calibration_resealed', 'code', 'symlink'])
def test_changed_and_resealed_prior_evidence_rejected_before_new_fit(tmp_path, monkeypatch, damage):
    reference, review, digest = source_fixture(tmp_path/'source')
    if damage == 'verification':
        value = json.loads(review.read_text())
        value['complete'] = False
        save_json(review, value)
    elif damage == 'wrong_digest':
        digest = '0'*64
    elif damage == 'raw_training':
        with (reference/'early_training_used.parquet').open('ab') as handle:
            handle.write(b'tamper')
    elif damage == 'weights_resealed':
        path = reference/'early_training_weights.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'sample_weight'] *= 2
        frame.to_parquet(path, index=False)
        seal(reference)
    elif damage == 'calibration_resealed':
        path = reference/'calibration/metrics.json'
        value = json.loads(path.read_text())
        value['always_first']['selected'] += 1
        save_json(path, value)
        seal(path.parent)
    elif damage == 'code':
        path = reference/'code_snapshot/first_opportunity_close.py'
        path.write_text(path.read_text()+'\n# 변조\n')
    else:
        path = reference/'early_training_used.parquet'
        backup = tmp_path/'original.parquet'
        path.rename(backup)
        path.symlink_to(backup)
    calls = []
    monkeypatch.setattr(FirstOpportunityCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises(ValueError):
        run_first_opportunity_diagnosis(reference, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()


def test_changed_future_features_labels_and_separate_attestation_leave_training_model_equal(tmp_path):
    first = source_fixture(tmp_path/'original')
    second = source_fixture(tmp_path/'future', future_changed=True)
    result_a = run_first_opportunity_diagnosis(*first, tmp_path/'run-a')
    result_b = run_first_opportunity_diagnosis(*second, tmp_path/'run-b')
    assert json.loads((result_a/'model.json').read_text()) == json.loads((result_b/'model.json').read_text())
    for name in ['first_training_ledger.parquet', 'all_training_positions.parquet', 'first_training_contribution.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(result_a/name), pd.read_parquet(result_b/name), check_exact=True)
    assert json.loads((result_a/'decision.json').read_text())['diagnosis_evaluated'] is False
    assert json.loads((result_b/'decision.json').read_text())['diagnosis_evaluated'] is False
