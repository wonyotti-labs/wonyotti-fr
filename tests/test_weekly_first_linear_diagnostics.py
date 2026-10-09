import json

import pandas as pd
import pytest
from test_first_linear_diagnostics import linear_source_fixture
from test_first_opportunity_diagnostics import seal

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_linear_close import FirstLinearCloseModel
from wonyotti_fr.first_linear_diagnostics import run_first_linear_diagnosis
from wonyotti_fr.weekly_first_linear_diagnostics import run_weekly_first_linear_diagnosis
from wonyotti_fr.weekly_first_linear_reference import WEEKLY_FIRST_REVIEW_CHECKS


def weekly_source_fixture(path):
    prior = linear_source_fixture(path/'previous')
    source = run_first_linear_diagnosis(*prior, path/'linear')
    assert json.loads((source/'decision.json').read_text())['calibration_passed'] is False
    # 합성 이전 검산 기록으로 연결 계약만 검사하며 실제 이전 재현을 주장하지 않는다.
    review = path/'synthetic-linear-verification.json'
    save_json(review, {**dict.fromkeys(WEEKLY_FIRST_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def test_full_weekly_pipeline_keeps_inputs_nine_controls_and_no_last_diagnosis(tmp_path, monkeypatch):
    source, review, digest = weekly_source_fixture(tmp_path/'source')
    original_read = pd.read_parquet

    def no_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_diagnosis)
    result = run_weekly_first_linear_diagnosis(source, review, digest, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 9 and summary['first_model_exact']
    assert summary['sequential_calibration_labels_used_only_after_maturity']
    assert not summary['diagnosis_evaluated'] and not summary['previous_models_refitted'] and not summary['profitability_accepted']
    assert len(summary['checks']) == 9
    old = pd.read_parquet(source/'calibration/predictions.parquet')
    new = pd.read_parquet(result/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=True)
    assert json.loads((result/'weekly_models.json').read_text())['week-00'] == json.loads((source/'model.json').read_text())
    for name in ['first_training_ledger.parquet', 'all_training_positions.parquet', 'first_training_contribution.parquet']:
        assert (result/name).read_bytes() == (source/name).read_bytes()
    assert not any((result/name).exists() for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet'])
    assert all(sha256(result/name) == value for name, value in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['verification', 'synthetic_attestation', 'weights_resealed', 'calibration_resealed', 'model', 'snapshot', 'symlink'])
def test_changed_linear_evidence_rejected_before_weekly_fit(tmp_path, monkeypatch, damage):
    source, review, digest = weekly_source_fixture(tmp_path/'source')
    if damage in ['verification', 'synthetic_attestation']:
        data = json.loads(review.read_text())
        data['complete' if damage == 'verification' else 'synthetic_only'] = damage == 'synthetic_attestation'
        save_json(review, data)
        if damage == 'synthetic_attestation':
            digest = sha256(review)
    elif damage == 'weights_resealed':
        path = source/'early_training_weights.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'sample_weight'] *= 2
        frame.to_parquet(path, index=False)
        seal(source)
    elif damage == 'calibration_resealed':
        path = source/'calibration/first_probability_metrics.json'
        data = json.loads(path.read_text())
        data['first_linear']['cost_log_loss'] += .1
        save_json(path, data)
        seal(path.parent)
    elif damage == 'model':
        path = source/'model.json'
        data = json.loads(path.read_text())
        data['intercept'] += .1
        save_json(path, data)
    elif damage == 'snapshot':
        path = source/'code_snapshot/weekly_first_linear.py'
        path.write_text(path.read_text()+'\n# 변조\n')
    else:
        path = source/'first_training_ledger.parquet'
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises(ValueError):
        run_weekly_first_linear_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
