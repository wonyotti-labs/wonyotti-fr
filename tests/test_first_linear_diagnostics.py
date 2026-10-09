import json

import pandas as pd
import pytest
from test_first_opportunity_diagnostics import seal
from test_first_opportunity_diagnostics import source_fixture as regression_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_linear_close import FirstLinearCloseModel
from wonyotti_fr.first_linear_diagnostics import run_first_linear_diagnosis
from wonyotti_fr.first_opportunity_diagnostics import run_first_opportunity_diagnosis


def linear_source_fixture(path, *, future_changed=False):
    previous = regression_fixture(path/'regression', future_changed=future_changed)
    source = run_first_opportunity_diagnosis(*previous, path/'first')
    assert json.loads((source/'decision.json').read_text())['calibration_passed'] is False
    # 실제 이전 학습 경로와 별개인 합성 검산 지문의 재사용 계약이다.
    review = path/'synthetic-first-verification.json'
    fields = ['complete', 'previous_verified_v80_inputs_and_calibration_reused', 'all_first_rows_all_positions_and_equal_weights_verified',
        'all64_new_trees_and_first_calibration_scores_verified', 'all_eight_policies_six_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']
    save_json(review, {**dict.fromkeys(fields, True), 'synthetic_only': False, 'calibration_passed': False,
        'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def test_full_linear_pipeline_keeps_first_ledger_eight_policies_and_no_diagnosis(tmp_path, monkeypatch):
    source, review, digest = linear_source_fixture(tmp_path/'source')
    original_read = pd.read_parquet

    def no_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_diagnosis)
    root = run_first_linear_diagnosis(source, review, digest, tmp_path/'run')
    summary = json.loads((root/'summary.json').read_text())
    assert summary['complete'] and summary['previous_first_ledger_exact'] and summary['new_models_fitted'] == 1
    assert not summary['diagnosis_evaluated'] and not summary['previous_models_refitted'] and not summary['profitability_accepted']
    assert len(summary['checks']) == 7
    for name in ['first_training_ledger.parquet', 'all_training_positions.parquet', 'first_training_contribution.parquet']:
        assert (root/name).read_bytes() == (source/name).read_bytes()
    old = pd.read_parquet(source/'calibration/predictions.parquet')
    new = pd.read_parquet(root/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=True)
    for name in ['predictions.parquet', 'intervals.json', 'diagnosis_used.parquet']:
        assert not (root/name).exists()
    assert json.loads((root/'previous_calibration_decision.json').read_text()) == json.loads((source/'decision.json').read_text())
    assert all(sha256(root/name) == value for name, value in json.loads((root/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['verification', 'synthetic_attestation', 'weight_resealed', 'first_ledger', 'calibration_resealed', 'snapshot', 'symlink'])
def test_changed_first_evidence_rejected_before_linear_fit(tmp_path, monkeypatch, damage):
    source, review, digest = linear_source_fixture(tmp_path/'source')
    if damage in ['verification', 'synthetic_attestation']:
        data = json.loads(review.read_text())
        if damage == 'verification':
            data['complete'] = False
        else:
            data['synthetic_only'] = True
        save_json(review, data)
        if damage == 'synthetic_attestation':
            digest = sha256(review)
    elif damage == 'weight_resealed':
        path = source/'early_training_weights.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'sample_weight'] *= 2
        frame.to_parquet(path, index=False)
        seal(source)
    elif damage == 'first_ledger':
        path = source/'first_training_ledger.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'first_target_common_bps'] += 1
        frame.to_parquet(path, index=False)
    elif damage == 'calibration_resealed':
        path = source/'calibration/first_probability_metrics.json'
        data = json.loads(path.read_text())
        data['first_opportunity']['cost_log_loss'] += 1
        save_json(path, data)
        seal(path.parent)
    elif damage == 'snapshot':
        path = source/'code_snapshot/first_linear_close.py'
        path.write_text(path.read_text()+'\n# 변조\n')
    else:
        path = source/'first_training_ledger.parquet'
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises(ValueError):
        run_first_linear_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()


def test_future_calibration_does_not_change_new_scaler_coefficients_or_old_ledger(tmp_path):
    first = linear_source_fixture(tmp_path/'original')
    other = linear_source_fixture(tmp_path/'future', future_changed=True)
    a = run_first_linear_diagnosis(*first, tmp_path/'a')
    b = run_first_linear_diagnosis(*other, tmp_path/'b')
    assert json.loads((a/'model.json').read_text()) == json.loads((b/'model.json').read_text())
    pd.testing.assert_frame_equal(pd.read_parquet(a/'first_training_ledger.parquet'), pd.read_parquet(b/'first_training_ledger.parquet'), check_exact=True)
