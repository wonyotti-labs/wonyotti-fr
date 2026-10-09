import json

import pandas as pd
import pytest
from test_expanded_first_diagnostics import expanded_source_fixture
from test_first_opportunity_diagnostics import seal

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.expanded_first_diagnostics import run_expanded_first_diagnosis
from wonyotti_fr.expanded_tree_diagnostics import run_expanded_tree_diagnosis
from wonyotti_fr.expanded_tree_reference import EXPANDED_TREE_REVIEW_CHECKS
from wonyotti_fr.first_opportunity_close import FirstOpportunityCloseModel


def expanded_tree_source_fixture(path):
    inputs = expanded_source_fixture(path/'previous')
    reference = run_expanded_first_diagnosis(*inputs, path/'expanded')
    assert json.loads((reference/'decision.json').read_text())['calibration_passed'] is False
    review = path/'synthetic-expanded-verification.json'
    # 이전 검산의 합성 대체 기록으로 재사용 계약을 검사한다.
    save_json(review, {**dict.fromkeys(EXPANDED_TREE_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(reference/'files.json')})
    return reference, review, sha256(review)


def test_complete_tree_pipeline_all_inputs_fourteen_policies_and_no_last_diagnosis(tmp_path, monkeypatch):
    inputs = expanded_tree_source_fixture(tmp_path/'source')
    original_read = pd.read_parquet

    def no_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_diagnosis)
    result = run_expanded_tree_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1 and summary['combined_first_positions'] == 755
    assert not summary['previous_models_refitted'] and not summary['generation_engine_rerun'] and not summary['diagnosis_evaluated']
    assert not summary['profitability_accepted'] and len(summary['checks']) == 11
    model = json.loads((result/'model.json').read_text())
    assert len(model['models'][0]['trees']) == 64
    before = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    after = pd.read_parquet(result/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(after[before.columns], before, check_exact=True)
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 14
    for name in ['combined_first_training.parquet', 'combined_training_costs.parquet', 'added_training_membership.parquet', 'first_training_ledger.parquet']:
        assert (result/name).read_bytes() == (inputs[0]/name).read_bytes()
    assert not any((result/name).exists() for name in ['diagnosis_used.parquet', 'predictions.parquet', 'intervals.json'])
    assert all(sha256(result/name) == value for name, value in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['proof', 'synthetic_proof', 'combined_resealed', 'calibration_resealed', 'model', 'snapshot', 'symlink'])
def test_changed_previous_combined_evidence_rejected_before_tree_fit(tmp_path, monkeypatch, damage):
    source, review, digest = expanded_tree_source_fixture(tmp_path/'source')
    if damage in ['proof', 'synthetic_proof']:
        data = json.loads(review.read_text())
        data['complete' if damage == 'proof' else 'synthetic_only'] = damage == 'synthetic_proof'
        save_json(review, data)
        if damage == 'synthetic_proof':
            digest = sha256(review)
    elif damage == 'combined_resealed':
        path = source/'combined_first_training.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'first_target_common_bps'] += 1
        frame.to_parquet(path, index=False)
        seal(source)
    elif damage == 'calibration_resealed':
        path = source/'calibration/first_probability_metrics.json'
        values = json.loads(path.read_text())
        values['expanded_first_linear']['cost_log_loss'] += .1
        save_json(path, values)
        seal(path.parent)
    elif damage == 'model':
        path = source/'model.json'
        values = json.loads(path.read_text())
        values['intercept'] += .1
        save_json(path, values)
    elif damage == 'snapshot':
        path = source/'code_snapshot/expanded_first_linear.py'
        path.write_text(path.read_text()+'\n# 변조\n')
    else:
        path = source/'combined_training_costs.parquet'
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    calls = []
    monkeypatch.setattr(FirstOpportunityCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_expanded_tree_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
