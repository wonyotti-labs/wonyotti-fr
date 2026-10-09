import json
from unittest.mock import patch

import pandas as pd
import pytest
from test_first_opportunity_diagnostics import seal
from test_managed_first_diagnostics import managed_source_fixture, synthetic_parent_loader

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.managed_first_diagnostics import run_managed_first_diagnosis
from wonyotti_fr.probability_first_diagnostics import run_probability_first_diagnosis
from wonyotti_fr.probability_first_model import ProbabilityFirstLinearModel
from wonyotti_fr.probability_first_reference import PROBABILITY_REVIEW_CHECKS


def probability_source_fixture(path):
    inputs = managed_source_fixture(path/'previous')
    with patch('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader):
        source = run_managed_first_diagnosis(*inputs, path/'managed')
    assert json.loads((source/'decision.json').read_text())['calibration_passed'] is False
    # 이전 검산의 합성 대체 기록으로 확률 원장 재사용 계약을 검사한다.
    review = path/'synthetic-managed-verification.json'
    save_json(review, {**dict.fromkeys(PROBABILITY_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def test_complete_three_input_pipeline_preserves77_inputs_and_sixteen_policies(tmp_path, monkeypatch):
    inputs = probability_source_fixture(tmp_path/'source')
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    original = pd.read_parquet

    def no_last_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_last_diagnosis)
    result = run_probability_first_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1 and len(summary['checks']) == 13
    assert not summary['previous_models_refitted'] and not summary['generation_engine_rerun'] and not summary['diagnosis_evaluated']
    assert not summary['profitability_accepted']
    assert len(json.loads((result/'model.json').read_text())['features']) == 3
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 16
    old = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(result/'calibration/predictions.parquet')[old.columns], old, check_exact=True)
    for name in ['combined_first_training.parquet', 'combined_training_costs.parquet', 'managed_first_training.parquet', 'managed_first_calibration.parquet']:
        assert (result/name).read_bytes() == (inputs[0]/name).read_bytes()
    assert all(sha256(result/name) == digest for name, digest in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['proof', 'training_probabilities', 'calibration_probabilities', 'parent_evidence', 'previous_parent_link', 'symlink'])
def test_changed_probabilities_or_parent_evidence_blocked_before_fit(tmp_path, monkeypatch, damage):
    source, review, digest = probability_source_fixture(tmp_path/'source')
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    previous = json.loads((source/'reference_evidence.json').read_text())['reference']
    path = {'proof': review, 'training_probabilities': source/'managed_first_training.parquet',
        'calibration_probabilities': source/'managed_first_calibration.parquet', 'parent_evidence': source/'manager_evidence.json',
        'previous_parent_link': source.parent.parent/'previous'/'unused', 'symlink': source/'managed_first_calibration.parquet'}[damage]
    if damage == 'previous_parent_link':
        from pathlib import Path
        path = Path(previous)/'reference_evidence.json'
    if damage == 'symlink':
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_bytes(path.read_bytes()+b' ')
    if damage in ['training_probabilities', 'calibration_probabilities']:
        seal(source)
    calls = []
    monkeypatch.setattr(ProbabilityFirstLinearModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_probability_first_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
