import json
from unittest.mock import patch

import pandas as pd
import pytest
from test_first_opportunity_close import first_examples
from test_managed_first_diagnostics import synthetic_parent_loader
from test_minute_first_diagnostics import minute_source_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_first_diagnostics import run_minute_first_diagnosis
from wonyotti_fr.regularized_first_diagnostics import run_regularized_first_diagnosis
from wonyotti_fr.regularized_first_model import REGULARIZATION_STRENGTHS, RegularizedFirstModel
from wonyotti_fr.regularized_first_reference import REGULARIZED_REVIEW_CHECKS


def regularized_source_fixture(path):
    with patch('test_expanded_first_linear.first_examples', lambda *_a, **_kw: first_examples('2020-04-01', 600, 8)):
        inputs = minute_source_fixture(path/'previous')
    with patch('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader):
        source = run_minute_first_diagnosis(*inputs, path/'minute')
    assert json.loads((source/'decision.json').read_text())['calibration_passed'] is False
    # 앞 기간의 합성 표본을 늘려 실제 최소 지원 조건을 그대로 검사한다.
    review = path/'synthetic-minute-verification.json'
    save_json(review, {**dict.fromkeys(REGULARIZED_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def test_complete13_models_selection_seals_and_eighteen_unchanged_population_policies(tmp_path, monkeypatch):
    inputs = regularized_source_fixture(tmp_path/'source')
    original_read = pd.read_parquet

    def no_final_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_final_diagnosis)
    result = run_regularized_first_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 13 and len(summary['checks']) == 15
    assert not summary['diagnosis_evaluated'] and not summary['profitability_accepted']
    assert not summary['external_calibration_used_for_selection'] and summary['selected_strength'] in REGULARIZATION_STRENGTHS
    assert summary['combined_first_positions'] == 1079 and summary['selection_files_sha256'] == sha256(result/'selection/files.json')
    folds = json.loads((result/'selection/folds.json').read_text())
    for name, digest in folds.items():
        assert digest == sha256(result/'selection'/name/'files.json')
        assert len(json.loads((result/'selection'/name/'models.json').read_text())) == 4
        for file, expected in json.loads((result/'selection'/name/'files.json').read_text()).items():
            assert sha256(result/'selection'/name/file) == expected
    for name, digest in json.loads((result/'selection/files.json').read_text()).items():
        assert sha256(result/'selection'/name) == digest
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 18
    old = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(result/'calibration/predictions.parquet')[old.columns], old, check_exact=True)
    for name in ['minute_first_training.parquet', 'minute_first_calibration.parquet', 'combined_training_costs.parquet', 'minute_source.json']:
        assert (result/name).read_bytes() == (inputs[0]/name).read_bytes()
    assert all(sha256(result/name) == digest for name, digest in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['proof', 'training', 'costs', 'baseline', 'snapshot', 'symlink'])
def test_changed_prior79_inputs_or_evidence_blocked_before_selection(tmp_path, monkeypatch, damage):
    source, review, digest = regularized_source_fixture(tmp_path/'source')
    path = {'proof': review, 'training': source/'minute_first_training.parquet', 'costs': source/'combined_training_costs.parquet',
        'baseline': source/'calibration/first_metrics.json', 'snapshot': source/'code_snapshot/minute_first_inputs.py',
        'symlink': source/'minute_first_calibration.parquet'}[damage]
    if damage == 'symlink':
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_bytes(path.read_bytes()+b' ')
    calls = []
    monkeypatch.setattr(RegularizedFirstModel, 'fit', lambda *_a, **_kw: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_regularized_first_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
