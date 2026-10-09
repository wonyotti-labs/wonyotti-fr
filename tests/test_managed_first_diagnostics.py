import json
from types import SimpleNamespace

import pandas as pd
import pytest
from test_expanded_tree_diagnostics import expanded_tree_source_fixture
from test_first_opportunity_diagnostics import seal
from test_managed_first_model import fixed_manager

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.expanded_tree_diagnostics import run_expanded_tree_diagnosis
from wonyotti_fr.managed_first_diagnostics import run_managed_first_diagnosis
from wonyotti_fr.managed_first_model import MANAGER_FEATURES, ManagedFirstLinearModel
from wonyotti_fr.managed_first_reference import MANAGED_REVIEW_CHECKS


def managed_source_fixture(path):
    inputs = expanded_tree_source_fixture(path/'previous')
    source = run_expanded_tree_diagnosis(*inputs, path/'tree')
    assert json.loads((source/'decision.json').read_text())['calibration_passed'] is False
    # 이전 검산과 부모 연결의 합성 대체 기록으로 소비 계약을 검사한다.
    parent, generation, linear = [path/name for name in ['parent', 'generation', 'linear']]
    for folder in [parent, generation, linear]:
        folder.mkdir()
    save_json(parent/'frozen_selection.json', {'protocol': 'exit_move_v54'})
    manager = fixed_manager()
    save_json(parent/'model.json', manager.model.to_dict())
    save_json(parent/'offset.json', manager.offset.to_dict())
    save_json(generation/'manifest.json', {'settings': {'reference': str(parent), 'reference_sha256': sha256(parent/'frozen_selection.json')}})
    save_json(linear/'manifest.json', {'settings': {'expansion': str(generation)}})
    save_json(linear/'expansion_evidence.json', {'expansion': str(generation), 'files': {'manifest.json': sha256(generation/'manifest.json')}})
    save_json(source/'reference_evidence.json', {'reference': str(linear), 'files': {name: sha256(linear/name)
        for name in ['manifest.json', 'expansion_evidence.json']}})
    seal(source)
    review = path/'synthetic-tree-verification.json'
    save_json(review, {**dict.fromkeys(MANAGED_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def synthetic_parent_loader(path):
    manager = fixed_manager()
    assert json.loads((path/'model.json').read_text()) == manager.model.to_dict()
    assert json.loads((path/'offset.json').read_text()) == manager.offset.to_dict()
    return json.loads((path/'frozen_selection.json').read_text()), SimpleNamespace(manager=manager)


def test_complete77_input_pipeline_all_old_rows_and_no_final_diagnosis(tmp_path, monkeypatch):
    inputs = managed_source_fixture(tmp_path/'source')
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    original_read = pd.read_parquet

    def no_final(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_final)
    result = run_managed_first_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1 and len(summary['checks']) == 12
    assert not summary['previous_models_refitted'] and not summary['generation_engine_rerun'] and not summary['diagnosis_evaluated']
    assert not summary['profitability_accepted']
    assert len(json.loads((result/'model.json').read_text())['features']) == 77
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 15
    old = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(result/'calibration/predictions.parquet')[old.columns], old, check_exact=True)
    for name in ['combined_first_training.parquet', 'combined_training_costs.parquet', 'added_training_membership.parquet']:
        assert (result/name).read_bytes() == (inputs[0]/name).read_bytes()
    before = pd.read_parquet(result/'combined_first_training.parquet')
    after = pd.read_parquet(result/'managed_first_training.parquet')
    assert list(after) == [*before.columns, *MANAGER_FEATURES]
    pd.testing.assert_frame_equal(after[before.columns], before, check_exact=True)
    assert all(sha256(result/name) == digest for name, digest in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['proof', 'combined', 'calibration', 'linear_manifest', 'expansion_evidence', 'generation_manifest', 'parent', 'symlink'])
def test_changed_evidence_or_parent_chain_rejected_before_any_fit(tmp_path, monkeypatch, damage):
    root = tmp_path/'source'
    source, review, digest = managed_source_fixture(root)
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    path = {'proof': review, 'combined': source/'combined_first_training.parquet', 'calibration': source/'calibration/first_metrics.json',
        'linear_manifest': root/'linear/manifest.json', 'expansion_evidence': root/'linear/expansion_evidence.json',
        'generation_manifest': root/'generation/manifest.json', 'parent': root/'parent/frozen_selection.json',
        'symlink': root/'generation/manifest.json'}[damage]
    if damage == 'symlink':
        target = tmp_path/'preserved.json'
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_bytes(path.read_bytes()+b' ')
    calls = []
    monkeypatch.setattr(ManagedFirstLinearModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_managed_first_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
