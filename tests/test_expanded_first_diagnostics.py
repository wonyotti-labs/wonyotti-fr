import json
from pathlib import Path

import pandas as pd
import pytest
from test_expanded_first_linear import added_examples
from test_first_opportunity_diagnostics import seal
from test_weekly_first_linear_diagnostics import weekly_source_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.expanded_first_diagnostics import run_expanded_first_diagnosis
from wonyotti_fr.expanded_first_reference import EXPANDED_REFERENCE_CHECKS, EXPANSION_PHASE_CHECKS
from wonyotti_fr.first_linear_close import FirstLinearCloseModel
from wonyotti_fr.first_opportunity_collection import EXPANDED_FIRST_FEATURES
from wonyotti_fr.first_opportunity_expansion import BASELINE_FILES, EXPANSION_PERIODS
from wonyotti_fr.weekly_first_linear_diagnostics import run_weekly_first_linear_diagnosis


def expanded_source_fixture(path):
    prior = weekly_source_fixture(path/'previous')
    reference = run_weekly_first_linear_diagnosis(*prior, path/'weekly')
    assert json.loads((reference/'decision.json').read_text())['calibration_passed'] is False
    review = path/'synthetic-weekly-verification.json'
    save_json(review, {**dict.fromkeys(EXPANDED_REFERENCE_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(reference/'files.json')})
    expansion = path/'synthetic-expansion'
    expansion.mkdir()
    sources = {p.name: sha256(p) for p in Path('src/wonyotti_fr').glob('*.py')}
    snapshot = expansion/'generation_source/wonyotti_fr'
    snapshot.mkdir(parents=True)
    for name in sources:
        (snapshot/name).write_bytes((Path('src/wonyotti_fr')/name).read_bytes())
    previous = json.loads((reference/'manifest.json').read_text())['settings']
    settings = {'periods': EXPANSION_PERIODS, 'features_used': EXPANDED_FIRST_FEATURES, 'new_models_fitted': False,
        'whole_policy_historically_available_claimed': False, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V84.md')),
        'implementation_sha256': sources, 'diagnosis_files_sha256': previous['reference_files_sha256']}
    save_json(expansion/'manifest.json', {'settings': settings})
    save_json(expansion/'reference_evidence.json', {'synthetic_only': True})
    added, positions = added_examples()
    phases, phase_proofs = {}, {}
    for name in EXPANSION_PERIODS:
        folder = expansion/name
        folder.mkdir()
        replay, baseline = folder/'replay', folder/'baseline'
        replay.mkdir()
        baseline.mkdir()
        for directory in [replay, baseline]:
            for file in BASELINE_FILES:
                (directory/file).write_text('synthetic previous verified payload\n')
        for file in ['management_membership.parquet', 'parity.json']:
            (replay/file).write_text('synthetic previous verified payload\n')
        added.to_parquet(folder/'first_opportunity_ledger.parquet', index=False)
        positions.to_parquet(folder/'all_positions.parquet', index=False)
        added[added.label_status.eq('closed')].reset_index(drop=True).to_parquet(folder/'training_labels.parquet', index=False)
        (folder/'outcomes.sqlite').write_text('synthetic previous verified journal\n')
        save_json(folder/'input_verification.json', {'synthetic_only': True})
        baseline_hashes = {file: sha256(baseline/file) for file in BASELINE_FILES}
        if name == 'expansion_2020':
            save_json(folder/'baseline_source.json', {'baseline': str(baseline), 'sha256': baseline_hashes})
        parity = {'complete': True, 'splits': {'synthetic_parity': True}} if name == 'parity_2021' else None
        summary = {'complete': True, 'phase': name, 'selected_first': len(added), 'closed': int(added.label_status.eq('closed').sum()),
            'positions': len(positions), 'counts': {'management_rows': 720}, 'replay': str(replay), 'baseline': str(baseline),
            'baseline_sha256': baseline_hashes, 'replay_sha256': {p.name: sha256(p) for p in replay.iterdir()}, 'reference_parity': parity}
        save_json(folder/'summary.json', summary)
        seal(folder)
        phases[name] = {'complete': True, 'files_sha256': sha256(folder/'files.json')}
        phase_proofs[name] = {**dict.fromkeys(EXPANSION_PHASE_CHECKS, True), 'first_eligible': len(added), 'closed': summary['closed'],
            'all_positions': len(positions), 'management_rows': 720}
        if name == 'parity_2021':
            phase_proofs[name]['previous_reference_parity'] = parity['splits']
    save_json(expansion/'summary.json', {'complete': True, 'phases': phases, 'new_models_fitted': False,
        'forced_boundary_closes_as_labels': False, 'profitability_accepted': False})
    seal(expansion)
    expansion_review = path/'synthetic-expansion-verification.json'
    # 합성 봉인과 이전 검산 기록은 재사용 계약의 검사이며 실제 시세 생성 검산을 대신하지 않는다.
    save_json(expansion_review, {'complete': True, 'source_files_sha256': sha256(expansion/'files.json'),
        'profitability_accepted': False, 'new_first_selection_features_cash_independently_verified': True,
        'original_files_unchanged': 5, 'phases': phase_proofs})
    return reference, review, sha256(review), expansion, expansion_review, sha256(expansion_review)


def test_full_added_pipeline_keeps_thirteen_policies_all_sources_and_no_last_diagnosis(tmp_path, monkeypatch):
    inputs = expanded_source_fixture(tmp_path/'source')
    original_read = pd.read_parquet

    def no_diagnosis(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_diagnosis)
    result = run_expanded_first_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1 and summary['combined_first_positions'] == 755
    assert summary['phase_first_positions'] == {'original_2021': 540, 'expansion_2020': 215}
    assert not summary['previous_models_refitted'] and not summary['generation_engine_rerun'] and not summary['diagnosis_evaluated']
    assert not summary['profitability_accepted'] and len(summary['checks']) == 10
    previous = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    current = pd.read_parquet(result/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(current[previous.columns], previous, check_exact=True)
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 13
    for name in ['first_training_ledger.parquet', 'all_training_positions.parquet', 'first_training_contribution.parquet']:
        assert (result/name).read_bytes() == (inputs[0]/name).read_bytes()
    assert not any((result/name).exists() for name in ['diagnosis_used.parquet', 'predictions.parquet', 'intervals.json'])
    assert all(sha256(result/name) == value for name, value in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['weekly_proof', 'expansion_proof', 'added_resealed', 'baseline', 'snapshot', 'symlink', 'phase_proof'])
def test_changed_previous_evidence_and_expansion_rejected_before_fit(tmp_path, monkeypatch, damage):
    inputs = list(expanded_source_fixture(tmp_path/'source'))
    reference, review, _, expansion, expansion_review, _ = inputs
    if damage in ['weekly_proof', 'expansion_proof']:
        path = review if damage == 'weekly_proof' else expansion_review
        data = json.loads(path.read_text())
        data['complete'] = False
        save_json(path, data)
    elif damage == 'added_resealed':
        path = expansion/'expansion_2020/first_opportunity_ledger.parquet'
        values = pd.read_parquet(path)
        values.loc[0, 'first_target_common_bps'] += 1
        values.to_parquet(path, index=False)
        seal(path.parent)
    elif damage == 'baseline':
        (expansion/'expansion_2020/baseline/equity.parquet').write_text('changed')
    elif damage == 'snapshot':
        path = expansion/'generation_source/wonyotti_fr/first_opportunity_collection.py'
        path.write_text(path.read_text()+'\n# 변조\n')
    elif damage == 'symlink':
        path = expansion/'expansion_2020/all_positions.parquet'
        target = tmp_path/'preserved.parquet'
        path.rename(target)
        path.symlink_to(target)
    else:
        data = json.loads(expansion_review.read_text())
        data['phases']['expansion_2020']['all_74_features_independently_recomputed'] = False
        save_json(expansion_review, data)
        inputs[-1] = sha256(expansion_review)
    calls = []
    monkeypatch.setattr(FirstLinearCloseModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_expanded_first_diagnosis(*inputs, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
