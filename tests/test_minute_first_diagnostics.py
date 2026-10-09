import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from test_first_opportunity_diagnostics import seal
from test_managed_first_diagnostics import synthetic_parent_loader
from test_minute_first_inputs import minute_bars
from test_probability_first_diagnostics import probability_source_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.managed_first_parent import checked_manager_parent
from wonyotti_fr.minute_first_diagnostics import run_minute_first_diagnosis
from wonyotti_fr.minute_first_inputs import MinuteFirstLinearModel
from wonyotti_fr.minute_first_reference import MINUTE_FIRST_REVIEW_CHECKS
from wonyotti_fr.probability_first_diagnostics import run_probability_first_diagnosis


def minute_source_fixture(path):
    inputs = probability_source_fixture(path/'previous')
    with patch('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader):
        source = run_probability_first_diagnosis(*inputs, path/'probability')
    manager = json.loads((source/'manager_evidence.json').read_text())
    market, generation, linear, tree, managed = [path/name for name in ['market', 'generation', 'linear', 'tree', 'managed']]
    for folder in [market, generation, linear, tree, managed]:
        folder.mkdir()
    training, calibration = [pd.read_parquet(source/('managed_first_'+name+'.parquet')) for name in ['training', 'calibration']]
    bars = minute_bars([*training.decision_time, *calibration.decision_time])
    bars.to_parquet(market/'BTCUSDT-1m.parquet', index=False)
    pd.DataFrame({'time': pd.Series([], dtype='datetime64[ns, UTC]'), 'rate': pd.Series([], dtype=float)}).to_parquet(market/'funding.parquet', index=False)
    save_json(market/'manifest-1m.json', {'interval': '1m', 'summary': {'BTCUSDT': {
        'klines': {'file': 'BTCUSDT-1m.parquet', 'sha256': sha256(market/'BTCUSDT-1m.parquet')},
        'fundingRate': {'file': 'funding.parquet', 'sha256': sha256(market/'funding.parquet')}}}})
    save_json(generation/'manifest.json', {'settings': {'reference': manager['parent'], 'reference_sha256': manager['parent_sha256'],
        'market': str(market), 'market_manifest_sha256': sha256(market/'manifest-1m.json')}})
    save_json(linear/'manifest.json', {'settings': {'expansion': str(generation)}})
    save_json(linear/'expansion_evidence.json', {'expansion': str(generation), 'files': {'manifest.json': sha256(generation/'manifest.json')}})
    save_json(tree/'reference_evidence.json', {'reference': str(linear), 'files': {name: sha256(linear/name)
        for name in ['manifest.json', 'expansion_evidence.json']}})
    with patch('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader):
        _, parent = checked_manager_parent(tree)
    save_json(managed/'reference_evidence.json', {'reference': str(tree), 'files': {'reference_evidence.json': sha256(tree/'reference_evidence.json')}})
    save_json(managed/'manager_evidence.json', parent)
    # 별도 합성 연결에서 시세·부모 소비 계약을 검사하며 과거 실제 검산으로 표시하지 않는다.
    save_json(source/'reference_evidence.json', {'reference': str(managed), 'files': {name: sha256(managed/name)
        for name in ['reference_evidence.json', 'manager_evidence.json']}})
    save_json(source/'manager_evidence.json', parent)
    for name, item in parent['metadata'].items():
        (source/name).write_bytes(Path(item['path']).read_bytes())
    seal(source)
    review = path/'synthetic-probability-verification.json'
    save_json(review, {**dict.fromkeys(MINUTE_FIRST_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, review, sha256(review)


def test_complete79_inputs_pipeline_all_original_rows_and_no_final_diagnosis(tmp_path, monkeypatch):
    inputs = minute_source_fixture(tmp_path/'source')
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    original = pd.read_parquet

    def no_final(path, *args, **kwargs):
        assert str(path).split('/')[-1] != 'diagnosis_used.parquet'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', no_final)
    result = run_minute_first_diagnosis(*inputs, tmp_path/'run')
    summary = json.loads((result/'summary.json').read_text())
    assert summary['complete'] and summary['new_models_fitted'] == 1 and len(summary['checks']) == 14
    assert not summary['diagnosis_evaluated'] and not summary['profitability_accepted'] and not summary['previous_models_refitted']
    assert len(json.loads((result/'model.json').read_text())['features']) == 79
    assert len(json.loads((result/'calibration/first_metrics.json').read_text())) == 17
    old = pd.read_parquet(inputs[0]/'calibration/predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(result/'calibration/predictions.parquet')[old.columns], old, check_exact=True)
    for phase in ['training', 'calibration']:
        before = pd.read_parquet(inputs[0]/('managed_first_'+phase+'.parquet'))
        after = pd.read_parquet(result/('minute_first_'+phase+'.parquet'))
        pd.testing.assert_frame_equal(after[before.columns], before, check_exact=True)
    assert all(sha256(result/name) == digest for name, digest in json.loads((result/'files.json').read_text()).items())


@pytest.mark.parametrize('damage', ['proof', 'training', 'market_manifest', 'market_parquet', 'market_path_escape', 'parent_chain'])
def test_changed_market_or_prior_evidence_rejected_before_fit(tmp_path, monkeypatch, damage):
    source, review, digest = minute_source_fixture(tmp_path/'source')
    monkeypatch.setattr('wonyotti_fr.managed_first_parent.load_selection', synthetic_parent_loader)
    generation = json.loads((source/'generation_manifest.json').read_text())['settings']
    market = Path(generation['market'])
    if damage == 'market_path_escape':
        path = market/'BTCUSDT-1m.parquet'
        moved = tmp_path/'outside.parquet'
        path.rename(moved)
        path.symlink_to(moved)
    else:
        path = {'proof': review, 'training': source/'managed_first_training.parquet', 'market_manifest': market/'manifest-1m.json',
            'market_parquet': market/'BTCUSDT-1m.parquet', 'parent_chain': tmp_path/'source/managed/reference_evidence.json'}[damage]
        path.write_bytes(path.read_bytes()+b' ')
    calls = []
    monkeypatch.setattr(MinuteFirstLinearModel, 'fit', lambda *_: calls.append(True))
    with pytest.raises((ValueError, AssertionError)):
        run_minute_first_diagnosis(source, review, digest, tmp_path/'run')
    assert not calls and not (tmp_path/'run').exists()
