import copy
import json
from dataclasses import asdict

import pandas as pd
import pytest
from test_close_effect import CollectionPolicy, collection_fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.first_exit_reference import (
    FIRST_EXIT_BASELINES,
    FIRST_EXIT_REVIEW_CHECKS,
    FIRST_EXIT_RUN_FILES,
    checked_first_exit_reference,
)
from wonyotti_fr.first_exit_research import run_first_exit_control


def fixture(tmp_path, monkeypatch):
    bars, cfg, parent = collection_fixture(tmp_path)
    later = bars.copy()
    later[['time', 'end']] += pd.Timedelta(days=365)
    backtest(later, CollectionPolicy(), cfg, parent/'confirmation-2022')
    roots = {}
    for year in ['2021', '2022']:
        root = tmp_path/year
        root.mkdir()
        for name in ['manifest-1m.json', 'manifest-5m.json']:
            save_json(root/name, {'year': year, 'synthetic_contract': True})
        roots[year] = root
    manifests = {str(root/name): sha256(root/name) for root in roots.values() for name in ['manifest-1m.json', 'manifest-5m.json']}
    save_json(parent/'manifest.json', {'settings': {'input_manifests': manifests}})
    frozen = json.loads((parent/'frozen_selection.json').read_text())
    monkeypatch.setattr('wonyotti_fr.first_exit_reference.load_selection', lambda _: (copy.deepcopy(frozen), CollectionPolicy()))
    monkeypatch.setattr('wonyotti_fr.first_exit_research.load_selection', lambda _: (copy.deepcopy(frozen), CollectionPolicy()))
    monkeypatch.setattr('wonyotti_fr.first_exit_research.prepare_minute_period', lambda *a, **_: (bars.copy() if a[3] == '2021-01-01' else later.copy(), {'synthetic_contract': True}))
    source = tmp_path/'diagnosis'
    source.mkdir()
    calibration, selection = source/'calibration', source/'selection'
    calibration.mkdir()
    selection.mkdir()
    save_json(calibration/'files.json', {})
    folds = {}
    for n in range(3):
        folder = selection/f'fold-{n:02}'
        folder.mkdir()
        save_json(folder/'files.json', {})
        folds[folder.name] = sha256(folder/'files.json')
    save_json(selection/'folds.json', folds)
    save_json(selection/'files.json', {'folds.json': sha256(selection/'folds.json')})
    from pathlib import Path
    save_json(source/'manifest.json', {'settings': {'protocol_sha256': sha256(Path('docs/EXPERIMENT_V90.md'))}})
    save_json(source/'summary.json', {'complete': True, 'profitability_accepted': False, 'diagnosis_evaluated': False,
        'calibration_files_sha256': sha256(calibration/'files.json'), 'selection_files_sha256': sha256(selection/'files.json')})
    save_json(source/'decision.json', {'calibration_passed': False, 'candidate': 'regularized_first_linear'})
    save_json(source/'generation_manifest.json', {'settings': {'reference': str(parent), 'reference_sha256': sha256(parent/'frozen_selection.json'),
        'market': str(roots['2021']), 'features': str(roots['2021']),
        'reference_outputs_sha256': {n: sha256(parent/'candidate-00'/n) for n in ['equity.parquet', 'fills.parquet', 'trades.parquet', 'final_state.json']}}})
    (source/'manager_parent_selection.json').write_bytes((parent/'frozen_selection.json').read_bytes())
    save_json(source/'files.json', {p.name: sha256(p) for p in source.iterdir() if p.is_file()})
    proof = tmp_path/'verification.json'
    # 출처 계약만 검사하는 합성 증명이며 실제 독립 검산 결과로 사용하지 않는다.
    save_json(proof, {**dict.fromkeys(FIRST_EXIT_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(source/'files.json')})
    return source, proof, parent, cfg, roots


def test_reference_chain_and_all_four_runs_preserve_inputs_and_parent_outputs(tmp_path, monkeypatch):
    source, proof, parent, cfg, roots = fixture(tmp_path, monkeypatch)
    digest = sha256(proof)
    evidence = checked_first_exit_reference(source, proof, digest)
    assert evidence['risk'] == asdict(cfg) and evidence['inputs']['2021']['market'] == str(roots['2021'])
    out = run_first_exit_control(source, proof, digest, tmp_path/'runs')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['runs'] == 4 and summary['parent_replays_exact']
    assert summary['failed_model_applied'] is False and summary['profitability_accepted'] is False
    assert checked_first_exit_reference(source, proof, digest) == evidence
    runs = json.loads((out/'runs.json').read_text())
    assert len(runs) == 6
    for name, value in runs.items():
        assert sha256(out/name/'files.json') == value
    for year, previous in FIRST_EXIT_BASELINES.items():
        for name in FIRST_EXIT_RUN_FILES:
            if name.endswith('.parquet'):
                pd.testing.assert_frame_equal(pd.read_parquet(out/year/'parent'/name), pd.read_parquet(parent/previous/name), check_exact=True)
            else:
                assert json.loads((out/year/'parent'/name).read_text()) == json.loads((parent/previous/name).read_text())
        trace = pd.read_parquet(out/year/'control_decisions.parquet')
        assert trace.changed.any() and trace.loc[trace.changed, 'final_intent'].eq('exit').all()
        assert trace.loc[trace.changed, 'features_supported'].all()


@pytest.mark.parametrize('damage', ['proof', 'root', 'admitted', 'diagnosis', 'parent', 'market', 'baseline', 'fold'])
def test_changed_or_wrong_reference_fails_before_backtest(tmp_path, monkeypatch, damage):
    source, proof, parent, _, roots = fixture(tmp_path, monkeypatch)
    digest = sha256(proof)
    if damage == 'proof':
        proof.write_text('{}')
    elif damage == 'root':
        (source/'summary.json').write_text('{}')
    elif damage == 'admitted':
        value = json.loads(proof.read_text())
        value['calibration_passed'] = True
        save_json(proof, value)
        digest = sha256(proof)
    elif damage == 'diagnosis':
        (source/'diagnosis_used.parquet').touch()
    elif damage == 'parent':
        (parent/'frozen_selection.json').write_text('{}')
    elif damage == 'market':
        (roots['2022']/'manifest-1m.json').write_text('{}')
    elif damage == 'baseline':
        (parent/'candidate-00/final_state.json').write_text('{}')
    else:
        (source/'selection/fold-01/files.json').write_text('{"unexpected": "missing"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        checked_first_exit_reference(source, proof, digest)


def test_parent_replay_error_is_preserved_and_stops_completion(tmp_path, monkeypatch):
    source, proof, parent, _, _ = fixture(tmp_path, monkeypatch)
    value = json.loads((parent/'candidate-00/metrics.json').read_text())
    value['fees'] += 1
    save_json(parent/'candidate-00/metrics.json', value)
    with pytest.raises(ValueError, match='부모 재생'):
        run_first_exit_control(source, proof, sha256(proof), tmp_path/'runs')
    output = next((tmp_path/'runs').iterdir())
    assert (output/'failure.json').exists() and not (output/'summary.json').exists()
