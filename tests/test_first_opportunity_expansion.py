import copy
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from test_first_opportunity_close import setup
from test_first_opportunity_collection import execute, fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_linear_evaluation import LINEAR_POLICIES
from wonyotti_fr.first_opportunity_close import first_opportunity_rows
from wonyotti_fr.first_opportunity_expansion import (
    BASELINE_FILES,
    run_first_opportunity_expansion,
    sealed_phase,
    time_frame,
    verify_first_expansion_parity,
)
from wonyotti_fr.weekly_first_linear_reference import (
    LINEAR_CALIBRATION_FILES,
    LINEAR_FAILURE_FILES,
    WEEKLY_FIRST_REVIEW_CHECKS,
)


def parity_examples(path):
    path.mkdir()
    training, _, calibration = setup()
    labels, populations = [], []
    for name, frame in [('early_training', training), ('calibration', calibration)]:
        frame.to_parquet(path/f'{name}_used.parquet', index=False)
        first, positions = first_opportunity_rows(frame)
        first['label_status'] = 'closed'
        population = positions.copy()
        population['management_rows'] = 3
        population['available_rows'] = 3
        population['eligible_rows'] = population.has_eligible_opportunity.astype(int)*3
        ends = frame.groupby('position_entry_time').label_end.first()
        population['natural_exit_time'] = population.position_entry_time.map(ends)-pd.Timedelta(minutes=1)
        population['natural_exit_reason'] = 'signal_exit'
        labels.append(first)
        populations.append(population)
    save_json(path/'files.json', {p.name: sha256(p) for p in path.iterdir()})
    return pd.concat(labels, ignore_index=True), pd.concat(populations, ignore_index=True)


def test_2021_first_rows_all_features_cash_and_population_match_without_diagnosis(tmp_path, monkeypatch):
    reference = tmp_path/'reference'
    labels, positions = parity_examples(reference)
    read = pd.read_parquet

    def only_first_splits(path, *args, **kwargs):
        assert str(path).endswith(('early_training_used.parquet', 'calibration_used.parquet'))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', only_first_splits)
    result = verify_first_expansion_parity(labels, positions, reference)
    assert result['complete'] and not result['diagnosis_model_predictions_generated']
    assert result['splits']['early_training']['all_positions'] == 600
    assert result['splits']['early_training']['first_eligible'] == 540
    extra = positions.iloc[:1].copy()
    extra['position_entry_time'] += pd.Timedelta(minutes=1)
    extra['available_rows'] = 0
    assert verify_first_expansion_parity(labels, pd.concat([positions, extra]), reference) == result


@pytest.mark.parametrize('damage', ['feature', 'cash', 'first_time', 'equity', 'missing_first', 'eligibility', 'natural_end'])
def test_first_parity_rejects_feature_target_anchor_and_population_changes(tmp_path, damage):
    reference = tmp_path/'reference'
    labels, positions = parity_examples(reference)
    if damage == 'feature':
        labels.loc[0, 'ret_5m'] += 1e-8
    elif damage == 'cash':
        labels.loc[0, 'close_advantage_pnl'] += 1e-8
    elif damage == 'first_time':
        positions['first_available_time'] = positions.first_available_time.astype('datetime64[ns, UTC]')
        positions.loc[1, 'first_available_time'] += pd.Timedelta(nanoseconds=1)
    elif damage == 'equity':
        positions.loc[1, 'reference_equity'] += 1e-8
    elif damage == 'missing_first':
        labels = labels.iloc[1:]
    elif damage == 'eligibility':
        positions.loc[0, 'has_eligible_opportunity'] = True
    else:
        positions.loc[1, 'natural_exit_time'] = pd.Timestamp('2021-07-31', tz='UTC')
    with pytest.raises((ValueError, AssertionError)):
        verify_first_expansion_parity(labels, positions, reference)


def pipeline_fixture(path, monkeypatch):
    from test_close_effect import CollectionPolicy

    import wonyotti_fr.first_opportunity_expansion as module
    from wonyotti_fr.common import new_run

    data = fixture(path/'actual')
    bars, cfg, baseline, context = data
    rows, positions, _, complete = execute(path/'actual', data)
    assert complete
    # 새 수집 경로는 실제로 실행하고 이전 검산의 출처 계약만 합성 자료로 대체한다.
    selection = path/'selection'
    selection.mkdir()
    for name in BASELINE_FILES:
        (selection/'candidate-00').mkdir(exist_ok=True)
        (selection/'candidate-00'/name).write_bytes((baseline/name).read_bytes())
    frozen = {'protocol': 'exit_move_v54', 'risk': asdict(cfg)}
    save_json(selection/'frozen_selection.json', frozen)
    reference = new_run(path, 'synthetic-linear', {'policies': LINEAR_POLICIES, 'score_rows': 'first_eligible_only'})
    for name in LINEAR_FAILURE_FILES-{'manifest.json'}:
        (reference/name).write_text('synthetic earlier attestation binding\n')
    folder = reference/'calibration'
    folder.mkdir()
    for name in LINEAR_CALIBRATION_FILES:
        (folder/name).write_text('synthetic earlier policy\n')
    save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir()})
    save_json(reference/'summary.json', {'complete': True, 'profitability_accepted': False, 'diagnosis_evaluated': False,
        'calibration_files_sha256': sha256(folder/'files.json')})
    save_json(reference/'decision.json', {'candidate': 'first_linear', 'calibration_passed': False,
        'threshold_search': False, 'fallback_used': False, 'threshold': .5})
    save_json(reference/'files.json', {p.name: sha256(p) for p in reference.iterdir() if p.is_file()})
    review = path/'synthetic-verification.json'
    save_json(review, {**dict.fromkeys(WEEKLY_FIRST_REVIEW_CHECKS, True), 'synthetic_only': False,
        'calibration_passed': False, 'profitability_accepted': False, 'source_files_sha256': sha256(reference/'files.json')})
    market, features = path/'market', path/'features'
    market.mkdir()
    features.mkdir()
    save_json(market/'manifest-1m.json', {})
    save_json(features/'manifest-5m.json', {})
    monkeypatch.setattr(module, 'load_selection', lambda _path: (copy.deepcopy(frozen), CollectionPolicy()))
    monkeypatch.setattr(module, 'prepare_minute_period', lambda *_args, **_kwargs: (bars.copy(), {'synthetic': True}))
    monkeypatch.setattr(module, 'load_market', lambda *_: (bars.copy(), None))
    monkeypatch.setattr(module, 'first_context_lookup', lambda _: copy.deepcopy(context))
    expected = time_frame(pd.DataFrame(rows), ['decision_time', 'position_entry_time', 'first_available_time', 'label_end', 'continue_end'])
    expected.loc[expected.label_status.eq('right_censored'), 'label_end'] = pd.Timestamp('2021-12-31', tz='UTC')
    expected_population = time_frame(pd.DataFrame(positions), ['position_entry_time', 'first_available_time', 'first_eligible_time', 'natural_exit_time'])

    def synthetic_parity(first, population, original):
        assert original == reference
        pd.testing.assert_frame_equal(first[expected.columns], expected, check_exact=True)
        pd.testing.assert_frame_equal(population, expected_population, check_exact=True)
        return {'complete': True, 'synthetic_first_collection_exact': True}

    monkeypatch.setattr(module, 'verify_first_expansion_parity', synthetic_parity)
    return selection, reference, review, sha256(review), market, features


def test_two_phase_generation_partial_resume_complete_and_immutable_sources(tmp_path, monkeypatch):
    args = pipeline_fixture(tmp_path, monkeypatch)
    before = {str(p): sha256(p) for folder in [args[0], args[1]] for p in folder.rglob('*') if p.is_file()}
    root = run_first_opportunity_expansion(*args, tmp_path/'output', max_opportunities=1)
    summary = json.loads((root/'summary.json').read_text())
    assert not summary['complete'] and set(summary['phases']) == {'parity_2021'}
    assert not (root/'expansion_2020').exists()
    run_first_opportunity_expansion(*args, tmp_path/'output', resume=root)
    summary = json.loads((root/'summary.json').read_text())
    assert summary['complete'] and not summary['new_models_fitted'] and not summary['profitability_accepted']
    for phase in ['parity_2021', 'expansion_2020']:
        assert sealed_phase(root/phase)['complete']
        assert len(list((root/phase).glob('baseline-*'))) == int(phase == 'expansion_2020')
    assert len(list((root/'parity_2021').glob('replay-*'))) == 2
    assert all(sha256(Path(name)) == value for name, value in before.items())
    with pytest.raises(ValueError, match='완료 실행'):
        run_first_opportunity_expansion(*args, tmp_path/'output', resume=root)


def test_first_parity_failure_preserved_and_blocks_2020_generation(tmp_path, monkeypatch):
    import wonyotti_fr.first_opportunity_expansion as module
    args = pipeline_fixture(tmp_path, monkeypatch)

    def fail(*_):
        raise ValueError('합성 첫 기회 동일성 실패')

    monkeypatch.setattr(module, 'verify_first_expansion_parity', fail)
    with pytest.raises(ValueError, match='동일성'):
        run_first_opportunity_expansion(*args, tmp_path/'output')
    root = next((tmp_path/'output').iterdir())
    assert list(root.glob('failure-*.json')) and not (root/'expansion_2020').exists()


def test_source_attestation_and_partial_resume_changes_rejected(tmp_path, monkeypatch):
    args = pipeline_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        run_first_opportunity_expansion(*args[:3], '0'*64, *args[4:], tmp_path/'output')
    assert not (tmp_path/'output').exists()
    root = run_first_opportunity_expansion(*args, tmp_path/'output', max_opportunities=1)
    save_json(args[4]/'manifest-1m.json', {'changed': True})
    with pytest.raises(ValueError, match='변경 입력'):
        run_first_opportunity_expansion(*args, tmp_path/'output', resume=root)
