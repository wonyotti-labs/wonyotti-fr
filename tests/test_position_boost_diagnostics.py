from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.activity_diagnostics import ACTIVITY_PERIODS
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.position_boost_diagnostics import run_position_boost
from wonyotti_fr.position_boost_reference import (
    POSITION_REVIEW_CHECKS,
    checked_position_reference,
    position_baseline,
    position_comparisons,
    read,
)
from wonyotti_fr.position_boost_verification import verify_position_boost
from wonyotti_fr.position_target import PositionTargetModel, position_admission
from wonyotti_fr.retained_weight_close_reference import checked_files


def synthetic_position_source(folder):
    source, review = folder/'source', folder/'review'
    source.mkdir(parents=True)
    review.mkdir()
    rng = np.random.default_rng(944)
    train_x = rng.normal(size=(1200, len(MARKET_FEATURES)))
    validation_x = np.repeat(rng.normal(size=(80, len(MARKET_FEATURES))), 3, axis=0)
    parts = {}
    for name, year, matrix, offset in [('training', 2020, train_x, 0), ('diagnosis', 2021, validation_x, 10000)]:
        count = len(matrix)
        per_month = count//12
        times = pd.DatetimeIndex(np.concatenate([pd.date_range(f'{year}-{month:02}-02',
            periods=per_month, freq='5min', tz='UTC').to_numpy() for month in range(1, 13)]))
        labels = np.tile(np.arange(3), count//3)
        frame = pd.DataFrame(matrix, columns=MARKET_FEATURES)
        frame['end'], frame['label_end'] = times, times + pd.Timedelta(minutes=5)
        frame['episode_id'] = np.arange(1, count+1) + offset
        frame['position_target'] = labels
        frame['position_target_episode_id'] = np.where(labels == 0, 0, frame.episode_id)
        frame['position_target_time'] = frame.label_end-pd.Timedelta(minutes=1)
        frame.to_parquet(source/f'{name}_used.parquet', index=False)
        frame[['end', 'label_end', 'episode_id', 'position_target_episode_id', 'position_target_time', 'position_target']].assign(
            reason='included').to_parquet(source/f'{name}_ledger.parquet', index=False)
        frame.groupby('position_target_episode_id').agg(rows=('end', 'size'), first_end=('end', 'min'),
            last_end=('end', 'max'), target=('position_target', 'first'), target_classes=('position_target', 'nunique')).reset_index().to_parquet(
                source/f'{name}_position_support.parquet', index=False)
        parts[name] = frame
    train, validation = parts.values()
    model, support = PositionTargetModel.fit(train_x, train.position_target.to_numpy())
    frequencies = np.bincount(train.position_target, minlength=3)/len(train)
    support['training_frequencies'] = frequencies.tolist()
    predictions, metrics, monthly = position_comparisons(validation, {
        'position': model.probabilities(validation_x), 'constant': np.tile(frequencies, (len(validation_x), 1))})
    decision = position_admission(metrics, monthly)
    assert not decision['position_target_admitted']
    for name, data in [('model', model.to_dict()), ('training_support', support), ('metrics', metrics),
        ('monthly', monthly), ('decision', decision), ('summary', {'complete': True,
            'rows': {key: len(value) for key, value in parts.items()}, **decision})]:
        save_json(source/f'{name}.json', data)
    predictions.to_parquet(source/'predictions.parquet', index=False)
    save_json(source/'manifest.json', {'settings': {'protocol_sha256': sha256(Path('docs/EXPERIMENT_V44.md')),
        'periods': ACTIVITY_PERIODS, 'new_model_count': 1, 'request_threshold': .65, 'trading_returns_evaluated': False}})
    (source/'REPORT.md').write_text('합성 계약 검사 자료. 실제 원본 검산 근거 아님.\n')
    seal(source)
    # 이전 원본 검산의 데이터 계약만 모사하며 실제 원본 검증으로 집계하지 않는다.
    save_json(review/'verification.json', {**dict.fromkeys(POSITION_REVIEW_CHECKS, True),
        'original_files_unchanged': 5, 'position_target_admitted': False, 'profitability_accepted': False,
        'synthetic_contract_fixture': True})
    save_json(review/'manifest.json', {'settings': {'diagnosis': str(source), 'files_sha256': sha256(source/'files.json')}})
    (review/'reproduce.py').write_text('# 합성 출처 계약의 설명용 파일\n')
    return write_reference(folder/'reference.json', source, review)


def write_reference(path, source, review):
    save_json(path, {'source': str(source), 'source_files_sha256': sha256(source/'files.json'), 'review': str(review),
        'review_sha256': {name: sha256(review/name) for name in ['manifest.json', 'verification.json', 'reproduce.py']},
        'profitability_accepted': False, 'all_source_periods_already_observed': True})
    return path


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    return synthetic_position_source(tmp_path_factory.mktemp('position_boost_source'))


def copied_reference(fixture, folder):
    spec = read(fixture)
    source, review = folder/'source', folder/'review'
    shutil.copytree(spec['source'], source)
    shutil.copytree(spec['review'], review)
    save_json(review/'manifest.json', {'settings': {'diagnosis': str(source), 'files_sha256': sha256(source/'files.json')}})
    return write_reference(folder/'reference.json', source, review)


def test_entire_synthetic_pipeline_preserves_baselines_and_rows(fixture, tmp_path):
    spec, files = checked_position_reference(fixture, sha256(fixture))
    out = run_position_boost(fixture, sha256(fixture), tmp_path/'results')
    proof = read(out/'summary.json')
    assert proof['complete'] and proof['training_rows'] == 1200 and proof['diagnosis_rows'] == 240
    assert not proof['profitability_accepted'] and not proof['trading_returns_evaluated']
    assert proof['prior_states_and_ledgers_verification_reused']
    assert read(out/'baseline_files.json') == files
    for name, digest in files.items():
        assert sha256(out/'baseline_source'/name) == digest
    checked_files(out)
    assert read(out/'reference_review.json')['synthetic_contract_fixture']
    assert checked_position_reference(fixture, sha256(fixture)) == (spec, files)


@pytest.mark.parametrize('change', ['caller', 'unsealed_data', 'review_link', 'missing_proof', 'old_plan', 'old_period'])
def test_reference_tampering_rejected(fixture, tmp_path, change):
    reference = copied_reference(fixture, tmp_path)
    spec = read(reference)
    source, review = Path(spec['source']), Path(spec['review'])
    expected = sha256(reference)
    if change == 'caller':
        expected = '0'*64
    elif change == 'unsealed_data':
        (source/'REPORT.md').write_text('changed')
    elif change in ['review_link', 'missing_proof']:
        name = 'manifest' if change == 'review_link' else 'verification'
        data = read(review/f'{name}.json')
        if change == 'review_link':
            data['settings']['files_sha256'] = '0'*64
        else:
            data[POSITION_REVIEW_CHECKS[1]] = False
        save_json(review/f'{name}.json', data)
        write_reference(reference, source, review)
        expected = sha256(reference)
    else:
        data = read(source/'manifest.json')
        if change == 'old_plan':
            data['settings']['protocol_sha256'] = '0'*64
        else:
            data['settings']['periods']['training'][0] = '2019-01-01'
        save_json(source/'manifest.json', data)
        seal(source)
        save_json(review/'manifest.json', {'settings': {'diagnosis': str(source), 'files_sha256': sha256(source/'files.json')}})
        write_reference(reference, source, review)
        expected = sha256(reference)
    with pytest.raises(ValueError):
        checked_position_reference(reference, expected)


@pytest.mark.parametrize('change', ['score', 'metric', 'month', 'position_overlap', 'time_boundary'])
def test_baseline_recalculation_rejects_resealed_data(fixture, tmp_path, change):
    reference = copied_reference(fixture, tmp_path)
    source = Path(read(reference)['source'])
    if change == 'score':
        p = pd.read_parquet(source/'predictions.parquet')
        p.loc[0, 'position_flat'] += .01
        p.to_parquet(source/'predictions.parquet', index=False)
    elif change in ['metric', 'month']:
        name = 'metrics' if change == 'metric' else 'monthly'
        data = read(source/f'{name}.json')
        if change == 'metric':
            data['position']['log_loss'] += .01
        else:
            data[0]['position_log_loss'] += .01
        save_json(source/f'{name}.json', data)
    else:
        frame = pd.read_parquet(source/'diagnosis_used.parquet')
        if change == 'position_overlap':
            frame.loc[0, 'episode_id'] = 1
        else:
            frame.loc[0, 'end'] = pd.Timestamp('2021-01-01', tz='UTC')
            frame.loc[0, 'label_end'] = frame.loc[0, 'end']+pd.Timedelta(minutes=5)
        frame.to_parquet(source/'diagnosis_used.parquet', index=False)
    seal(source)
    with pytest.raises((ValueError, AssertionError)):
        position_baseline(source)


@pytest.fixture(scope='module')
def diagnosis(fixture, tmp_path_factory):
    return run_position_boost(fixture, sha256(fixture), tmp_path_factory.mktemp('position_boost_audit'))


def test_independent_full_multiclass_audit(diagnosis, tmp_path):
    result = verify_position_boost(diagnosis, sha256(diagnosis/'files.json'), tmp_path)
    proof = read(result/'verification.json')
    assert proof['complete'] and proof['independent_192_trees_and_baselines_exact']
    assert proof['all_predictions_metrics_months_and_decision_verified']
    assert not proof['profitability_accepted']
    checked_files(result)


@pytest.mark.parametrize('change', ['model', 'score', 'metric', 'month', 'decision', 'baseline_copy', 'missing_output', 'code_snapshot'])
def test_resealed_outputs_rejected_by_independent_audit(diagnosis, tmp_path, change):
    source = tmp_path/'tampered'
    shutil.copytree(diagnosis, source)
    if change == 'baseline_copy':
        (source/'baseline_source'/'REPORT.md').write_text('changed')
    elif change == 'code_snapshot':
        (source/'code_snapshot'/'event_features.py').write_text('changed')
    elif change == 'missing_output':
        (source/'predictions.parquet').unlink()
    elif change == 'score':
        predictions = pd.read_parquet(source/'predictions.parquet')
        predictions.loc[0, 'boosted_long'] += .001
        predictions.to_parquet(source/'predictions.parquet', index=False)
    else:
        name = {'model': 'model', 'metric': 'metrics', 'month': 'monthly', 'decision': 'decision'}[change]
        data = read(source/f'{name}.json')
        if change == 'model':
            data['models'][0]['baseline'] += .001
        elif change == 'metric':
            data['boosted']['long']['roc_auc'] += .01
        elif change == 'month':
            data[0]['boosted_log_loss'] += .01
        else:
            data['position_boost_admitted'] = not data['position_boost_admitted']
        save_json(source/f'{name}.json', data)
    seal(source)
    with pytest.raises((ValueError, AssertionError)):
        verify_position_boost(source, sha256(source/'files.json'), tmp_path/'review')
