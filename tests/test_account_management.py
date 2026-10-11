import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.account_management import (
    ACCOUNT_FEATURES,
    AccountManagementLinear,
    AccountManagementModel,
    account_admission,
    account_split,
    evaluation_masks,
    subset_metrics,
)
from wonyotti_fr.account_management_reference import (
    independent_account_models,
    independent_admission,
    verify_account_split,
    verify_subset,
)
from wonyotti_fr.minute_management import ACTIONS, FEATURES
from wonyotti_fr.paired_management import PairedManagementModel
from wonyotti_fr.paired_management_evaluation import choose_thresholds
from wonyotti_fr.paired_management_reference import manual_threshold
from wonyotti_fr.position_boost_verification import compare_statistics


def synthetic():
    rng = np.random.default_rng(47)
    parts = []
    for part, start, count, offset in [('training', '2020-01-03', 1200, 0),
        ('calibration', '2021-01-03', 300, 100), ('diagnosis', '2021-07-03', 300, 200)]:
        f = pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES)
        f['end'] = pd.date_range(start, periods=count, freq='5min', tz='UTC')
        f['label_end'] = f.end+pd.Timedelta(minutes=5)
        f['direction'] = np.where(f.ret_5m > 0, 1, -1)
        f['state_episode_id'] = np.arange(count)//60+1+offset
        f['paired_part'] = part
        f['paired_usable'] = f['account_supported'] = f['account_usable'] = True
        f['account_reason'] = 'included'
        f['futures_leverage'] = rng.uniform(.001, 3, count)
        f['log_futures_leverage'] = np.log1p(f.futures_leverage)
        f['account_band'] = pd.cut(f.futures_leverage, [-np.inf, 0, .5, 1, 2, 5, np.inf],
            labels=['zero', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,5]', '(5,inf)']).astype('str')
        for i, action in enumerate(ACTIONS):
            f['all_y_'+action] = (f.iloc[:, i]+.2*f.log_futures_leverage) > .1
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def source_summary(frame):
    records = []
    for part in ('training', 'calibration', 'diagnosis'):
        f = frame.loc[frame.paired_part.eq(part) & frame.account_usable]
        records.append({'kind': 'split', 'group': part, 'joint_usable': {'rows': len(f),
            'episodes': int(f.state_episode_id.nunique()), 'positive': {a: int(f['all_y_'+a].sum()) for a in ACTIONS}}})
    return {'results': {'groups': records}}


def test_joint_support_preserves_original_rows_and_excludes_unavailable_accounts():
    frame = synthetic()
    frame.loc[0, ['account_usable', 'account_supported']] = False
    frame.loc[0, 'account_reason'] = 'uncertain_external'
    frame.loc[0, 'log_futures_leverage'] = np.nan
    original = frame.copy(deep=True)
    rows, support = account_split(frame)
    verify_account_split(frame, rows, support, source_summary(frame))
    assert support['supported'] and len(rows['training']) == 1199
    assert len(rows['calibration']) == len(rows['diagnosis']) == 300
    pd.testing.assert_frame_equal(original, frame, check_exact=True)


@pytest.mark.parametrize('problem', ['inconsistent_support', 'overlap_episode', 'bad_period', 'invalid_log',
    'bad_direction', 'bad_feature', 'bad_label', 'bad_boundary', 'duplicate', 'negative_size'])
def test_invalid_or_leaking_split_rejected(problem):
    f = synthetic()
    if problem == 'inconsistent_support':
        f.loc[0, 'account_usable'] = False
    elif problem == 'overlap_episode':
        f.loc[1200, 'state_episode_id'] = f.state_episode_id.iloc[0]
    elif problem == 'bad_period':
        f.loc[0, 'paired_part'] = 'calibration'
    elif problem == 'invalid_log':
        f.loc[0, 'log_futures_leverage'] += .01
    elif problem == 'bad_direction':
        f.loc[0, 'direction'] = 0
    elif problem == 'bad_feature':
        f.loc[0, FEATURES[0]] = np.nan
    elif problem == 'bad_label':
        f.all_y_exit = f.all_y_exit.astype(int)
        f.loc[0, 'all_y_exit'] = 2
    elif problem == 'bad_boundary':
        f.loc[0, 'label_end'] += pd.Timedelta(seconds=1)
    elif problem == 'duplicate':
        f.loc[1, 'end'] = f.end.iloc[0]
    else:
        f.loc[0, 'futures_leverage'] = -1
    with pytest.raises(ValueError):
        account_split(f)


def test_future_feature_and_label_changes_cannot_change_training_rows():
    source = synthetic()
    before, support = account_split(source)
    source.loc[1500:, 'ret_5m'] += 10
    source.loc[1500:, 'all_y_exit'] = ~source.loc[1500:, 'all_y_exit']
    after, _ = account_split(source)
    pd.testing.assert_frame_equal(before['training'], after['training'], check_exact=True)
    pd.testing.assert_frame_equal(before['calibration'], after['calibration'], check_exact=True)
    assert support['supported']


def test_class_support_failure_does_not_silently_train():
    frame = synthetic()
    frame.loc[frame.paired_part.eq('diagnosis'), 'all_y_exit'] = False
    _, support = account_split(frame)
    assert not support['supported']
    assert not support['periods']['diagnosis']['classes_supported']


@pytest.fixture(scope='module')
def fitted():
    frame = synthetic()
    rows, _ = account_split(frame)
    train, cal = rows['training'].copy(), rows['calibration'].copy()
    validation = pd.concat([cal, rows['diagnosis']], ignore_index=True)
    for action in ACTIONS:
        train['y_'+action], cal['y_'+action] = train['all_y_'+action], cal['all_y_'+action]
    x = np.ascontiguousarray(train[ACCOUNT_FEATURES].to_numpy(dtype=float))
    y = train[['y_'+a for a in ACTIONS]].to_numpy(dtype=int)
    vx = np.ascontiguousarray(validation[ACCOUNT_FEATURES].to_numpy(dtype=float))
    vy = validation[['all_y_'+a for a in ACTIONS]].to_numpy(dtype=int)
    candidate, _ = AccountManagementModel.fit(x, y, vx)
    baseline, _ = PairedManagementModel.fit(np.ascontiguousarray(x[:, :32]), y, np.ascontiguousarray(vx[:, :32]))
    linear, _, _ = AccountManagementLinear.fit(train, cal)
    models = {'account': candidate, 'baseline': baseline, 'linear': linear}
    predictions = {name: model.probabilities(np.ascontiguousarray(vx[:, :len(model.features)])) for name, model in models.items()}
    predictions['constant'] = np.tile(y.mean(axis=0), (len(vx), 1))
    return frame, validation, x, y, vx, vy, models, predictions


def test_full_independent_models_and_scalar_inference(fitted):
    _, _, x, y, vx, _, models, predictions = fitted
    references = independent_account_models({k: v.to_dict() for k, v in models.items()}, x, y, vx)
    for name in predictions:
        np.testing.assert_allclose(predictions[name], references[name], atol=1e-12, rtol=0)
    for name, model in models.items():
        assert model.to_dict() == type(model).from_dict(model.to_dict()).to_dict()
        scalar = np.vstack([model.probabilities(row[None, :len(model.features)]) for row in vx])
        np.testing.assert_allclose(scalar, predictions[name], atol=1e-12, rtol=0)
    assert models['account'].features == [*FEATURES, 'log_futures_leverage']


def test_all_subset_metrics_empty_bands_and_independent_admission(fitted):
    _, validation, _, _, _, y, _, predictions = fitted
    thresholds, details = choose_thresholds(y[:300], {k: v[:300] for k, v in predictions.items()})
    for name, scores in predictions.items():
        for i, a in enumerate(ACTIONS):
            threshold, detail = manual_threshold(y[:300, i], scores[:300, i], (2., 1., .5)[i])
            assert threshold == thresholds[name][a]
            compare_statistics(details[name][a], detail)
    empty = 0
    for key, mask in evaluation_masks(validation):
        result, decisions = subset_metrics(validation, mask, y, predictions, thresholds)
        verify_subset(validation, key, mask, result, decisions, y, predictions, thresholds)
        if result['rows'] == 0:
            empty += 1
            assert result['metrics'] is None and decisions is None
        if key == {'kind': 'period', 'group': 'diagnosis', 'band': 'all'}:
            assert account_admission(result['metrics']) == independent_admission(result['metrics'])
            assert len(account_admission(result['metrics'])['checks']) == 17
    assert empty > 0


def test_no_positive_subgroup_does_not_invent_average_precision(fitted):
    _, validation, _, _, _, y, _, predictions = fitted
    thresholds, _ = choose_thresholds(y[:300], {k: v[:300] for k, v in predictions.items()})
    frame, labels = validation.copy(), y.copy()
    frame['all_y_exit'] = False
    labels[:, 0] = 0
    key = {'kind': 'all', 'group': 'all', 'band': 'all'}
    mask = np.ones(len(frame), dtype=bool)
    result, decisions = subset_metrics(frame, mask, labels, predictions, thresholds)
    verify_subset(frame, key, mask, result, decisions, labels, predictions, thresholds)
    for metrics in result['metrics']['exit'].values():
        assert metrics['average_precision'] is None and metrics['true_positive'] == 0


@pytest.mark.parametrize('name', ['account', 'linear'])
@pytest.mark.parametrize('mutation', ['feature_order', 'format', 'nan', 'dimensions'])
def test_model_schema_and_invalid_inference(fitted, name, mutation):
    _, _, _, _, vx, _, models, _ = fitted
    model = models[name]
    data = copy.deepcopy(model.to_dict())
    if mutation == 'feature_order':
        data['features'][0], data['features'][-1] = data['features'][-1], data['features'][0]
    elif mutation == 'format':
        data['format'] = 'paired_five_minute_management_v1'
    elif mutation == 'nan':
        sample = vx[:1].copy()
        sample[0, -1] = np.nan
        assert np.isnan(model.probabilities(sample)).all()
        return
    else:
        with pytest.raises(ValueError):
            model.probabilities(vx[:, :32])
        return
    with pytest.raises(ValueError):
        type(model).from_dict(data)


def test_independent_group_mask_rejects_wrong_boundary(fitted):
    _, frame, _, _, _, y, _, predictions = fitted
    thresholds, _ = choose_thresholds(y[:300], {k: v[:300] for k, v in predictions.items()})
    key, mask = next(evaluation_masks(frame))
    result, decisions = subset_metrics(frame, mask, y, predictions, thresholds)
    mask[0] = False
    with pytest.raises(AssertionError):
        verify_subset(frame, key, mask, result, decisions, y, predictions, thresholds)


def test_public_runner_serializes_verified_models_scores_groups_and_seal(tmp_path, monkeypatch):
    from wonyotti_fr import account_management_research as research
    from wonyotti_fr.common import save_json, sha256
    from wonyotti_fr.holding_stress import read_json, seal

    frame = synthetic()
    parent = tmp_path/'parent'
    parent.mkdir()
    frame.to_parquet(parent/'all_windows.parquet', index=False)
    summary = {'complete': True, 'profitability_accepted': False, **source_summary(frame)}
    save_json(parent/'summary.json', summary)
    seal(parent)
    marker = tmp_path/'proof.json'
    save_json(marker, {'complete': True, 'profitability_accepted': False, 'run': str(parent),
        'files_sha256': sha256(parent/'files.json'), 'summary_sha256': sha256(parent/'summary.json')})
    reference, protocol = tmp_path/'reference.json', tmp_path/'protocol.md'
    reference.write_text('{}')
    protocol.write_text('합성 파이프라인 검사')
    monkeypatch.setattr(research, 'checked_reference', lambda *args: {'sha256': {}})
    root = research.run(reference, sha256(reference), marker, sha256(marker), protocol, tmp_path, progress=lambda _: None)
    report = read_json(root/'summary.json')
    assert report['complete'] and not report['profitability_accepted']
    assert report['verification']['prediction_rows_verified'] == 600
    assert report['verification']['independent_trees_refitted'] == 384
    assert report['verification']['groups_verified'] == 35
    assert len(read_json(root/'decision.json')['checks']) == 17
    assert len(pd.read_parquet(root/'predictions.parquet')) == 600
    assert len(read_json(root/'models.json')['account']['features']) == 33
    for name, digest in read_json(root/'files.json').items():
        assert sha256(root/name) == digest
