import copy
import json

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.sparse_exposure import SparseExposureClassifier, action_labels
from wonyotti_fr.sparse_exposure_evaluation import classification_metrics
from wonyotti_fr.sparse_exposure_reference import reference_classification, verify_metrics
from wonyotti_fr.sparse_history import (
    HISTORY_FEATURES,
    HistoryClassifier,
    admission,
    history_features,
    linear_baseline,
    training_support,
)
from wonyotti_fr.sparse_history_reference import (
    independent_refit,
    reference_admission,
    verify_history,
)


def rows(n=60):
    end = pd.date_range('2020-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame({'end': end, 'label_end': end+pd.Timedelta(minutes=5),
        'decomposition_supported': True, 'execution_change': np.zeros(n)})


def test_exact_60minute_window_and_signed_counts():
    frame = rows()
    frame.loc[[0, 5, 11, 12], 'execution_change'] = [-.2, .3, -.4, 100.]
    result = history_features(frame, frame)
    verify_history(frame, frame, result)
    assert not result.history_supported.iloc[:12].any() and result.history_supported.iloc[12:].all()
    np.testing.assert_allclose(result.loc[12, HISTORY_FEATURES].astype(float), [1., 0., np.log1p(2), 1., np.log1p(30), np.log1p(1)])
    np.testing.assert_allclose(result.loc[13, HISTORY_FEATURES].astype(float), [1., np.log1p(5), np.log1p(1), 1., 0., np.log1p(2)])


def test_missing_rows_and_unsupported_past_are_not_flat():
    frame = rows().drop(index=2).reset_index(drop=True)
    frame.loc[4, ['decomposition_supported', 'execution_change']] = [False, np.nan]
    result = history_features(frame, frame)
    verify_history(frame, frame, result)
    point = result[result.end == pd.Timestamp('2020-01-01T01:00Z')].iloc[0]
    assert point.history_missing_windows == point.history_unsupported_windows == 1
    assert point.history_reason == 'missing_prior_windows|unsupported_prior_decomposition'
    assert point[HISTORY_FEATURES].isna().all()
    assert result[result.end >= pd.Timestamp('2020-01-01T01:30Z')].history_supported.all()


def test_current_future_labels_and_support_do_not_change_past_features():
    frame = rows()
    original = history_features(frame, frame)
    changed = frame.copy()
    changed.loc[30:, 'execution_change'] = -1000.
    changed.loc[40:, 'decomposition_supported'] = False
    actual = history_features(changed, changed)
    fields = HISTORY_FEATURES+['history_supported', 'history_reason', 'history_missing_windows', 'history_unsupported_windows']
    pd.testing.assert_frame_equal(original.loc[:30, fields], actual.loc[:30, fields], check_exact=True)
    assert not original.loc[31, HISTORY_FEATURES].equals(actual.loc[31, HISTORY_FEATURES])


@pytest.mark.parametrize('kind', ['duplicate', 'reverse', 'long_window', 'off_boundary', 'invalid_support', 'nonfinite'])
def test_invalid_history_source_rejected(kind):
    frame = rows()
    if kind == 'duplicate':
        frame.loc[1, 'end'] = frame.end.iloc[0]
    elif kind == 'reverse':
        frame = frame.iloc[::-1]
    elif kind == 'long_window':
        frame.loc[0, 'label_end'] += pd.Timedelta(minutes=5)
    elif kind == 'off_boundary':
        frame.loc[0, ['end', 'label_end']] += pd.Timedelta(seconds=1)
    elif kind == 'invalid_support':
        frame['decomposition_supported'] = 1
    elif kind == 'nonfinite':
        frame.loc[0, 'execution_change'] = np.nan
    with pytest.raises(ValueError):
        history_features(frame, frame)


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(101)
    x, vx = rng.normal(size=(1800, 21)), rng.normal(size=(97, 21))
    y = np.where(x[:, 15] < -.7, -.1, np.where(x[:, 15] > .7, .2, 0.))
    labels = action_labels(y)
    model, _ = HistoryClassifier.fit(x, labels, vx)
    market, _ = SparseExposureClassifier.fit(x[:, :15], labels, vx[:, :15])
    linear, p = linear_baseline(x, labels, vx)
    return x, y, vx, model, market, linear, p


def test_full_384trees_refit_matched_inputs_and_all_metrics(fitted):
    x, y, vx, model, market, linear, p = fitted
    restored = HistoryClassifier.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    independent = independent_refit(restored.to_dict(), market.to_dict(), linear, x, y, vx)
    np.testing.assert_allclose(restored.probabilities(vx), independent['history'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(market.probabilities(vx[:, :15]), independent['matched_market'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, independent['logistic'], atol=1e-12, rtol=0)
    for i, row in enumerate(vx):
        np.testing.assert_allclose(restored.probabilities_one(row), independent['history'][i], atol=1e-12, rtol=0)
    target = np.where(vx[:, 15] < -.7, -.1, np.where(vx[:, 15] > .7, .2, 0.))
    for prediction in independent.values():
        verify_metrics(classification_metrics(target, prediction), reference_classification(target, prediction))


def test_future_diagnosis_input_cannot_change_models(fitted):
    x, y, vx, model, _, linear, _ = fitted
    x_before, y_before = x.copy(), y.copy()
    other, _ = HistoryClassifier.fit(x, action_labels(y), vx*17+3)
    other_linear, _ = linear_baseline(x, action_labels(y), vx*17+3)
    assert model.to_dict() == other.to_dict() and linear == other_linear
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_serialized_wrong_feature_order_and_short_support_rejected(fitted):
    x, y, vx, model, *_ = fitted
    raw = model.to_dict()
    raw['features'][15], raw['features'][16] = raw['features'][16], raw['features'][15]
    with pytest.raises(ValueError):
        HistoryClassifier.from_dict(raw)
    labels = action_labels(y)
    labels[labels == 0] = 1
    labels[:127] = 0
    with pytest.raises(ValueError, match='지원'):
        HistoryClassifier.fit(x, labels, vx)


@pytest.mark.parametrize('fail', ['none', 'training', 'diagnosis', 'cold'])
def test_support_requires_cold_first_actions(fail):
    train = np.tile([-.2, 0., .2], 400)
    val = np.tile([-.2, 0., .2], 30)
    cold = np.ones(len(val), dtype=bool)
    if fail == 'training':
        train[train < 0] = 0
    elif fail == 'diagnosis':
        val[val < 0] = 0
    elif fail == 'cold':
        cold[val < 0] = False
    support = training_support(train, val, cold)
    assert support['supported'] == (fail == 'none')


@pytest.mark.parametrize('failure', ['none', 'v100', 'matched_market', 'logistic', 'ap', 'cold', 'overall', 'frequency'])
def test_all_admission_conditions_and_strict_boundary(failure):
    good = {'log_loss': 1., 'confusion': [[10, 0, 0], [0, 50, 0], [0, 0, 10]],
        'directions': {'-1': {'ap': .3}, '1': {'ap': .3}}}
    overall = {name: copy.deepcopy(good) for name in ['history', 'v100', 'matched_market', 'logistic', 'training_frequency']}
    for name in ['v100', 'matched_market', 'logistic', 'training_frequency']:
        overall[name]['log_loss'] = 1.2
    cold = copy.deepcopy(overall)
    if failure in ['v100', 'matched_market', 'logistic']:
        overall[failure]['log_loss'] = 1./.99
    elif failure == 'ap':
        overall['matched_market']['directions']['-1']['ap'] = .31
    elif failure in ['cold', 'overall']:
        (cold if failure == 'cold' else overall)['history']['confusion'][2][2] = 9
    elif failure == 'frequency':
        overall['training_frequency']['log_loss'] = 1.
    actual = admission(overall, cold)
    assert actual == reference_admission(overall, cold)
    assert actual['passed'] == (failure == 'none')
