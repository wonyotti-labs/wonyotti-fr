from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.exposure_change_model import EXPOSURE_FEATURES
from wonyotti_fr.sparse_exposure import SparseExposureClassifier, action_labels

HISTORY_FEATURES = [f'past_{side}_{name}_60m' for side in ['decrease', 'increase']
    for name in ['exists', 'log_minutes_since', 'log_window_count']]
FEATURES = EXPOSURE_FEATURES + HISTORY_FEATURES


def history_features(frame, source):
    for part in [frame, source]:
        if (not len(part) or part.end.isna().any() or part.end.duplicated().any() or not part.end.is_monotonic_increasing
            or str(part.end.dt.tz) != 'UTC' or (part.end.astype('datetime64[ns, UTC]').array.asi8 % (300*10**9)).any()
            or not part.label_end.sub(part.end).eq(pd.Timedelta(minutes=5)).all()):
            raise ValueError('과거 체결 이력의 시각·중복·창 경계 오류')
    if (source.decomposition_supported.isna().any()
        or any(type(v) not in (bool, np.bool_) for v in source.decomposition_supported)
        or not np.isfinite(source.loc[source.decomposition_supported, 'execution_change']).all()):
        raise ValueError('과거 체결 이력의 지원·정답 오류')
    needed = np.stack([(frame.end-pd.Timedelta(minutes=5*lag)).astype('datetime64[ns, UTC]').array.asi8 for lag in range(1, 13)], axis=1)
    locations = pd.Index(source.end.astype('datetime64[ns, UTC]').array.asi8).get_indexer(needed.ravel()).reshape(len(frame), 12)
    present = locations >= 0
    found = np.maximum(locations, 0)
    supported = present & source.decomposition_supported.to_numpy()[found]
    missing, unavailable = (~present).sum(axis=1), (present & ~supported).sum(axis=1)
    available = supported.all(axis=1)
    values = np.full((len(frame), len(HISTORY_FEATURES)), np.nan)
    prior = source.execution_change.to_numpy(dtype=float)[found]
    for index, sign in enumerate([-1, 1]):
        active = prior*sign > 1e-12
        counts = active.sum(axis=1)
        exists = counts > 0
        # 첫 열은 현재 경계에 끝난 직전 창이며 현재 정답 창은 포함하지 않는다.
        minutes = np.where(exists, active.argmax(axis=1)*5, 0)
        part = np.column_stack([exists.astype(float), np.log1p(minutes), np.log1p(counts)])
        values[available, index*3:index*3+3] = part[available]
    out = frame.copy()
    fields = HISTORY_FEATURES+['history_supported', 'history_reason', 'history_missing_windows', 'history_unsupported_windows']
    if any(name in out for name in fields):
        raise ValueError('과거 체결 이력의 기존 열 덮어쓰기 오류')
    out[HISTORY_FEATURES] = values
    out['history_supported'] = available
    out['history_missing_windows'], out['history_unsupported_windows'] = missing, unavailable
    out['history_reason'] = ['|'.join(reason for reason, use in [('missing_prior_windows', a > 0),
        ('unsupported_prior_decomposition', b > 0)] if use) or 'included' for a, b in zip(missing, unavailable, strict=True)]
    return out


def feature_matrix(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not len(x) or not np.isfinite(x).all():
        raise ValueError('과거 체결 모형의 입력 오류')
    return x


class HistoryClassifier(SparseExposureClassifier):
    format = 'past_execution_history_classifier_v1'
    features = FEATURES

    @classmethod
    def fit(cls, values, labels, validation_values):
        x, vx, y = feature_matrix(values), feature_matrix(validation_values), np.asarray(labels)
        if y.shape != (len(x),) or y.dtype.kind not in 'iu' or not np.isin(y, [0, 1, 2]).all() or min(np.bincount(y, minlength=3)) < 128:
            raise ValueError('과거 체결 모형의 행동별 지원 부족')
        return super().fit(x, y, vx)


def linear_baseline(x, labels, validation):
    x, vx = feature_matrix(x), feature_matrix(validation)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, class_weight=None, random_state=0).fit(scaler.transform(x), labels)
        probability = learner.predict_proba(scaler.transform(vx))
    if not np.array_equal(learner.classes_, [0, 1, 2]) or (learner.n_iter_ >= 2000).any():
        raise ValueError('과거 체결 선형 대조의 클래스·수렴 오류')
    payload = {'features': FEATURES, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'coef': learner.coef_.tolist(), 'intercept': learner.intercept_.tolist(), 'iterations': learner.n_iter_.tolist()}
    logits = ((vx-scaler.mean_)/scaler.scale_)@learner.coef_.T+learner.intercept_
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(exp/exp.sum(axis=1, keepdims=True), probability, atol=1e-12, rtol=0)
    return payload, probability


def training_support(training_y, diagnosis_y, cold):
    train, diagnosis = action_labels(training_y), action_labels(diagnosis_y)
    cold = np.asarray(cold, dtype=bool)
    if cold.shape != diagnosis.shape:
        raise ValueError('최초 행동 진단 지원의 차원 오류')
    counts = {'training': np.bincount(train, minlength=3).tolist(), 'diagnosis': np.bincount(diagnosis, minlength=3).tolist(),
        'cold': np.bincount(diagnosis[cold], minlength=3).tolist()}
    checks = {'training_rows': len(train) >= 1000, 'training_classes': min(counts['training']) >= 128,
        'diagnosis_directions': min(counts['diagnosis'][k] for k in [0, 2]) >= 20,
        'cold_directions': min(counts['cold'][k] for k in [0, 2]) >= 20}
    return {'supported': all(checks.values()), 'counts': counts, 'checks': checks}


def admission(overall, cold):
    candidate = overall['history']
    checks, gains = {}, {}
    for baseline in ['v100', 'matched_market', 'logistic']:
        reference = overall[baseline]
        gain = 1-candidate['log_loss']/reference['log_loss'] if reference['log_loss'] > 0 else float('-inf')
        gains[baseline] = gain if np.isfinite(gain) else None
        checks['loss_gain_'+baseline] = gain > .01+1e-12
        for direction in ['-1', '1']:
            a, b = candidate['directions'][direction]['ap'], reference['directions'][direction]['ap']
            checks['ap_'+baseline+'_'+direction] = a is not None and b is not None and a >= b-1e-12
    checks['loss_better_frequency'] = candidate['log_loss'] < overall['training_frequency']['log_loss']-1e-12
    for part, rows in [('overall', overall), ('cold', cold)]:
        for index in [0, 2]:
            checks[f'{part}_correct_{index-1}'] = rows['history']['confusion'][index][index] >= 10
    return {'passed': all(checks.values()), 'relative_log_loss_gains': gains, 'checks': checks}
