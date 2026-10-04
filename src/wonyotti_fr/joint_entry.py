from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .event_features import MARKET_FEATURES

JOINT_CLASSES = ['hold', 'short', 'long']
JOINT_SETTINGS = {'C': .1, 'solver': 'lbfgs', 'max_iter': 2000,
                  'random_state': 0, 'class_weight': None}


def joint_targets(frame):
    if not np.isin(frame.active, [0, 1]).all() or not np.isin(frame.loc[frame.active.eq(1), 'buy'], [0, 1]).all():
        raise ValueError('결합 진입 정답의 활동·방향 오류')
    return np.where(frame.active.eq(0), 0, np.where(frame.buy.eq(1), 2, 1)).astype(int)


class JointEntryModel:
    format = 'joint_entry_multinomial_v1'
    classes = JOINT_CLASSES
    features = MARKET_FEATURES

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def fit(cls, values, labels):
        x, y = np.asarray(values, dtype=float), np.asarray(labels)
        if (x.ndim != 2 or x.shape[1] != len(cls.features) or len(x) < 1000
            or not np.isfinite(x).all() or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or set(np.unique(y)) != {0, 1, 2} or (np.bincount(y, minlength=3) < 20).any()):
            raise ValueError('결합 진입 모형의 차원·클래스·지원 부족')
        with threadpool_limits(limits=1):
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(**JOINT_SETTINGS).fit(scaler.transform(x), y)
            expected = learner.predict_proba(scaler.transform(x))
        if learner.n_iter_.max() >= 2000 or not np.array_equal(learner.classes_, np.arange(3)):
            raise ValueError('결합 진입 모형의 수렴·클래스 순서 오류')
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'classes': cls.classes,
            'settings': JOINT_SETTINGS, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coefficients': learner.coef_.tolist(), 'intercepts': learner.intercept_.tolist()})
        error = float(np.max(np.abs(model.probabilities(x)-expected)))
        if error > 1e-12:
            raise ValueError('결합 진입 모형의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'class_counts': np.bincount(y, minlength=3).tolist(),
            'export_max_error': error, 'iterations': int(learner.n_iter_.max())}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('features') != cls.features
            or data.get('classes') != cls.classes or data.get('settings') != JOINT_SETTINGS):
            raise ValueError('결합 진입 모형의 형식·클래스·설정 오류')
        width = len(cls.features)
        for key, shape in [('mean', (width,)), ('scale', (width,)), ('coefficients', (3, width)), ('intercepts', (3,))]:
            value = np.asarray(data[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all() or (key == 'scale' and (value <= 0).any()):
                raise ValueError('결합 진입 모형의 계수·표준화 오류')
        return cls(data)

    def probabilities(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError('결합 진입 모형의 예측 차원 오류')
        result = np.full((len(x), 3), np.nan)
        valid = np.flatnonzero(np.isfinite(x).all(axis=1))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            z = (x[valid]-np.asarray(self.data['mean']))/np.asarray(self.data['scale'])
            logits = z @ np.asarray(self.data['coefficients']).T+np.asarray(self.data['intercepts'])
        available = np.isfinite(logits).all(axis=1)
        selected = logits[available]
        # 최댓값을 빼면 클래스 비율을 유지하면서 지수 오버플로를 막는다.
        exponent = np.exp(selected-selected.max(axis=1, keepdims=True))
        result[valid[available]] = exponent/exponent.sum(axis=1, keepdims=True)
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)


def validate_joint_scores(scores):
    p = np.asarray(scores, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError('결합 진입 확률의 차원·범위·총합 오류')
    return p


def factorized_scores(activity, direction):
    a, b = np.asarray(activity, dtype=float), np.asarray(direction, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all()
        or not np.isfinite(b).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any()):
        raise ValueError('분해 진입 확률의 차원·범위 오류')
    return validate_joint_scores(np.column_stack([1-a, a*(1-b), a*b]))
