from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.exposure_change_model import (
    EXPOSURE_FEATURES,
    SETTINGS,
    ExposureChangeModel,
    matrix,
)
from wonyotti_fr.position_boost import PositionBoostModel

EPSILON = 1e-12


def action_labels(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('희소 행동 정답의 차원·유한성 오류')
    return np.where(a < -EPSILON, 0, np.where(a > EPSILON, 2, 1))


def choose_actions(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
        or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError('희소 행동 확률 오류')
    tied = (values == values.max(axis=1, keepdims=True)).sum(axis=1) > 1
    return np.where(tied, 1, values.argmax(axis=1))-1


class SparseExposureClassifier(PositionBoostModel):
    format = 'sparse_execution_change_classifier_v1'
    features = EXPOSURE_FEATURES
    classes = ['decrease', 'no_change', 'increase']
    actions = classes

    def __init__(self, data):
        super().__init__(data)
        self._packed = tuple((m['baseline'], tuple(tuple(tuple(t[key]) for key in
            ['left', 'right', 'feature', 'threshold', 'value']) for t in m['trees'])) for m in self.data['models'])

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {'format', 'features', 'settings', 'models'} or not isinstance(data['models'], list):
            raise ValueError('희소 행동 모형 필드 오류')
        for model in data['models']:
            if not isinstance(model, dict) or set(model) != {'action', 'baseline', 'trees'} or not isinstance(model['trees'], list):
                raise ValueError('희소 행동 모형 구조 오류')
            for tree in model['trees']:
                if (not isinstance(tree, dict) or set(tree) != {'left', 'right', 'feature', 'threshold', 'value'}
                    or any(not isinstance(a, list) for a in tree.values())
                    or any(type(v) not in (int, float) for key in ['threshold', 'value'] for v in tree[key])):
                    raise ValueError('희소 행동 숫자 트리 구조 오류')
        return super().from_dict(data)

    def probabilities_one(self, values):
        row = np.asarray(values, dtype=float)
        if row.shape != (len(self.features),) or not np.isfinite(row).all():
            raise ValueError('희소 행동 단일 계좌 입력 오류')
        logits = []
        for initial, trees in self._packed:
            score = initial
            for left, right, feature, threshold, value in trees:
                at = 0
                while left[at] != -1:
                    at = left[at] if row[feature[at]] <= threshold[at] else right[at]
                score += value[at]
            logits.append(score)
        if not np.isfinite(logits).all():
            raise ValueError('희소 행동 단일 점수 오류')
        exponent = np.exp(np.asarray(logits)-max(logits))
        return exponent/exponent.sum()


class ExecutionMagnitudeModel(ExposureChangeModel):
    format = 'sparse_execution_magnitude_histogram_v1'

    @classmethod
    def fit(cls, x, y, validation):
        x, vx, y = matrix(x), matrix(validation), np.asarray(y, dtype=float)
        if len(x) < 128 or not len(vx) or y.shape != (len(x),) or not np.isfinite(y).all() or (y <= EPSILON).any():
            raise ValueError('체결 크기 방향별 학습 지원 오류')
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingRegressor(**SETTINGS).fit(x, y)
            expected = [learner.predict(part) for part in [x, vx]]
        if learner.n_iter_ != 64 or learner.n_trees_per_iteration_ != 1 or learner.do_early_stopping_:
            raise ValueError('체결 크기 반복·구성 오류')
        trees = []
        for part in learner._predictors:
            nodes = part[0].nodes
            if nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                raise ValueError('체결 크기 범주·깊이 오류')
            leaf = nodes['is_leaf'].astype(bool)
            trees.append({'left': np.where(leaf, -1, nodes['left'].astype(int)).tolist(),
                'right': np.where(leaf, -1, nodes['right'].astype(int)).tolist(),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)).tolist(),
                'threshold': np.where(leaf, 0., nodes['num_threshold']).tolist(),
                'value': np.where(leaf, nodes['value'], 0.).tolist()})
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': SETTINGS,
            'baseline': float(learner._baseline_prediction[0, 0]), 'trees': trees})
        errors = [float(np.max(np.abs(model.predict(part)-reference))) for part, reference in zip([x, vx], expected, strict=True)]
        if max(errors) > 1e-12:
            raise ValueError('체결 크기 숫자 내보내기 불일치')
        return model, {'training_rows': len(x), 'training_export_error': errors[0], 'diagnosis_export_error': errors[1]}


class SparseExposureModel:
    format = 'sparse_execution_change_policy_v1'

    def __init__(self, classifier, negative, positive):
        self.classifier, self.negative, self.positive = classifier, negative, positive

    @classmethod
    def fit(cls, x, y, validation):
        labels = action_labels(y)
        x, vx, y = matrix(x), matrix(validation), np.asarray(y, dtype=float)
        if len(labels) != len(x) or min(int((labels == k).sum()) for k in [0, 2]) < 128:
            raise ValueError('희소 정책 양방향 크기 지원 부족')
        classifier, cp = SparseExposureClassifier.fit(x, labels, vx)
        negative, np_ = ExecutionMagnitudeModel.fit(x[labels == 0], -y[labels == 0], vx)
        positive, pp = ExecutionMagnitudeModel.fit(x[labels == 2], y[labels == 2], vx)
        return cls(classifier, negative, positive), {'classifier': cp, 'negative': np_, 'positive': pp}

    def to_dict(self):
        return {'format': self.format, 'classifier': self.classifier.to_dict(),
            'negative': self.negative.to_dict(), 'positive': self.positive.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'format', 'classifier', 'negative', 'positive'} or value['format'] != cls.format:
            raise ValueError('희소 체결 정책 형식 오류')
        value = copy.deepcopy(value)
        return cls(SparseExposureClassifier.from_dict(value['classifier']),
            ExecutionMagnitudeModel.from_dict(value['negative']), ExecutionMagnitudeModel.from_dict(value['positive']))

    def predict(self, values):
        x = matrix(values)
        probabilities = self.classifier.probabilities(x)
        negative, positive = np.maximum(self.negative.predict(x), 0), np.maximum(self.positive.predict(x), 0)
        actions = choose_actions(probabilities)
        selected = np.where(actions < 0, -negative, np.where(actions > 0, positive, 0))
        return {'probabilities': probabilities, 'negative_size': negative, 'positive_size': positive,
            'actions': actions, 'decision_change': selected,
            'expected_change': probabilities[:, 2]*positive-probabilities[:, 0]*negative}

    def predict_one(self, market, current):
        values = np.asarray([*market, current], dtype=float)
        probabilities = self.classifier.probabilities_one(values)
        negative = max(0., self.negative.predict_one(market, current))
        positive = max(0., self.positive.predict_one(market, current))
        # 최대 확률 동률은 주문을 만들지 않는다.
        action = int(choose_actions(probabilities.reshape(1, -1))[0])
        return {'probabilities': probabilities.tolist(), 'negative_size': negative, 'positive_size': positive,
            'action': action, 'decision_change': -negative if action < 0 else positive if action > 0 else 0.,
            'expected_change': float(probabilities[2]*positive-probabilities[0]*negative)}


def linear_baseline(x, labels, validation):
    x, vx = matrix(x), matrix(validation)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, class_weight=None, random_state=0).fit(scaler.transform(x), labels)
        expected = learner.predict_proba(scaler.transform(vx))
    if not np.array_equal(learner.classes_, [0, 1, 2]) or (learner.n_iter_ >= 2000).any():
        raise ValueError('희소 선형 대조의 클래스·수렴 오류')
    payload = {'features': EXPOSURE_FEATURES, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'coef': learner.coef_.tolist(), 'intercept': learner.intercept_.tolist(), 'iterations': learner.n_iter_.tolist()}
    logits = ((vx-scaler.mean_)/scaler.scale_)@learner.coef_.T+learner.intercept_
    exponent = np.exp(logits-logits.max(axis=1, keepdims=True))
    prediction = exponent/exponent.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
    return payload, prediction
