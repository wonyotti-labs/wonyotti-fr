from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.net_exposure_model import SETTINGS, NetExposureModel, matrix
from wonyotti_fr.sparse_exposure import (
    EPSILON,
    SparseExposureClassifier,
    action_labels,
    choose_actions,
)


class MarketChangeClassifier(SparseExposureClassifier):
    format = 'market_only_execution_classifier_v1'
    features = MARKET_FEATURES


class MarketMagnitudeModel(NetExposureModel):
    format = 'market_only_execution_magnitude_v1'

    def __init__(self, data):
        super().__init__(data)
        self._packed = tuple(tuple(tuple(t[k]) for k in ['left', 'right', 'feature', 'threshold', 'value'])
            for t in self.data['trees'])

    def predict_one(self, market):
        values = np.asarray(market, dtype=float)
        if values.shape != (len(MARKET_FEATURES),) or not np.isfinite(values).all():
            raise ValueError('시장 행동 크기의 입력 차원·유한성 오류')
        result = self.data['baseline']
        for left, right, feature, threshold, value in self._packed:
            at = 0
            while left[at] != -1:
                at = left[at] if values[feature[at]] <= threshold[at] else right[at]
            result += value[at]
        if not np.isfinite(result):
            raise ValueError('시장 행동 크기 예측 유한성 오류')
        return result

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


class MarketChangeModel:
    format = 'market_only_execution_policy_v1'

    def __init__(self, classifier, negative, positive, timing):
        if (timing not in ('window_start', 'window_end') or type(classifier) is not MarketChangeClassifier
            or type(negative) is not MarketMagnitudeModel or type(positive) is not MarketMagnitudeModel):
            raise ValueError('시장 행동 학습 시각·모형 구성 오류')
        self.classifier, self.negative, self.positive = classifier, negative, positive
        self.timing = timing

    @classmethod
    def fit(cls, x, y, validation, *, timing):
        if timing not in ('window_start', 'window_end'):
            raise ValueError('시장 행동 학습 시각 오류')
        labels = action_labels(y)
        x, vx, y = matrix(x), matrix(validation), np.asarray(y, dtype=float)
        if len(labels) != len(x) or len(x) < 1000 or min(int((labels == k).sum()) for k in [0, 1, 2]) < 128:
            raise ValueError('희소 정책 양방향 크기 지원 부족')
        classifier, cp = MarketChangeClassifier.fit(x, labels, vx)
        negative, np_ = MarketMagnitudeModel.fit(x[labels == 0], -y[labels == 0], vx)
        positive, pp = MarketMagnitudeModel.fit(x[labels == 2], y[labels == 2], vx)
        return cls(classifier, negative, positive, timing), {'classifier': cp, 'negative': np_, 'positive': pp}

    def to_dict(self):
        return {'format': self.format, 'timing': self.timing, 'classifier': self.classifier.to_dict(),
            'negative': self.negative.to_dict(), 'positive': self.positive.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'format', 'timing', 'classifier', 'negative', 'positive'} or value['format'] != cls.format:
            raise ValueError('희소 체결 정책 형식 오류')
        value = copy.deepcopy(value)
        return cls(MarketChangeClassifier.from_dict(value['classifier']),
            MarketMagnitudeModel.from_dict(value['negative']), MarketMagnitudeModel.from_dict(value['positive']), value['timing'])

    def predict(self, values):
        x = matrix(values)
        probabilities = self.classifier.probabilities(x)
        negative, positive = np.maximum(self.negative.predict(x), 0), np.maximum(self.positive.predict(x), 0)
        actions = choose_actions(probabilities)
        selected = np.where(actions < 0, -negative, np.where(actions > 0, positive, 0))
        return {'probabilities': probabilities, 'negative_size': negative, 'positive_size': positive,
            'actions': actions, 'decision_change': selected,
            'expected_change': probabilities[:, 2]*positive-probabilities[:, 0]*negative}

    def predict_one(self, market):
        values = np.asarray(market, dtype=float)
        probabilities = self.classifier.probabilities_one(values)
        negative = max(0., self.negative.predict_one(market))
        positive = max(0., self.positive.predict_one(market))
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
    payload = {'features': MARKET_FEATURES, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'coef': learner.coef_.tolist(), 'intercept': learner.intercept_.tolist(), 'iterations': learner.n_iter_.tolist()}
    logits = ((vx-scaler.mean_)/scaler.scale_)@learner.coef_.T+learner.intercept_
    exponent = np.exp(logits-logits.max(axis=1, keepdims=True))
    prediction = exponent/exponent.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
    return payload, prediction
