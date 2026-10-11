from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.market_change_model import (
    MarketChangeClassifier,
    MarketChangeModel,
    MarketMagnitudeModel,
)


class WeightedMarketClassifier(MarketChangeClassifier):
    format = 'inverse_frequency_market_classifier_v1'

    @classmethod
    def fit(cls, values, labels, validation_values):
        x, y = np.asarray(values, dtype=float), np.asarray(labels)
        vx = np.asarray(validation_values, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(cls.features) or len(x) < 1000
            or not np.isfinite(x).all() or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or set(np.unique(y)) != {0, 1, 2} or (np.bincount(y, minlength=3) < 128).any()
            or vx.ndim != 2 or vx.shape[1] != len(cls.features) or not len(vx)
            or not np.isfinite(vx).all()):
            raise ValueError('보유 방향 부스팅의 입력·클래스·지원 오류')
        counts = np.bincount(y, minlength=3)
        weights = len(y)/(3*counts)
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y, sample_weight=weights[y])
            expected, validation_expected = learner.predict_proba(x), learner.predict_proba(vx)
        if (not np.array_equal(learner.classes_, np.arange(3)) or learner.n_iter_ != 64
            or learner.n_trees_per_iteration_ != 3 or learner.do_early_stopping_
            or learner._baseline_prediction.shape != (1, 3)
            or len(learner._predictors) != 64 or any(len(part) != 3 for part in learner._predictors)):
            raise ValueError('보유 방향 부스팅의 클래스·반복·초기 점수 오류')
        models = []
        for index, action in enumerate(cls.classes):
            trees = []
            for iteration in learner._predictors:
                nodes = iteration[index].nodes
                if nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                    raise ValueError('보유 방향 부스팅의 범주·트리 깊이 오류')
                leaves = nodes['is_leaf'].astype(bool)
                # 라이브러리의 말단 값에는 학습률이 이미 반영되어 있다.
                trees.append({'left': np.where(leaves, -1, nodes['left'].astype(int)).tolist(),
                    'right': np.where(leaves, -1, nodes['right'].astype(int)).tolist(),
                    'feature': np.where(leaves, -2, nodes['feature_idx'].astype(int)).tolist(),
                    'threshold': np.where(leaves, 0., nodes['num_threshold']).tolist(),
                    'value': np.where(leaves, nodes['value'], 0.).tolist()})
            models.append({'action': action, 'baseline': float(learner._baseline_prediction[0, index]), 'trees': trees})
        model = cls.from_dict({'format': cls.format, 'features': cls.features,
            'settings': HISTOGRAM_SETTINGS, 'models': models})
        errors = [float(np.max(np.abs(model.probabilities(part) - scores)))
                  for part, scores in [(x, expected), (vx, validation_expected)]]
        if max(errors) > 1e-12:
            raise ValueError('보유 방향 부스팅의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'class_counts': np.bincount(y, minlength=3).tolist(),
            'iterations': 64, 'trees_per_iteration': 3, 'export_max_error': errors[0],
            'validation_export_max_error': errors[1], 'early_stopping': False, 'class_weights': weights.tolist(),
            'training_objective': 'inverse_training_frequency'}


def class_weights(counts):
    if (not isinstance(counts, list) or len(counts) != 3
        or any(type(n) is not int or n < 128 for n in counts) or sum(counts) < 1000):
        raise ValueError('행동 가중치의 학습 클래스 지원 오류')
    return sum(counts)/(3*np.asarray(counts, dtype=float))


def transform_scores(probabilities, counts, condition):
    from wonyotti_fr.sparse_exposure import choose_actions

    p = np.asarray(probabilities, dtype=float)
    choose_actions(p)
    weights = class_weights(counts)
    if condition not in ('A', 'B', 'C', 'D'):
        raise ValueError('학습·선택 대조 조건 오류')
    if condition in ('A', 'C'):
        return p.copy()
    scores = p*weights if condition == 'B' else p/weights
    return scores/scores.sum(axis=1, keepdims=True)


class DecisionScores:
    def __init__(self, base, counts, condition):
        self.base, self.counts, self.condition = base, counts, condition

    def probabilities(self, values):
        return transform_scores(self.base.probabilities(values), self.counts, self.condition)

    def probabilities_one(self, values):
        return transform_scores(self.base.probabilities_one(values).reshape(1, 3), self.counts, self.condition)[0]


class WeightedActionModel(MarketChangeModel):
    format = 'weighted_action_factorial_v1'

    def __init__(self, classifier, negative, positive, counts, condition):
        class_weights(counts)
        expected = MarketChangeClassifier if condition in ('A', 'B') else WeightedMarketClassifier
        if (condition not in ('A', 'B', 'C', 'D') or type(classifier) is not expected
            or type(negative) is not MarketMagnitudeModel or type(positive) is not MarketMagnitudeModel):
            raise ValueError('행동 가중 모형의 조건·학습 구성 오류')
        self.base_classifier = classifier
        self.counts, self.condition = counts.copy(), condition
        self.classifier = DecisionScores(classifier, self.counts, condition)
        self.negative, self.positive, self.timing = negative, positive, 'window_end'

    @classmethod
    def fit(cls, *args, **kwargs):
        raise ValueError('분류기만 별도 학습하고 고정 크기 모형으로 조립 필요')

    def to_dict(self):
        return {'format': self.format, 'timing': self.timing, 'condition': self.condition,
            'class_counts': self.counts.copy(), 'classifier': self.base_classifier.to_dict(),
            'negative': self.negative.to_dict(), 'positive': self.positive.to_dict()}

    @classmethod
    def from_dict(cls, raw):
        keys = {'format', 'timing', 'condition', 'class_counts', 'classifier', 'negative', 'positive'}
        if (not isinstance(raw, dict) or set(raw) != keys or raw['format'] != cls.format
            or raw['timing'] != 'window_end' or raw['condition'] not in ('A', 'B', 'C', 'D')):
            raise ValueError('행동 가중 모형의 직렬화 형식 오류')
        raw = copy.deepcopy(raw)
        parser = MarketChangeClassifier if raw['condition'] in ('A', 'B') else WeightedMarketClassifier
        return cls(parser.from_dict(raw['classifier']), MarketMagnitudeModel.from_dict(raw['negative']),
            MarketMagnitudeModel.from_dict(raw['positive']), raw['class_counts'], raw['condition'])

