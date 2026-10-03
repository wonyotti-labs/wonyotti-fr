from __future__ import annotations

import copy

import numpy as np

from .minute_management import ACTIONS


def probability_logits(values):
    scores = np.asarray(values, dtype=float)
    if (scores.ndim != 2 or scores.shape[1] != 3 or not len(scores)
        or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('관리 빈도 보정의 점수 차원·범위 오류')
    clipped = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


class ManagementOffset:
    format = 'management_offset_v1'

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.offsets = np.asarray(data['offsets'], dtype=float)

    @classmethod
    def fit(cls, scores, labels):
        logits = probability_logits(scores)
        y = np.asarray(labels)
        if (len(y) < 100 or y.shape != logits.shape or not np.isin(y, [0, 1]).all()
            or (y.sum(axis=0) < 20).any() or ((1 - y).sum(axis=0) < 20).any()):
            raise ValueError('관리 빈도 보정의 정답·지원 부족')
        target = y.mean(axis=0)
        low, high = np.full(3, -64.), np.full(3, 64.)
        if (sigmoid(logits + low).mean(axis=0) >= target).any() or (sigmoid(logits + high).mean(axis=0) <= target).any():
            raise ValueError('관리 빈도 보정 절편의 탐색 범위 초과')
        # 기울기를 고정하면 로그 손실의 도함수는 평균 예측과 실제 빈도의 차이다.
        for _ in range(100):
            middle = (low + high) / 2
            below = sigmoid(logits + middle).mean(axis=0) < target
            low, high = np.where(below, middle, low), np.where(below, high, middle)
        model = cls.from_dict({'format': cls.format, 'actions': ACTIONS, 'slope': 1.,
            'epsilon': float(np.finfo(float).eps), 'bounds': [-64., 64.], 'iterations': 100,
            'offsets': ((low + high) / 2).tolist()})
        mean = model.predict(scores).mean(axis=0)
        residual = float(np.max(np.abs(mean - target)))
        if residual >= 1e-12:
            raise ValueError('관리 빈도 보정 절편의 수렴 실패')
        return model, {'rows': len(y), 'positive': y.sum(axis=0).astype(int).tolist(),
            'actual_fraction': target.tolist(), 'calibrated_mean': mean.tolist(), 'max_mean_residual': residual}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('actions') != ACTIONS
            or type(data.get('slope')) not in (int, float) or data['slope'] != 1.
            or type(data.get('epsilon')) is not float or data['epsilon'] != np.finfo(float).eps
            or data.get('bounds') != [-64., 64.] or type(data.get('iterations')) is not int or data['iterations'] != 100
            or not isinstance(data.get('offsets'), list) or len(data['offsets']) != 3
            or any(type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 64 for v in data['offsets'])):
            raise ValueError('관리 빈도 보정의 형식·고정 설정·절편 오류')
        return cls(data)

    def predict(self, scores):
        return sigmoid(probability_logits(scores) + self.offsets)

    def to_dict(self):
        return copy.deepcopy(self.data)
