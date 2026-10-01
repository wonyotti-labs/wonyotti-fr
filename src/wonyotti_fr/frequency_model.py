from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .event_model import EventModel


def adjust_scores(probabilities: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    values, prior = np.asarray(probabilities, dtype=float), np.asarray(prior, dtype=float)
    if (values.ndim != 2 or prior.shape != (values.shape[1],)
        or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
        or not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1)
        or type(alpha) not in (int, float) or alpha not in (0, 0.5, 1)):
        raise ValueError('점수·빈도·alpha 설정 오류')
    total = values.sum(axis=1)
    valid = total > 0
    if not np.allclose(total[valid], 1):
        raise ValueError('점수의 행 합계가 1이 아닙니다.')
    if alpha == 0:
        return values.copy()
    result = np.zeros_like(values)
    # 결측 입력의 0 벡터는 빈도만으로 행동을 생성하지 않는다.
    weighted = values[valid] * prior**alpha
    result[valid] = weighted / weighted.sum(axis=1, keepdims=True)
    return result


@dataclass
class FrequencyModel:
    base: EventModel
    prior: np.ndarray
    alpha: float

    def __post_init__(self):
        self.prior = np.asarray(self.prior, dtype=float).copy()
        adjust_scores(np.zeros((1, len(self.classes))), self.prior, self.alpha)
        self.prior.setflags(write=False)
        self._weights = self.prior**self.alpha

    @classmethod
    def from_counts(cls, base: EventModel, counts: dict[str, int], alpha: float) -> FrequencyModel:
        if (set(counts) != set(base.classes)
            or any(type(value) is not int or value <= 0 for value in counts.values())):
            raise ValueError('학습 클래스의 빈도 구성이 다릅니다.')
        prior = np.array([counts[label] for label in base.classes], dtype=float)
        prior /= prior.sum()
        return cls(base, prior, alpha)

    @property
    def features(self):
        return self.base.features

    @property
    def classes(self):
        return self.base.classes

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        scores = self.base.probabilities(values)
        if self.alpha == 0:
            return scores
        scores *= self._weights
        total = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, total, out=np.zeros_like(scores), where=total > 0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.probabilities(frame[self.features].to_numpy(dtype=float))
        result = np.array(self.classes, dtype=object)[scores.argmax(axis=1)]
        result[scores.sum(axis=1) == 0] = 'hold'
        return result


def score_quality(model: EventModel | FrequencyModel, data: pd.DataFrame) -> dict:
    if data.empty:
        return {'rows': 0, 'brier_multiclass': None, 'log_loss': None, 'reliability': []}
    if not set(data.target) <= set(model.classes):
        raise ValueError('학습하지 않은 정답 클래스가 평가 자료에 있습니다.')
    scores = model.probabilities(data[model.features].to_numpy(dtype=float))
    if not np.allclose(scores.sum(axis=1), 1):
        raise ValueError('유효하지 않은 평가 특징이 있습니다.')
    truth = np.column_stack([data.target.eq(label).to_numpy() for label in model.classes])
    bins = []
    for index, label in enumerate(model.classes):
        bucket = np.minimum((scores[:, index] * 10).astype(int), 9)
        for level in range(10):
            selected = bucket == level
            bins.append({'class': label, 'lower': level / 10, 'upper': (level + 1) / 10,
                         'count': int(selected.sum()),
                         'mean_score': float(scores[selected, index].mean()) if selected.any() else None,
                         'observed_frequency': float(truth[selected, index].mean()) if selected.any() else None})
    return {'rows': len(data), 'brier_multiclass': float(np.square(scores - truth).sum(axis=1).mean()),
            'log_loss': float(-np.log(np.clip(scores[truth], 1e-15, 1)).mean()),
            'reliability': bins, 'interpretation': '행동 점수와 관측 빈도의 진단이며 금융 수익 확률이 아님'}
