from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .engine import INTENTS
from .event_features import MARKET_FEATURES, STATE_FEATURES


@dataclass
class EventModel:
    features: list[str]
    classes: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, data: pd.DataFrame, management: bool = False) -> EventModel:
        features = MARKET_FEATURES + (STATE_FEATURES if management else [])
        values = data[features].to_numpy(dtype=float)
        if len(data) < 1000 or data.target.nunique() < 2 or not np.isfinite(values).all():
            raise ValueError('학습 자료의 수량·종류·유한성 조건이 부족합니다.')
        scaler = StandardScaler()
        learner = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=0)
        learner.fit(scaler.fit_transform(values), data.target.to_numpy())
        if int(learner.n_iter_.max()) >= 2000:
            raise ValueError('사건별 분류기가 수렴하지 않았습니다.')
        return cls.from_dict({'format': 'event_logistic_v1', 'features': features,
                              'classes': learner.classes_.tolist(), 'mean': scaler.mean_.tolist(),
                              'scale': scaler.scale_.tolist(), 'coefficients': learner.coef_.tolist(),
                              'intercept': learner.intercept_.tolist()})

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('특징 배열의 차원이 다릅니다.')
        valid = np.isfinite(values).all(axis=1)
        probs = np.zeros((len(values), len(self.classes)))
        if not valid.any():
            return probs
        z = ((values[valid] - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        if len(self.classes) == 2:
            right = 1 / (1 + np.exp(-np.clip(z[:, 0], -700, 700)))
            probs[valid] = np.column_stack([1 - right, right])
        else:
            z -= z.max(axis=1, keepdims=True)
            weights = np.exp(z)
            probs[valid] = weights / weights.sum(axis=1, keepdims=True)
        return probs

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.probabilities(frame[self.features].to_numpy(dtype=float))
        labels = np.array(self.classes, dtype=object)[probs.argmax(axis=1)]
        labels[probs.sum(axis=1) == 0] = 'hold'
        return labels

    def to_dict(self) -> dict:
        return {'format': 'event_logistic_v1', 'features': self.features, 'classes': self.classes,
                **{key: getattr(self, key).tolist() for key in ['mean', 'scale', 'coefficients', 'intercept']},
                'interpretation': '독립 주문 의도의 통계적 모사. 클래스 가중 확률은 수익 확률이 아님.'}

    @classmethod
    def from_dict(cls, data: dict) -> EventModel:
        features, classes = data.get('features'), data.get('classes')
        if (data.get('format') != 'event_logistic_v1'
            or features not in [MARKET_FEATURES, MARKET_FEATURES + STATE_FEATURES]
            or not isinstance(classes, list) or not 2 <= len(classes) <= len(INTENTS)
            or any(not isinstance(label, str) or label not in INTENTS for label in classes)
            or len(set(classes)) != len(classes) or 'hold' not in classes):
            raise ValueError('지원하지 않는 사건별 모델 형식입니다.')
        if features == MARKET_FEATURES and not set(classes) <= {'hold', 'enter_long', 'enter_short'}:
            raise ValueError('진입 모델에 관리 주문이 포함됐습니다.')
        arrays = [np.asarray(data[key], dtype=float) for key in ['mean', 'scale', 'coefficients', 'intercept']]
        mean, scale, coef, intercept = arrays
        n, rows = len(features), 1 if len(classes) == 2 else len(classes)
        if (mean.shape != (n,) or scale.shape != (n,) or coef.shape != (rows, n)
            or intercept.shape != (rows,) or (scale <= 0).any()
            or any(not np.isfinite(value).all() for value in arrays)):
            raise ValueError('사건별 모델의 수치 배열이 유효하지 않습니다.')
        return cls(list(features), list(classes), *arrays)
