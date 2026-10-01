from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


@dataclass
class DirectionModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    @classmethod
    def fit(cls, data: pd.DataFrame) -> DirectionModel:
        clean = data.dropna(subset=FEATURES + ["label"])
        if len(clean) < 1000 or clean.label.nunique() < 2:
            raise ValueError("학습 표본 또는 방향 종류가 부족합니다.")
        scaler = StandardScaler()
        x = scaler.fit_transform(clean[FEATURES].to_numpy(dtype=float))
        learner = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=0)
        learner.fit(x, clean.label.to_numpy(dtype=int))
        if int(learner.n_iter_.max()) >= 2000:
            raise ValueError("학습이 수렴하지 않았습니다.")
        return cls(scaler.mean_, scaler.scale_, learner.coef_, learner.intercept_, learner.classes_)

    def signals(self, data: pd.DataFrame, threshold: float, volatility_cap: float | None = None) -> np.ndarray:
        if not 0 <= threshold <= 1:
            raise ValueError("신뢰도 기준은 0~1이어야 합니다.")
        values = data[FEATURES].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        result = np.zeros(len(data), dtype=int)
        if volatility_cap is not None and (not np.isfinite(volatility_cap) or volatility_cap <= 0):
            raise ValueError("변동성 한도는 유한한 양수여야 합니다.")
        if not valid.any():
            return result
        z = ((values[valid] - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        if len(self.classes) == 2:
            probability = 1 / (1 + np.exp(-np.clip(z[:, 0], -700, 700)))
            probs = np.column_stack([1 - probability, probability])
        else:
            z = z - z.max(axis=1, keepdims=True)
            probs = np.exp(z)
            probs /= probs.sum(axis=1, keepdims=True)
        labels = self.classes[probs.argmax(axis=1)]
        labels[probs.max(axis=1) < threshold] = 0
        result[valid] = labels
        if volatility_cap is not None:
            result[data.volatility_1d.to_numpy() > volatility_cap] = 0
        return result

    def to_dict(self) -> dict:
        return {"format": "direction_logistic_v1", "features": FEATURES,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coefficients": self.coefficients.tolist(), "intercept": self.intercept.tolist(),
                "classes": self.classes.tolist(),
                "interpretation": "공개 포지션 방향의 통계적 모사 후보. 본인의 실제 판단 규칙이 아님."}

    @classmethod
    def from_dict(cls, value: dict) -> DirectionModel:
        if value.get("format") != "direction_logistic_v1" or value.get("features") != FEATURES:
            raise ValueError("지원하지 않는 모델 형식")
        model = cls(*(np.asarray(value[key], dtype=float) for key in
                      ["mean", "scale", "coefficients", "intercept", "classes"]))
        n = len(FEATURES)
        classes = model.classes
        expected_rows = 1 if len(classes) == 2 else len(classes)
        if (model.mean.shape != (n,) or model.scale.shape != (n,)
            or model.coefficients.shape != (expected_rows, n)
            or model.intercept.shape != (expected_rows,)
            or len(classes) not in (2, 3) or len(np.unique(classes)) != len(classes)
            or not np.isin(classes, [-1, 0, 1]).all() or (model.scale <= 0).any()
            or any(not np.isfinite(x).all() for x in [model.mean, model.scale, model.coefficients, model.intercept, classes])):
            raise ValueError("모델 배열이 유효하지 않습니다.")
        model.classes = classes.astype(int)
        return model
