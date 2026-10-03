from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .engine import PolicyDecision
from .path_management import PathActionModels
from .rate_policy import RateActionPolicy

TRAINING_PERIODS = [('2019-01-01', '2020-07-01'), ('2020-01-01', '2021-07-01')]
CALIBRATION_PERIODS = [('2020-07-01', '2021-01-01'), ('2021-07-01', '2022-01-01')]


class InventoryActionModels(PathActionModels):
    features = PathActionModels.features + ['remaining_fraction']
    format = 'minute_inventory_action_v1'


class ReductionModel:
    features = InventoryActionModels.features
    format = 'actual_reduction_ridge_v1'

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.mean, self.scale, self.coef = (np.asarray(data[k], dtype=float) for k in ['mean', 'scale', 'coef'])
        self.intercept = data['intercept']

    @classmethod
    def fit(cls, frame, training_period=TRAINING_PERIODS[0]):
        if training_period not in TRAINING_PERIODS:
            raise ValueError('축소 크기 학습의 사전 고정 기간 오류')
        first, last = (pd.Timestamp(value, tz='UTC') for value in training_period)
        x, y = frame[cls.features].to_numpy(dtype=float), frame.reduction_target.to_numpy(dtype=float)
        if (len(frame) < 100 or frame.end.max() - frame.end.min() < pd.Timedelta(days=90)
            or frame.end.min() < first + pd.Timedelta(days=1)
            or frame.label_end.max() >= last - pd.Timedelta(days=1)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not ((y > 0) & (y <= 1)).all()):
            raise ValueError('축소 크기 학습의 지원 표본·시간·목표 오류')
        scaler = StandardScaler().fit(x)
        learner = Ridge(alpha=100.).fit(scaler.transform(x), y)
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'alpha': 100.,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': learner.coef_.tolist(),
            'intercept': float(learner.intercept_)})
        error = float(np.max(np.abs(model.predict(x) - np.clip(learner.predict(scaler.transform(x)), 0, 1))))
        if error > 1e-12:
            raise ValueError('축소 크기 모델의 내보내기 불일치')
        return model, {'rows': len(frame), 'first_end': frame.end.min(), 'last_end': frame.end.max(),
            'last_label_end': frame.label_end.max(), 'export_max_error': error,
            'mean_target': float(y.mean()), 'median_target': float(np.median(y))}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('features') != cls.features
            or type(data.get('alpha')) not in (int, float) or data['alpha'] != 100.):
            raise ValueError('축소 크기 모델의 형식·특징·고정 계수 오류')
        for name in ['mean', 'scale', 'coef']:
            value = np.asarray(data[name], dtype=float)
            if value.shape != (len(cls.features),) or not np.isfinite(value).all():
                raise ValueError('축소 크기 모델의 차원·숫자 오류')
        if (np.any(np.asarray(data['scale']) <= 0) or type(data['intercept']) not in (int, float)
            or not np.isfinite(data['intercept'])):
            raise ValueError('축소 크기 모델의 척도·절편 오류')
        return cls(data)

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features) or not np.isfinite(values).all():
            raise ValueError('축소 크기 예측 입력의 차원·숫자 오류')
        with np.errstate(over='ignore', invalid='ignore'):
            raw = ((values - self.mean) / self.scale) @ self.coef + self.intercept
        if not np.isfinite(raw).all():
            raise ValueError('축소 크기 예측의 유한 범위 초과')
        return np.clip(raw, 0, 1)

    def to_dict(self):
        return copy.deepcopy(self.data)


class InventoryRatePolicy(RateActionPolicy):
    def __init__(self, entry, manager, thresholds, multiplier, scales, size_model=None):
        super().__init__(entry, manager, thresholds, multiplier, scales)
        self.size_model = size_model

    def feature_values(self, bar, state):
        fraction = state['remaining_fraction']
        if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('관리 입력의 실제 잔여 수량 비율 오류')
        return np.r_[super().feature_values(bar, state), fraction]

    def __call__(self, bar, state):
        decision = super().__call__(bar, state)
        if decision.intent != 'reduce' or self.size_model is None:
            return decision
        # 행동 판단 때 이미 갱신한 과거 가격 경로로 같은 입력을 구성한다.
        context = {**state, '_path_bounds': (decision.state['path_low'], decision.state['path_high'])}
        fraction = float(self.size_model.predict(self.feature_values(bar, context).reshape(1, -1))[0])
        if fraction == 0:
            return PolicyDecision('hold', decision.state, 'action_zero_reduction')
        return PolicyDecision('reduce', decision.state, 'action_sized_reduce', fraction)
