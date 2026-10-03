from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .addition_effect import position_weights
from .engine import PolicyDecision
from .minute_inventory import MinuteInventoryModels, MinuteInventoryPolicy


class AdditionEffectModel:
    features = MinuteInventoryModels.features

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.mean, self.scale, self.coef = (np.asarray(data[k], dtype=float) for k in ['mean', 'scale', 'coef'])
        self.intercept = data['intercept']

    @classmethod
    def fit(cls, frame):
        first, cutoff = pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2021-12-31', tz='UTC')
        if any(frame[name].isna().any() or str(frame[name].dt.tz) != 'UTC'
               or not frame[name].eq(frame[name].dt.floor('min')).all()
               for name in ['decision_time', 'label_end', 'position_entry_time']):
            raise ValueError('추가 순효과 학습 시각의 UTC 분 경계 오류')
        if (len(frame) < 100 or frame.position_entry_time.nunique() < 30
            or frame.decision_time.max() - frame.decision_time.min() < pd.Timedelta(days=180)
            or frame.decision_time.lt(first).any() or frame.label_end.ge(cutoff).any()
            or frame.decision_time.ge(frame.label_end).any() or frame.decision_time.duplicated().any()
            or not frame.decision_time.is_monotonic_increasing or not frame.label_status.eq('closed').all()
            or frame.position_entry_time.gt(frame.decision_time).any()):
            raise ValueError('추가 순효과 학습의 시간 경계·지원 표본 오류')
        x, y = frame[cls.features].to_numpy(dtype=float), frame.incremental_bps.to_numpy(dtype=float)
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(frame.decision_equity).all() or frame.decision_equity.le(0).any()
            or not np.allclose(y, frame.incremental_pnl / frame.decision_equity * 10000, atol=1e-8, rtol=0)):
            raise ValueError('추가 순효과 학습 입력·순자산 정규화 오류')
        weights = position_weights(frame)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        learner = Ridge(alpha=100.).fit(scaler.transform(x), y, sample_weight=weights)
        model = cls.from_dict({'format': 'addition_effect_ridge_v1', 'features': cls.features, 'alpha': 100,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': learner.coef_.tolist(),
            'intercept': float(learner.intercept_)})
        predicted = model.predict(x)
        error = float(np.max(np.abs(predicted - learner.predict(scaler.transform(x)))))
        if error > 1e-10:
            raise ValueError('추가 순효과 모델의 내보내기 예측 불일치')
        return model, weights, {'rows': len(frame), 'positions': int(frame.position_entry_time.nunique()),
            'export_max_error': error, 'first_decision': frame.decision_time.min(), 'last_label_end': frame.label_end.max(),
            'positive_labels': int((y > 0).sum()), 'negative_labels': int((y < 0).sum()), 'zero_labels': int((y == 0).sum()),
            'weighting': 'equal_original_position_mean_one_v1', 'weight_sum': float(weights.sum()),
            'weighted_mean_target': float(np.average(y, weights=weights)),
            'weighted_training_mse': float(np.average((y-predicted)**2, weights=weights)),
            'independent_samples_claimed': False}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != 'addition_effect_ridge_v1' or data.get('features') != cls.features
            or type(data.get('alpha')) is not int or data['alpha'] != 100):
            raise ValueError('추가 순효과 모델의 형식·고정 설정 오류')
        arrays = [np.asarray(data[k], dtype=float) for k in ['mean', 'scale', 'coef']]
        if (any(x.shape != (len(cls.features),) or not np.isfinite(x).all() for x in arrays)
            or (arrays[1] <= 0).any() or type(data.get('intercept')) not in (int, float)
            or not np.isfinite(data['intercept'])):
            raise ValueError('추가 순효과 모델의 차원·숫자 오류')
        return cls(data)

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('추가 순효과 예측 입력의 차원 오류')
        with np.errstate(over='ignore', invalid='ignore'):
            result = ((values-self.mean)/self.scale) @ self.coef + self.intercept
        result[~np.isfinite(values).all(axis=1) | ~np.isfinite(result)] = np.nan
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)


class AdditionEffectPolicy(MinuteInventoryPolicy):
    def __init__(self, parent, effect_model, enabled=True):
        if not isinstance(parent, MinuteInventoryPolicy) or type(enabled) is not bool:
            raise ValueError('추가 순효과 정책의 기반·활성 설정 오류')
        super().__init__(parent, parent.manager, parent.thresholds, parent.multiplier, parent.scales, parent.size_model)
        self.effect_model, self.enabled, self.audit = effect_model, enabled, []

    def prepare(self, frame):
        self.audit = []
        super().prepare(frame)

    def __call__(self, bar, state):
        decision = super().__call__(bar, state)
        if decision.intent != 'increase' or not self.enabled:
            return decision
        context = {**state, '_path_bounds': (decision.state['path_low'], decision.state['path_high'])}
        values = self.feature_values(bar, context)
        predicted = float(self.effect_model.predict(values.reshape(1, -1))[0])
        accepted = bool(np.isfinite(predicted) and predicted > 0)
        self.audit.append({'decision_time': bar['end'], 'position_entry_time': state['position_entry_time'],
            'predicted_incremental_bps': predicted if np.isfinite(predicted) else None, 'accepted': accepted,
            **dict(zip(AdditionEffectModel.features, values.tolist(), strict=True))})
        # 거절한 추가의 누적량 초기화·관리 대기는 유지해 같은 요청을 반복하지 않는다.
        return decision if accepted else PolicyDecision('hold', decision.state, 'action_add_rejected')
