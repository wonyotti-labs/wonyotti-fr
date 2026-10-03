from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .engine import PolicyDecision
from .event_features import MARKET_FEATURES, STATE_FEATURES
from .pullback_policy import PullbackPolicy

MANAGEMENT_FEATURES = MARKET_FEATURES + STATE_FEATURES
ACTIONS = ['exit', 'increase', 'reduce']


class ManagementModel:
    def __init__(self, payload: dict):
        self.data = copy.deepcopy(payload)
        self.mean = np.asarray(payload['mean'])
        self.scale = np.asarray(payload['scale'])
        self.activity_coef = np.asarray(payload['activity_coef'])
        self.action_coef = np.asarray(payload['action_coef'])
        self.action_intercept = np.asarray(payload['action_intercept'])

    @classmethod
    def fit(cls, train: pd.DataFrame) -> tuple[ManagementModel, dict]:
        values = train[MANAGEMENT_FEATURES].to_numpy(dtype=float)
        labels = train.target.replace({'enter_long': 'exit', 'enter_short': 'exit'}).to_numpy()
        counts = {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
        if (len(train) < 1000 or not np.isfinite(values).all()
            or set(counts) != {'hold', *ACTIONS} or min(counts.values()) < 20
            or not train.direction.isin([-1, 1]).all()):
            raise ValueError('관리 학습의 특징·행동·방향 지원 표본 부족')
        active = labels != 'hold'
        scaler = StandardScaler().fit(values)
        scaled = scaler.transform(values)
        activity = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaled, active.astype(int))
        action = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaled[active], labels[active])
        if max(activity.n_iter_.max(), action.n_iter_.max()) >= 2000 or list(action.classes_) != ACTIONS:
            raise ValueError('관리 분류기의 수렴·클래스 오류')
        model = cls.from_dict({
            'format': 'lifecycle_management_v1', 'features': MANAGEMENT_FEATURES, 'actions': ACTIONS,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'activity_coef': activity.coef_[0].tolist(), 'activity_intercept': float(activity.intercept_[0]),
            'action_coef': action.coef_.tolist(), 'action_intercept': action.intercept_.tolist(),
        })
        scores, actions = model.probabilities(values)
        error = max(float(np.max(np.abs(scores - activity.predict_proba(scaled)[:, 1]))),
                    float(np.max(np.abs(actions[active] - action.predict_proba(scaled[active])))))
        if error > 1e-12:
            raise ValueError('관리 모델 내보내기 예측 불일치')
        rate = float(active.mean())
        thresholds = {str(factor): float(np.quantile(scores, 1 - factor * rate)) for factor in [.5, 1.]}
        return model, {'rows': len(train), 'class_counts': counts, 'active_fraction': rate,
                       'thresholds': thresholds, 'export_max_error': error,
                       'first_time': train.end.min(), 'last_label_end': train.label_end.max()}

    @classmethod
    def from_dict(cls, data: dict) -> ManagementModel:
        if (data.get('format') != 'lifecycle_management_v1' or data.get('features') != MANAGEMENT_FEATURES
            or data.get('actions') != ACTIONS):
            raise ValueError('관리 모델 형식·특징·행동 오류')
        shapes = {'mean': (18,), 'scale': (18,), 'activity_coef': (18,), 'activity_intercept': (),
                  'action_coef': (3, 18), 'action_intercept': (3,)}
        for name, shape in shapes.items():
            value = np.asarray(data[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('관리 모델의 차원·유한성 오류')
        if np.any(np.asarray(data['scale']) <= 0):
            raise ValueError('관리 모델 표준화 척도 오류')
        return cls(data)

    def probabilities(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != 18:
            raise ValueError('관리 예측 특징 차원 오류')
        valid = np.isfinite(values).all(axis=1)
        activity = np.full(len(values), np.nan)
        actions = np.full((len(values), 3), np.nan)
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = (values[valid] - self.mean) / self.scale
            z = scaled @ self.activity_coef + self.data['activity_intercept']
            logits = scaled @ self.action_coef.T + self.action_intercept
        finite = np.isfinite(z) & np.isfinite(logits).all(axis=1)
        indices = np.flatnonzero(valid)[finite]
        activity[indices] = 1 / (1 + np.exp(-np.clip(z[finite], -700, 700)))
        logits = logits[finite]
        if len(logits):
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            actions[indices] = weights / weights.sum(axis=1, keepdims=True)
        return activity, actions

    def to_dict(self):
        return copy.deepcopy(self.data)


class LifecyclePolicy(PullbackPolicy):
    def __init__(self, entry: PullbackPolicy, manager: ManagementModel, activity_threshold: float,
                 enabled: bool = True):
        super().__init__(entry.base, entry.offset_bps, entry.ttl_minutes, entry.baseline)
        if not np.isfinite(activity_threshold) or not 0 <= activity_threshold <= 1 or type(enabled) is not bool:
            raise ValueError('관리 정책 문턱·활성 설정 오류')
        self.manager, self.activity_threshold, self.enabled = manager, activity_threshold, enabled

    def __call__(self, bar: dict, state: dict) -> PolicyDecision:
        if (not self.enabled or not state['direction'] or state['halted']
            or state['policy_state'] or self.baseline == 'cash'):
            return super().__call__(bar, state)
        if state['bar_seconds'] != 60:
            raise ValueError('관리 정책은 1분 실행만 지원합니다.')
        if pd.Timestamp(bar['end']).value % pd.Timedelta(minutes=5).value or state['pending'] != 'hold':
            return PolicyDecision('hold', {}, 'idle')
        market = np.asarray(bar['features'], dtype=float)
        if market.shape != (14,):
            raise ValueError('관리 시장 특징 차원 오류')
        # 보유 시간은 실행 봉 수를 분으로 변환하며 미래 원본 상태를 참조하지 않는다.
        values = np.r_[market, state['direction'], state['favorable_move'],
                       np.log1p(state['hold_bars'] * state['bar_seconds'] / 60), min(state['adds'], 5)]
        activity, actions = self.manager.probabilities(values.reshape(1, -1))
        if not np.isfinite(activity[0]) or activity[0] < self.activity_threshold:
            return PolicyDecision('hold', {}, 'manage_hold')
        index = int(np.argmax(actions[0]))
        if actions[0, index] < .5:
            return PolicyDecision('hold', {}, 'manage_uncertain')
        intent = ACTIONS[index]
        return PolicyDecision(intent, {}, f'manage_{intent}')
