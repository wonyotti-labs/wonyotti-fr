from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .edge_model import EDGE_FEATURES, edge_values
from .engine import PolicyDecision
from .event_features import MARKET_FEATURES
from .pullback_policy import PullbackPolicy

NET_FEATURES = EDGE_FEATURES + ['favorable_bps', 'wait_minutes']


def net_values(market, direction, favorable, minutes) -> np.ndarray:
    values = edge_values(market, direction)
    extra = np.column_stack([favorable, minutes]).astype(float)
    if extra.shape != (len(values), 2) or not np.isfinite(extra).all() or (extra[:, 1] <= 0).any():
        raise ValueError('순손익 예측의 대기 특징 오류')
    return np.column_stack([values, extra])


@dataclass
class NetEdgeModel:
    data: dict

    @classmethod
    def fit(cls, frame: pd.DataFrame, alpha: int, *, sample_weight=None) -> tuple[NetEdgeModel, dict]:
        counts = frame.order_direction.value_counts()
        if (len(frame) < 200 or any(counts.get(side, 0) < 20 for side in [-1, 1])
            or frame.decision_time.max() - frame.decision_time.min() < pd.Timedelta(days=180)
            or type(alpha) is not int or alpha not in (10, 100)):
            raise ValueError('순손익 학습의 표본·방향·기간·설정 부족')
        values = net_values(frame[MARKET_FEATURES].to_numpy(), frame.order_direction.to_numpy(),
                            frame.favorable_bps.to_numpy(), frame.wait_minutes.to_numpy())
        target = frame.net_bps.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError('순손익 학습의 비유한 값')
        fit_options = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(frame),) or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('순손익 학습의 양수 표본 가중치 오류')
            fit_options['sample_weight'] = weights
        scaler = StandardScaler().fit(values, **fit_options)
        learner = Ridge(alpha=alpha).fit(scaler.transform(values), target, **fit_options)
        model = cls.from_dict({'format': 'net_edge_ridge_v1', 'features': NET_FEATURES, 'alpha': alpha,
                              'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                              'coefficients': learner.coef_.tolist(), 'intercept': float(learner.intercept_)})
        predicted = model.predict(values)
        error = float(np.max(np.abs(predicted - learner.predict(scaler.transform(values)))))
        if error > 1e-10:
            raise ValueError('순손익 모델의 내보내기 예측 불일치')
        support = {'rows': len(frame), 'directions': {str(k): int(v) for k, v in counts.items()},
                       'export_max_error': error, 'mean_net_bps': float(target.mean()),
                       'last_label_end': frame.label_end.max(), 'training_mse': float(np.mean((target-predicted)**2)),
                       'constant_mean_mse': float(np.mean((target-target.mean())**2))}
        if sample_weight is not None:
            support.update(weighted_target_mean_bps=float(np.average(target, weights=weights)),
                           weighted_training_mse=float(np.average((target-predicted)**2, weights=weights)))
        return model, support

    @classmethod
    def from_dict(cls, data: dict) -> NetEdgeModel:
        if (data.get('format') != 'net_edge_ridge_v1' or data.get('features') != NET_FEATURES
            or type(data.get('alpha')) is not int or data['alpha'] not in (10, 100)):
            raise ValueError('순손익 모델의 형식 오류')
        values = [np.asarray(data[k], dtype=float) for k in ['mean', 'scale', 'coefficients']]
        if (any(x.shape != (len(NET_FEATURES),) or not np.isfinite(x).all() for x in values)
            or (values[1] <= 0).any() or type(data.get('intercept')) not in (int, float)
            or not np.isfinite(data['intercept'])):
            raise ValueError('순손익 모델의 계수 오류')
        return cls(copy.deepcopy(data))

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(NET_FEATURES):
            raise ValueError('순손익 모델의 특징 차원 오류')
        with np.errstate(over='ignore', invalid='ignore'):
            result = ((values-self.data['mean']) / self.data['scale']) @ np.asarray(self.data['coefficients']) + self.data['intercept']
        result[~np.isfinite(result)] = np.nan
        return result

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)


class NetEdgePolicy(PullbackPolicy):
    def __init__(self, pullback: PullbackPolicy, model: NetEdgeModel, margin_bps: int, enabled: bool = True):
        if (pullback.offset_bps != 16 or pullback.ttl_minutes != 5 or pullback.baseline is not None
            or type(margin_bps) is not int or margin_bps not in (0, 8) or type(enabled) is not bool):
            raise ValueError('순손익 필터의 고정 기반·기준 오류')
        super().__init__(pullback.base, pullback.offset_bps, pullback.ttl_minutes)
        self.model, self.margin_bps, self.enabled = model, margin_bps, enabled

    def __call__(self, bar: dict, state: dict) -> PolicyDecision:
        decision = super().__call__(bar, state)
        if decision.event != 'triggered' or not self.enabled:
            return decision
        waiting = state['policy_state']
        direction = waiting['direction']
        favorable = direction * np.log(waiting['reference_price'] / bar['close']) * 10000
        minutes = (pd.Timestamp(bar['end']) - pd.Timestamp(waiting['signal_time'])).total_seconds() / 60
        values = net_values(np.asarray(bar['features'], dtype=float).reshape(1, -1), [direction], [favorable], [minutes])
        predicted = self.model.predict(values)[0]
        # 거절한 기회는 끝내고 미래의 더 유리한 가격으로 재판정하지 않는다.
        return decision if np.isfinite(predicted) and predicted >= self.margin_bps else PolicyDecision('hold', {}, 'filtered')
