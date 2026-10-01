from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .event_features import MARKET_FEATURES
from .expansion_model import ExpansionPolicy

EDGE_FEATURES = MARKET_FEATURES + [f'signed_{name}' for name in MARKET_FEATURES] + ['order_direction']


def edge_values(market: np.ndarray, direction: np.ndarray) -> np.ndarray:
    market, direction = np.asarray(market, dtype=float), np.asarray(direction, dtype=float)
    if (market.ndim != 2 or market.shape[1] != len(MARKET_FEATURES) or direction.shape != (len(market),)
        or not np.isin(direction, [-1, 1]).all()):
        raise ValueError('가격 변화 예측 특징의 차원·방향 오류')
    return np.column_stack([market, market * direction[:, None], direction])


def edge_targets(events: pd.DataFrame, bars: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    if horizon_bars not in (6, 12, 24) or bars.end.duplicated().any():
        raise ValueError('가격 정답의 간격 또는 중복 시각 오류')
    frame = events.copy()
    frame['outcome_time'] = frame.end + pd.Timedelta(minutes=5 * horizon_bars)
    prices = bars.set_index('end').close
    entry = prices.reindex(frame.end).to_numpy(dtype=float)
    outcome = prices.reindex(frame.outcome_time).to_numpy(dtype=float)
    frame['order_direction'] = np.where(frame.buy.eq(1), 1, -1)
    frame['outcome_bps'] = frame.order_direction * np.log(outcome / entry) * 10000
    frame['label_end'] = frame.outcome_time
    frame['usable'] &= frame.active.eq(1) & np.isfinite(frame.outcome_bps)
    return frame


@dataclass
class EdgeModel:
    data: dict

    @classmethod
    def fit(cls, frame: pd.DataFrame, alpha: float) -> tuple[EdgeModel, dict]:
        values = edge_values(frame[MARKET_FEATURES].to_numpy(), frame.order_direction.to_numpy())
        target = frame.outcome_bps.to_numpy(dtype=float)
        if len(frame) < 1000 or not np.isfinite(values).all() or not np.isfinite(target).all() or alpha not in (10, 100):
            raise ValueError('가격 변화 학습의 표본·정밀도·설정 오류')
        scaler = StandardScaler().fit(values)
        learner = Ridge(alpha=alpha).fit(scaler.transform(values), target)
        model = cls.from_dict({'format': 'edge_ridge_v1', 'features': EDGE_FEATURES, 'alpha': alpha,
                               'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                               'coefficients': learner.coef_.tolist(), 'intercept': float(learner.intercept_)})
        expected = learner.predict(scaler.transform(values))
        predicted = model.predict(values)
        error = float(np.max(np.abs(expected - predicted)))
        if error > 1e-10:
            raise ValueError('저장한 가격 모델과 라이브러리의 예측 불일치')
        return model, {'rows': len(frame), 'export_max_error': error, 'mean_target_bps': float(target.mean()),
                        'last_outcome_time': frame.outcome_time.max(),
                        'training_mse': float(np.mean((target - predicted) ** 2)),
                        'training_zero_forecast_mse': float(np.mean(target ** 2))}

    @classmethod
    def from_dict(cls, data: dict) -> EdgeModel:
        if (data.get('format') != 'edge_ridge_v1' or data.get('features') != EDGE_FEATURES or data.get('alpha') not in (10, 100)):
            raise ValueError('지원하지 않는 가격 예측 형식')
        arrays = [np.asarray(data[name], dtype=float) for name in ['mean', 'scale', 'coefficients']]
        if (any(array.shape != (len(EDGE_FEATURES),) or not np.isfinite(array).all() for array in arrays)
            or (arrays[1] <= 0).any() or not np.isscalar(data['intercept']) or not np.isfinite(data['intercept'])):
            raise ValueError('가격 모델의 비정상 계수')
        return cls(copy.deepcopy(data))

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(EDGE_FEATURES):
            raise ValueError('가격 모델의 특징 차원 오류')
        result = np.full(len(values), np.nan)
        valid = np.isfinite(values).all(axis=1)
        with np.errstate(over='ignore', invalid='ignore'):
            result[valid] = ((values[valid] - self.data['mean']) / self.data['scale']) @ np.asarray(self.data['coefficients']) + self.data['intercept']
        result[~np.isfinite(result)] = np.nan
        return result

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)


class EdgePolicy:
    def __init__(self, base: ExpansionPolicy, model: EdgeModel, margin_bps: int, filter_enabled: bool = True):
        if margin_bps not in (0, 8) or type(filter_enabled) is not bool:
            raise ValueError('비용 예측 정책의 설정 오류')
        self.base, self.model, self.margin_bps, self.filter_enabled = base, model, margin_bps, filter_enabled

    def prepare(self, frame):
        self.base.prepare(frame)

    def __call__(self, bar: dict, state: dict) -> str:
        if state['direction'] or state['halted']:
            return 'hold'
        intent = self.base(bar, state)
        if intent not in {'enter_long', 'enter_short'} or not self.filter_enabled:
            return intent
        direction = 1 if intent == 'enter_long' else -1
        values = edge_values(np.asarray(bar['features']).reshape(1, -1), np.array([direction]))
        predicted = self.model.predict(values)[0]
        return intent if np.isfinite(predicted) and predicted >= 16 + self.margin_bps else 'hold'
