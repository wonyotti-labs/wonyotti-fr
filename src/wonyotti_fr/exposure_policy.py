from __future__ import annotations

from datetime import datetime

import numpy as np

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.exposure_change_model import ExposureChangeModel
from wonyotti_fr.net_exposure_model import NetExposureModel


class ExposurePolicy:
    def __init__(self, kind, model=None, allocation=.5, *, constant_exposure=None):
        if kind not in ('change', 'absolute', 'constant') or allocation != .5:
            raise ValueError('고정 자기 계좌 정책 설정 오류')
        if (kind == 'change' and not isinstance(model, ExposureChangeModel)
            or kind == 'absolute' and not isinstance(model, NetExposureModel)
            or kind == 'constant' and model is not None):
            raise ValueError('자기 계좌 정책 모형 불일치')
        if kind == 'constant':
            if (type(constant_exposure) not in (int, float) or not np.isfinite(constant_exposure)
                or not -allocation <= constant_exposure <= allocation):
                raise ValueError('학습 대조 노출 상수 오류')
        elif constant_exposure is not None:
            raise ValueError('비상수 정책의 대조 노출 설정 오류')
        self.kind, self.model, self.allocation = kind, model, allocation
        self.constant_exposure = float(constant_exposure) if kind == 'constant' else None
        self.trees = () if model is None else tuple(tuple(tuple(tree[key])
            for key in ['left', 'right', 'feature', 'threshold', 'value']) for tree in model.to_dict()['trees'])
        self.baseline = None if model is None else model.to_dict()['baseline']

    def to_dict(self):
        value = {'kind': self.kind, 'model': None if self.model is None else self.model.to_dict()}
        if self.kind == 'constant':
            value['constant_exposure'] = self.constant_exposure
        return value

    def __call__(self, bar, own):
        end = datetime.fromisoformat(bar['end'])
        if end.second or end.microsecond or end.minute % 5:
            return None
        if self.kind == 'constant':
            return self.constant_exposure
        if bar.get('feature_end') != bar['end']:
            raise ValueError('5분 의사결정의 확정 특징 시각 불일치')
        features = np.asarray(bar['features'], dtype=float)
        if features.shape != (len(MARKET_FEATURES),):
            raise ValueError('자기 계좌 정책 특징 차원 오류')
        if not np.isfinite(features).all():
            return None
        if self.kind == 'change':
            equity, quantity = own['equity'], own['quantity']
            if not np.isfinite([equity, quantity]).all() or equity <= 0:
                raise ValueError('자기 계좌 정책 평가액 오류')
            # 원본 계좌 상태 대신 현재 봇 수량과 확정 종가만 사용한다.
            current = float(quantity*bar['close']/equity)
            desired = current+self.model.predict_one(features, current)
        else:
            desired = self.baseline
            for left, right, feature, threshold, value in self.trees:
                node = 0
                while left[node] != -1:
                    node = left[node] if features[feature[node]] <= threshold[node] else right[node]
                desired += value[node]
        return float(np.clip(desired, -self.allocation, self.allocation))
