from datetime import datetime

import numpy as np

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.sparse_exposure import EPSILON, SparseExposureModel


class SparseExposurePolicy:
    kind = 'sparse_execution'

    def __init__(self, model):
        if not isinstance(model, SparseExposureModel):
            raise ValueError('희소 체결 정책의 모형 오류')
        self.model, self.audit_sink = model, None

    def __call__(self, bar, own):
        end = datetime.fromisoformat(bar['end'])
        if end.second or end.microsecond or end.minute % 5:
            return None
        if bar.get('feature_end') != bar['end']:
            raise ValueError('희소 체결 정책의 확정 특징 시각 오류')
        values = np.asarray(bar['features'], dtype=float)
        if values.shape != (len(MARKET_FEATURES),):
            raise ValueError('희소 체결 정책의 특징 차원 오류')
        equity, quantity = own['equity'], own['quantity']
        if not np.isfinite([equity, quantity, bar['close']]).all() or equity <= 0 or bar['close'] <= 0:
            raise ValueError('희소 체결 정책의 자체 계좌 오류')
        current = float(quantity*bar['close']/equity)
        record = {'time': bar['end'], 'equity': equity, 'quantity': quantity, 'current_exposure': current,
            **dict.fromkeys(['probability_decrease', 'probability_no_change', 'probability_increase',
                'negative_size', 'positive_size', 'action', 'expected_change', 'decision_change', 'target']),
            'reason': 'missing_features'}
        if np.isfinite(values).all():
            prediction = self.model.predict_one(values, current)
            action, change = prediction['action'], prediction['decision_change']
            record.update({k: prediction[k] for k in ['negative_size', 'positive_size', 'action', 'expected_change', 'decision_change']})
            record.update(zip(['probability_decrease', 'probability_no_change', 'probability_increase'], prediction['probabilities'], strict=True))
            if action == 0:
                record['reason'] = 'no_change'
            elif abs(change) <= EPSILON:
                record['reason'] = 'zero_size'
            else:
                record['reason'] = 'target'
                record['target'] = float(np.clip(current+change, -.5, .5))
        if self.audit_sink is not None:
            self.audit_sink(record)
        return record['target']
