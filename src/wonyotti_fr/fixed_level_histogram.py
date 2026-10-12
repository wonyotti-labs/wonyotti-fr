from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .first_state_models import FEATURES


class FixedLevelHistogram(HistGradientBoostingClassifier):
    def _bin_data(self, X, sample_weight, is_training_data):
        x = np.asarray(X)
        if (
            self.max_bins != 255
            or self.early_stopping is not False
            or self.categorical_features is not None
            or x.ndim != 2
            or x.shape[1] != len(FEATURES)
            or not np.isfinite(x).all()
            or ((x < 0) | (x > 254) | (x != np.floor(x))).any()
        ):
            raise ValueError("고정 수준 학습의 구간·입력·설정 오류")
        if sample_weight is not None:
            weights = np.asarray(sample_weight)
            if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("고정 수준 학습의 양수 가중치 오류")
        if is_training_data:
            for values in x.T:
                distinct = np.unique(values)
                if not np.array_equal(distinct, np.arange(len(distinct))):
                    raise ValueError("고정 수준 학습의 누락된 정수 수준")
            # 재표집이 드문 수준을 버리면 원래 구간이 합쳐지므로 전체 수준을 사용한다.
            self._bin_mapper.subsample = None
        encoded = super()._bin_data(X, sample_weight, is_training_data)
        np.testing.assert_array_equal(encoded, x.astype(np.uint8))
        return encoded
