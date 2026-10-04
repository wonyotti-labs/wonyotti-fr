from __future__ import annotations

import copy

import numpy as np

from .continuation_inputs import ContinuationCloseModel, validate_pending_matrix

EXPOSURE_FEATURE = 'current_gross_exposure'
EXPOSURE_TRANSFORM = {'target': 'account_bps/current_gross_exposure',
    'training_weight': 'original_weight*current_gross_exposure**2/mean_training_weight',
    'prediction': 'unit_prediction*current_gross_exposure'}


def current_exposure(values):
    x = np.asarray(values, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(ContinuationCloseModel.features)
        or not len(x) or not np.isfinite(x).all()):
        raise ValueError('노출 회귀의 입력 차원·숫자 오류')
    validate_pending_matrix(x[:, -4:])
    exposure = x[:, ContinuationCloseModel.features.index(EXPOSURE_FEATURE)]
    if (exposure <= 0).any():
        raise ValueError('노출 회귀의 비양수 현재 노출')
    return exposure


def exposure_training(values, target, weights):
    exposure = current_exposure(values)
    y, w = (np.asarray(v, dtype=float) for v in [target, weights])
    if (y.shape != exposure.shape or w.shape != exposure.shape
        or not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any()
        or not np.isclose(w.mean(), 1., rtol=0, atol=1e-12)):
        raise ValueError('노출 회귀의 정답·원래 가중치 오류')
    # 계좌 기준 제곱 오차의 상대 비중을 노출 변환 뒤에도 보존한다.
    with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
        unit_target, raw_weight = y/exposure, w*exposure**2
        normalizer = float(raw_weight.mean())
        unit_weight = raw_weight/normalizer
    if (not np.isfinite(normalizer) or normalizer <= 0
        or not np.isfinite(unit_target).all() or not np.isfinite(unit_weight).all()
        or (unit_weight <= 0).any()):
        raise ValueError('노출 회귀의 변환 넘침·가중치 소실')
    return unit_target, unit_weight, normalizer


class ExposureUnitModel(ContinuationCloseModel):
    format = 'close_exposure_unit_histogram_v1'


class ExposureCloseModel:
    features = ContinuationCloseModel.features
    format = 'close_exposure_scaled_histogram_v1'

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.unit_model = ExposureUnitModel.from_dict(data['unit_model'])

    @classmethod
    def fit(cls, values, target, weights, validation_values):
        unit_target, unit_weight, normalizer = exposure_training(values, target, weights)
        current_exposure(validation_values)
        unit_model, support = ExposureUnitModel.fit(values, unit_target, unit_weight, validation_values)
        model = cls.from_dict({'format': cls.format, 'features': cls.features,
            'transform': EXPOSURE_TRANSFORM, 'unit_model': unit_model.to_dict()})
        return model, {**support, 'training_weight_normalizer': normalizer,
            'original_account_error_relative_weights_preserved': True}

    @classmethod
    def from_dict(cls, data):
        if (set(data) != {'format', 'features', 'transform', 'unit_model'} or data['format'] != cls.format
            or data['features'] != cls.features or data['transform'] != EXPOSURE_TRANSFORM):
            raise ValueError('노출 회귀의 형식·입력·변환 오류')
        return cls(data)

    def predict(self, values):
        exposure = current_exposure(values)
        with np.errstate(over='ignore', invalid='ignore'):
            result = exposure*self.unit_model.predict(values)
        if not np.isfinite(result).all():
            raise ValueError('노출 회귀의 계좌 예측 넘침')
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)
