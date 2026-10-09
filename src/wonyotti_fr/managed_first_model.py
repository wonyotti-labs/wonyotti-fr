from __future__ import annotations

import numpy as np

from .close_effect import CLOSE_FEATURES
from .first_linear_close import FirstLinearCloseModel
from .minute_management import ACTIONS

MANAGER_FEATURES = ['current_manager_'+action+'_probability' for action in ACTIONS]


class ManagedFirstLinearModel(FirstLinearCloseModel):
    features = [*FirstLinearCloseModel.features, *MANAGER_FEATURES]
    format = 'managed_first_opportunity_cost_logistic_v1'

    @classmethod
    def matrix(cls, values):
        matrix = super().matrix(values)
        if ((matrix[:, -3:] < 0) | (matrix[:, -3:] > 1)).any():
            raise ValueError('관리 확률 첫 선형 모델의 행동 확률 범위 오류')
        return matrix


def augment_manager_inputs(frame, manager):
    if manager.features != CLOSE_FEATURES or any(name in frame for name in MANAGER_FEATURES):
        raise ValueError('관리 확률 첫 입력의 부모 특징·기존 확률 충돌')
    FirstLinearCloseModel.matrix(frame[FirstLinearCloseModel.features].to_numpy())
    scores = manager.probabilities(frame[CLOSE_FEATURES].to_numpy(dtype=float))
    if (scores.shape != (len(frame), 3) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('고정 관리 모델의 첫 입력 확률 오류')
    result = frame.copy()
    for index, name in enumerate(MANAGER_FEATURES):
        result[name] = scores[:, index]
    return result
