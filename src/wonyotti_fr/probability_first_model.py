from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .first_linear_close import FirstLinearCloseModel
from .managed_first_model import MANAGER_FEATURES
from .managed_first_parent import checked_manager_parent, pinned_json


class ProbabilityFirstLinearModel(FirstLinearCloseModel):
    features = MANAGER_FEATURES
    format = 'probability_first_opportunity_cost_logistic_v1'

    @classmethod
    def matrix(cls, values):
        matrix = np.asarray(values, dtype=float)
        if (matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all()
            or ((matrix < 0) | (matrix > 1)).any()):
            raise ValueError('세 확률 첫 선형 모델의 입력 차원·범위 오류')
        return matrix


def checked_existing_manager(reference):
    proof = json.loads((reference/'reference_evidence.json').read_text())
    previous = Path(proof['reference'])
    pinned_json(previous/'reference_evidence.json', proof['files']['reference_evidence.json'])
    manager, parent = checked_manager_parent(previous)
    if parent != json.loads((reference/'manager_evidence.json').read_text()):
        raise ValueError('세 확률 첫 선형 모델의 기존 부모 연결 불일치')
    return manager, parent
