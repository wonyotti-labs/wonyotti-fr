from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .addition_effect import position_weights
from .close_flow import validate_flow_matrix
from .common import save_json
from .continuation_inputs import PENDING_FEATURES, validate_pending_matrix
from .early_stopping_close import validate_early_stopping_training
from .first_opportunity_close import (
    FirstOpportunityCloseModel,
    first_cost_training,
    first_opportunity_rows,
)

FIRST_LINEAR_SETTINGS = {'C': .1, 'l1_ratio': 0., 'solver': 'lbfgs', 'fit_intercept': True,
    'class_weight': None, 'random_state': 0, 'warm_start': False, 'max_iter': 2000, 'tol': 1e-9}


class FirstLinearCloseModel:
    features = FirstOpportunityCloseModel.features
    format = 'first_opportunity_cost_logistic_v1'

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def matrix(cls, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(cls.features) or not np.isfinite(matrix).all():
            raise ValueError('첫 기회 선형 모델의 현재 입력 오류')
        validate_flow_matrix(matrix, cls.features)
        validate_pending_matrix(matrix[:, [cls.features.index(name) for name in PENDING_FEATURES]])
        return matrix

    @classmethod
    def fit(cls, values, target, validation_values):
        x, vx = cls.matrix(values), cls.matrix(validation_values)
        ledger, support = first_cost_training(target)
        if len(x) != len(ledger) or not len(vx):
            raise ValueError('첫 기회 선형 모델의 학습·숫자 검증 행 오류')
        fit = ledger.fit_used.to_numpy()
        # 표준화는 0 비용을 포함한 모든 첫 포지션에 같은 비중으로 적합한다.
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error', ConvergenceWarning)
            scaler = StandardScaler().fit(x)
            transformed = scaler.transform(x)
            learner = LogisticRegression(**FIRST_LINEAR_SETTINGS).fit(transformed[fit], ledger.positive_effect.to_numpy()[fit],
                sample_weight=ledger.fit_weight.to_numpy()[fit])
            if learner.n_iter_.shape != (1,) or not 0 < learner.n_iter_[0] < FIRST_LINEAR_SETTINGS['max_iter']:
                raise ValueError('첫 기회 선형 모델의 수렴·반복 한도 오류')
            expected, validation_expected = learner.predict_proba(transformed)[:, 1], learner.predict_proba(scaler.transform(vx))[:, 1]
        if learner.coef_.shape != (1, len(cls.features)) or learner.intercept_.shape != (1,) or learner.classes_.tolist() != [False, True]:
            raise ValueError('첫 기회 선형 모델의 계수·클래스 오류')
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': FIRST_LINEAR_SETTINGS,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coefficient': learner.coef_[0].tolist(),
            'intercept': float(learner.intercept_[0])})
        error = float(np.max(np.abs(model.probabilities(x)[:, 0]-expected)))
        validation_error = float(np.max(np.abs(model.probabilities(vx)[:, 0]-validation_expected)))
        if error > 1e-12 or validation_error > 1e-12:
            raise ValueError('첫 기회 선형 모델의 숫자 내보내기 불일치')
        return model, {**support, 'iterations': int(learner.n_iter_[0]), 'converged': True,
            'export_max_error': error, 'validation_export_max_error': validation_error,
            'scaler_rows': len(x), 'scaler_zero_cost_rows_included': True}, ledger

    @classmethod
    def from_dict(cls, data):
        if (not isinstance(data, dict) or set(data) != {'format', 'features', 'settings', 'mean', 'scale', 'coefficient', 'intercept'}
            or data['format'] != cls.format or data['features'] != cls.features or data['settings'] != FIRST_LINEAR_SETTINGS):
            raise ValueError('첫 기회 선형 모델의 형식·특징·설정 오류')
        for name in ['mean', 'scale', 'coefficient']:
            values = data[name]
            if (not isinstance(values, list) or len(values) != len(cls.features)
                or any(type(value) not in (int, float) for value in values) or not np.isfinite(values).all()
                or (name == 'scale' and np.any(np.asarray(values) <= 0))):
                raise ValueError('첫 기회 선형 모델의 숫자 배열·척도 오류')
        if type(data['intercept']) not in (int, float) or not np.isfinite(data['intercept']):
            raise ValueError('첫 기회 선형 모델의 절편 오류')
        return cls(data)

    def to_dict(self):
        return copy.deepcopy(self.data)

    def probabilities(self, values):
        matrix = self.matrix(values)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            transformed = (matrix-np.asarray(self.data['mean']))/np.asarray(self.data['scale'])
            linear = transformed@np.asarray(self.data['coefficient'])+self.data['intercept']
        if not np.isfinite(transformed).all() or not np.isfinite(linear).all():
            raise ValueError('첫 기회 선형 모델의 표준화·예측 넘침')
        return np.exp(-np.logaddexp(0., -linear))[:, None]


def fit_first_linear(training, weights, calibration, previous_ledger, output):
    validate_early_stopping_training(training)
    pd.testing.assert_frame_equal(training[['decision_time', 'position_entry_time']], weights.drop(columns='sample_weight'), check_exact=True)
    np.testing.assert_array_equal(weights.sample_weight, position_weights(training))
    first, positions = first_opportunity_rows(training)
    validation, _ = first_opportunity_rows(calibration)
    directions = first.groupby('direction').position_entry_time.nunique()
    if (len(first) < 500 or any(directions.get(side, 0) < 20 for side in [-1, 1])
        or first.decision_time.max()-first.decision_time.min() < pd.Timedelta(days=180)):
        raise ValueError('첫 기회 선형 모델의 포지션·방향·기간 지원 부족')
    costs, _ = first_cost_training(first.first_target_common_bps)
    keys = ['opportunity_index', 'decision_time', 'position_entry_time', 'label_end', 'direction',
        'decision_equity', 'reference_equity', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']
    expected = pd.concat([first[keys], costs], axis=1)
    pd.testing.assert_frame_equal(expected, previous_ledger, check_exact=True)
    model, support, fitted_costs = FirstLinearCloseModel.fit(first[FirstLinearCloseModel.features].to_numpy(),
        first.first_target_common_bps.to_numpy(), validation[FirstLinearCloseModel.features].to_numpy())
    pd.testing.assert_frame_equal(fitted_costs, costs, check_exact=True)
    support = {**support, 'all_positions': len(positions), 'no_eligible_positions': int((~positions.has_eligible_opportunity).sum()),
        'original_rows': len(training), 'fit_time': '2021-08-02T00:00:00+00:00', 'last_label_end': training.label_end.max(),
        'diagnosis_used_for_export': False, 'export_validation_period': 'first_calibration_opportunities',
        'new_models_fitted': 1, 'refit_after_calibration': False, 'previous_first_ledger_exact': True}
    save_json(output/'model.json', model.to_dict())
    save_json(output/'training_support.json', support)
    return model, support
