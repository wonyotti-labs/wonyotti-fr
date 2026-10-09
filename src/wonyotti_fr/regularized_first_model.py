from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .first_linear_close import FIRST_LINEAR_SETTINGS, FirstLinearCloseModel
from .first_opportunity_close import first_cost_training
from .minute_first_inputs import MinuteFirstLinearModel

REGULARIZATION_STRENGTHS = (.001, .01, .1, 1.)


def regularized_settings(strength):
    if type(strength) not in (int, float) or strength not in REGULARIZATION_STRENGTHS:
        raise ValueError('시간순 정규화 모델의 허용하지 않는 강도')
    return {**FIRST_LINEAR_SETTINGS, 'C': float(strength)}


class RegularizedFirstModel(FirstLinearCloseModel):
    features = MinuteFirstLinearModel.features
    format = 'regularized_first_opportunity_cost_logistic_v1'

    @classmethod
    def matrix(cls, values):
        return MinuteFirstLinearModel.matrix(values)

    @classmethod
    def fit(cls, values, target, validation_values, *, strength):
        settings = regularized_settings(strength)
        x, vx = cls.matrix(values), cls.matrix(validation_values)
        ledger, support = first_cost_training(target)
        if len(x) != len(ledger) or not len(vx):
            raise ValueError('시간순 정규화 모델의 학습·숫자 검증 행 오류')
        fit = ledger.fit_used.to_numpy()
        # 표준화는 0 비용을 포함한 모든 첫 포지션에 같은 비중으로 적합한다.
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error', ConvergenceWarning)
            scaler = StandardScaler().fit(x)
            transformed = scaler.transform(x)
            learner = LogisticRegression(**settings).fit(transformed[fit], ledger.positive_effect.to_numpy()[fit],
                sample_weight=ledger.fit_weight.to_numpy()[fit])
            if learner.n_iter_.shape != (1,) or not 0 < learner.n_iter_[0] < settings['max_iter']:
                raise ValueError('시간순 정규화 모델의 수렴·반복 한도 오류')
            expected, validation_expected = learner.predict_proba(transformed)[:, 1], learner.predict_proba(scaler.transform(vx))[:, 1]
        if learner.coef_.shape != (1, len(cls.features)) or learner.intercept_.shape != (1,) or learner.classes_.tolist() != [False, True]:
            raise ValueError('시간순 정규화 모델의 계수·클래스 오류')
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': settings,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coefficient': learner.coef_[0].tolist(),
            'intercept': float(learner.intercept_[0])})
        error = float(np.max(np.abs(model.probabilities(x)[:, 0]-expected)))
        validation_error = float(np.max(np.abs(model.probabilities(vx)[:, 0]-validation_expected)))
        if error > 1e-12 or validation_error > 1e-12:
            raise ValueError('시간순 정규화 모델의 숫자 내보내기 불일치')
        return model, {**support, 'iterations': int(learner.n_iter_[0]), 'converged': True,
            'export_max_error': error, 'validation_export_max_error': validation_error,
            'scaler_rows': len(x), 'scaler_zero_cost_rows_included': True}, ledger

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get('settings'), dict):
            raise ValueError('시간순 정규화 모델의 설정 형식 오류')
        settings = regularized_settings(data['settings'].get('C'))
        if (set(data) != {'format', 'features', 'settings', 'mean', 'scale', 'coefficient', 'intercept'}
            or data['format'] != cls.format or data['features'] != cls.features or data['settings'] != settings):
            raise ValueError('시간순 정규화 모델의 형식·특징·설정 오류')
        for name in ['mean', 'scale', 'coefficient']:
            values = data[name]
            if (not isinstance(values, list) or len(values) != len(cls.features)
                or any(type(value) not in (int, float) for value in values) or not np.isfinite(values).all()
                or (name == 'scale' and np.any(np.asarray(values) <= 0))):
                raise ValueError('시간순 정규화 모델의 숫자 배열·척도 오류')
        if type(data['intercept']) not in (int, float) or not np.isfinite(data['intercept']):
            raise ValueError('시간순 정규화 모델의 절편 오류')
        return cls(data)

