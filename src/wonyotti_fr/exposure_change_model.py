from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.event_features import MARKET_FEATURES

EXPOSURE_FEATURES = MARKET_FEATURES + ['current_net_exposure']

SETTINGS = {'loss': 'squared_error', 'max_iter': 64, 'learning_rate': .05,
    'max_leaf_nodes': 4, 'max_depth': 2, 'min_samples_leaf': 64, 'l2_regularization': 1.,
    'max_bins': 255, 'max_features': 1., 'categorical_features': None,
    'early_stopping': False, 'random_state': 0}


def matrix(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(EXPOSURE_FEATURES) or not np.isfinite(data).all():
        raise ValueError('계좌 노출 변경량 시장 입력 차원·유한성 오류')
    return data


def validate_training(x, y):
    data, target = matrix(x), np.asarray(y, dtype=float)
    if (target.shape != (len(data),) or not np.isfinite(target).all() or len(data) < 1000
        or (target > 0).sum() < 20 or (target < 0).sum() < 20):
        raise ValueError('계좌 노출 변경량 학습 행·양방향 지원 오류')
    return data, target


class ExposureChangeModel:
    format = 'account_btc_exposure_change_histogram_v1'
    features = EXPOSURE_FEATURES

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self._trees = tuple(tuple(tuple(tree[key]) for key in ['left', 'right', 'feature', 'threshold', 'value'])
            for tree in self.data['trees'])
        self._baseline = float(self.data['baseline'])

    @classmethod
    def fit(cls, x, y, validation):
        x, y = validate_training(x, y)
        vx = matrix(validation)
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingRegressor(**SETTINGS).fit(x, y)
            reference = [learner.predict(part) for part in (x, vx)]
        if learner.n_iter_ != 64 or learner.n_trees_per_iteration_ != 1 or learner.do_early_stopping_:
            raise ValueError('계좌 노출 변경량 회귀 학습 반복·구성 오류')
        trees = []
        for iteration in learner._predictors:
            nodes = iteration[0].nodes
            if nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                raise ValueError('계좌 노출 변경량 회귀 범주·깊이 오류')
            leaf = nodes['is_leaf'].astype(bool)
            trees.append({'left': np.where(leaf, -1, nodes['left'].astype(int)).tolist(),
                'right': np.where(leaf, -1, nodes['right'].astype(int)).tolist(),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)).tolist(),
                'threshold': np.where(leaf, 0., nodes['num_threshold']).tolist(),
                'value': np.where(leaf, nodes['value'], 0.).tolist()})
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': SETTINGS,
            'baseline': float(learner._baseline_prediction[0, 0]), 'trees': trees})
        errors = [float(np.max(np.abs(model.predict(part)-expected))) for part, expected in zip((x, vx), reference, strict=True)]
        if max(errors) > 1e-12:
            raise ValueError('계좌 노출 변경량 숫자 트리 예측 불일치')
        return model, {'training_rows': len(x), 'positive_rows': int((y > 0).sum()), 'negative_rows': int((y < 0).sum()),
            'training_export_error': errors[0], 'diagnosis_export_error': errors[1]}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) != {'format', 'features', 'settings', 'baseline', 'trees'}
            or value['format'] != cls.format or value['features'] != cls.features or value['settings'] != SETTINGS
            or type(value['baseline']) not in (int, float) or not np.isfinite(value['baseline'])
            or not isinstance(value['trees'], list) or len(value['trees']) != 64):
            raise ValueError('계좌 노출 변경량 숫자 모형 형식 오류')
        for tree in value['trees']:
            if not isinstance(tree, dict) or set(tree) != {'left', 'right', 'feature', 'threshold', 'value'}:
                raise ValueError('계좌 노출 변경량 숫자 트리 필드 오류')
            if any(not isinstance(a, list) for a in tree.values()):
                raise ValueError('계좌 노출 변경량 트리 배열 오류')
            n = len(tree['left'])
            if not 1 <= n <= 7 or any(len(a) != n for a in tree.values()):
                raise ValueError('계좌 노출 변경량 숫자 트리 크기 오류')
            if any(type(v) is not int for key in ['left', 'right', 'feature'] for v in tree[key]):
                raise ValueError('계좌 노출 변경량 트리 인덱스 오류')
            if any(type(v) not in (int, float) or not np.isfinite(v) for key in ['threshold', 'value'] for v in tree[key]):
                raise ValueError('계좌 노출 변경량 트리 숫자 오류')
            seen, pending = set(), [(0, 0)]
            while pending:
                i, depth = pending.pop()
                if not 0 <= i < n or i in seen or depth > 2:
                    raise ValueError('계좌 노출 변경량 트리 순환·깊이 오류')
                seen.add(i)
                left, right, feature = (tree[key][i] for key in ['left', 'right', 'feature'])
                if left == -1:
                    if right != -1 or feature != -2 or tree['threshold'][i] != 0:
                        raise ValueError('계좌 노출 변경량 트리 말단 오류')
                else:
                    if not 0 <= feature < len(cls.features) or tree['value'][i] != 0:
                        raise ValueError('계좌 노출 변경량 트리 분기 오류')
                    pending.extend([(left, depth+1), (right, depth+1)])
            if len(seen) != n:
                raise ValueError('계좌 노출 변경량 트리 미사용 노드 오류')
        return cls(value)

    def to_dict(self):
        return copy.deepcopy(self.data)

    def predict(self, values):
        x = matrix(values)
        result = np.full(len(x), self.data['baseline'], dtype=float)
        for tree in self.data['trees']:
            left, right, feature = (np.asarray(tree[key], dtype=int) for key in ['left', 'right', 'feature'])
            threshold = np.asarray(tree['threshold'])
            at = np.zeros(len(x), dtype=int)
            for _ in range(2):
                active = left[at] != -1
                indexes = at[active]
                at[active] = np.where(x[active, feature[indexes]] <= threshold[indexes], left[indexes], right[indexes])
            result += np.asarray(tree['value'])[at]
        if not np.isfinite(result).all():
            raise ValueError('계좌 노출 변경량 예측 유한성 오류')
        return result

    def predict_one(self, market, current_exposure):
        market = np.asarray(market, dtype=float)
        if (market.shape != (len(MARKET_FEATURES),) or not np.isfinite(market).all()
            or type(current_exposure) not in (int, float) or not np.isfinite(current_exposure)):
            raise ValueError('자기 계좌 노출 변경량의 현재 입력 오류')
        values = (*market, current_exposure)
        result = self._baseline
        for left, right, feature, threshold, value in self._trees:
            node = 0
            while left[node] != -1:
                node = left[node] if values[feature[node]] <= threshold[node] else right[node]
            result += value[node]
        if not np.isfinite(result):
            raise ValueError('자기 계좌 노출 변경량의 예측 오류')
        return result


def fit_ridge(x, y, validation):
    x, y = validate_training(x, y)
    vx = matrix(validation)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        learner = Ridge(alpha=1., solver='svd').fit(scaler.transform(x), y)
    data = {'format': 'account_btc_exposure_change_ridge_v1', 'features': EXPOSURE_FEATURES,
        'alpha': 1., 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'coef': learner.coef_.tolist(), 'intercept': float(learner.intercept_)}
    prediction = ((vx-np.asarray(data['mean']))/np.asarray(data['scale']))@np.asarray(data['coef'])+data['intercept']
    if not np.allclose(prediction, learner.predict(scaler.transform(vx)), atol=1e-12, rtol=0):
        raise ValueError('계좌 노출 변경량 릿지 내보내기 오류')
    return data, prediction
