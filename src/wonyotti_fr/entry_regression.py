from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .net_edge_model import NET_FEATURES

REGRESSION_SETTINGS = {'loss': 'squared_error', 'max_iter': 64, 'learning_rate': .05,
    'max_leaf_nodes': 4, 'max_depth': 2, 'min_samples_leaf': 64, 'l2_regularization': 1.,
    'max_bins': 255, 'max_features': 1., 'categorical_features': None,
    'early_stopping': False, 'random_state': 0}


class EntryRegressionModel:
    features = NET_FEATURES
    format = 'entry_histogram_regression_v1'
    settings = REGRESSION_SETTINGS

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def fit(cls, values, target, weights, validation_values):
        x, y, w, vx = (np.asarray(v, dtype=float) for v in [values, target, weights, validation_values])
        if (x.ndim != 2 or x.shape[1] != len(cls.features) or len(x) < 200
            or y.shape != (len(x),) or w.shape != (len(x),)
            or not all(np.isfinite(v).all() for v in [x, y, w]) or (w <= 0).any()
            or not np.isclose(w.mean(), 1., rtol=0, atol=1e-12)
            or vx.ndim != 2 or vx.shape[1] != len(cls.features) or not len(vx) or not np.isfinite(vx).all()):
            raise ValueError('진입 회귀의 학습 차원·지원·숫자·가중치 오류')
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingRegressor(**cls.settings).fit(x, y, sample_weight=w)
            expected, validation_expected = learner.predict(x), learner.predict(vx)
        if (learner.n_iter_ != cls.settings['max_iter'] or learner._baseline_prediction.shape != (1, 1)
            or len(learner._predictors) != cls.settings['max_iter'] or any(len(part) != 1 for part in learner._predictors)):
            raise ValueError('진입 회귀 내보내기의 반복·초기 점수 구조 오류')
        trees = []
        required = {'value', 'count', 'feature_idx', 'num_threshold', 'missing_go_to_left',
            'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold', 'is_categorical', 'bitset_idx'}
        for part in learner._predictors:
            nodes = part[0].nodes
            if set(nodes.dtype.names) != required or nodes['is_categorical'].any() or nodes['depth'].max() > cls.settings['max_depth']:
                raise ValueError('진입 회귀 내보내기의 지원하지 않는 노드 구조')
            leaf = nodes['is_leaf'].astype(bool)
            # 말단 값에 반영된 학습률을 다시 곱하지 않는다.
            trees.append({'left': np.where(leaf, -1, nodes['left'].astype(int)).tolist(),
                'right': np.where(leaf, -1, nodes['right'].astype(int)).tolist(),
                'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)).tolist(),
                'threshold': np.where(leaf, 0., nodes['num_threshold']).tolist(),
                'value': np.where(leaf, nodes['value'], 0.).tolist()})
        model = cls.from_dict({'format': cls.format, 'features': cls.features, 'settings': cls.settings,
            'baseline': float(learner._baseline_prediction[0, 0]), 'trees': trees})
        error = float(np.max(np.abs(model.predict(x)-expected)))
        validation_error = float(np.max(np.abs(model.predict(vx)-validation_expected)))
        if max(error, validation_error) > 1e-10:
            raise ValueError('진입 회귀의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'export_max_error': error,
                       'validation_export_max_error': validation_error, 'early_stopping': False}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('features') != cls.features
            or data.get('settings') != cls.settings
            or type(data.get('baseline')) not in (int, float) or not np.isfinite(data['baseline'])
            or not isinstance(data.get('trees'), list) or len(data['trees']) != cls.settings['max_iter']):
            raise ValueError('진입 회귀의 형식·특징·설정·초기 점수 오류')
        for tree in data['trees']:
            if not isinstance(tree, dict) or set(tree) != {'left', 'right', 'feature', 'threshold', 'value'}:
                raise ValueError('진입 회귀의 트리 구조 오류')
            count = len(tree['left'])
            if (not 1 <= count <= 2*cls.settings['max_leaf_nodes']-1 or any(len(tree[k]) != count for k in ['right', 'feature', 'threshold', 'value'])
                or not np.isfinite(tree['threshold']).all() or not np.isfinite(tree['value']).all()):
                raise ValueError('진입 회귀의 노드 크기·숫자 오류')
            visited, pending = set(), [(0, 0)]
            while pending:
                node, depth = pending.pop()
                if node in visited or not 0 <= node < count or depth > cls.settings['max_depth']:
                    raise ValueError('진입 회귀의 순환·공유·범위·깊이 오류')
                visited.add(node)
                left, right, feature = (tree[k][node] for k in ['left', 'right', 'feature'])
                if any(type(v) is not int for v in [left, right, feature]):
                    raise ValueError('진입 회귀의 노드 인덱스 오류')
                if left == right == -1:
                    if feature != -2:
                        raise ValueError('진입 회귀의 말단 특징 오류')
                elif min(left, right) < 0 or not 0 <= feature < len(cls.features):
                    raise ValueError('진입 회귀의 특징·자식 범위 오류')
                else:
                    pending.extend([(left, depth+1), (right, depth+1)])
            if len(visited) != count:
                raise ValueError('진입 회귀의 도달 불가능한 노드')
        return cls(data)

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('진입 회귀의 예측 차원 오류')
        valid = np.isfinite(values).all(axis=1)
        matrix = values[valid]
        prediction = np.full(len(matrix), self.data['baseline'], dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            for tree in self.data['trees']:
                left, right, feature, threshold, value = (np.asarray(tree[k]) for k in ['left', 'right', 'feature', 'threshold', 'value'])
                node = np.zeros(len(matrix), dtype=int)
                for _ in range(self.settings['max_depth']):
                    split = left[node] != -1
                    indices = np.flatnonzero(split)
                    current = node[indices]
                    node[indices] = np.where(matrix[indices, feature[current]] <= threshold[current], left[current], right[current])
                prediction += value[node]
        result = np.full(len(values), np.nan)
        result[valid] = prediction
        result[~np.isfinite(result)] = np.nan
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)
