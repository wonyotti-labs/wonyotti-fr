from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .minute_inventory import MinuteInventoryModels
from .minute_management import ACTIONS

HISTOGRAM_SETTINGS = {'loss': 'log_loss', 'max_iter': 64, 'learning_rate': .05,
    'max_leaf_nodes': 4, 'max_depth': 2, 'min_samples_leaf': 64, 'l2_regularization': 1.,
    'max_bins': 255, 'max_features': 1., 'categorical_features': None,
    'early_stopping': False, 'random_state': 0, 'class_weight': None}


class HistogramManagementModels:
    features = MinuteInventoryModels.features
    format = 'histogram_management_v1'
    kind = 'histogram'

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def fit(cls, values, labels, validation_values):
        x, y = np.asarray(values, dtype=float), np.asarray(labels)
        if (x.ndim != 2 or x.shape[1] != len(cls.features) or len(x) < 1000
            or y.shape != (len(x), len(ACTIONS)) or not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any()
            or ((1 - y).sum(axis=0) < 20).any()):
            raise ValueError('관리 부스팅의 학습 차원·유한성·지원 부족')
        vx = np.asarray(validation_values, dtype=float)
        if vx.ndim != 2 or vx.shape[1] != len(cls.features) or not len(vx) or not np.isfinite(vx).all():
            raise ValueError('관리 부스팅의 숫자 검증 입력 오류')
        binaries, expected, validation_expected = [], [], []
        for i, action in enumerate(ACTIONS):
            with threadpool_limits(limits=1):
                learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y[:, i])
                expected.append(learner.predict_proba(x)[:, 1])
                validation_expected.append(learner.predict_proba(vx)[:, 1])
            if (learner.n_iter_ != 64 or learner._baseline_prediction.shape != (1, 1)
                or len(learner._predictors) != 64 or any(len(part) != 1 for part in learner._predictors)):
                raise ValueError('관리 부스팅 내보내기의 반복·초기 점수 구조 오류')
            trees = []
            required = {'value', 'count', 'feature_idx', 'num_threshold', 'missing_go_to_left',
                'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold', 'is_categorical', 'bitset_idx'}
            for part in learner._predictors:
                nodes = part[0].nodes
                if set(nodes.dtype.names) != required or nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                    raise ValueError('관리 부스팅 내보내기의 지원하지 않는 노드 구조')
                leaf = nodes['is_leaf'].astype(bool)
                # 말단 값에 학습률이 이미 반영되어 있어 한 번만 합산한다.
                trees.append({'left': np.where(leaf, -1, nodes['left'].astype(int)).tolist(),
                    'right': np.where(leaf, -1, nodes['right'].astype(int)).tolist(),
                    'feature': np.where(leaf, -2, nodes['feature_idx'].astype(int)).tolist(),
                    'threshold': np.where(leaf, 0., nodes['num_threshold']).tolist(),
                    'value': np.where(leaf, nodes['value'], 0.).tolist()})
            binaries.append({'action': action, 'baseline': float(learner._baseline_prediction[0, 0]), 'trees': trees})
        model = cls.from_dict({'format': cls.format, 'features': cls.features,
                              'settings': HISTOGRAM_SETTINGS, 'models': binaries})
        error = float(np.max(np.abs(model.probabilities(x) - np.column_stack(expected))))
        validation_error = float(np.max(np.abs(model.probabilities(vx) - np.column_stack(validation_expected))))
        if error > 1e-12 or validation_error > 1e-12:
            raise ValueError('관리 부스팅의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'positive': dict(zip(ACTIONS, y.sum(axis=0).astype(int).tolist(), strict=True)),
            'negative': dict(zip(ACTIONS, (1 - y).sum(axis=0).astype(int).tolist(), strict=True)),
            'export_max_error': error, 'validation_export_max_error': validation_error, 'early_stopping': False}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('features') != cls.features
            or data.get('settings') != HISTOGRAM_SETTINGS or not isinstance(data.get('models'), list)
            or len(data['models']) != len(ACTIONS)):
            raise ValueError('관리 부스팅의 형식·특징·고정 설정 오류')
        for model, action in zip(data['models'], ACTIONS, strict=True):
            if (model.get('action') != action or type(model.get('baseline')) not in (int, float)
                or not np.isfinite(model['baseline']) or not isinstance(model.get('trees'), list)
                or len(model['trees']) != 64):
                raise ValueError('관리 부스팅의 행동·초기 점수·트리 수 오류')
            for tree in model['trees']:
                count = len(tree['left'])
                if (not 1 <= count <= 7 or any(len(tree[k]) != count for k in ['right', 'feature', 'threshold', 'value'])
                    or not np.isfinite(tree['threshold']).all() or not np.isfinite(tree['value']).all()):
                    raise ValueError('관리 부스팅의 노드 크기·숫자 오류')
                visited, pending = set(), [(0, 0)]
                while pending:
                    node, depth = pending.pop()
                    if node in visited or not 0 <= node < count or depth > 2:
                        raise ValueError('관리 부스팅의 순환·공유·범위·깊이 오류')
                    visited.add(node)
                    left, right, feature = (tree[k][node] for k in ['left', 'right', 'feature'])
                    if any(type(v) is not int for v in [left, right, feature]):
                        raise ValueError('관리 부스팅의 노드 인덱스 오류')
                    if left == right == -1:
                        if feature != -2:
                            raise ValueError('관리 부스팅의 말단 특징 오류')
                    elif min(left, right) < 0 or not 0 <= feature < len(cls.features):
                        raise ValueError('관리 부스팅의 특징·자식 범위 오류')
                    else:
                        pending.extend([(left, depth + 1), (right, depth + 1)])
                if len(visited) != count:
                    raise ValueError('관리 부스팅의 도달 불가능한 노드')
        return cls(data)

    def probabilities(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('관리 부스팅의 예측 차원 오류')
        valid = np.isfinite(values).all(axis=1)
        result = np.full((len(values), len(ACTIONS)), np.nan)
        matrix = values[valid]
        for i, model in enumerate(self.data['models']):
            logits = np.full(len(matrix), model['baseline'], dtype=float)
            for tree in model['trees']:
                left, right, feature = (np.asarray(tree[k]) for k in ['left', 'right', 'feature'])
                threshold, nodes = np.asarray(tree['threshold']), np.zeros(len(matrix), dtype=int)
                for _ in range(2):
                    active = left[nodes] != -1
                    selected = nodes[active]
                    go_left = matrix[active, feature[selected]] <= threshold[selected]
                    nodes[active] = np.where(go_left, left[selected], right[selected])
                with np.errstate(over='ignore', invalid='ignore'):
                    logits += np.asarray(tree['value'])[nodes]
            result[valid, i] = np.where(np.isfinite(logits), 1 / (1 + np.exp(-np.clip(logits, -700, 700))), np.nan)
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)
