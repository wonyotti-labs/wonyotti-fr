from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS, HistogramManagementModels


class PositionBoostModel(HistogramManagementModels):
    format = 'position_target_histogram_v1'
    features = MARKET_FEATURES
    classes = ['flat', 'short', 'long']
    actions = classes

    @classmethod
    def fit(cls, values, labels, validation_values):
        x, y = np.asarray(values, dtype=float), np.asarray(labels)
        vx = np.asarray(validation_values, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(cls.features) or len(x) < 1000
            or not np.isfinite(x).all() or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or set(np.unique(y)) != {0, 1, 2} or (np.bincount(y, minlength=3) < 20).any()
            or vx.ndim != 2 or vx.shape[1] != len(cls.features) or not len(vx)
            or not np.isfinite(vx).all()):
            raise ValueError('보유 방향 부스팅의 입력·클래스·지원 오류')
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y)
            expected, validation_expected = learner.predict_proba(x), learner.predict_proba(vx)
        if (not np.array_equal(learner.classes_, np.arange(3)) or learner.n_iter_ != 64
            or learner.n_trees_per_iteration_ != 3 or learner.do_early_stopping_
            or learner._baseline_prediction.shape != (1, 3)
            or len(learner._predictors) != 64 or any(len(part) != 3 for part in learner._predictors)):
            raise ValueError('보유 방향 부스팅의 클래스·반복·초기 점수 오류')
        models = []
        for index, action in enumerate(cls.classes):
            trees = []
            for iteration in learner._predictors:
                nodes = iteration[index].nodes
                if nodes['is_categorical'].any() or nodes['depth'].max() > 2:
                    raise ValueError('보유 방향 부스팅의 범주·트리 깊이 오류')
                leaves = nodes['is_leaf'].astype(bool)
                # 라이브러리의 말단 값에는 학습률이 이미 반영되어 있다.
                trees.append({'left': np.where(leaves, -1, nodes['left'].astype(int)).tolist(),
                    'right': np.where(leaves, -1, nodes['right'].astype(int)).tolist(),
                    'feature': np.where(leaves, -2, nodes['feature_idx'].astype(int)).tolist(),
                    'threshold': np.where(leaves, 0., nodes['num_threshold']).tolist(),
                    'value': np.where(leaves, nodes['value'], 0.).tolist()})
            models.append({'action': action, 'baseline': float(learner._baseline_prediction[0, index]), 'trees': trees})
        model = cls.from_dict({'format': cls.format, 'features': cls.features,
            'settings': HISTOGRAM_SETTINGS, 'models': models})
        errors = [float(np.max(np.abs(model.probabilities(part) - scores)))
                  for part, scores in [(x, expected), (vx, validation_expected)]]
        if max(errors) > 1e-12:
            raise ValueError('보유 방향 부스팅의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'class_counts': np.bincount(y, minlength=3).tolist(),
            'iterations': 64, 'trees_per_iteration': 3, 'export_max_error': errors[0],
            'validation_export_max_error': errors[1], 'early_stopping': False}

    def probabilities(self, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError('보유 방향 부스팅의 예측 차원 오류')
        result = np.full((len(matrix), 3), np.nan)
        valid = np.flatnonzero(np.isfinite(matrix).all(axis=1))
        selected = matrix[valid]
        if not len(selected):
            return result
        baseline = np.asarray([model['baseline'] for model in self.data['models']], dtype=float)
        logits = np.tile(baseline, (len(selected), 1))
        for index, model in enumerate(self.data['models']):
            for tree in model['trees']:
                nodes = np.zeros(len(selected), dtype=int)
                left, right, feature = (np.asarray(tree[key]) for key in ['left', 'right', 'feature'])
                threshold = np.asarray(tree['threshold'])
                for _ in range(2):
                    active = left[nodes] != -1
                    current = nodes[active]
                    choose_left = selected[active, feature[current]] <= threshold[current]
                    nodes[active] = np.where(choose_left, left[current], right[current])
                with np.errstate(over='ignore', invalid='ignore'):
                    logits[:, index] += np.asarray(tree['value'])[nodes]
        available = np.isfinite(logits).all(axis=1)
        current = logits[available]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            exponent = np.exp(current - current.max(axis=1, keepdims=True))
        result[valid[available]] = exponent / exponent.sum(axis=1, keepdims=True)
        return result
