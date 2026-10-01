from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .event_features import MARKET_FEATURES


@dataclass
class BinaryModel:
    data: dict

    @classmethod
    def fit(cls, values: np.ndarray, labels: np.ndarray, kind: str) -> tuple[BinaryModel, dict]:
        values, labels = np.asarray(values, dtype=float), np.asarray(labels)
        if (values.ndim != 2 or values.shape[1] != len(MARKET_FEATURES) or len(values) < 1000
            or labels.shape != (len(values),) or not np.isfinite(values).all()
            or set(np.unique(labels)) != {0, 1} or min(np.bincount(labels.astype(int))) < 20):
            raise ValueError('이진 학습 자료의 차원·유한성·지원 표본 부족')
        payload = {'format': 'expansion_binary_v1', 'kind': kind, 'features': MARKET_FEATURES}
        if kind == 'logistic':
            scaler = StandardScaler().fit(values)
            learner = LogisticRegression(C=0.1, max_iter=2000, random_state=0).fit(scaler.transform(values), labels)
            if learner.n_iter_.max() >= 2000:
                raise ValueError('이진 로지스틱 모델의 수렴 실패')
            payload.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                           coefficients=learner.coef_[0].tolist(), intercept=float(learner.intercept_[0]))
            expected = learner.predict_proba(scaler.transform(values))[:, 1]
        elif kind == 'boosted':
            learner = GradientBoostingClassifier(n_estimators=64, learning_rate=0.05, max_depth=2,
                                                 min_samples_leaf=32, init='zero', random_state=0).fit(values, labels)
            trees = []
            for estimator in learner.estimators_[:, 0]:
                tree = estimator.tree_
                trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
                              'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
                              'value': tree.value[:, 0, 0].tolist()})
            payload.update(learning_rate=0.05, trees=trees)
            expected = learner.predict_proba(values)[:, 1]
        else:
            raise ValueError('지원하지 않는 이진 모델')
        result = cls.from_dict(payload)
        error = float(np.max(np.abs(result.probabilities(values) - expected)))
        if error > 1e-12:
            raise ValueError('저장 모델과 학습 라이브러리의 예측 불일치')
        return result, {'rows': len(labels), 'negative': int((labels == 0).sum()),
                        'positive': int((labels == 1).sum()), 'export_max_error': error}

    @classmethod
    def from_dict(cls, data: dict) -> BinaryModel:
        if (data.get('format') != 'expansion_binary_v1' or data.get('features') != MARKET_FEATURES
            or data.get('kind') not in {'logistic', 'boosted'}):
            raise ValueError('지원하지 않는 이진 모델 형식')
        if data['kind'] == 'logistic':
            arrays = [np.asarray(data[key], dtype=float) for key in ['mean', 'scale', 'coefficients']]
            if (any(a.shape != (len(MARKET_FEATURES),) or not np.isfinite(a).all() for a in arrays)
                or (arrays[1] <= 0).any() or not np.isscalar(data['intercept']) or not np.isfinite(data['intercept'])):
                raise ValueError('이진 모델의 계수 오류')
        else:
            trees = data.get('trees')
            if (not isinstance(trees, list) or not 1 <= len(trees) <= 64
                or type(data.get('learning_rate')) not in (int, float) or not 0 < data['learning_rate'] <= 1):
                raise ValueError('트리 수 또는 학습률 오류')
            for tree in trees:
                count = len(tree['left'])
                if not 1 <= count <= 7 or any(len(tree[key]) != count for key in ['right', 'feature', 'threshold', 'value']):
                    raise ValueError('트리 크기 오류')
                if not np.isfinite(np.asarray([tree['threshold'], tree['value']], dtype=float)).all():
                    raise ValueError('트리의 비정상 숫자')
                visited, pending = set(), [(0, 0)]
                while pending:
                    node, depth = pending.pop()
                    if node in visited or not 0 <= node < count or depth > 2:
                        raise ValueError('트리의 순환·공유·범위·깊이 오류')
                    visited.add(node)
                    left, right, feature = (tree[key][node] for key in ['left', 'right', 'feature'])
                    if any(type(value) is not int for value in [left, right, feature]):
                        raise ValueError('트리 인덱스는 정수여야 합니다.')
                    if left == right == -1:
                        if feature != -2:
                            raise ValueError('말단 특징 오류')
                    elif not 0 <= feature < len(MARKET_FEATURES) or min(left, right) < 0:
                        raise ValueError('트리 특징 또는 자식 오류')
                    else:
                        pending.extend([(left, depth + 1), (right, depth + 1)])
                if len(visited) != count:
                    raise ValueError('도달할 수 없는 트리 노드')
        return cls(copy.deepcopy(data))

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(MARKET_FEATURES):
            raise ValueError('이진 예측 특징 차원 오류')
        valid = np.isfinite(values).all(axis=1)
        result = np.full(len(values), np.nan)
        if not valid.any():
            return result
        data = self.data
        if data['kind'] == 'logistic':
            with np.errstate(over='ignore', invalid='ignore'):
                logits = ((values[valid] - data['mean']) / data['scale']) @ np.array(data['coefficients']) + data['intercept']
        else:
            # 학습 라이브러리의 트리 입력 정밀도와 비교 경계를 유지한다.
            with np.errstate(over='ignore'):
                matrix = values[valid].astype(np.float32)
            logits = np.zeros(len(matrix))
            for tree in data['trees']:
                nodes = np.zeros(len(matrix), dtype=int)
                left, right, feature = (np.asarray(tree[key]) for key in ['left', 'right', 'feature'])
                threshold = np.asarray(tree['threshold'])
                for _ in range(2):
                    active = left[nodes] != -1
                    selected = nodes[active]
                    go_left = matrix[active, feature[selected]] <= threshold[selected]
                    nodes[active] = np.where(go_left, left[selected], right[selected])
                logits += data['learning_rate'] * np.asarray(tree['value'])[nodes]
            logits[~np.isfinite(matrix).all(axis=1)] = np.nan
        with np.errstate(over='ignore', invalid='ignore'):
            result[valid] = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        result[valid & ~np.isfinite(result)] = np.nan
        return result

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)


class ExpansionPolicy:
    def __init__(self, activity: BinaryModel, direction: BinaryModel, activity_threshold: float,
                 direction_threshold: float, min_hold_bars: int = 12):
        if (not np.isfinite([activity_threshold, direction_threshold]).all()
            or not 0 <= activity_threshold <= 1 or not 0.5 < direction_threshold <= 1
            or type(min_hold_bars) is not int or min_hold_bars != 12):
            raise ValueError('노출 확대 정책의 기준 오류')
        self.activity, self.direction = activity, direction
        self.activity_threshold, self.direction_threshold = activity_threshold, direction_threshold
        self.min_hold_bars = min_hold_bars
        self._cache = {}

    def prepare(self, frame):
        values = frame[MARKET_FEATURES].to_numpy(dtype=float)
        self._values = values
        self._scores = np.column_stack([self.activity.probabilities(values), self.direction.probabilities(values)])
        self._cache = {stamp.isoformat(): index for index, stamp in enumerate(frame.end)}

    def __call__(self, bar: dict, state: dict) -> str:
        values = np.asarray(bar['features'], dtype=float)
        if state['halted'] or values.shape != (len(MARKET_FEATURES),) or not np.isfinite(values).all():
            return 'hold'
        index = self._cache.get(bar['end'])
        if index is not None and np.array_equal(values, self._values[index]):
            activity, buy = self._scores[index]
        else:
            activity, buy = (model.probabilities(values.reshape(1, -1))[0] for model in [self.activity, self.direction])
        if not np.isfinite([activity, buy]).all() or activity < self.activity_threshold:
            return 'hold'
        desired = 1 if buy >= self.direction_threshold else (-1 if 1 - buy >= self.direction_threshold else 0)
        if (not desired or desired == state['direction']
            or (state['direction'] and state['hold_bars'] < self.min_hold_bars)):
            return 'hold'
        return 'enter_long' if desired > 0 else 'enter_short'
