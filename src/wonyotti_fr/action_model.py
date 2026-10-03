from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .engine import PolicyDecision
from .minute_management import ACTIONS, FEATURES, management_values
from .pullback_policy import PullbackPolicy


def select_threshold(labels, scores, beta):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all()
        or set(np.unique(labels)) != {0, 1} or labels.sum() < 20 or beta not in (.5, 1., 2.)):
        raise ValueError('행동 문턱 조정의 지원 표본·점수 오류')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    counts = len(scores) - np.searchsorted(np.sort(scores), thresholds, side='left')
    p, r = precision[:-1], recall[:-1]
    denominator = beta**2 * p + r
    value = np.divide((1 + beta**2) * p * r, denominator, out=np.zeros_like(p), where=denominator > 0)
    value[counts < 20] = -1
    index = np.flatnonzero(value == value.max())[-1]
    if value[index] < 0:
        raise ValueError('행동 문턱의 예측 양성 지원 표본 부족')
    return float(thresholds[index]), {'beta': beta, 'f_beta': float(value[index]),
                                     'precision': float(p[index]), 'recall': float(r[index]),
                                     'predicted_positive': int(counts[index]), 'actual_positive': int(labels.sum())}


class ActionModels:
    features = FEATURES
    format = 'minute_action_v1'

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.kind = data['kind']
        if self.kind == 'logistic':
            self.mean, self.scale = np.asarray(data['mean']), np.asarray(data['scale'])
            self.coef, self.intercept = np.asarray(data['coef']), np.asarray(data['intercept'])

    @classmethod
    def fit(cls, train, calibration, kind):
        x = train[cls.features].to_numpy(dtype=float)
        y = train[[f'y_{a}' for a in ACTIONS]].to_numpy(dtype=int)
        if (len(x) < 1000 or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
            or (y.sum(axis=0) < 20).any() or ((1 - y).sum(axis=0) < 20).any()
            or train.label_end.max() >= calibration.end.min()):
            raise ValueError('행동별 학습의 특징·표본·시간 분리 오류')
        data = {'format': cls.format, 'features': cls.features, 'actions': ACTIONS, 'kind': kind}
        learners = []
        if kind == 'logistic':
            scaler = StandardScaler().fit(x)
            scaled = scaler.transform(x)
            for i in range(3):
                learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaled, y[:, i])
                if learner.n_iter_.max() >= 2000:
                    raise ValueError('행동 모델 수렴 실패')
                learners.append(learner)
            data.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                        coef=[m.coef_[0].tolist() for m in learners], intercept=[float(m.intercept_[0]) for m in learners])
            expected = np.column_stack([m.predict_proba(scaled)[:, 1] for m in learners])
        elif kind == 'tree':
            trees = []
            for i in range(3):
                learner = DecisionTreeClassifier(max_depth=4, min_samples_leaf=64, random_state=0).fit(x, y[:, i])
                tree = learner.tree_
                values = tree.value[:, 0, :]
                trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
                              'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
                              'probability': (values[:, 1] / values.sum(axis=1)).tolist()})
                learners.append(learner)
            data['trees'] = trees
            expected = np.column_stack([m.predict_proba(x)[:, 1] for m in learners])
        else:
            raise ValueError('지원하지 않는 행동 모델')
        model = cls.from_dict(data)
        error = float(np.max(np.abs(model.probabilities(x) - expected)))
        if error > 1e-12:
            raise ValueError('행동 모델 내보내기 불일치')
        scores = model.probabilities(calibration[cls.features].to_numpy(dtype=float))
        thresholds, details = {}, {}
        for i, action in enumerate(ACTIONS):
            thresholds[action], details[action] = select_threshold(calibration[f'y_{action}'], scores[:, i], [2., 1., .5][i])
        return model, thresholds, {'train_rows': len(train), 'calibration_rows': len(calibration),
                                    'export_max_error': error, 'threshold_selection': details,
                                    'train_last_label_end': train.label_end.max(), 'calibration_first_end': calibration.end.min()}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != cls.format or data.get('features') != cls.features
            or data.get('actions') != ACTIONS or data.get('kind') not in ('logistic', 'tree')):
            raise ValueError('행동 모델의 형식·특징·행동 오류')
        if data['kind'] == 'logistic':
            count = len(cls.features)
            for name, shape in [('mean', (count,)), ('scale', (count,)), ('coef', (3, count)), ('intercept', (3,))]:
                value = np.asarray(data[name], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('행동 모델의 숫자·차원 오류')
            if np.any(np.asarray(data['scale']) <= 0):
                raise ValueError('행동 모델 표준화 척도 오류')
        else:
            if not isinstance(data.get('trees'), list) or len(data['trees']) != 3:
                raise ValueError('행동 트리 수 오류')
            for tree in data['trees']:
                count = len(tree['left'])
                if not 1 <= count <= 31 or any(len(tree[k]) != count for k in ['right', 'feature', 'threshold', 'probability']):
                    raise ValueError('행동 트리 크기 오류')
                probabilities = np.asarray(tree['probability'], dtype=float)
                if (not np.isfinite(tree['threshold']).all() or not np.isfinite(probabilities).all()
                    or ((probabilities < 0) | (probabilities > 1)).any()):
                    raise ValueError('행동 트리 숫자 오류')
                visited, pending = set(), [(0, 0)]
                while pending:
                    node, depth = pending.pop()
                    if node in visited or not 0 <= node < count or depth > 4:
                        raise ValueError('행동 트리 순환·공유·깊이 오류')
                    visited.add(node)
                    left, right, feature = (tree[k][node] for k in ['left', 'right', 'feature'])
                    if any(type(v) is not int for v in [left, right, feature]):
                        raise ValueError('행동 트리 인덱스 오류')
                    if left == right == -1:
                        if feature != -2:
                            raise ValueError('행동 트리 말단 오류')
                    elif min(left, right) < 0 or not 0 <= feature < len(cls.features):
                        raise ValueError('행동 트리 특징 오류')
                    else:
                        pending.extend([(left, depth + 1), (right, depth + 1)])
                if len(visited) != count:
                    raise ValueError('도달 불가능한 행동 트리 노드')
        return cls(data)

    def probabilities(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError('행동 예측 차원 오류')
        valid = np.isfinite(values).all(axis=1)
        result = np.full((len(values), 3), np.nan)
        if self.kind == 'logistic':
            with np.errstate(over='ignore', invalid='ignore'):
                logits = ((values[valid] - self.mean) / self.scale) @ self.coef.T + self.intercept
                result[valid] = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
            result[np.flatnonzero(valid)[~np.isfinite(logits).all(axis=1)]] = np.nan
        else:
            with np.errstate(over='ignore'):
                matrix = values[valid].astype(np.float32)
            if len(values) == 1 and valid[0] and np.isfinite(matrix).all():
                for i, tree in enumerate(self.data['trees']):
                    node = 0
                    while tree['left'][node] != -1:
                        node = tree['left'][node] if float(matrix[0, tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
                    result[0, i] = tree['probability'][node]
                return result
            for i, tree in enumerate(self.data['trees']):
                left, right, feature = (np.asarray(tree[k]) for k in ['left', 'right', 'feature'])
                threshold, nodes = np.asarray(tree['threshold']), np.zeros(len(matrix), dtype=int)
                for _ in range(4):
                    active = left[nodes] != -1
                    selected = nodes[active]
                    nodes[active] = np.where(matrix[active, feature[selected]] <= threshold[selected], left[selected], right[selected])
                result[valid, i] = np.asarray(tree['probability'])[nodes]
            result[np.flatnonzero(valid)[~np.isfinite(matrix).all(axis=1)]] = np.nan
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)


class MinuteActionPolicy(PullbackPolicy):
    def __init__(self, entry, manager, thresholds, multiplier=1.):
        super().__init__(entry.base, entry.offset_bps, entry.ttl_minutes, entry.baseline)
        if (set(thresholds) != set(ACTIONS) or not np.isfinite(list(thresholds.values())).all()
            or any(not 0 <= x <= 1 for x in thresholds.values()) or multiplier not in (1., 1.5)):
            raise ValueError('행동별 정책 문턱 오류')
        self.manager, self.thresholds, self.multiplier = manager, dict(thresholds), multiplier

    def feature_values(self, bar, state):
        return management_values(bar['features'], state['direction'], state['favorable_move'],
                                 state['hold_bars'], state['adds'])

    def __call__(self, bar, state):
        if state['bar_seconds'] != 60:
            raise ValueError('분별 관리 정책의 실행 간격 오류')
        stored = state['policy_state']
        managing = 'management_after' in stored
        if not state['direction'] or state['halted'] or self.baseline == 'cash':
            return super().__call__(bar, {**state, 'policy_state': {} if managing else stored})
        current = pd.Timestamp(bar['end'])
        if managing:
            if set(stored) != {'management_after', 'management_direction'} or stored['management_direction'] not in (-1, 1):
                raise ValueError('관리 대기 상태 필드 오류')
            until = pd.Timestamp(stored['management_after'])
            if (until.tzinfo is None or until.utcoffset().total_seconds() != 0
                or until.value % pd.Timedelta(minutes=1).value or until > current + pd.Timedelta(minutes=3)):
                raise ValueError('관리 대기 시각 오류')
            if stored['management_direction'] == state['direction'] and current < until:
                return PolicyDecision('hold', stored, 'action_cooldown')
        elif stored:
            if set(stored) == {'signal_time', 'expires_at', 'direction', 'reference_price'}:
                # 지연된 진입 체결 전에 새로 생긴 대기는 보유가 시작되면 취소한다.
                return super().__call__(bar, state)
            raise ValueError('보유 중 알 수 없는 관리 상태')
        if state['pending'] != 'hold':
            return PolicyDecision('hold', stored if managing else {}, 'action_pending')
        values = self.feature_values(bar, state)
        scores = self.manager.probabilities(values.reshape(1, -1))[0]
        if not np.isfinite(scores).all():
            return PolicyDecision('hold', {}, 'action_unavailable')
        for action, score in zip(ACTIONS, scores, strict=True):
            if score >= self.thresholds[action] * self.multiplier:
                return PolicyDecision(action, {'management_after': (current + pd.Timedelta(minutes=3)).isoformat(),
                                               'management_direction': state['direction']}, f'action_{action}')
        return PolicyDecision('hold', {}, 'action_hold')
