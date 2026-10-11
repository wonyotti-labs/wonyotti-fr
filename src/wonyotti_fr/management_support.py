from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .engine import PolicyDecision
from .order_history_boost import OrderHistoryBoostModels

FEATURES = OrderHistoryBoostModels.features
QUANTILES = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
GROUPS = {'market': (0, 14), 'directional_market': (14, 28), 'position': (28, 32),
          'path_inventory': (32, 36), 'minute': (36, 44), 'history': (44, 50)}


def matrix(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != len(FEATURES):
        raise ValueError('관리 입력 지원 범위의 특징 차원 오류')
    return result


def joint_codes(values):
    x = matrix(values)
    direction, adds, fraction, increase, reduce = x[:, [28, 31, 35, 44, 47]].T
    if (not np.isfinite(x).all() or not np.isin(direction, [-1, 1]).all()
        or not np.isin(adds, np.arange(6)).all() or not ((fraction > 0) & (fraction <= 1)).all()
        or not np.isin(increase, [0, 1]).all() or not np.isin(reduce, [0, 1]).all()):
        raise ValueError('관리 입력 지원 범위의 공동 상태 오류')
    return (((direction > 0).astype(int)*6 + adds.astype(int))*8
            + (fraction == 1).astype(int)*4 + increase.astype(int)*2 + reduce.astype(int))


def training_profile(values):
    x = matrix(values)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('관리 입력 지원 범위의 학습 값 오류')
    counts = np.bincount(joint_codes(x), minlength=96)
    return {'format': 'management_support_v1', 'features': list(FEATURES), 'rows': len(x),
            'quantiles': QUANTILES, 'bounds': np.quantile(x, QUANTILES, axis=0, method='linear').tolist(),
            'joint_counts': counts.tolist(), 'fitted_from': 'training_only'}


class SupportCounts:
    def __init__(self, profile):
        self.bounds = np.asarray(profile['bounds'], dtype=float)
        self.joint = np.asarray(profile['joint_counts'])
        if (profile.get('format') != 'management_support_v1' or profile.get('features') != FEATURES
            or profile.get('quantiles') != QUANTILES or profile.get('fitted_from') != 'training_only'
            or self.bounds.shape != (9, 50) or not np.isfinite(self.bounds).all()
            or (np.diff(self.bounds, axis=0) < 0).any() or self.joint.shape != (96,)
            or not np.issubdtype(self.joint.dtype, np.integer) or (self.joint < 0).any()
            or type(profile.get('rows')) is not int or profile['rows'] <= 0
            or self.joint.sum() != profile['rows']):
            raise ValueError('관리 입력 지원 범위의 학습 경계 오류')
        self.rows = self.finite_rows = 0
        self.nonfinite = np.zeros(50, dtype=np.int64)
        self.range_out = np.zeros(50, dtype=np.int64)
        self.central_out = np.zeros(50, dtype=np.int64)
        self.group_range = dict.fromkeys(['all', *GROUPS], 0)
        self.group_central = dict.fromkeys(['all', *GROUPS], 0)
        self.joint_observed = np.zeros(96, dtype=np.int64)

    def update(self, values):
        x = matrix(values)
        finite = np.isfinite(x)
        self.rows += len(x)
        self.nonfinite += (~finite).sum(axis=0)
        good = x[finite.all(axis=1)]
        self.finite_rows += len(good)
        self.joint_observed += np.bincount(joint_codes(good), minlength=96)
        outside = (good < self.bounds[0]) | (good > self.bounds[-1])
        central = (good < self.bounds[1]) | (good > self.bounds[-2])
        self.range_out += outside.sum(axis=0)
        self.central_out += central.sum(axis=0)
        for name, (start, stop) in {'all': (0, 50), **GROUPS}.items():
            self.group_range[name] += int(outside[:, start:stop].any(axis=1).sum())
            self.group_central[name] += int(central[:, start:stop].any(axis=1).sum())

    def result(self):
        return {'rows': self.rows, 'finite_rows': self.finite_rows,
                'nonfinite_rows': self.rows - self.finite_rows,
                'feature_counts': {name: {'nonfinite': int(self.nonfinite[i]),
                    'outside_training_range': int(self.range_out[i]),
                    'outside_training_central98': int(self.central_out[i])}
                    for i, name in enumerate(FEATURES)},
                'group_outside_training_range': self.group_range.copy(),
                'group_outside_training_central98': self.group_central.copy(),
                'unseen_joint_rows': int(self.joint_observed[self.joint == 0].sum()),
                'joint_counts': self.joint_observed.tolist(),
                'range_denominator': 'finite_rows', 'independent_trade_samples': False}


def reconstruct_input(bar, state, decision):
    market = np.asarray(bar['features'], dtype=float)
    minute = np.asarray(bar['minute_features'], dtype=float)
    if market.shape != (14,) or minute.shape != (4,):
        raise ValueError('관리 입력 수식 검산의 시세 차원 오류')
    direction, entry = state['direction'], state['average_entry']
    if direction not in (-1, 1) or entry <= 0:
        raise ValueError('관리 입력 수식 검산의 보유 상태 오류')
    stored = decision.state
    best = stored['path_high'] if direction == 1 else stored['path_low']
    worst = stored['path_low'] if direction == 1 else stored['path_high']
    favorable = direction*(bar['close']/entry-1)
    path = [direction*(best/entry-1), direction*(worst/entry-1)]
    stamp = pd.Timestamp(bar['end'])
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() or stamp.value % 60_000_000_000:
        raise ValueError('관리 입력 수식 검산의 UTC 분 경계 오류')
    end = stamp.value // 60_000_000_000
    history = []
    for action in ['increase', 'reduce']:
        last = stored[f'fill_{action}_last']
        mask = stored[f'fill_{action}_mask']
        if type(last) is not int or not 0 <= last < end or type(mask) is not int or not 0 <= mask < 2**15:
            raise ValueError('관리 입력 수식 검산의 과거 체결 오류')
        history.extend([float(last != 0), float(np.log1p(end-last)) if last else 0.,
                        float(np.log1p(mask.bit_count()))])
    return np.asarray([*market, *(market*direction), direction, favorable,
        np.log1p(state['hold_bars']), min(state['adds'], 5), *path,
        path[0]-favorable, state['remaining_fraction'], *minute, *(minute*direction), *history])


class ManagementInputTrace:
    def __init__(self, parent, record):
        if parent.manager.features != FEATURES or not callable(record) or isinstance(parent, ManagementInputTrace):
            raise ValueError('관리 입력 관찰의 부모·저장 함수 오류')
        self.parent, self.record = parent, record
        self.calls, self.features = 0, 0
        self.no_features = Counter()

    def prepare(self, data):
        self.parent.prepare(data)

    def __call__(self, bar, state):
        captured = []
        original = self.parent.feature_values
        had_instance_method = 'feature_values' in vars(self.parent)
        previous = vars(self.parent).get('feature_values')

        def capture(current, view):
            values = original(current, view)
            captured.append(np.array(values, dtype=float, copy=True))
            return values

        # 예외가 나도 원래 메서드의 인스턴스·클래스 탐색 상태를 복원한다.
        self.parent.feature_values = capture
        try:
            decision = self.parent(bar, state)
        finally:
            if had_instance_method:
                self.parent.feature_values = previous
            else:
                del self.parent.feature_values
        if not isinstance(decision, PolicyDecision) or len(captured) > 1:
            raise ValueError('관리 입력 관찰의 중복 호출·판단 오류')
        self.calls += 1
        if captured:
            value = captured[0]
            reference = reconstruct_input(bar, state, decision)
            if value.shape != (50,) or not np.allclose(value, reference, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError('관리 입력 관찰과 별도 수식 불일치')
            self.features += 1
            self.record({'end': bar['end'], 'entry_time': state['position_entry_time'],
                'intent': decision.intent, 'event': decision.event,
                **dict(zip(FEATURES, value, strict=True))})
        else:
            self.no_features[decision.event] += 1
        return decision

    def coverage(self):
        return {'policy_calls': self.calls, 'management_input_calls': self.features,
                'without_management_input': dict(sorted(self.no_features.items())),
                'all_captured_inputs_formula_verified': True}
