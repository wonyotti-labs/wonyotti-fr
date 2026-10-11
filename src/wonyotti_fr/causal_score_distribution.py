from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort_right
from collections import deque

import numpy as np
import pandas as pd

WINDOW_NS = 30 * 24 * 60 * 60 * 10**9
MIN_HISTORY = 1000
MINUTE_NS = 60 * 10**9


def scores_array(values):
    p = np.asarray(values)
    if (p.ndim != 1 or p.dtype.kind not in "fiu" or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()):
        raise ValueError("과거 분포 보정의 점수 차원·범위 오류")
    return p.astype(float)


def time_array(times):
    if (not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC"
        or times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing
        or not times.eq(times.dt.floor("min")).all()):
        raise ValueError("과거 분포 보정의 시간 순서·경계 오류")
    return times.array.as_unit("ns").asi8


class PastScoreWindow:
    def __init__(self):
        self.queue = deque()
        self.ordered = []
        self.last = None

    def observe(self, timestamp, score):
        if (isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer))
            or timestamp % MINUTE_NS or self.last is not None and timestamp <= self.last
            or isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.integer, np.floating))
            or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("과거 점수 창의 현재 시간·확률 오류")
        timestamp, score = int(timestamp), float(score)
        boundary = timestamp - WINDOW_NS
        while self.queue and self.queue[0][0] < boundary:
            _, expired = self.queue.popleft()
            index = bisect_left(self.ordered, expired)
            if index == len(self.ordered) or self.ordered[index] != expired:
                raise ValueError("과거 점수 창의 만료 상태 불일치")
            self.ordered.pop(index)
        low, high = bisect_left(self.ordered, score), bisect_right(self.ordered, score)
        counts = (len(self.ordered), low, high - low)
        # 현재 분은 순위 조회가 끝난 뒤 다음 분의 과거에 추가한다.
        self.queue.append((timestamp, score))
        insort_right(self.ordered, score)
        self.last = timestamp
        return counts


def map_past_scores(times, probabilities, reference):
    t, p, pool = time_array(times), scores_array(probabilities), scores_array(reference)
    if len(t) != len(p) or len(pool) < MIN_HISTORY or (np.diff(pool) < 0).any():
        raise ValueError("과거 분포 보정의 기준 분포·행 수 오류")
    window = PastScoreWindow()
    counts = np.empty((len(t), 3), dtype=np.int64)
    for index, (timestamp, score) in enumerate(zip(t, p, strict=True)):
        counts[index] = window.observe(timestamp, score)
    u = (counts[:, 1] + .5 * counts[:, 2] + .5) / (counts[:, 0] + 1)
    mapped = np.interp(u, np.linspace(0., 1., len(pool)), pool)
    return pd.DataFrame({"count": counts[:, 0], "less": counts[:, 1], "equal": counts[:, 2],
                         "mapped": mapped})


def verify_past_mapping(times, probabilities, reference, mapped):
    t, p = time_array(times), scores_array(probabilities)
    pool = scores_array(reference)
    levels, indices = np.unique(p, return_inverse=True)
    tree = np.zeros(len(levels) + 1, dtype=np.int64)

    def update(index, value):
        index += 1
        while index < len(tree):
            tree[index] += value
            index += index & -index

    def prefix(end):
        result = 0
        while end:
            result += int(tree[end])
            end -= end & -end
        return result

    expected = np.empty((len(t), 3), dtype=np.int64)
    left = 0
    for index, timestamp in enumerate(t):
        while left < index and t[left] < timestamp - WINDOW_NS:
            update(int(indices[left]), -1)
            left += 1
        level = int(indices[index])
        low, high = prefix(level), prefix(level + 1)
        expected[index] = (index - left, low, high - low)
        update(level, 1)
    np.testing.assert_array_equal(mapped[["count", "less", "equal"]].to_numpy(), expected)
    percentile = (expected[:, 1] + expected[:, 2] / 2 + .5) / (expected[:, 0] + 1)
    position = percentile * (len(pool) - 1)
    low, high = np.floor(position).astype(int), np.ceil(position).astype(int)
    value = pool[low] + (pool[high] - pool[low]) * (position - low)
    np.testing.assert_allclose(mapped.mapped.to_numpy(), value, rtol=0, atol=1e-12)
    return {"rows": len(p), "supported": int((expected[:, 0] >= MIN_HISTORY).sum())}
