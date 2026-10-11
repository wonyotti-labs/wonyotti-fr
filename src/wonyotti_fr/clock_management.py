from __future__ import annotations

import numpy as np

from .entry_boundary_support import utc_times
from .minute_management import FEATURES
from .paired_management import PairedManagementLinear, PairedManagementModel
from .paired_management_evaluation import admission

CLOCK_FEATURES = ['utc_day_sin', 'utc_day_cos', 'utc_week_sin', 'utc_week_cos']
CLOCK_MANAGEMENT_FEATURES = FEATURES+CLOCK_FEATURES
DAY_NS = 86400*10**9


class ClockManagementModel(PairedManagementModel):
    features = CLOCK_MANAGEMENT_FEATURES
    format = 'clock_management_v1'


class ClockManagementLinear(PairedManagementLinear):
    features = CLOCK_MANAGEMENT_FEATURES
    format = 'clock_management_linear_v1'


def clock_values(times):
    stamps = utc_times(times, unique=True)
    if not len(stamps):
        raise ValueError('주기 관리 시각의 빈 입력')
    day = (stamps % DAY_NS)/DAY_NS
    # 유닉스 기준일은 목요일이므로 월요일부터의 주간 위상으로 맞춘다.
    week = ((stamps % (7*DAY_NS)+3*DAY_NS) % (7*DAY_NS))/(7*DAY_NS)
    return np.column_stack([np.sin(2*np.pi*day), np.cos(2*np.pi*day),
        np.sin(2*np.pi*week), np.cos(2*np.pi*week)])


def extend_clock_frame(frame):
    if 'end' not in frame or any(name in frame for name in CLOCK_FEATURES):
        raise ValueError('주기 관리 시각 누락·중복 특징')
    result = frame.copy()
    result[CLOCK_FEATURES] = clock_values(frame.end)
    return result


def clock_admission(metrics):
    mapped = {action: {key: row[name] for key, name in [('first', 'baseline'), ('all', 'clock'),
        ('linear', 'linear'), ('constant', 'constant')]} for action, row in metrics.items()}
    result = admission(mapped)
    result['checks'] = {name.replace('_first', '_baseline'): value for name, value in result['checks'].items()}
    return result
