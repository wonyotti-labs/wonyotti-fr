from __future__ import annotations

import numpy as np
import pandas as pd

from .entry_boundary_support import utc_times
from .management_account_state import BANDS
from .minute_management import ACTIONS, FEATURES
from .paired_management import PERIODS, PairedManagementLinear, PairedManagementModel
from .paired_management_evaluation import admission, evaluate

ACCOUNT_FEATURES = FEATURES+['log_futures_leverage']


class AccountManagementModel(PairedManagementModel):
    features = ACCOUNT_FEATURES
    format = 'account_size_management_v1'


class AccountManagementLinear(PairedManagementLinear):
    features = ACCOUNT_FEATURES
    format = 'account_size_management_linear_v1'


def account_split(frame):
    utc_times(frame.end, unique=True)
    if (not frame.label_end.sub(frame.end).eq(pd.Timedelta(minutes=5)).all()
        or not frame.account_usable.equals(frame.paired_usable & frame.account_supported)
        or not frame.paired_part.isin([*PERIODS, 'outside_periods']).all()
        or not frame.account_supported.equals(frame.account_reason.eq('included'))):
        raise ValueError('계좌 크기 관리의 지원·기간 연결 오류')
    rows, support = {}, {}
    for part, (left, right) in PERIODS.items():
        selected = frame.loc[frame.paired_part.eq(part) & frame.account_usable].copy().reset_index(drop=True)
        x = selected[ACCOUNT_FEATURES].to_numpy(dtype=float)
        y = selected[['all_y_'+a for a in ACTIONS]].to_numpy()
        if (not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
            or not selected.direction.isin([-1, 1]).all() or not selected.state_episode_id.gt(0).all()
            or not selected.end.between(pd.Timestamp(left, tz='UTC'), pd.Timestamp(right, tz='UTC'), inclusive='left').all()
            or not selected.futures_leverage.gt(0).all()
            or not np.array_equal(selected.log_futures_leverage, np.log1p(selected.futures_leverage))):
            raise ValueError('계좌 크기 관리의 특징·정답·보유 상태 오류')
        rows[part] = selected
        positive = {a: int(selected['all_y_'+a].sum()) for a in ACTIONS}
        support[part] = {'rows': len(selected), 'episodes': int(selected.state_episode_id.nunique()),
            'positive': positive, 'row_minimum_passed': len(selected) >= (1000 if part == 'training' else 100),
            'classes_supported': all(20 <= n <= len(selected)-20 for n in positive.values())}
    for left, right in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        if (set(rows[left].state_episode_id) & set(rows[right].state_episode_id)
            or len(rows[left]) and len(rows[right]) and rows[left].label_end.max() >= rows[right].end.min()):
            raise ValueError('계좌 크기 관리의 기간·에피소드 중첩')
    return rows, {'periods': support, 'supported': all(r['row_minimum_passed'] and r['classes_supported'] for r in support.values())}


def account_admission(metrics):
    mapped = {action: {key: row[name] for key, name in [('first', 'baseline'), ('all', 'account'),
        ('linear', 'linear'), ('constant', 'constant')]} for action, row in metrics.items()}
    result = admission(mapped)
    result['checks'] = {key.replace('_first', '_baseline'): value for key, value in result['checks'].items()}
    return result


def evaluation_masks(frame):
    groups = [('all', 'all', np.ones(len(frame), dtype=bool))]
    groups.extend(('period', p, frame.paired_part.eq(p).to_numpy()) for p in ('calibration', 'diagnosis'))
    months = frame.end.dt.strftime('%Y-%m')
    groups.extend(('month', month, months.eq(month).to_numpy()) for month in sorted(months.unique()))
    for kind, group, mask in groups:
        yield {'kind': kind, 'group': group, 'band': 'all'}, mask
        for band in BANDS:
            yield {'kind': kind, 'group': group, 'band': band}, mask & frame.account_band.eq(band).to_numpy()


def subset_metrics(frame, mask, labels, predictions, thresholds):
    part = frame.loc[mask]
    support = {'rows': len(part), 'episodes': int(part.state_episode_id.nunique()),
        'positive': {a: int(part['all_y_'+a].sum()) for a in ACTIONS}}
    if not len(part):
        return {**support, 'metrics': None}, None
    metrics, decisions = evaluate(labels[mask], {k: v[mask] for k, v in predictions.items()}, thresholds)
    return {**support, 'metrics': metrics}, decisions
