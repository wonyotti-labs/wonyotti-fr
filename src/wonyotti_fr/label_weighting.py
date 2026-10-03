from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTING = 'mean_inverse_concurrency_v1'


def lifecycle_weights(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    start, end = frame.decision_time, frame.label_end
    first, cutoff = pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2021-12-31', tz='UTC')
    if (frame.empty or str(start.dtype) not in {'datetime64[ns, UTC]', 'datetime64[us, UTC]'}
        or str(end.dtype) not in {'datetime64[ns, UTC]', 'datetime64[us, UTC]'}
        or start.isna().any() or end.isna().any() or not start.is_monotonic_increasing
        or start.duplicated().any() or start.lt(first).any() or end.ge(cutoff).any()
        or start.ge(end).any() or not start.eq(start.dt.floor('min')).all()
        or not end.eq(end.dt.floor('min')).all()):
        raise ValueError('중첩 가중치의 학습 시각·순서·분 경계 오류')
    left = ((start - first) / pd.Timedelta(minutes=1)).to_numpy(dtype=int)
    right = ((end - first) / pd.Timedelta(minutes=1)).to_numpy(dtype=int)
    changes = np.zeros(right.max()+1, dtype=int)
    np.add.at(changes, left, 1)
    np.add.at(changes, right, -1)
    # 오른쪽 경계를 제외해 종료와 다음 진입이 맞닿은 분은 겹치지 않는다.
    concurrency = changes.cumsum()
    inverse = np.divide(1., concurrency, out=np.zeros_like(concurrency, dtype=float), where=concurrency > 0)
    prefix = np.r_[0., inverse.cumsum()]
    unique = (prefix[right]-prefix[left]) / (right-left)
    if not np.isfinite(unique).all() or (unique <= 0).any() or (unique > 1+1e-10).any():
        raise ValueError('중첩 가중치의 비중 오류')
    weight = unique / unique.mean()
    ledger = pd.DataFrame({'decision_time': start.to_numpy(), 'label_end': end.to_numpy(),
                           'mean_inverse_concurrency': unique, 'sample_weight': weight})
    monthly = ledger.assign(month=start.dt.strftime('%Y-%m').to_numpy()).groupby('month').agg(
        rows=('sample_weight', 'size'), weight_sum=('sample_weight', 'sum')).reset_index()
    monthly['sample_share'] = monthly.rows / len(frame)
    monthly['weight_share'] = monthly.weight_sum / weight.sum()
    report = {'algorithm': WEIGHTING, 'rows': len(frame), 'max_concurrent_labels': int(concurrency.max()),
              'covered_minutes': int((concurrency > 0).sum()), 'sum_label_minutes': int((right-left).sum()),
              'mean_uniqueness': float(unique.mean()), 'min_weight': float(weight.min()),
              'max_weight': float(weight.max()), 'weight_sum': float(weight.sum()),
              'weight_kish_count': float(weight.sum()**2 / np.square(weight).sum()),
              'kish_limit': '가중치 불균형 진단이며 독립 표본 수가 아님',
              'training_period': ['2021-01-01', '2022-01-01'], 'loss_or_future_weighting': False,
              'monthly': monthly.to_dict('records')}
    return weight, ledger, report
