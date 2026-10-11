import numpy as np


def exposure_metrics(target, prediction, low, high):
    y, p, a, b = (np.asarray(v, dtype=float) for v in (target, prediction, low, high))
    if (not len(y) or y.ndim != 1 or any(v.shape != y.shape for v in [p, a, b])
        or not np.isfinite([y, p, a, b]).all() or (a > b).any()
        or (y < a-1e-12).any() or (y > b+1e-12).any()):
        raise ValueError('계좌 노출 지표의 정답·예측·민감도 구간 오류')
    difference = p-y
    distance = np.maximum(a-p, 0)+np.maximum(p-b, 0)
    positive, negative = y > 0, y < 0
    return {'rows': len(y), 'mse': float(np.mean(difference**2)), 'mae': float(np.mean(np.abs(difference))),
        'interval_mse': float(np.mean(distance**2)), 'interval_mae': float(np.mean(distance)),
        'direction_accuracy': float(np.mean(np.sign(y) == np.sign(p))),
        'long_rows': int(positive.sum()), 'short_rows': int(negative.sum()), 'flat_rows': int((y == 0).sum()),
        'long_recall': float(np.mean(p[positive] > 0)) if positive.any() else None,
        'short_recall': float(np.mean(p[negative] < 0)) if negative.any() else None,
        'mean_prediction': float(p.mean()), 'predicted_long_fraction': float(np.mean(p > 0)),
        'prediction_quantiles': dict(zip(['min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'],
            map(float, np.quantile(p, [0, .01, .1, .5, .9, .99, 1])), strict=True)),
        'independent_trade_count': None}
