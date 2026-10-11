import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss

from wonyotti_fr.sparse_exposure import action_labels, choose_actions


def classification_metrics(target, probability):
    y, p = action_labels(target), np.asarray(probability, dtype=float)
    selected = choose_actions(p) + 1
    if not len(y) or len(y) != len(p):
        raise ValueError('희소 행동 지표의 행 수 오류')
    matrix = confusion_matrix(y, selected, labels=[0, 1, 2])
    return {'rows': len(y), 'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
        'confusion': matrix.tolist(), 'true_counts': np.bincount(y, minlength=3).tolist(),
        'predicted_counts': np.bincount(selected, minlength=3).tolist(),
        'directions': {str(k-1): {'ap': float(average_precision_score(y == k, p[:, k])) if (y == k).any() else None,
            'recall': float(np.mean(selected[y == k] == k)) if (y == k).any() else None} for k in [0, 2]}}


def regression_metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if not len(y) or y.ndim != 1 or y.shape != p.shape or not np.isfinite([y, p]).all():
        raise ValueError('희소 변경량 지표의 행·유한성 오류')
    return {'rows': len(y), 'mse': float(np.mean((p-y)**2)), 'mae': float(np.mean(np.abs(p-y)))}


def diagnostic_metrics(target, probabilities, changes, sizes):
    y = np.asarray(target, dtype=float)
    labels = action_labels(y)
    return {'classification': {name: classification_metrics(y, p) for name, p in probabilities.items()},
        'changes': {name: regression_metrics(y, p) for name, p in changes.items()},
        'conditional_size': {str(k-1): {name: regression_metrics(np.abs(y[labels == k]), p[labels == k])
            for name, p in sizes[k].items()} if (labels == k).any() else None for k in [0, 2]}}
