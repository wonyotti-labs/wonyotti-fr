import numpy as np

from wonyotti_fr.sparse_exposure import action_labels, choose_actions
from wonyotti_fr.sparse_exposure_evaluation import classification_metrics, regression_metrics


def calibration(target, probabilities):
    y, p = action_labels(target), np.asarray(probabilities, dtype=float)
    choose_actions(p)
    if not len(y) or len(y) != len(p):
        raise ValueError('행동 확률 보정 평가의 행 수 오류')
    truth = np.eye(3)[y]
    bins = np.minimum((p*10).astype(int), 9)
    reliability = {}
    for k in range(3):
        rows = []
        for b in range(10):
            mask = bins[:, k] == b
            rows.append({'bin': b, 'rows': int(mask.sum()),
                'mean_probability': float(p[mask, k].mean()) if mask.any() else None,
                'observed_frequency': float(truth[mask, k].mean()) if mask.any() else None})
        reliability[str(k-1)] = rows
    return {'rows': len(y), 'multiclass_brier': float(np.mean(np.sum((p-truth)**2, axis=1))),
        'reliability': reliability}


def evaluate(target, predictions, frequency):
    if set(predictions) != {'A', 'B', 'C', 'D'}:
        raise ValueError('행동 가중 평가의 네 조건 누락')
    scores, changes = {}, {}
    for condition, prediction in predictions.items():
        metrics = classification_metrics(target, prediction['probabilities'])
        for k in (0, 2):
            count = metrics['predicted_counts'][k]
            metrics['directions'][str(k-1)]['precision'] = metrics['confusion'][k][k]/count if count else None
        scores[condition] = metrics
        changes[condition] = {key: regression_metrics(target, prediction[key])
            for key in ('decision_change', 'expected_change')}
    prior = np.tile(frequency, (len(target), 1))
    return {'classification': scores, 'changes': changes,
        'zero_change': regression_metrics(target, np.zeros(len(target))),
        'original_population_calibration': {name: calibration(target, values)
            for name, values in {'A': predictions['A']['probabilities'], 'D': predictions['D']['probabilities'], 'frequency': prior}.items()}}
