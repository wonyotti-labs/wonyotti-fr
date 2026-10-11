import math
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from wonyotti_fr.sparse_exposure_reference import (
    exact_trees,
    raw_forest,
    reference_classification,
    reference_regression,
)


def independent_weighted_fit(raw, x, y, validation):
    labels = np.array([0 if n < -1e-12 else 2 if n > 1e-12 else 1 for n in y])
    counts = [int(np.sum(labels == k)) for k in range(3)]
    weights = np.asarray([len(labels)/(3*counts[k]) for k in labels])
    with threadpool_limits(limits=1):
        learner = HistGradientBoostingClassifier(loss='log_loss', max_iter=64, learning_rate=.05,
            max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
            max_features=1., categorical_features=None, early_stopping=False, random_state=0,
            class_weight=None).fit(x, labels, sample_weight=weights)
        probabilities = learner.predict_proba(validation)
    assert np.array_equal(learner.classes_, [0, 1, 2]) and not learner.do_early_stopping_
    for k in range(3):
        exact_trees(raw['models'][k], learner, k)
    return probabilities


def reference_prediction(raw, x):
    logits = np.column_stack([raw_forest(part, x) for part in raw['classifier']['models']])
    probabilities = []
    factors = [sum(raw['class_counts'])/(3*n) for n in raw['class_counts']]
    for row in logits:
        scores = [math.exp(float(v-max(row))) for v in row]
        if raw['condition'] in ('B', 'D'):
            scores = [p*w if raw['condition'] == 'B' else p/w for p, w in zip(scores, factors, strict=True)]
        probabilities.append([p/sum(scores) for p in scores])
    p = np.asarray(probabilities)
    negative, positive = [np.maximum(raw_forest(raw[side], x), 0) for side in ('negative', 'positive')]
    actions = np.asarray([0 if list(row).count(max(row)) > 1 else int(np.argmax(row))-1 for row in p])
    return {'probabilities': p, 'negative_size': negative, 'positive_size': positive, 'actions': actions,
        'decision_change': np.select([actions == -1, actions == 1], [-negative, positive], default=0.),
        'expected_change': p[:, 2]*positive-p[:, 0]*negative}


def reference_calibration(target, probabilities):
    labels = [0 if v < -1e-12 else 2 if v > 1e-12 else 1 for v in target]
    p = probabilities
    brier = math.fsum((float(row[k])-int(k == y))**2 for row, y in zip(p, labels, strict=True) for k in range(3))/len(labels)
    reliability = {}
    for k in range(3):
        rows = []
        for b in range(10):
            members = [i for i, row in enumerate(p) if b <= row[k]*10 < b+1 or b == 9 and row[k] == 1.]
            rows.append({'bin': b, 'rows': len(members),
                'mean_probability': math.fsum(float(p[i][k]) for i in members)/len(members) if members else None,
                'observed_frequency': sum(labels[i] == k for i in members)/len(members) if members else None})
        reliability[str(k-1)] = rows
    return {'rows': len(labels), 'multiclass_brier': brier, 'reliability': reliability}


def reference_evaluate(target, predictions, frequency):
    scores, changes = {}, {}
    for condition, prediction in predictions.items():
        score = reference_classification(target, prediction['probabilities'])
        for k in (0, 2):
            count = score['predicted_counts'][k]
            score['directions'][str(k-1)]['precision'] = score['confusion'][k][k]/count if count else None
        scores[condition] = score
        changes[condition] = {key: reference_regression(target, prediction[key]) for key in ('decision_change', 'expected_change')}
    return {'classification': scores, 'changes': changes, 'zero_change': reference_regression(target, [0.]*len(target)),
        'original_population_calibration': {name: reference_calibration(target, values)
            for name, values in {'A': predictions['A']['probabilities'], 'D': predictions['D']['probabilities'],
                'frequency': [frequency]*len(target)}.items()}}


class ReferenceWeightedPolicy:
    def __init__(self, raw):
        self.counts, self.condition = raw['class_counts'], raw['condition']
        self.classifiers = [self.pack(part) for part in raw['classifier']['models']]
        self.negative, self.positive = self.pack(raw['negative']), self.pack(raw['positive'])

    @staticmethod
    def pack(part):
        return (part['baseline'], [[tuple(node) for node in zip(tree['feature'], tree['threshold'], tree['left'], tree['right'], tree['value'], strict=True)]
            for tree in part['trees']])

    @staticmethod
    def value(head, x):
        result, forest = head
        for tree in forest:
            at = 0
            while tree[at][0] != -2:
                feature, threshold, left, right, _ = tree[at]
                at = left if x[feature] <= threshold else right
            result += tree[at][4]
        return result

    def __call__(self, bar, cash, quantity):
        end = datetime.fromisoformat(bar['end'])
        if end.minute % 5 or end.second or end.microsecond:
            return None, None
        assert bar['feature_end'] == bar['end']
        equity = cash+quantity*bar['close']
        current = quantity*bar['close']/equity
        journal = {'time': bar['end'], 'equity': equity, 'quantity': quantity, 'current_exposure': current,
            **dict.fromkeys(['probability_decrease', 'probability_no_change', 'probability_increase',
                'negative_size', 'positive_size', 'action', 'expected_change', 'decision_change', 'target']),
            'reason': 'missing_features'}
        if not all(v is not None and math.isfinite(v) for v in bar['features']):
            return None, journal
        x = bar['features']
        logits = [self.value(head, x) for head in self.classifiers]
        exponent = [math.exp(value-max(logits)) for value in logits]
        probabilities = [v/sum(exponent) for v in exponent]
        if self.condition in ('B', 'D'):
            weights = [sum(self.counts)/(3*n) for n in self.counts]
            scores = [p*w if self.condition == 'B' else p/w for p, w in zip(probabilities, weights, strict=True)]
            probabilities = [p/sum(scores) for p in scores]
        negative, positive = max(0., self.value(self.negative, x)), max(0., self.value(self.positive, x))
        action = 0 if probabilities.count(max(probabilities)) > 1 else probabilities.index(max(probabilities))-1
        change = -negative if action == -1 else positive if action == 1 else 0.
        journal.update({'probability_decrease': probabilities[0], 'probability_no_change': probabilities[1],
            'probability_increase': probabilities[2], 'negative_size': negative, 'positive_size': positive,
            'action': action, 'expected_change': probabilities[2]*positive-probabilities[0]*negative,
            'decision_change': change, 'reason': 'no_change' if action == 0 else 'zero_size' if abs(change) <= 1e-12 else 'target'})
        if journal['reason'] == 'target':
            journal['target'] = min(.5, max(-.5, current+change))
        return journal['target'], journal
