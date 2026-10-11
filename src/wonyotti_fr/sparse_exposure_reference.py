import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def verify_metrics(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('희소 검산 지표 필드 불일치')
        for key in expected:
            verify_metrics(expected[key], actual[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError('희소 검산 지표 배열 불일치')
        for left, right in zip(expected, actual, strict=True):
            verify_metrics(left, right)
    elif expected is None or type(expected) is int:
        if actual != expected:
            raise ValueError('희소 검산 지표 분모 불일치')
    elif not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('희소 검산 지표 수치 불일치')


def raw_forest(model, rows):
    x = np.asarray(rows, dtype=float)
    result = np.full(len(x), model['baseline'], dtype=float)
    for tree in model['trees']:
        for i, row in enumerate(x):
            node = 0
            while tree['feature'][node] >= 0:
                node = tree['left'][node] if row[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
            result[i] += tree['value'][node]
    return result


def exact_trees(raw, learner, channel=0):
    assert raw['baseline'] == float(learner._baseline_prediction[0, channel])
    assert len(raw['trees']) == len(learner._predictors) == 64
    for saved, iteration in zip(raw['trees'], learner._predictors, strict=True):
        nodes = iteration[channel].nodes
        assert len(saved['left']) == len(nodes)
        for i, node in enumerate(nodes):
            leaf = bool(node['is_leaf'])
            expected = {'left': -1 if leaf else int(node['left']), 'right': -1 if leaf else int(node['right']),
                'feature': -2 if leaf else int(node['feature_idx']),
                'threshold': 0. if leaf else float(node['num_threshold']), 'value': float(node['value']) if leaf else 0.}
            assert all(saved[key][i] == value for key, value in expected.items())


def independent_refit(raw, linear, x, y, vx):
    labels = np.ones(len(y), dtype=int)
    labels[y < -1e-12], labels[y > 1e-12] = 0, 2
    fixed = {'max_iter': 64, 'learning_rate': .05, 'max_leaf_nodes': 4, 'max_depth': 2,
        'min_samples_leaf': 64, 'l2_regularization': 1., 'max_bins': 255, 'max_features': 1.,
        'categorical_features': None, 'early_stopping': False, 'random_state': 0}
    with threadpool_limits(limits=1):
        classifier = HistGradientBoostingClassifier(loss='log_loss', class_weight=None, **fixed).fit(x, labels)
        negative = HistGradientBoostingRegressor(loss='squared_error', **fixed).fit(x[labels == 0], -y[labels == 0])
        positive = HistGradientBoostingRegressor(loss='squared_error', **fixed).fit(x[labels == 2], y[labels == 2])
        scaler = StandardScaler().fit(x)
        logistic = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, class_weight=None, random_state=0).fit(scaler.transform(x), labels)
        reference = {'probabilities': classifier.predict_proba(vx), 'linear': logistic.predict_proba(scaler.transform(vx)),
            'negative_size': np.maximum(negative.predict(vx), 0), 'positive_size': np.maximum(positive.predict(vx), 0)}
    assert np.array_equal(classifier.classes_, [0, 1, 2]) and np.array_equal(logistic.classes_, [0, 1, 2])
    for k in range(3):
        exact_trees(raw['classifier']['models'][k], classifier, k)
    exact_trees(raw['negative'], negative)
    exact_trees(raw['positive'], positive)
    for key, values in [('mean', scaler.mean_), ('scale', scaler.scale_), ('coef', logistic.coef_),
        ('intercept', logistic.intercept_), ('iterations', logistic.n_iter_)]:
        np.testing.assert_array_equal(linear[key], values)
    p = reference['probabilities']
    actions = np.array([0 if sum(v == max(row) for v in row) > 1 else int(np.argmax(row))-1 for row in p])
    reference['actions'] = actions
    reference['expected_change'] = p[:, 2]*reference['positive_size']-p[:, 0]*reference['negative_size']
    reference['decision_change'] = np.select([actions == -1, actions == 1], [-reference['negative_size'], reference['positive_size']], default=0.)
    return reference


def reference_classification(y, probability):
    p = np.asarray(probability)
    labels = np.array([0 if v < -1e-12 else 2 if v > 1e-12 else 1 for v in y])
    chosen = np.array([1 if sum(v == max(row) for v in row) > 1 else int(np.argmax(row)) for row in p])
    eps = np.finfo(np.float64).eps
    clipped = np.clip(p, eps, 1-eps)
    loss = math.fsum(-math.log(clipped[i, label]) for i, label in enumerate(labels))/len(y)
    matrix = [[int(np.sum((labels == a) & (chosen == b))) for b in range(3)] for a in range(3)]
    directions = {}
    for k in [0, 2]:
        truth = labels == k
        if not truth.any():
            directions[str(k-1)] = {'ap': None, 'recall': None}
            continue
        order = np.argsort(-p[:, k], kind='stable')
        scores, actual = p[order, k], truth[order]
        cumulative = np.cumsum(actual)
        boundaries = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
        positives = np.diff(np.r_[0, cumulative[boundaries]])
        ap = math.fsum(float(n*t/(i+1)) for n, t, i in zip(positives, cumulative[boundaries], boundaries, strict=True))/int(truth.sum())
        directions[str(k-1)] = {'ap': ap, 'recall': float(np.sum(truth & (chosen == k))/truth.sum())}
    return {'rows': len(y), 'log_loss': loss, 'confusion': matrix,
        'true_counts': [int(np.sum(labels == k)) for k in range(3)],
        'predicted_counts': [int(np.sum(chosen == k)) for k in range(3)], 'directions': directions}


def reference_regression(y, p):
    return {'rows': len(y), 'mse': math.fsum((float(a)-float(b))**2 for a, b in zip(y, p, strict=True))/len(y),
        'mae': math.fsum(abs(float(a)-float(b)) for a, b in zip(y, p, strict=True))/len(y)}


def reference_metrics(y, probabilities, changes, sizes):
    masks = {0: y < -1e-12, 2: y > 1e-12}
    return {'classification': {name: reference_classification(y, p) for name, p in probabilities.items()},
        'changes': {name: reference_regression(y, p) for name, p in changes.items()},
        'conditional_size': {str(k-1): {name: reference_regression(np.abs(y[mask]), p[mask]) for name, p in sizes[k].items()}
            if mask.any() else None for k, mask in masks.items()}}
