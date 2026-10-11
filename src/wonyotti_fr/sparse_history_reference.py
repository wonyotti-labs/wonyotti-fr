import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.sparse_exposure_reference import exact_trees


def verify_history(source, original, actual):
    pd.testing.assert_frame_equal(actual[original.columns], original, check_exact=True)
    lookup = {row.end: (row.label_end, row.decomposition_supported, row.execution_change) for row in source.itertuples(index=False)}
    selected = [f'past_{side}_{name}_60m' for side in ['decrease', 'increase']
        for name in ['exists', 'log_minutes_since', 'log_window_count']]
    for row in actual.itertuples(index=False):
        seen = {-1: [], 1: []}
        missing = unavailable = 0
        for lag in range(12, 0, -1):
            start = row.end-pd.Timedelta(minutes=lag*5)
            past = lookup.get(start)
            if past is None:
                missing += 1
            elif not past[1]:
                unavailable += 1
            else:
                end, _, change = past
                assert start+pd.Timedelta(minutes=5) == end and end <= row.end
                if change < -1e-12:
                    seen[-1].append(end)
                elif change > 1e-12:
                    seen[1].append(end)
        available = missing == unavailable == 0
        assert row.history_supported == available
        assert row.history_missing_windows == missing and row.history_unsupported_windows == unavailable
        reasons = []
        if missing:
            reasons.append('missing_prior_windows')
        if unavailable:
            reasons.append('unsupported_prior_decomposition')
        assert row.history_reason == ('|'.join(reasons) or 'included')
        values = [getattr(row, name) for name in selected]
        if not available:
            assert np.isnan(values).all()
            continue
        expected = []
        for sign in [-1, 1]:
            events = seen[sign]
            elapsed = (row.end-max(events)).total_seconds()/60 if events else 0
            expected += [float(bool(events)), math.log1p(elapsed), math.log1p(len(events))]
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0)


def independent_refit(raw, market_raw, linear, x, y, vx):
    labels = np.array([0 if v < -1e-12 else 2 if v > 1e-12 else 1 for v in y])
    settings = {'loss': 'log_loss', 'max_iter': 64, 'learning_rate': .05, 'max_depth': 2, 'max_leaf_nodes': 4,
        'min_samples_leaf': 64, 'l2_regularization': 1., 'max_bins': 255, 'max_features': 1.,
        'categorical_features': None, 'early_stopping': False, 'random_state': 0, 'class_weight': None}
    with threadpool_limits(limits=1):
        history = HistGradientBoostingClassifier(**settings).fit(x, labels)
        market = HistGradientBoostingClassifier(**settings).fit(x[:, :15], labels)
        scaler = StandardScaler().fit(x)
        logistic = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, class_weight=None, random_state=0).fit(scaler.transform(x), labels)
        expected = {'history': history.predict_proba(vx), 'matched_market': market.predict_proba(vx[:, :15]),
            'logistic': logistic.predict_proba(scaler.transform(vx)),
            'training_frequency': np.tile(np.bincount(labels, minlength=3)/len(labels), (len(vx), 1))}
    for saved, learner in [(raw, history), (market_raw, market)]:
        for k in range(3):
            exact_trees(saved['models'][k], learner, k)
    for key, values in [('mean', scaler.mean_), ('scale', scaler.scale_), ('coef', logistic.coef_),
        ('intercept', logistic.intercept_), ('iterations', logistic.n_iter_)]:
        np.testing.assert_array_equal(linear[key], values)
    return expected


def reference_admission(overall, cold):
    candidate = overall['history']
    checks, gains = {}, {}
    for name in ['v100', 'matched_market', 'logistic']:
        baseline = overall[name]
        gain = 1-candidate['log_loss']/baseline['log_loss'] if baseline['log_loss'] else None
        gains[name] = gain
        checks['loss_gain_'+name] = gain is not None and gain > .010000000001
        for k in [-1, 1]:
            a, b = candidate['directions'][str(k)]['ap'], baseline['directions'][str(k)]['ap']
            checks['ap_'+name+'_'+str(k)] = a is not None and b is not None and b-a <= 1e-12
    checks['loss_better_frequency'] = overall['training_frequency']['log_loss']-candidate['log_loss'] > 1e-12
    for name, part in [('overall', overall), ('cold', cold)]:
        for k in [0, 2]:
            checks[f'{name}_correct_{k-1}'] = part['history']['confusion'][k][k] >= 10
    return {'passed': all(checks.values()), 'relative_log_loss_gains': gains, 'checks': checks}
