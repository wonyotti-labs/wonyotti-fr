from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .paired_management_reference import numeric_linear_scores
from .sparse_exposure_reference import exact_trees, raw_forest


def verify_clock_frame(original, extended):
    pd.testing.assert_frame_equal(original, extended[original.columns], check_exact=True)
    names = ['utc_day_sin', 'utc_day_cos', 'utc_week_sin', 'utc_week_cos']
    assert list(extended.columns) == [*original.columns, *names]
    expected = []
    for t in original.end:
        seconds = t.hour*3600+t.minute*60+t.second+t.microsecond/1e6+t.nanosecond/1e9
        day = seconds/86400
        week = (t.dayofweek+day)/7
        expected.append([math.sin(2*math.pi*day), math.cos(2*math.pi*day),
            math.sin(2*math.pi*week), math.cos(2*math.pi*week)])
    np.testing.assert_allclose(expected, extended[names].to_numpy(), rtol=0, atol=2e-15)
    return {'all_clock_rows_verified': len(original), 'original_columns_exact': True}


def independent_clock_models(tree, linear, x, y, validation):
    x = np.ascontiguousarray(x, dtype=float)
    probabilities = []
    with threadpool_limits(limits=1):
        for i in range(3):
            learner = HistGradientBoostingClassifier(loss='log_loss', max_iter=64, learning_rate=.05,
                max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                class_weight=None).fit(x, y[:, i])
            assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
            exact_trees(tree['models'][i], learner)
            logits = raw_forest(tree['models'][i], validation)
            p = 1/(1+np.exp(-np.clip(logits, -700, 700)))
            np.testing.assert_allclose(p, learner.predict_proba(validation)[:, 1], atol=1e-12, rtol=0)
            probabilities.append(p)
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(scaler.mean_, linear['mean'])
        np.testing.assert_array_equal(scaler.scale_, linear['scale'])
        expected = []
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(learner.coef_[0], linear['coef'][i])
            assert float(learner.intercept_[0]) == linear['intercept'][i]
            expected.append(learner.predict_proba(scaler.transform(validation))[:, 1])
        canonical = numeric_linear_scores(linear, validation)
        np.testing.assert_allclose(canonical, np.column_stack(expected), atol=1e-12, rtol=0)
    return {'clock': np.column_stack(probabilities), 'linear': canonical}


def independent_admission(metrics):
    checks = {}
    for action in ('exit', 'reduce', 'increase'):
        candidate = metrics[action]['clock']
        for name in ('baseline', 'linear'):
            base = metrics[action][name]
            improvement = 1-candidate['log_loss']/base['log_loss'] if base['log_loss'] else 0.
            checks[f'{action}_log_loss_{name}'] = bool(improvement > .01+1e-12)
            checks[f'{action}_ap_{name}'] = bool(candidate['average_precision'] >= base['average_precision']-1e-12)
        checks[action+'_constant'] = bool(candidate['log_loss'] < metrics[action]['constant']['log_loss']-1e-12)
    for name in ('recall', 'f_beta'):
        checks['exit_'+name] = bool(metrics['exit']['clock'][name] > metrics['exit']['baseline'][name]+1e-12)
    return {'management_admitted': all(checks.values()), 'checks': checks, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}
