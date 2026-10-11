from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .paired_management_reference import manual_metrics, numeric_linear_scores
from .position_boost_verification import compare_statistics
from .sparse_exposure_reference import exact_trees, raw_forest


def verify_account_split(frame, rows, actual, parent_summary):
    summary = {}
    old = {g['group']: g['joint_usable'] for g in parent_summary['results']['groups'] if g['kind'] == 'split'}
    for part in ('training', 'calibration', 'diagnosis'):
        indices = [i for i, r in enumerate(frame[['paired_part', 'paired_usable', 'account_reason']].itertuples(index=False))
            if r.paired_part == part and r.paired_usable and r.account_reason == 'included']
        expected = frame.iloc[indices].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows[part], expected, check_exact=True)
        positive = {a: sum(bool(v) for v in expected['all_y_'+a]) for a in ('exit', 'reduce', 'increase')}
        count = {'rows': len(indices), 'episodes': len(set(expected.state_episode_id)), 'positive': positive}
        compare_statistics(old[part], count)
        summary[part] = {**count, 'row_minimum_passed': len(indices) >= (1000 if part == 'training' else 100),
            'classes_supported': all(20 <= n <= len(indices)-20 for n in positive.values())}
    compare_statistics(actual, {'periods': summary,
        'supported': all(r['row_minimum_passed'] and r['classes_supported'] for r in summary.values())})
    return {'all_source_rows_checked': len(frame), 'matched_rows_and_source_support_exact': True}


def independent_account_models(models, x, y, validation):
    x, validation = np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(validation, dtype=float)
    scores = {}
    with threadpool_limits(limits=1):
        for name, width in [('baseline', 32), ('account', 33)]:
            train = np.ascontiguousarray(x[:, :width])
            val = np.ascontiguousarray(validation[:, :width])
            values = []
            for i in range(3):
                learner = HistGradientBoostingClassifier(loss='log_loss', max_iter=64, learning_rate=.05,
                    max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                    max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                    class_weight=None).fit(train, y[:, i])
                assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
                exact_trees(models[name]['models'][i], learner)
                p = 1/(1+np.exp(-np.clip(raw_forest(models[name]['models'][i], val), -700, 700)))
                np.testing.assert_allclose(p, learner.predict_proba(val)[:, 1], atol=1e-12, rtol=0)
                values.append(p)
            scores[name] = np.column_stack(values)
        linear = models['linear']
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(scaler.mean_, linear['mean'])
        np.testing.assert_array_equal(scaler.scale_, linear['scale'])
        values = []
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(learner.coef_[0], linear['coef'][i])
            assert float(learner.intercept_[0]) == linear['intercept'][i]
            values.append(learner.predict_proba(scaler.transform(validation))[:, 1])
        scores['linear'] = numeric_linear_scores(linear, validation)
        np.testing.assert_allclose(scores['linear'], np.column_stack(values), atol=1e-12, rtol=0)
    frequency = [sum(y[:, i])/len(y) for i in range(3)]
    scores['constant'] = np.tile(frequency, (len(validation), 1))
    return scores


def verify_subset(frame, key, mask, result, decisions, y, scores, thresholds):
    expected = []
    for row in frame[['end', 'paired_part', 'futures_leverage']].itertuples(index=False):
        inside = key['kind'] == 'all' or (key['kind'] == 'period' and row.paired_part == key['group']) or (
            key['kind'] == 'month' and row.end.strftime('%Y-%m') == key['group'])
        boundaries = {'zero': (None, 0), '(0,0.5]': (0, .5), '(0.5,1]': (.5, 1),
            '(1,2]': (1, 2), '(2,5]': (2, 5), '(5,inf)': (5, None)}
        band = key['band']
        if band != 'all':
            low, high = boundaries[band]
            inside = inside and (low is None or row.futures_leverage > low) and (high is None or row.futures_leverage <= high)
        expected.append(inside)
    expected = np.asarray(expected, dtype=bool)
    np.testing.assert_array_equal(mask, expected)
    support = {'rows': int(sum(expected)), 'episodes': len(set(frame.loc[expected, 'state_episode_id'])),
        'positive': {a: int(sum(y[expected, i])) for i, a in enumerate(('exit', 'reduce', 'increase'))}}
    if not expected.any():
        assert decisions is None
        compare_statistics(result, {**support, 'metrics': None})
        return
    metrics = {a: {} for a in ('exit', 'reduce', 'increase')}
    for name, probabilities in scores.items():
        selected = probabilities[expected]
        for i, action in enumerate(('exit', 'reduce', 'increase')):
            metrics[action][name] = manual_metrics(y[expected, i], selected[:, i], thresholds[name][action], (2., 1., .5)[i])
            np.testing.assert_array_equal(decisions[name]['requests'][:, i], selected[:, i] >= thresholds[name][action])
        choices = [next((a for i, a in enumerate(('exit', 'reduce', 'increase')) if r[i] >= thresholds[name][a]), 'hold') for r in selected]
        np.testing.assert_array_equal(decisions[name]['chosen'], choices)
    compare_statistics(result, {**support, 'metrics': metrics})


def independent_admission(metrics):
    checks = {}
    for action in ('exit', 'reduce', 'increase'):
        candidate = metrics[action]['account']
        for name in ('baseline', 'linear'):
            comparison = metrics[action][name]
            gain = 1-candidate['log_loss']/comparison['log_loss'] if comparison['log_loss'] else 0.
            checks[f'{action}_log_loss_{name}'] = bool(gain > .01+1e-12)
            checks[f'{action}_ap_{name}'] = bool(candidate['average_precision'] >= comparison['average_precision']-1e-12)
        checks[action+'_constant'] = bool(candidate['log_loss'] < metrics[action]['constant']['log_loss']-1e-12)
    for name in ('recall', 'f_beta'):
        checks['exit_'+name] = bool(metrics['exit']['account'][name] > metrics['exit']['baseline'][name]+1e-12)
    return {'management_admitted': all(checks.values()), 'checks': checks, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}
