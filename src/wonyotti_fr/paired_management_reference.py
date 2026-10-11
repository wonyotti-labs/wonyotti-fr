import math
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.sparse_exposure_reference import exact_trees, raw_forest


def verify_frame(windows, filled, requested, events, episodes, bars, saved):
    pd.testing.assert_frame_equal(saved[windows.columns], windows.reset_index(drop=True), check_exact=True)
    times, states, quantity, unit, entry, adds, seen = [], [], 0, 0., None, 0, set()
    for event in events.itertuples(index=False):
        before = quantity
        if event.exectype != 'Funding':
            quantity += int(event.quantity) if event.side == 'Buy' else -int(event.quantity)
            if before == 0 or quantity*before < 0:
                unit, entry, adds, seen = abs(int(event.cost_satoshi))/int(event.quantity), event.time, 0, set()
            elif quantity == 0:
                unit, entry, adds = 0., None, 0
            elif abs(quantity) > abs(before):
                unit = (unit*abs(before)+abs(int(event.cost_satoshi)))/abs(quantity)
                adds += event.order_key not in seen
            seen.add(event.order_key)
        times.append(event.time.value)
        states.append((quantity, unit, entry, adds))
    prices = {r.end.value: float(r.close) for r in bars.itertuples(index=False)}
    first, all_labels, related = {}, defaultdict(set), defaultdict(set)
    for row in requested.itertuples(index=False):
        if row.window_index >= 0:
            labels = {a for a in ('exit', 'reduce', 'increase') if getattr(row, a)}
            first.setdefault(row.window_index, labels)
            all_labels[row.window_index].update(labels)
    for row in filled.itertuples(index=False):
        if row.window_index >= 0:
            related[row.window_index].update((row.before_episode_id, row.after_episode_id))
    periods = {'training': ('2018-01-01', '2021-01-01'),
        'calibration': ('2021-01-01', '2021-07-01'), 'diagnosis': ('2021-07-01', '2022-01-01')}
    bounds = {name: tuple(pd.Timestamp(t, tz='UTC').value for t in pair) for name, pair in periods.items()}
    crossing = {boundary: {r.episode_id for r in episodes.itertuples(index=False)
        if r.entry_time.value < boundary and (not r.closed or pd.isna(r.exit_time) or r.exit_time.value >= boundary)}
        for boundary in {b for pair in bounds.values() for b in pair}}
    max_error = 0.
    for i, row in enumerate(saved.itertuples(index=False)):
        index = bisect_left(times, row.end.value)-1
        q, basis, entered, count = states[index] if index >= 0 else (0, 0., None, None)
        direction = (q > 0)-(q < 0)
        average = 1e8/basis if basis else 0. if index >= 0 else np.nan
        price = prices.get(row.end.value, np.nan)
        gain = (price/average-1)*direction if average else np.nan
        hold = math.log1p(max((row.end.value-entered.value)/60e9, 0)) if entered is not None else np.nan
        capped = min(count, 5) if count is not None else np.nan
        expected = [average, float(count) if count is not None else np.nan, price, direction, gain, hold, capped]
        actual = [row.basis_average_entry, row.basis_adds, row.observed_close, row.direction,
            row.favorable_move, row.log_hold_minutes, row.adds_capped]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
        for a, b in zip(actual, expected, strict=True):
            if math.isfinite(a) and math.isfinite(b):
                max_error = max(max_error, abs(a-b))
        assert (pd.isna(row.basis_entry_time) if entered is None else row.basis_entry_time == entered)
        assert (pd.isna(row.basis_state_time) if index < 0 else row.basis_state_time.value == times[index])
        values = []
        for key in MARKET_FEATURES:
            market = getattr(row, 'start_market_'+key)
            values += [market, market*direction]
            np.testing.assert_equal(getattr(row, key), market)
            np.testing.assert_equal(getattr(row, 'directional_'+key), market*direction)
        for action in ('exit', 'reduce', 'increase'):
            assert getattr(row, 'first_y_'+action) == (action in first.get(i, set()))
            assert getattr(row, 'all_y_'+action) == (action in all_labels[i])
        part = next((name for name, (left, right) in bounds.items() if left <= row.end.value < right), 'outside_periods')
        reason = 'outside_periods'
        if part in bounds:
            left, right = bounds[part]
            ids = related[i] | ({int(row.state_episode_id)} if not pd.isna(row.state_episode_id) else set())
            checks = [(not row.usable, 'original_unsupported'), (direction == 0, 'flat_state'),
                (not all(math.isfinite(v) for v in [*values, direction, gain, hold, capped]), 'nonfinite_state_features'),
                (row.end.value < left+86400e9, 'left_embargo'), (row.label_end.value >= right-86400e9, 'right_embargo'),
                (bool(ids & crossing[left]), 'left_episode_boundary'), (bool(ids & crossing[right]), 'right_episode_boundary')]
            reason = next((message for failed, message in checks if failed), 'included')
        assert row.paired_part == part and row.paired_reason == reason and row.paired_usable == (reason == 'included')
    return {'complete': True, 'all_windows_verified': len(saved), 'independent_cost_state_rows': len(states),
        'maximum_state_absolute_error': max_error, 'state_tolerance_relative': 1e-12, 'state_tolerance_absolute': 1e-12,
        'all_original_columns_preserved': True, 'all_start_features_labels_and_periods_verified': True}


def independent_models(raw_models, linear, x, first_y, all_y, validation):
    x = np.ascontiguousarray(x, dtype=float)
    output = {}
    with threadpool_limits(limits=1):
        for name, labels in [('first', first_y), ('all', all_y)]:
            scores = []
            for i in range(3):
                learner = HistGradientBoostingClassifier(loss='log_loss', max_iter=64, learning_rate=.05,
                    max_leaf_nodes=4, max_depth=2, min_samples_leaf=64, l2_regularization=1., max_bins=255,
                    max_features=1., categorical_features=None, early_stopping=False, random_state=0,
                    class_weight=None).fit(x, labels[:, i])
                assert np.array_equal(learner.classes_, [0, 1]) and not learner.do_early_stopping_
                raw = raw_models[name]['models'][i]
                exact_trees(raw, learner)
                p = learner.predict_proba(validation)[:, 1]
                logits = raw_forest(raw, validation)
                np.testing.assert_allclose(p, 1/(1+np.exp(-np.clip(logits, -700, 700))), rtol=0, atol=1e-12)
                scores.append(p)
            output[name] = np.column_stack(scores)
        scaler = StandardScaler().fit(x)
        np.testing.assert_array_equal(linear['mean'], scaler.mean_)
        np.testing.assert_array_equal(linear['scale'], scaler.scale_)
        scores = []
        for i in range(3):
            learner = LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), all_y[:, i])
            assert learner.n_iter_.max() < 2000
            np.testing.assert_array_equal(linear['coef'][i], learner.coef_[0])
            assert linear['intercept'][i] == float(learner.intercept_[0])
            scores.append(learner.predict_proba(scaler.transform(validation))[:, 1])
        # 문턱과 같은 값에서는 라이브러리의 단일 열 곱셈과 공개 숫자 모형의
        # 세 열 곱셈 차이도 요청을 바꿀 수 있으므로 직렬화된 연산을 검산한다.
        output['linear'] = numeric_linear_scores(linear, validation)
        np.testing.assert_allclose(output['linear'], np.column_stack(scores), rtol=0, atol=1e-12)
    output['constant'] = np.tile(all_y.mean(axis=0), (len(validation), 1))
    return output


def numeric_linear_scores(raw, values):
    matrix = np.asarray(values, dtype=float)
    scaled = (matrix-np.asarray(raw['mean']))/np.asarray(raw['scale'])
    logits = scaled@np.asarray(raw['coef']).T+np.asarray(raw['intercept'])
    return 1/(1+np.exp(-np.clip(logits, -700, 700)))


def independent_support(frame):
    from collections import Counter

    result, reasons = {}, {}
    for part in ('training', 'calibration', 'diagnosis'):
        selected = [r for r in frame.itertuples(index=False) if r.paired_part == part and r.paired_usable]
        counts = {mode: {a: sum(bool(getattr(r, f'{mode}_y_{a}')) for r in selected)
            for a in ('exit', 'reduce', 'increase')} for mode in ('first', 'all')}
        modes = ('first', 'all') if part == 'training' else ('all',)
        result[part] = {'rows': len(selected), 'episodes': len({r.state_episode_id for r in selected}), 'positive': counts,
            'row_minimum_passed': len(selected) >= (1000 if part == 'training' else 100),
            'classes_supported': all(20 <= n <= len(selected)-20 for mode in modes for n in counts[mode].values())}
        reasons[part] = dict(Counter(r.paired_reason for r in frame.itertuples(index=False) if r.paired_part == part))
    return {'periods': result, 'supported': all(v['row_minimum_passed'] and v['classes_supported'] for v in result.values()), 'reasons': reasons}


def manual_threshold(y, p, beta):
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p)-1]
    positives = int(np.sum(y))
    tp = np.cumsum(y[order])[ends]
    counts = ends+1
    f = (1+beta**2)*tp/(beta**2*positives+counts)
    f[counts < 20] = -1
    best = int(np.flatnonzero(f == f.max())[0])
    threshold = float(p[order][ends[best]])
    return threshold, {'beta': beta, 'f_beta': float(f[best]), 'precision': float(tp[best]/counts[best]),
        'recall': float(tp[best]/positives), 'predicted_positive': int(counts[best]), 'actual_positive': positives}


def manual_metrics(y, p, threshold, beta):
    positive = int(sum(y))
    q = np.clip(p, np.finfo(float).eps, 1-np.finfo(float).eps)
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p)-1]
    tp_by_end = np.cumsum(y[order])[ends]
    ap = float(np.sum(tp_by_end/(ends+1)*np.diff(np.r_[0, tp_by_end]))/positive) if positive else None
    predictions = p >= threshold
    tp = int(sum(predictions & (y == 1)))
    fp = int(sum(predictions & (y == 0)))
    fn = positive-tp
    den = (1+beta**2)*tp+beta**2*fn+fp
    return {'rows': len(y), 'positive': positive, 'log_loss': float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
        'average_precision': ap, 'brier': float(np.mean((p-y)**2)), 'predicted_positive': int(sum(predictions)),
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'precision': tp/(tp+fp) if tp+fp else 0.,
        'recall': tp/positive if positive else 0., 'f_beta': (1+beta**2)*tp/den if den else 0.}


def manual_admission(metrics):
    checks = {}
    for action in ('exit', 'reduce', 'increase'):
        m = metrics[action]
        b = m['all']
        for comparison in ('first', 'linear'):
            gain = 1-b['log_loss']/m[comparison]['log_loss'] if m[comparison]['log_loss'] else 0.
            checks[f'{action}_log_loss_{comparison}'] = bool(gain > .01+1e-12)
            checks[f'{action}_ap_{comparison}'] = bool(b['average_precision'] >= m[comparison]['average_precision']-1e-12)
        checks[f'{action}_constant'] = bool(b['log_loss'] < m['constant']['log_loss']-1e-12)
    for metric in ('recall', 'f_beta'):
        checks['exit_'+metric] = metrics['exit']['all'][metric] > metrics['exit']['first'][metric]+1e-12
    return {'management_admitted': all(checks.values()), 'checks': checks, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}
