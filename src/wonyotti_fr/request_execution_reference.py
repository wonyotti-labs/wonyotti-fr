from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

MODELS = ('first', 'all', 'linear', 'constant')


def equal(expected, actual):
    if expected is None:
        assert pd.isna(actual)
    else:
        assert expected == actual, (expected, actual)


def verify_orders(raw, ledger, requests, ended):
    observed = {}
    columns = ['time', 'symbol', 'exectype', 'order_key', 'side', 'lastqty', 'orderqty', 'cumqty', 'ordstatus', 'lastliquidityind']
    for time, symbol, kind, key, side, qty, requested, cumulative, status, liquidity in raw[columns].itertuples(index=False, name=None):
        if symbol != 'XBTUSD' or kind != 'Trade' or key.startswith('special:'):
            continue
        if key not in observed:
            observed[key] = {'first_observed_time': time, 'first_order_quantity': requested,
                'observed_fills': 0, 'observed_quantity': 0, 'maker_fills': 0, 'taker_fills': 0,
                'other_liquidity_fills': 0, 'cumulative_mismatch_fills': 0, 'versions': set(), 'side': side}
        item = observed[key]
        assert item['side'] == side
        item['last_observed_time'] = time
        item['last_order_quantity'] = requested
        item['last_cumulative_quantity'] = cumulative
        item['last_observed_status'] = status
        item['observed_fills'] += 1
        item['observed_quantity'] += qty
        item['cumulative_mismatch_fills'] += cumulative != item['observed_quantity']
        item['versions'].add(requested)
        item[{'AddedLiquidity': 'maker_fills', 'RemovedLiquidity': 'taker_fills'}.get(liquidity, 'other_liquidity_fills')] += 1
    starts, stops = {}, {}
    for row in ledger.itertuples(index=False):
        if row.action == 'funding':
            continue
        if row.order_key not in starts:
            starts[row.order_key] = row
        if row.before_qty and (not row.after_qty or (row.before_qty > 0) != (row.after_qty > 0)):
            assert row.before_episode_id not in stops
            stops[row.before_episode_id] = row
    assert len(requests) == len(observed) and len(ended) == len(stops)
    for row in requests.itertuples(index=False):
        item, first = observed[row.order_key], starts[row.order_key]
        for key in ('first_observed_time', 'last_observed_time', 'first_order_quantity', 'last_order_quantity',
            'last_cumulative_quantity', 'last_observed_status', 'observed_fills', 'observed_quantity',
            'cumulative_mismatch_fills', 'maker_fills', 'taker_fills', 'other_liquidity_fills'):
            equal(item[key], getattr(row, key))
        equal(len(item['versions']), row.quantity_versions)
        equal(item['last_observed_time'].value-item['first_observed_time'].value, row.fill_span_ns)
        equal(item['observed_quantity']/item['first_order_quantity'], row.observed_to_first_requested)
        equal(first.event_index, row.first_event_index)
        equal(first.before_episode_id, row.before_episode_id)
        equal(first.before_qty, row.before_qty)
        equal(first.time, row.time)
        sign = 1 if item['side'] == 'Buy' else -1
        before, qty = first.before_qty, item['first_order_quantity']
        kind = ('open' if not before else 'increase' if (before > 0) == (sign > 0)
            else 'reduce' if qty < abs(before) else 'close' if qty == abs(before) else 'reverse')
        equal(kind, row.requested_action)
        stop = stops.get(first.before_episode_id)
        owner = ('not_held' if not before else 'not_observed' if stop is None
            else 'same_order' if stop.order_key == row.order_key else 'other_order')
        equal(owner, row.end_owner)
        equal(None if stop is None else stop.time, row.initial_episode_end_time)
        equal(None if stop is None else stop.event_index, row.initial_episode_end_index)
        equal(None if stop is None else stop.order_key, row.initial_episode_end_order)
        equal(None if stop is None else stop.time.value-first.time.value, row.termination_delay_ns)
        equal(stop is not None and stop.time.value//300_000_000_000 == first.time.value//300_000_000_000,
            row.termination_in_first_window)
    for row in ended.itertuples(index=False):
        stop, first = stops[row.before_episode_id], starts[row.order_key]
        equal(stop.event_index, row.event_index)
        if row.order_key.startswith('special:'):
            expected, time, kind, episode = 'special_order', None, None, None
        else:
            source = observed[row.order_key]
            time, episode = first.time, first.before_episode_id
            side = source['side'] == 'Buy'
            before, qty = first.before_qty, source['first_order_quantity']
            kind = ('open' if not before else 'increase' if (before > 0) == side
                else 'reduce' if qty < abs(before) else 'close' if qty == abs(before) else 'reverse')
            expected = ('different_episode' if episode != row.before_episode_id
                else 'exit_request_same_episode' if kind in ('close', 'reverse') else 'other_action_same_episode')
        equal(expected, row.origin)
        equal(time, row.first_request_time)
        equal(kind, row.first_requested_action)
        equal(episode, row.first_request_episode)
        equal(time is not None and time.value//300_000_000_000 == stop.time.value//300_000_000_000, row.request_in_same_window)
        equal(None if time is None else stop.time.value-time.value, row.request_to_termination_ns)
    return {'orders_verified': len(requests), 'termination_events_verified': len(ended)}


def verify_windows(windows, requested, filled, comparison):
    assert len(windows) == len(comparison)
    pd.testing.assert_frame_equal(windows[comparison.columns.intersection(windows.columns)],
        comparison[comparison.columns.intersection(windows.columns)], check_exact=True)
    lookup = {r.end.value: r for r in windows.itertuples(index=False)}
    positives = {}
    for label, rows in [('requested', requested), ('filled', filled)]:
        active = set()
        for row in rows.itertuples(index=False):
            stamp = row.time.value//300_000_000_000*300_000_000_000
            start = lookup.get(stamp)
            kind = row.requested_action if label == 'requested' else row.action
            if (start is not None and start.state_present and start.state_quantity != 0
                and start.state_episode_id == row.before_episode_id
                and (start.state_quantity > 0) == (row.before_qty > 0)
                and kind in ('close', 'reverse')):
                active.add(stamp)
        positives[label] = active
    for row in comparison.itertuples(index=False):
        for label, active in positives.items():
            equal(row.end.value in active, getattr(row, f'y_{label}_exit'))
    return {'windows_verified': len(windows)}


def independent_summary(requests, ended, windows):
    def records(frame):
        return frame.to_dict('records')

    r, e, w = records(requests), records(ended), records(windows)

    def counts(values):
        return dict(sorted(Counter(str(v) for v in values).items()))

    def stats(values):
        ordered = sorted(float(v) for v in values if pd.notna(v))
        result = {'count': len(ordered)}
        for name, q in [('min', 0.), ('p50', .5), ('p90', .9), ('p99', .99), ('max', 1.)]:
            if not ordered:
                result[name] = None
                continue
            at = q*(len(ordered)-1)
            lower, upper = math.floor(at), math.ceil(at)
            result[name] = ordered[lower]+(at-lower)*(ordered[upper]-ordered[lower])
        return result

    def orders(rows):
        exits = [v for v in rows if v['requested_action'] in ('close', 'reverse')]
        return {'orders': len(rows), 'exit_requests': len(exits),
            'exit_first_filled_actions': counts(v['first_filled_action'] for v in exits),
            'exit_end_owner': counts(v['end_owner'] for v in exits),
            'exit_same_window_end': sum(v['termination_in_first_window'] for v in exits),
            'exit_same_order_same_window': sum(v['termination_in_first_window'] and v['end_owner'] == 'same_order' for v in exits),
            'exit_delay_seconds': stats((v['termination_delay_ns']/1e9 if pd.notna(v['termination_delay_ns']) else None) for v in exits),
            'all_fill_span_seconds': stats(v['fill_span_ns']/1e9 for v in rows),
            'exit_fill_span_seconds': stats(v['fill_span_ns']/1e9 for v in exits),
            'quantity_changed_orders': sum(v['quantity_versions'] > 1 for v in rows),
            'exit_quantity_changed_orders': sum(v['quantity_versions'] > 1 for v in exits),
            'exit_observed_less_than_first_requested': sum(v['observed_quantity'] < v['first_order_quantity'] for v in exits),
            'exit_observed_to_first_requested': stats(v['observed_to_first_requested'] for v in exits),
            'exit_last_observed_status': counts(v['last_observed_status'] if pd.notna(v['last_observed_status']) else 'unknown' for v in exits),
            **{name: sum(v[name] for v in rows) for name in ('observed_fills', 'maker_fills', 'taker_fills', 'other_liquidity_fills', 'cumulative_mismatch_fills')}}

    def terminations(rows):
        return {'events': len(rows), 'origin': counts(v['origin'] for v in rows),
            'request_in_same_window': sum(v['request_in_same_window'] for v in rows),
            'request_to_termination_seconds': stats((v['request_to_termination_ns']/1e9 if pd.notna(v['request_to_termination_ns']) else None) for v in rows)}

    def table(rows):
        tally = Counter((v['y_requested_exit'], v['y_filled_exit']) for v in rows)
        return {'windows': len(rows), 'both': tally[True, True], 'requested_only': tally[True, False],
            'filled_only': tally[False, True], 'neither': tally[False, False]}

    def metrics(rows, model, label):
        tally = Counter((bool(v[f'y_{label}_exit']), bool(v[f'{model}_exit_request'])) for v in rows)
        tp, fp, fn = tally[True, True], tally[False, True], tally[True, False]
        return {'positive': tp+fn, 'requests': tp+fp, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
            'f2': 5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else None}

    def predicted(rows):
        return {model: {label: metrics(rows, model, label) for label in ('requested', 'filled')} for model in MODELS}

    result = {'orders': orders(r), 'terminations': terminations(e),
        'v107_supported_held': table([v for v in w if v['usable'] and v['state_quantity'] != 0]),
        'paired_periods': {}, 'predictions': {}, 'yearly': [], 'monthly': [],
        'model_fitted': False, 'profitability_accepted': False}
    for part in ('training', 'calibration', 'diagnosis'):
        selected = [v for v in w if v['paired_usable'] and v['paired_part'] == part]
        result['paired_periods'][part] = table(selected)
        if part != 'training':
            result['predictions'][part] = predicted(selected)
    for name, format_ in [('yearly', '%Y'), ('monthly', '%Y-%m')]:
        groups = []
        for rows, time in [(r, 'time'), (e, 'time'), (w, 'end')]:
            group = {}
            for row in rows:
                group.setdefault(row[time].strftime(format_), []).append(row)
            groups.append(group)
        for key in sorted(set().union(*groups)):
            rs, es, ws = [group.get(key, []) for group in groups]
            scored = [v for v in ws if v['paired_usable'] and v['paired_part'] in ('calibration', 'diagnosis')]
            result[name].append({'period': key, 'orders': orders(rs), 'terminations': terminations(es),
                'v107_supported_held': table([v for v in ws if v['usable'] and v['state_quantity'] != 0]),
                'predictions': predicted(scored) if scored else None})
    return result


def compare_summary(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            compare_summary(expected[key], actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            compare_summary(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=1e-12)
    else:
        assert expected == actual, (expected, actual)
