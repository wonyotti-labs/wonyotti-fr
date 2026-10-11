from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES

TARGET_NAMES = ('enter_long', 'enter_short', 'exit', 'reduce', 'increase')
ACTION_NAMES = ('open', 'reverse', 'increase', 'reduce', 'close', 'funding')


def _compare(actual, expected):
    left, right = actual.copy(), expected.copy()
    for frame in (left, right):
        for column in frame:
            if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                frame[column] = frame[column].astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)


def verify_requests(executions, ledger, saved):
    events = {}
    for row in ledger.itertuples(index=False):
        if row.action != 'funding' and row.order_key not in events:
            events[row.order_key] = row
    seen, expected = set(), []
    for row in executions.itertuples(index=False):
        if row.symbol != 'XBTUSD' or row.exectype != 'Trade' or row.order_key.startswith('special:') or row.order_key in seen:
            continue
        seen.add(row.order_key)
        event = events[row.order_key]
        direction = 1 if row.side == 'Buy' else -1
        before, quantity = int(event.before_qty), int(row.orderqty)
        if before == 0:
            action = 'open'
        elif (before > 0) == (direction > 0):
            action = 'increase'
        elif quantity < abs(before):
            action = 'reduce'
        elif quantity == abs(before):
            action = 'close'
        else:
            action = 'reverse'
        expected.append({'time': row.time, 'order_key': row.order_key, 'side': row.side,
            'orderqty': row.orderqty, 'execid': row.execid, 'requested_quantity': quantity,
            'event_time': event.time, 'before_qty': before, 'after_qty': event.after_qty,
            'before_episode_id': event.before_episode_id, 'after_episode_id': event.after_episode_id,
            'first_filled_action': event.action, 'requested_action': action, 'requested_direction': direction})
    reference = pd.DataFrame(expected, columns=saved.columns)
    _compare(saved, reference)
    return {'complete': True, 'first_orders_verified': len(expected)}


def _linked(rows, starts, state, *, requested):
    extra = defaultdict(list)
    for row in rows.itertuples(index=False):
        stamp = row.time.value
        index = bisect_right(starts, stamp)-1
        if index < 0 or stamp >= starts[index]+300_000_000_000:
            index = -1
        quantity, episode, _ = state[index] if index >= 0 else (None, None, None)
        present = quantity is not None
        direction = None if not present else 1 if quantity > 0 else -1 if quantity < 0 else 0
        kind = row.requested_action if requested else row.action
        intended = row.requested_direction if requested else row.after_qty
        if index < 0:
            scope = 'outside_windows'
        elif not present:
            scope = 'unknown_window_state'
        elif kind == 'funding':
            scope = 'funding'
        elif quantity == 0 and kind == 'open':
            scope = 'flat_entry'
        elif quantity != 0 and episode == row.before_episode_id and (quantity > 0) == (row.before_qty > 0) and row.before_qty != 0:
            scope = 'same_episode'
        else:
            scope = 'new_or_different_episode'
        values = {'window_start': pd.Timestamp(stamp-stamp % 300_000_000_000, tz='UTC'),
            'window_index': index, 'state_present': present if index >= 0 else None,
            'state_quantity': quantity, 'state_direction': direction, 'state_episode_id': episode,
            'label_scope': scope, 'enter_long': scope == 'flat_entry' and intended > 0,
            'enter_short': scope == 'flat_entry' and intended < 0,
            'exit': scope == 'same_episode' and kind in ('close', 'reverse'),
            'reduce': scope == 'same_episode' and kind == 'reduce',
            'increase': scope == 'same_episode' and kind == 'increase'}
        for key, value in values.items():
            extra[key].append(value)
    result = rows.reset_index(drop=True).copy()
    names = ['window_start', 'window_index', 'state_present', 'state_quantity', 'state_direction',
        'state_episode_id', 'label_scope', *TARGET_NAMES]
    for name in names:
        if name in ('state_quantity', 'state_direction', 'state_episode_id'):
            result[name] = pd.array(extra[name], dtype='Int64')
        elif name == 'window_start':
            result[name] = pd.to_datetime(extra[name], utc=True)
        else:
            result[name] = extra[name]
    return result


def verify_windows(features, ledger, orders, episodes, windows, filled, requested):
    pd.testing.assert_series_equal(windows.end, features.end.reset_index(drop=True), check_exact=True)
    starts = [value.value for value in features.end]
    event_times = [value.value for value in ledger.time]
    after, episode_ids = ledger.after_qty.tolist(), ledger.after_episode_id.tolist()
    state = []
    for stamp in starts:
        previous = bisect_left(event_times, stamp)-1
        state.append((after[previous], episode_ids[previous], event_times[previous]) if previous >= 0 else (None, None, None))
    expected = features[['end']].reset_index(drop=True).copy()
    expected['label_end'] = pd.to_datetime([s+300_000_000_000 for s in starts], utc=True)
    for name, values in [('state_quantity', [s[0] for s in state]), ('state_episode_id', [s[1] for s in state]),
        ('state_direction', [None if s[0] is None else 1 if s[0] > 0 else -1 if s[0] < 0 else 0 for s in state])]:
        expected[name] = pd.array(values, dtype='Int64')
    expected['state_present'] = [s[0] is not None for s in state]
    expected['state_time'] = pd.to_datetime([pd.Timestamp(s[2], tz='UTC') if s[2] is not None else pd.NaT for s in state], utc=True)
    expected['source_covered'] = [s[0] is not None and stamp+300_000_000_000 <= event_times[-1] for stamp, s in zip(starts, state, strict=True)]
    market = features[MARKET_FEATURES].to_numpy(dtype=float)
    for timing, offset in [('start', 0), ('end', 300_000_000_000)]:
        values, present, finite = [], [], []
        for stamp in starts:
            i = bisect_left(starts, stamp+offset)
            exists = i < len(starts) and starts[i] == stamp+offset
            value = market[i] if exists else np.full(len(MARKET_FEATURES), np.nan)
            values.append(value)
            present.append(exists)
            finite.append(bool(np.isfinite(value).all()))
        expected[[f'{timing}_market_{key}' for key in MARKET_FEATURES]] = np.asarray(values)
        expected[f'{timing}_feature_present'] = present
        expected[f'{timing}_feature_finite'] = finite
    links = [_linked(ledger, starts, state, requested=False), _linked(orders, starts, state, requested=True)]
    for actual, reference in zip((filled, requested), links, strict=True):
        _compare(actual, reference)
    related = defaultdict(set)
    for row in links[0].itertuples(index=False):
        if row.window_index >= 0:
            related[row.window_index].update((row.before_episode_id, row.after_episode_id))
    for prefix, column, rows in zip(('filled', 'requested'), ('event', 'request'), links, strict=True):
        counts, unique, actions = Counter(), defaultdict(set), Counter()
        for row in rows.itertuples(index=False):
            if row.window_index < 0:
                continue
            actions[row.window_index, row.requested_action if prefix == 'requested' else row.action] += 1
            for target in TARGET_NAMES:
                if getattr(row, target):
                    counts[row.window_index, target] += 1
                    unique[row.window_index, target].add(row.order_key)
        for kind in ACTION_NAMES:
            expected[f'{column}_{kind}_count'] = [actions[i, kind] for i in range(len(starts))]
        for target in TARGET_NAMES:
            expected[f'{prefix}_{target}_count'] = [counts[i, target] for i in range(len(starts))]
            expected[f'{prefix}_{target}_orders'] = [len(unique[i, target]) for i in range(len(starts))]
            expected[f'y_{prefix}_{target}'] = [counts[i, target] > 0 for i in range(len(starts))]
    boundaries = {part: tuple(pd.Timestamp(value, tz='UTC').value for value in dates)
        for part, dates in [('training', ('2018-01-01', '2021-01-01')), ('diagnosis', ('2021-01-01', '2022-01-01'))]}
    crossing = {}
    for boundary in {value for pair in boundaries.values() for value in pair}:
        crossing[boundary] = {r.episode_id for r in episodes.itertuples(index=False)
            if r.entry_time.value < boundary and (not r.closed or pd.isna(r.exit_time) or r.exit_time.value >= boundary)}
    parts, reasons = [], []
    day = 86_400_000_000_000
    for i, row in enumerate(expected.itertuples(index=False)):
        stamp, end = starts[i], starts[i]+300_000_000_000
        part = next((name for name, (first, last) in boundaries.items() if first <= stamp < last), 'outside_periods')
        reason = 'outside_periods'
        if part in boundaries:
            first, last = boundaries[part]
            ids = related[i] | ({state[i][1]} if state[i][1] is not None else set())
            checks = [(stamp < first+day, 'left_embargo'), (end >= last-day, 'right_embargo'),
                (not row.source_covered, 'outside_source_coverage'), (not row.start_feature_present, 'missing_start_market'),
                (not row.start_feature_finite, 'nonfinite_start_market'), (not row.end_feature_present, 'missing_end_market'),
                (not row.end_feature_finite, 'nonfinite_end_market'), (bool(ids & crossing[first]), 'left_episode_boundary'),
                (bool(ids & crossing[last]), 'right_episode_boundary')]
            reason = next((message for failed, message in checks if failed), 'included')
        parts.append(part)
        reasons.append(reason)
    expected['period'], expected['support_reason'] = parts, reasons
    expected['usable'] = [reason == 'included' for reason in reasons]
    _compare(windows[expected.columns], expected)
    if set(windows) != set(expected):
        raise ValueError('독립 행동 창 검산의 추가·누락 열')
    return {'complete': True, 'windows_verified': len(expected), 'filled_rows_verified': len(filled),
        'requested_rows_verified': len(requested), 'all_features_states_counts_and_support_exact': True}


def independent_summary(windows, filled, requested):
    def empty():
        return {'windows': 0, 'usable': 0, 'flat': 0, 'held': 0, 'support_reasons': Counter(),
            'actions': {p: dict.fromkeys(ACTION_NAMES, 0) for p in ('filled', 'requested')},
            'labels': {p: {t: {'positive_windows': 0, 'events': 0, 'window_order_pairs': 0} for t in TARGET_NAMES} for p in ('filled', 'requested')},
            'coexisting_management': dict.fromkeys(('filled', 'requested'), 0),
            'both_entry_directions': dict.fromkeys(('filled', 'requested'), 0)}

    groups = defaultdict(empty)
    for row in windows.itertuples(index=False):
        keys = ['all', row.end.strftime('%Y'), row.end.strftime('%Y-%m')]
        if row.period in ('training', 'diagnosis'):
            keys.append(row.period)
        for key in keys:
            g = groups[key]
            g['windows'] += 1
            g['support_reasons'][row.support_reason] += 1
            if not row.usable:
                continue
            g['usable'] += 1
            g['flat' if row.state_quantity == 0 else 'held'] += 1
            for prefix, column in [('filled', 'event'), ('requested', 'request')]:
                for kind in ACTION_NAMES:
                    g['actions'][prefix][kind] += getattr(row, f'{column}_{kind}_count')
                active = set()
                for target in TARGET_NAMES:
                    count = getattr(row, f'{prefix}_{target}_count')
                    label = g['labels'][prefix][target]
                    label['positive_windows'] += int(count > 0)
                    label['events'] += count
                    label['window_order_pairs'] += getattr(row, f'{prefix}_{target}_orders')
                    if count:
                        active.add(target)
                g['coexisting_management'][prefix] += int(len(active & {'exit', 'reduce', 'increase'}) > 1)
                g['both_entry_directions'][prefix] += int({'enter_long', 'enter_short'} <= active)
    result = {'all': groups['all'], 'periods': {p: groups[p] for p in ('training', 'diagnosis')},
        'yearly': [{'period': k, **groups[k]} for k in sorted(groups) if len(k) == 4],
        'monthly': [{'period': k, **groups[k]} for k in sorted(groups) if len(k) == 7], 'ledgers': {}}
    mismatches = 0
    for name, frame in [('filled', filled), ('requested', requested)]:
        scopes, targets, seen, later = Counter(), Counter(), set(), 0
        for row in frame.itertuples(index=False):
            scopes[row.label_scope] += 1
            for target in TARGET_NAMES:
                targets[target] += int(getattr(row, target))
            if name == 'requested':
                mismatches += row.requested_action != row.first_filled_action
                if row.window_index in seen and any(getattr(row, k) for k in ('exit', 'reduce', 'increase')):
                    later += 1
                seen.add(row.window_index)
        result['ledgers'][name] = {'rows': len(frame), 'scopes': dict(scopes),
            'linked_targets': {t: targets[t] for t in TARGET_NAMES}, 'later_management_orders_in_window': later}
    result['requested_vs_first_fill_action_mismatches'] = mismatches
    result['profitability_accepted'] = False
    return result


def independent_request_sequence(windows, requested):
    supported = {i for i, row in enumerate(windows.itertuples(index=False)) if row.usable}
    seen = set()

    def empty():
        return {'orders': 0, 'supported_orders': 0, 'scopes': Counter(),
            'later_management': {scope: dict.fromkeys(('exit', 'reduce', 'increase'), 0) for scope in ('all', 'supported')}}

    groups = defaultdict(empty)
    for row in requested.itertuples(index=False):
        later = row.window_index >= 0 and row.window_index in seen
        seen.add(row.window_index)
        available = row.window_index in supported
        keys = ['all', row.time.strftime('%Y'), row.time.strftime('%Y-%m')]
        if 2018 <= row.time.year < 2021:
            keys.append('training')
        elif row.time.year == 2021:
            keys.append('diagnosis')
        for key in keys:
            g = groups[key]
            g['orders'] += 1
            g['supported_orders'] += int(available)
            g['scopes'][row.label_scope] += 1
            for target in ('exit', 'reduce', 'increase'):
                if later and getattr(row, target):
                    g['later_management']['all'][target] += 1
                    g['later_management']['supported'][target] += int(available)
    return {'all': groups['all'], 'periods': {p: groups[p] for p in ('training', 'diagnosis')},
        'yearly': [{'period': k, **groups[k]} for k in sorted(groups) if len(k) == 4],
        'monthly': [{'period': k, **groups[k]} for k in sorted(groups) if len(k) == 7]}
