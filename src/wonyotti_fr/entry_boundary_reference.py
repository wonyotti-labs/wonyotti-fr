from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES


def verify_aggregation(executions, events):
    required = {'symbol', 'time', 'execid', 'order_key', 'exectype', 'side', 'ordtype', 'lastqty', 'execcost', 'execcomm', 'source_row'}
    if (not required <= set(executions) or executions.empty or executions.execid.duplicated().any()
        or not isinstance(executions.time.dtype, pd.DatetimeTZDtype) or str(executions.time.dt.tz) != 'UTC'
        or executions.time.isna().any() or not executions.time.is_monotonic_increasing
        or any(executions[k].dtype.kind not in 'iu' for k in ('lastqty', 'execcost', 'execcomm', 'source_row'))):
        raise ValueError('진입 원본 묶음 검산의 정수·시각·중복 오류')
    groups = {}
    for row in executions.loc[executions.symbol.eq('XBTUSD')].itertuples(index=False):
        if row.exectype not in ('Trade', 'Funding'):
            raise ValueError('진입 검산의 미지원 원본 사건')
        key = (row.time, row.order_key, row.exectype, row.side, row.ordtype)
        if key not in groups:
            groups[key] = {'quantity': 0, 'cost_satoshi': 0, 'fee_satoshi': 0, 'fills': 0, 'source_row': int(row.source_row)}
        group = groups[key]
        group['quantity'] += int(row.lastqty)
        group['cost_satoshi'] += int(row.execcost)
        group['fee_satoshi'] += int(row.execcomm)
        group['fills'] += 1
        group['source_row'] = min(group['source_row'], int(row.source_row))
    ordered = sorted(groups, key=lambda k: (k[0], groups[k]['source_row']))
    if len(ordered) != len(events):
        raise ValueError('진입 검산의 원본 묶음 수 불일치')
    for key, row in zip(ordered, events.itertuples(index=False), strict=True):
        if key != (row.time, row.order_key, row.exectype, row.side, row.ordtype) or any(
            getattr(row, name) != value for name, value in groups[key].items()):
            raise ValueError('진입 검산의 원본 묶음 순서·정수 합계 불일치')
    return {'complete': True, 'event_groups': len(events), 'raw_xbtusd_rows': sum(g['fills'] for g in groups.values()),
        'all_quantity_cost_fee_counts_and_source_rows_exact': True}


def independent_ledger(events, actions):
    rows, q, episode, closed, seen = [], 0, 0, None, set()
    quantities = []
    for i, (event, old) in enumerate(zip(events.itertuples(index=False), actions.itertuples(index=False), strict=True)):
        prior_episode = episode if q else 0
        before = q
        if event.exectype == 'Funding':
            kind = 'funding'
        else:
            q += int(event.quantity) if event.side == 'Buy' else -int(event.quantity)
            if before == 0:
                kind = 'open'
                episode += 1
            elif q == 0:
                kind = 'close'
            elif (before > 0) != (q > 0):
                kind = 'reverse'
                episode += 1
            else:
                kind = 'increase' if abs(q) > abs(before) else 'reduce'
        if (old.time != event.time or old.order_key != event.order_key or old.before_qty != before
            or old.after_qty != q or old.action != kind or old.episode_id != episode):
            raise ValueError('진입 스칼라 수량·행동·에피소드 불일치')
        quantities.append(q)
        gap = event.time.value-closed if kind == 'open' and closed is not None else None
        band = 'not_open'
        if kind == 'open':
            if gap is None:
                band = 'unknown'
            elif gap == 0:
                band = 'zero'
            elif gap/10**9 <= 60:
                band = 'up_to_60s'
            elif gap/10**9 <= 300:
                band = 'up_to_300s'
            else:
                band = 'over_300s'
        first = event.exectype == 'Trade' and event.order_key not in seen
        if event.exectype == 'Trade':
            seen.add(event.order_key)
        rows.append({'event_index': i, 'before_episode_id': prior_episode, 'after_episode_id': episode if q else 0,
            'first_trade_of_order': first, 'special_order': event.order_key.startswith('special:'),
            'prior_flat_ns': gap, 'prior_flat_band': band})
        if kind == 'close':
            closed = event.time.value
        elif kind == 'open':
            closed = None
    times = [t.value for t in events.time]
    for i, row in enumerate(rows):
        for seconds in (60, 300):
            boundary = times[i]-times[i] % (seconds*10**9)
            previous = bisect_left(times, boundary)-1
            quantity = quantities[previous] if previous >= 0 else None
            row[f'boundary_{seconds}_quantity'] = quantity
            row[f'boundary_{seconds}_observed'] = previous >= 0
            row[f'boundary_{seconds}_flat'] = quantity == 0
    result = actions.reset_index(drop=True).copy()
    for column in rows[0]:
        values = [row[column] for row in rows]
        if column == 'prior_flat_ns' or column.endswith('_quantity'):
            result[column] = pd.array(values, dtype='Int64')
        elif column.endswith('_flat'):
            result[column] = pd.array(values, dtype='boolean')
        else:
            result[column] = values
    return result


def verify_entries(ledger, features, sources, saved):
    source = {row.end: (part, row) for part, frame in sources.items() for row in frame.itertuples(index=False)}
    market = {row.end: row for row in features.itertuples(index=False)}
    opens = ledger.loc[ledger.action.eq('open')].reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[opens.columns], opens, check_exact=True)
    for entry, actual in zip(opens.itertuples(index=False), saved.itertuples(index=False), strict=True):
        at = pd.Timestamp(entry.time.value-entry.time.value % 300_000_000_000, tz='UTC')
        bar = market.get(at)
        finite = bar is not None and all(np.isfinite(getattr(bar, key)) for key in MARKET_FEATURES)
        first = pd.Timestamp(year=entry.time.year, month=1, day=1, tz='UTC')
        last = pd.Timestamp(year=entry.time.year+1, month=1, day=1, tz='UTC')
        embargo = at >= first+pd.Timedelta(hours=24) and entry.time < last-pd.Timedelta(hours=24)
        part, original = source.get(at, (None, None))
        reason = 'outside_original_source' if original is None else 'jointly_supported' if original.alignment_supported else 'unsupported_original_window'
        fields = {'feature_end': at, 'feature_present': bar is not None, 'feature_finite': finite,
            'year_embargo_pass': embargo, 'market_reason': 'missing_feature_boundary' if bar is None
                else 'nonfinite_market' if not finite else 'year_embargo' if not embargo else 'available',
            'source_reason': reason, 'source_window_flat': original is not None and original.direction == 0,
            'source_net_nonzero': original is not None and abs(original.xbtusd_execution_change) > 1e-12}
        for key, value in fields.items():
            if getattr(actual, key) != value:
                raise ValueError('진입 사건 연결 독립 검산 불일치: '+key)
        for key, value in {'source_part': part, 'end': at if original is not None else None,
            'alignment_supported': original.alignment_supported if original is not None else None,
            'direction': original.direction if original is not None else None,
            'xbtusd_execution_change': original.xbtusd_execution_change if original is not None else None}.items():
            actual_value = getattr(actual, key)
            if not (pd.isna(actual_value) and (value is None or pd.isna(value))) and actual_value != value:
                raise ValueError('진입 원본 정답 연결 불일치: '+key)
    return {'complete': True, 'all_entries_verified': len(saved), 'exact_market_and_source_time_lookup': True}


def independent_summary(ledger, entries):
    action_names = ['open', 'reverse', 'increase', 'reduce', 'close', 'funding']
    gap_names = ['zero', 'up_to_60s', 'up_to_300s', 'over_300s', 'unknown']
    event_groups, entry_groups = defaultdict(list), defaultdict(list)
    times = defaultdict(list)
    for row in ledger.itertuples(index=False):
        for key in ('all', row.time.strftime('%Y'), row.time.strftime('%Y-%m')):
            event_groups[key].append(row)
        if row.action != 'funding':
            times[row.time].append(1 if row.after_qty > row.before_qty else -1)
    for row in entries.itertuples(index=False):
        for key in ('all', row.time.strftime('%Y'), row.time.strftime('%Y-%m')):
            entry_groups[key].append(row)

    def summarize(key):
        events, opens = event_groups[key], entry_groups[key]
        actions = {}
        for kind in action_names:
            chosen = [r for r in events if r.action == kind]
            actions[kind] = {'events': len(chosen), 'unique_orders': len({r.order_key for r in chosen}),
                'first_trade_events': sum(bool(r.first_trade_of_order) for r in chosen)}
        admitted = [r for r in opens if r.source_reason == 'jointly_supported']
        entry = {'events': len(opens), 'long': sum(r.after_qty > 0 for r in opens), 'short': sum(r.after_qty < 0 for r in opens),
            'special_orders': sum(bool(r.special_order) for r in opens),
            'flat_gaps': {band: sum(r.prior_flat_band == band for r in opens) for band in gap_names},
            'boundary': {str(seconds): {'observed_flat': sum(bool(getattr(r, f'boundary_{seconds}_flat')) for r in opens),
                'observed_held': sum(bool(getattr(r, f'boundary_{seconds}_observed')) and not bool(getattr(r, f'boundary_{seconds}_flat')) for r in opens),
                'unknown': sum(not bool(getattr(r, f'boundary_{seconds}_observed')) for r in opens)} for seconds in (60, 300)},
            'market_support': dict(Counter(r.market_reason for r in opens)),
            'market_available_directions': {str(side): sum(r.market_reason == 'available' and (1 if r.after_qty > 0 else -1) == side for r in opens) for side in (-1, 1)},
            'source_support': dict(Counter(r.source_reason for r in opens)),
            'supported_window_flat': sum(bool(r.source_window_flat) for r in admitted),
            'supported_window_held': sum(not bool(r.source_window_flat) for r in admitted),
            'supported_net_zero': sum(not bool(r.source_net_nonzero) for r in admitted)}
        return {'actions': actions, 'opens': entry}

    all_rows = summarize('all')
    return {'total_events': len(ledger), **all_rows,
        'yearly': [{'period': key, **summarize(key)} for key in sorted(event_groups) if len(key) == 4],
        'monthly': [{'period': key, **summarize(key)} for key in sorted(event_groups) if len(key) == 7],
        'opposite_trade_sides_same_time_groups': sum(len(set(rows)) > 1 for rows in times.values()),
        'multi_trade_event_time_groups': sum(len(rows) > 1 for rows in times.values()), 'profitability_accepted': False}
