from __future__ import annotations

import numpy as np
import pandas as pd

from wonyotti_fr.entry_boundary_support import ACTIONS, utc_times
from wonyotti_fr.event_features import MARKET_FEATURES

TARGETS = ['enter_long', 'enter_short', 'exit', 'reduce', 'increase']
PERIODS = {'training': ('2018-01-01', '2021-01-01'), 'diagnosis': ('2021-01-01', '2022-01-01')}


def validate_ledger(ledger):
    required = {'time', 'order_key', 'action', 'before_qty', 'after_qty', 'before_episode_id', 'after_episode_id'}
    if not required <= set(ledger) or ledger.empty:
        raise ValueError('상태별 사건 원장 열 누락·빈 입력')
    utc_times(ledger.time)
    integers = ['before_qty', 'after_qty', 'before_episode_id', 'after_episode_id']
    if (any(ledger[k].dtype.kind not in 'iu' or ledger[k].isna().any() for k in integers)
        or not ledger.action.isin(ACTIONS).all()
        or not ledger.order_key.map(lambda x: isinstance(x, str) and bool(x)).all()
        or not np.array_equal(ledger.before_qty.iloc[1:], ledger.after_qty.iloc[:-1])
        or not np.array_equal(ledger.before_episode_id.iloc[1:], ledger.after_episode_id.iloc[:-1])
        or any((ledger[f'{k}_qty'].eq(0) != ledger[f'{k}_episode_id'].eq(0)).any() for k in ('before', 'after'))):
        raise ValueError('상태별 사건 원장의 정수·연속 상태 오류')


def requested_orders(executions, ledger):
    required = {'time', 'symbol', 'exectype', 'order_key', 'side', 'orderqty', 'execid'}
    if not required <= set(executions):
        raise ValueError('상태별 주문 원본 열 누락')
    utc_times(executions.time)
    validate_ledger(ledger)
    if (executions.execid.isna().any() or executions.execid.duplicated().any()
        or not executions.order_key.map(lambda x: isinstance(x, str) and bool(x)).all()):
        raise ValueError('상태별 주문 원본 체결 중복')
    trade = executions.loc[executions.symbol.eq('XBTUSD') & executions.exectype.eq('Trade')
        & ~executions.order_key.str.startswith('special:')]
    first = trade.drop_duplicates('order_key', keep='first')[['time', 'order_key', 'side', 'orderqty', 'execid']].copy()
    if (first.orderqty.dtype.kind not in 'iuf' or not np.isfinite(first.orderqty).all()
        or first.orderqty.le(0).any() or (first.orderqty != np.trunc(first.orderqty)).any()
        or any(int(v) > np.iinfo(np.int64).max for v in first.orderqty) or not first.side.isin(['Buy', 'Sell']).all()):
        raise ValueError('상태별 최초 요청의 정수 수량·방향 오류')
    first['requested_quantity'] = first.orderqty.map(int).astype('int64')
    observed = ledger.loc[ledger.action.ne('funding')].drop_duplicates('order_key', keep='first')
    observed = observed[['time', 'order_key', 'before_qty', 'after_qty', 'before_episode_id', 'after_episode_id', 'action']].rename(
        columns={'time': 'event_time', 'action': 'first_filled_action'})
    result = first.merge(observed, on='order_key', how='left', validate='one_to_one')
    if result.before_qty.isna().any() or not result.time.equals(result.event_time.rename('time')):
        raise ValueError('상태별 최초 체결과 관측 수량의 연결 오류')
    side = np.where(result.side.eq('Buy'), 1, -1)
    before = result.before_qty.to_numpy()
    result['requested_action'] = np.select([before == 0, np.sign(before) == side,
        result.requested_quantity < np.abs(before), result.requested_quantity == np.abs(before)],
        ['open', 'increase', 'reduce', 'close'], default='reverse')
    result['requested_direction'] = side
    return result


def link_actions(rows, windows, *, requested=False):
    result = rows.reset_index(drop=True).copy()
    result['window_start'] = result.time.dt.floor('5min')
    result['window_index'] = pd.Index(windows.end).get_indexer(result.window_start)
    attached = windows[['end', 'state_present', 'state_quantity', 'state_direction', 'state_episode_id']].rename(columns={'end': 'window_start'})
    result = result.merge(attached, on='window_start', how='left', validate='many_to_one')
    kind = result.requested_action if requested else result.action
    direction = result.requested_direction if requested else np.sign(result.after_qty)
    same_episode = result.state_episode_id.eq(result.before_episode_id).fillna(False)
    same_direction = result.state_direction.eq(np.sign(result.before_qty)).fillna(False)
    result['label_scope'] = np.select([result.window_index.lt(0), ~result.state_present.eq(True), kind.eq('funding'),
        result.state_quantity.eq(0).fillna(False) & kind.eq('open'),
        result.state_quantity.ne(0).fillna(False) & same_episode & same_direction],
        ['outside_windows', 'unknown_window_state', 'funding', 'flat_entry', 'same_episode'], default='new_or_different_episode')
    flat, held = result.label_scope.eq('flat_entry'), result.label_scope.eq('same_episode')
    result['enter_long'] = flat & (direction > 0)
    result['enter_short'] = flat & (direction < 0)
    result['exit'] = held & kind.isin(['close', 'reverse'])
    result['reduce'] = held & kind.eq('reduce')
    result['increase'] = held & kind.eq('increase')
    return result


def action_windows(features, ledger, orders, episodes):
    if (not {'end', *MARKET_FEATURES} <= set(features)
        or not {'episode_id', 'entry_time', 'exit_time', 'closed'} <= set(episodes)):
        raise ValueError('상태별 행동 창의 필수 열 누락')
    validate_ledger(ledger)
    utc_times(orders.time)
    if orders.order_key.duplicated().any():
        raise ValueError('상태별 독립 주문의 중복')
    stamps = utc_times(features.end, unique=True)
    times = utc_times(ledger.time)
    if (features.empty or ledger.empty or not set(MARKET_FEATURES) <= set(features)
        or (stamps % 300_000_000_000).any() or episodes.episode_id.duplicated().any()
        or episodes.entry_time.isna().any() or not episodes.closed.isin([True, False]).all()):
        raise ValueError('상태별 행동 창의 입력·에피소드 오류')
    utc_times(episodes.sort_values('entry_time', kind='stable').entry_time)
    if (not ledger.loc[ledger.after_episode_id.ne(0), 'after_episode_id'].isin(episodes.episode_id).all()
        or not episodes.loc[episodes.closed, 'exit_time'].notna().all()
        or episodes.exit_time.lt(episodes.entry_time).any()):
        raise ValueError('상태별 행동 창의 에피소드 범위 오류')
    window = features[['end']].copy().reset_index(drop=True)
    window['label_end'] = window.end+pd.Timedelta(minutes=5)
    lookup = features.set_index('end')
    for timing, at in [('start', window.end), ('end', window.label_end)]:
        values = lookup.reindex(pd.DatetimeIndex(at))[MARKET_FEATURES].to_numpy(dtype=float)
        window[[f'{timing}_market_{name}' for name in MARKET_FEATURES]] = values
        window[f'{timing}_feature_present'] = at.isin(lookup.index)
        window[f'{timing}_feature_finite'] = np.isfinite(values).all(axis=1)
    state_index = np.searchsorted(times, stamps, side='left')-1
    window['state_present'] = state_index >= 0
    for name, column in [('state_quantity', 'after_qty'), ('state_episode_id', 'after_episode_id')]:
        values = pd.array(ledger[column].to_numpy()[np.maximum(state_index, 0)], dtype='Int64')
        values[state_index < 0] = pd.NA
        window[name] = values
    window['state_direction'] = np.sign(window.state_quantity)
    state_time = ledger.time.iloc[np.maximum(state_index, 0)].reset_index(drop=True)
    state_time[state_index < 0] = pd.NaT
    window['state_time'] = state_time
    window['source_covered'] = window.state_present & window.label_end.le(ledger.time.max())
    filled, requested = link_actions(ledger, window), link_actions(orders, window, requested=True)
    for kind in ACTIONS:
        counts = filled.loc[filled.action.eq(kind) & filled.window_index.ge(0)].groupby('window_index').size()
        window[f'event_{kind}_count'] = counts.reindex(window.index, fill_value=0).astype('int64')
        counts = requested.loc[requested.requested_action.eq(kind) & requested.window_index.ge(0)].groupby('window_index').size()
        window[f'request_{kind}_count'] = counts.reindex(window.index, fill_value=0).astype('int64')
    for prefix, rows in [('filled', filled), ('requested', requested)]:
        for target in TARGETS:
            selected = rows.loc[rows[target] & rows.window_index.ge(0)]
            counts = selected.groupby('window_index').size()
            unique = selected.groupby('window_index').order_key.nunique()
            window[f'{prefix}_{target}_count'] = counts.reindex(window.index, fill_value=0).astype('int64')
            window[f'{prefix}_{target}_orders'] = unique.reindex(window.index, fill_value=0).astype('int64')
            window[f'y_{prefix}_{target}'] = window[f'{prefix}_{target}_count'].gt(0)
    window['period'] = 'outside_periods'
    window['support_reason'] = 'outside_periods'
    for part, (begin, finish) in PERIODS.items():
        first, last = pd.Timestamp(begin, tz='UTC'), pd.Timestamp(finish, tz='UTC')
        period = window.end.ge(first) & window.end.lt(last)
        window.loc[period, 'period'] = part
        crossing = []
        for boundary in (first, last):
            ids = episodes.loc[episodes.entry_time.lt(boundary)
                & (~episodes.closed | episodes.exit_time.isna() | episodes.exit_time.ge(boundary)), 'episode_id']
            affected = filled.before_episode_id.isin(ids) | filled.after_episode_id.isin(ids)
            affected_windows = filled.loc[affected & filled.window_index.ge(0), 'window_index']
            crossing.append(window.state_episode_id.isin(ids) | window.index.isin(affected_windows))
        reasons = np.select([window.end.lt(first+pd.Timedelta(days=1)), window.label_end.ge(last-pd.Timedelta(days=1)),
            ~window.source_covered, ~window.start_feature_present, ~window.start_feature_finite,
            ~window.end_feature_present, ~window.end_feature_finite, crossing[0], crossing[1]],
            ['left_embargo', 'right_embargo', 'outside_source_coverage', 'missing_start_market', 'nonfinite_start_market',
                'missing_end_market', 'nonfinite_end_market', 'left_episode_boundary', 'right_episode_boundary'], default='included')
        window.loc[period, 'support_reason'] = reasons[period]
    window['usable'] = window.support_reason.eq('included')
    return window, filled, requested


def summarize_windows(windows, filled, requested):
    def rows(frame):
        valid = frame.loc[frame.usable]
        return {'windows': len(frame), 'usable': len(valid), 'flat': int(valid.state_quantity.eq(0).sum()),
            'held': int(valid.state_quantity.ne(0).sum()),
            'support_reasons': {str(k): int(v) for k, v in frame.support_reason.value_counts().sort_index().items()},
            'actions': {prefix: {kind: int(valid[f'{column}_{kind}_count'].sum()) for kind in ACTIONS}
                for prefix, column in [('filled', 'event'), ('requested', 'request')]},
            'labels': {prefix: {target: {'positive_windows': int(valid[f'y_{prefix}_{target}'].sum()),
                'events': int(valid[f'{prefix}_{target}_count'].sum()), 'window_order_pairs': int(valid[f'{prefix}_{target}_orders'].sum())}
                for target in TARGETS} for prefix in ('filled', 'requested')},
            'coexisting_management': {prefix: int(valid[[f'y_{prefix}_{k}' for k in ('exit', 'reduce', 'increase')]].sum(axis=1).gt(1).sum()) for prefix in ('filled', 'requested')},
            'both_entry_directions': {prefix: int((valid[f'y_{prefix}_enter_long'] & valid[f'y_{prefix}_enter_short']).sum()) for prefix in ('filled', 'requested')}}

    result = {'all': rows(windows), 'periods': {part: rows(windows.loc[windows.period.eq(part)]) for part in PERIODS},
        'yearly': [], 'monthly': [], 'ledgers': {}}
    for name, format_ in [('yearly', '%Y'), ('monthly', '%Y-%m')]:
        result[name] = [{'period': str(key), **rows(part)} for key, part in windows.groupby(windows.end.dt.strftime(format_), sort=True)]
    for name, frame in [('filled', filled), ('requested', requested)]:
        scopes = frame.label_scope.value_counts()
        result['ledgers'][name] = {'rows': len(frame), 'scopes': {str(k): int(v) for k, v in scopes.items()},
            'linked_targets': {target: int(frame[target].sum()) for target in TARGETS},
            'later_management_orders_in_window': 0}
    eligible = requested.loc[requested[['exit', 'reduce', 'increase']].any(axis=1)]
    # 같은 시각의 서로 다른 주문도 선언된 원본 순서에 따라 구분한다.
    ordinal = requested.groupby('window_index', sort=False).cumcount()
    later = ordinal.loc[eligible.index].gt(0)
    result['ledgers']['requested']['later_management_orders_in_window'] = int(later.sum())
    result['requested_vs_first_fill_action_mismatches'] = int(requested.requested_action.ne(requested.first_filled_action).sum())
    result['profitability_accepted'] = False
    return result


def summarize_request_sequence(windows, requested):
    rows = requested.copy()
    rows['supported'] = rows.window_index.isin(windows.index[windows.usable])
    rows['later'] = rows.window_index.ge(0) & rows.groupby('window_index', sort=False).cumcount().gt(0)

    def summarize(frame):
        return {'orders': len(frame), 'supported_orders': int(frame.supported.sum()),
            'scopes': {str(k): int(v) for k, v in frame.label_scope.value_counts().items()},
            'later_management': {scope: {target: int((frame.later & frame[target] & eligible).sum())
                for target in ('exit', 'reduce', 'increase')}
                for scope, eligible in [('all', True), ('supported', frame.supported)]}}

    return {'all': summarize(rows),
        'periods': {name: summarize(rows.loc[rows.time.ge(pd.Timestamp(first, tz='UTC')) & rows.time.lt(pd.Timestamp(last, tz='UTC'))])
            for name, (first, last) in PERIODS.items()},
        'yearly': [{'period': str(k), **summarize(v)} for k, v in rows.groupby(rows.time.dt.strftime('%Y'), sort=True)],
        'monthly': [{'period': str(k), **summarize(v)} for k, v in rows.groupby(rows.time.dt.strftime('%Y-%m'), sort=True)]}
