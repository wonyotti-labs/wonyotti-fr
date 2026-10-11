from __future__ import annotations

import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES

ACTIONS = ['open', 'reverse', 'increase', 'reduce', 'close', 'funding']
GAP_BANDS = ['zero', 'up_to_60s', 'up_to_300s', 'over_300s', 'unknown']


def utc_times(values, *, unique=False):
    if (not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != 'UTC'
        or values.isna().any() or not values.is_monotonic_increasing
        or unique and values.duplicated().any()):
        raise ValueError('진입 지원 시각의 시간대·정렬·중복 오류')
    return values.astype('datetime64[ns, UTC]').array.asi8


def transition_ledger(events, actions):
    required = {'time', 'order_key', 'exectype', 'side', 'ordtype', 'quantity', 'source_row'}
    if not required <= set(events) or not {'time', 'order_key', 'action', 'before_qty', 'after_qty', 'episode_id'} <= set(actions):
        raise ValueError('진입 지원 사건·행동 원장 열 누락')
    e, a = events.reset_index(drop=True), actions.reset_index(drop=True)
    times = utc_times(e.time)
    utc_times(a.time)
    if (e.empty or len(e) != len(a) or not e.time.equals(a.time) or not e.order_key.equals(a.order_key)
        or not e.exectype.isin(['Trade', 'Funding']).all()
        or e.duplicated(['time', 'order_key', 'exectype', 'side', 'ordtype']).any()
        or not e.order_key.map(lambda x: isinstance(x, str) and bool(x)).all()
        or any(e[c].dtype.kind not in 'iu' for c in ['quantity', 'source_row'])
        or e.source_row.lt(1).any()
        or not e[['time', 'source_row']].equals(e[['time', 'source_row']].sort_values(['time', 'source_row'], kind='stable').reset_index(drop=True))):
        raise ValueError('진입 지원 사건·행동의 정렬·입력 오류')
    trade = e.exectype.eq('Trade').to_numpy()
    if (not e.loc[trade, 'side'].isin(['Buy', 'Sell']).all() or e.loc[trade, 'quantity'].le(0).any()
        or sum(int(v) for v in e.loc[trade, 'quantity']) > np.iinfo(np.int64).max):
        raise ValueError('진입 지원 체결 방향·정수 수량 오류')
    change = np.where(trade, np.where(e.side.eq('Buy'), e.quantity, -e.quantity), 0).astype(np.int64)
    after = change.cumsum()
    before = np.r_[0, after[:-1]]
    starts = trade & ((before == 0) | ((after != 0) & (np.sign(before) != np.sign(after))))
    kinds = np.select([~trade, before == 0, after == 0, np.sign(before) != np.sign(after),
        np.abs(after) > np.abs(before)], ['funding', 'open', 'close', 'reverse', 'increase'], default='reduce')
    episodes = starts.cumsum()
    if (any(a[c].dtype.kind not in 'iu' for c in ['before_qty', 'after_qty', 'episode_id'])
        or not np.array_equal(a.before_qty, before) or not np.array_equal(a.after_qty, after)
        or not np.array_equal(a.action, kinds) or not np.array_equal(a.episode_id, episodes)):
        raise ValueError('진입 지원 수량·행동·에피소드 재생 불일치')
    result = a.copy()
    result['event_index'] = np.arange(len(e))
    result['before_episode_id'] = np.where(before == 0, 0, np.r_[0, episodes[:-1]])
    result['after_episode_id'] = np.where(after == 0, 0, episodes)
    result['first_trade_of_order'] = False
    result.loc[trade, 'first_trade_of_order'] = ~e.loc[trade, 'order_key'].duplicated().to_numpy()
    result['special_order'] = e.order_key.str.startswith('special:')
    flat_start, gaps, bands = None, [], []
    for stamp, kind in zip(times, kinds, strict=True):
        gap = int(stamp-flat_start) if kind == 'open' and flat_start is not None else None
        gaps.append(gap)
        bands.append('not_open' if kind != 'open' else 'unknown' if gap is None else 'zero' if gap == 0
            else 'up_to_60s' if gap <= 60_000_000_000 else 'up_to_300s' if gap <= 300_000_000_000 else 'over_300s')
        if kind == 'close':
            flat_start = stamp
        elif kind == 'open':
            flat_start = None
    result['prior_flat_ns'] = pd.array(gaps, dtype='Int64')
    result['prior_flat_band'] = bands
    for seconds in (60, 300):
        boundaries = times//(seconds*10**9)*(seconds*10**9)
        index = np.searchsorted(times, boundaries, side='left')-1
        values = pd.array(after[np.maximum(index, 0)], dtype='Int64')
        values[index < 0] = pd.NA
        result[f'boundary_{seconds}_quantity'] = values
        result[f'boundary_{seconds}_observed'] = index >= 0
        result[f'boundary_{seconds}_flat'] = (values == 0).fillna(False)
    return result


def entry_support(ledger, features, sources):
    utc_times(ledger.time)
    utc_times(features.end, unique=True)
    if not set(MARKET_FEATURES) <= set(features) or set(sources) != {'training', 'diagnosis'}:
        raise ValueError('진입 지원 시장 특징·원본 표 오류')
    entries = ledger.loc[ledger.action.eq('open')].copy().reset_index(drop=True)
    entries['feature_end'] = entries.time.dt.floor('5min')
    index = features.set_index('end')
    market = index.reindex(pd.DatetimeIndex(entries.feature_end))[MARKET_FEATURES].to_numpy(dtype=float)
    entries['feature_present'] = entries.feature_end.isin(index.index)
    entries['feature_finite'] = np.isfinite(market).all(axis=1)
    first = pd.to_datetime(entries.time.dt.year.astype(str)+'-01-01', utc=True)
    last = pd.to_datetime((entries.time.dt.year+1).astype(str)+'-01-01', utc=True)
    entries['year_embargo_pass'] = (entries.feature_end >= first+pd.Timedelta(days=1)) & (entries.time < last-pd.Timedelta(days=1))
    entries['market_reason'] = np.select([~entries.feature_present, ~entries.feature_finite, ~entries.year_embargo_pass],
        ['missing_feature_boundary', 'nonfinite_market', 'year_embargo'], default='available')
    pieces = []
    for part, frame in sources.items():
        utc_times(frame.end, unique=True)
        if not {'alignment_supported', 'direction', 'xbtusd_execution_change'} <= set(frame):
            raise ValueError('진입 지원 기존 정답 열 누락')
        pieces.append(frame[['end', 'alignment_supported', 'direction', 'xbtusd_execution_change']].assign(source_part=part))
    source = pd.concat(pieces, ignore_index=True).sort_values('end', kind='stable')
    utc_times(source.end, unique=True)
    entries = entries.merge(source, left_on='feature_end', right_on='end', how='left', validate='many_to_one')
    entries['source_reason'] = np.select([entries.source_part.isna(), ~entries.alignment_supported.eq(True)],
        ['outside_original_source', 'unsupported_original_window'], default='jointly_supported')
    entries['source_window_flat'] = entries.direction.eq(0)
    entries['source_net_nonzero'] = entries.xbtusd_execution_change.abs().gt(1e-12)
    matched = entries.source_part.notna()
    if (not np.array_equal(entries.loc[matched, 'direction'],
            np.sign(entries.loc[matched, 'boundary_300_quantity'].to_numpy(dtype=float)))
        or not entries.loc[matched, 'boundary_300_observed'].all()):
        raise ValueError('진입 사건과 기존 창의 직전 상태 불일치')
    return entries


def summarize_support(ledger, entries):
    def actions(frame):
        return {kind: {'events': int(frame.action.eq(kind).sum()),
            'unique_orders': int(frame.loc[frame.action.eq(kind), 'order_key'].nunique()),
            'first_trade_events': int(frame.loc[frame.action.eq(kind), 'first_trade_of_order'].sum())} for kind in ACTIONS}

    def opens(frame):
        supported = frame.source_reason.eq('jointly_supported')
        return {'events': len(frame), 'long': int(frame.after_qty.gt(0).sum()), 'short': int(frame.after_qty.lt(0).sum()),
            'special_orders': int(frame.special_order.sum()),
            'flat_gaps': {band: int(frame.prior_flat_band.eq(band).sum()) for band in GAP_BANDS},
            'boundary': {str(seconds): {'observed_flat': int(frame[f'boundary_{seconds}_flat'].sum()),
                'observed_held': int((frame[f'boundary_{seconds}_observed'] & ~frame[f'boundary_{seconds}_flat']).sum()),
                'unknown': int((~frame[f'boundary_{seconds}_observed']).sum())} for seconds in (60, 300)},
            'market_support': {str(k): int(v) for k, v in frame.market_reason.value_counts().sort_index().items()},
            'market_available_directions': {str(side): int((frame.market_reason.eq('available') & np.sign(frame.after_qty).eq(side)).sum()) for side in (-1, 1)},
            'source_support': {str(k): int(v) for k, v in frame.source_reason.value_counts().sort_index().items()},
            'supported_window_flat': int((supported & frame.source_window_flat).sum()),
            'supported_window_held': int((supported & ~frame.source_window_flat).sum()),
            'supported_net_zero': int((supported & ~frame.source_net_nonzero).sum())}

    result = {'total_events': len(ledger), 'actions': actions(ledger), 'opens': opens(entries), 'yearly': [], 'monthly': []}
    for label, format_ in [('yearly', '%Y'), ('monthly', '%Y-%m')]:
        keys = ledger.time.dt.strftime(format_)
        entry_keys = entries.time.dt.strftime(format_)
        result[label] = [{'period': str(period), 'actions': actions(part), 'opens': opens(entries.loc[entry_keys.eq(period)])}
            for period, part in ledger.groupby(keys, sort=True)]
    trade = ledger.loc[ledger.action.ne('funding')]
    mixed = trade.groupby('time').after_qty.agg('size')
    sides = pd.Series(np.sign(trade.after_qty.to_numpy()-trade.before_qty.to_numpy()), index=trade.index)
    count = sides.groupby(trade.time).nunique()
    result['opposite_trade_sides_same_time_groups'] = int(count.gt(1).sum())
    result['multi_trade_event_time_groups'] = int(mixed.gt(1).sum())
    result['profitability_accepted'] = False
    return result
