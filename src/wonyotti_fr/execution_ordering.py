from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .entry_boundary_support import utc_times


def inspect_cumulative_order(raw):
    required = {'time', 'symbol', 'exectype', 'order_key', 'execid', 'lastqty', 'cumqty', 'orderqty', 'ordstatus', 'side'}
    if not required <= set(raw):
        raise ValueError('누적 체결 순서 진단 열 누락')
    utc_times(raw.time)
    if (raw.execid.isna().any() or raw.execid.duplicated().any()
        or not raw.order_key.map(lambda v: isinstance(v, str) and bool(v)).all()):
        raise ValueError('누적 체결 순서의 중복·주문 식별 오류')
    data = raw.loc[raw.symbol.eq('XBTUSD') & raw.exectype.eq('Trade')
        & ~raw.order_key.str.startswith('special:')].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError('누적 체결 순서 진단의 정상 체결 부족')
    for name in ('lastqty', 'cumqty', 'orderqty'):
        if (data[name].dtype.kind not in 'iu' or data[name].isna().any() or data[name].le(0).any()
            or sum(int(v) for v in data[name]) > np.iinfo(np.int64).max):
            raise ValueError('누적 체결 순서의 정수 수량 오류')
    if not data.side.isin(['Buy', 'Sell']).all() or data.groupby('order_key').side.nunique().ne(1).any():
        raise ValueError('누적 체결 순서의 주문 방향 오류')
    data['source_ordinal'] = np.arange(len(data))
    data['source_prefix'] = data.groupby('order_key', sort=False).lastqty.cumsum()
    data['timestamp_end_quantity'] = data.groupby(['order_key', 'time'], sort=False).source_prefix.transform('max')
    # 같은 시각의 내부 순서만 대조하며 기존 원본 원장은 수정하지 않는다.
    ordered = data.sort_values(['time', 'order_key', 'cumqty', 'source_ordinal'], kind='stable').copy()
    ordered['cumulative_order_prefix'] = ordered.groupby('order_key', sort=False).lastqty.cumsum()
    data['cumulative_order_prefix'] = ordered.cumulative_order_prefix.reindex(data.index)
    data['source_mismatch'] = data.cumqty.ne(data.source_prefix)
    data['ordered_mismatch'] = data.cumqty.ne(data.cumulative_order_prefix)
    grouped = data.groupby(['order_key', 'time'], sort=False)
    size = grouped.size()
    first = data.groupby('order_key', sort=False).head(1).set_index('order_key')
    reordered_first = ordered.groupby('order_key', sort=False).head(1).set_index('order_key').reindex(first.index)
    last = data.groupby('order_key', sort=False).tail(1).set_index('order_key')
    reordered_last = ordered.groupby('order_key', sort=False).tail(1).set_index('order_key').reindex(last.index)
    summary = {'normal_rows': len(data), 'source_prefix_mismatches': int(data.source_mismatch.sum()),
        'mismatches_matching_timestamp_end': int((data.source_mismatch & data.cumqty.eq(data.timestamp_end_quantity)).sum()),
        'all_rows_matching_timestamp_end': int(data.cumqty.eq(data.timestamp_end_quantity).sum()),
        'sorted_cumqty_within_timestamp_mismatches': int(data.ordered_mismatch.sum()),
        'timestamp_groups': len(size), 'multiple_row_timestamp_groups': int(size.gt(1).sum()),
        'group_end_max_cumqty_mismatches': int(grouped.cumqty.max().ne(grouped.timestamp_end_quantity.first()).sum()),
        'last_order_max_cumqty_mismatches': int(data.groupby('order_key').cumqty.max().ne(data.groupby('order_key').lastqty.sum()).sum()),
        'status_counts': {str(k): int(v) for k, v in data.ordstatus.value_counts().items()},
        'orders': len(first), 'first_observed_time_changes': int(first.time.ne(reordered_first.time).sum()),
        'first_observed_quantity_changes': int(first.orderqty.ne(reordered_first.orderqty).sum()),
        'last_observed_status_changes': int(last.ordstatus.ne(reordered_last.ordstatus).sum()),
        'cumulative_order_last_status': {str(k): int(v) for k, v in reordered_last.ordstatus.value_counts().items()},
        'canonical_ledger_changed': False, 'missing_executions_proven': False}
    return data, summary


def verify_cumulative_order(raw, observed, summary):
    groups = defaultdict(list)
    rows = []
    for row in raw.itertuples(index=False):
        if row.symbol != 'XBTUSD' or row.exectype != 'Trade' or row.order_key.startswith('special:'):
            continue
        i = len(rows)
        rows.append(row)
        groups[(row.order_key, row.time)].append(i)
    assert len(observed) == len(rows)
    original_prefix, ordered_prefix, group_end = [np.zeros(len(rows), dtype=np.int64) for _ in range(3)]
    prior = {}
    for key, indexes in groups.items():
        order, _ = key
        start = prior.get(order, 0)
        total = start+sum(int(rows[i].lastqty) for i in indexes)
        value = start
        for i in indexes:
            value += int(rows[i].lastqty)
            original_prefix[i], group_end[i] = value, total
        value = start
        for i in sorted(indexes, key=lambda index: (rows[index].cumqty, index)):
            value += int(rows[i].lastqty)
            ordered_prefix[i] = value
        prior[order] = total
    for name, expected in [('source_prefix', original_prefix), ('cumulative_order_prefix', ordered_prefix), ('timestamp_end_quantity', group_end)]:
        np.testing.assert_array_equal(expected, observed[name])
    np.testing.assert_array_equal(observed.execid, [r.execid for r in rows])
    source_bad = [i for i,r in enumerate(rows) if r.cumqty != original_prefix[i]]
    ordered_bad = [i for i,r in enumerate(rows) if r.cumqty != ordered_prefix[i]]
    assert summary['source_prefix_mismatches'] == len(source_bad)
    assert summary['sorted_cumqty_within_timestamp_mismatches'] == len(ordered_bad)
    assert summary['group_end_max_cumqty_mismatches'] == sum(max(rows[i].cumqty for i in ix) != group_end[ix[0]] for ix in groups.values())
    assert summary['timestamp_groups'] == len(groups)
    assert summary['multiple_row_timestamp_groups'] == sum(len(ix) > 1 for ix in groups.values())
    by_order = defaultdict(list)
    ordered_indices = []
    for indexes in groups.values():
        ordered_indices.extend(sorted(indexes, key=lambda index: (rows[index].cumqty, index)))
    for i,row in enumerate(rows):
        by_order[row.order_key].append(i)
    reordered = defaultdict(list)
    for i in ordered_indices:
        reordered[rows[i].order_key].append(i)
    first_changes, time_changes, status_changes = 0, 0, 0
    last_status = Counter()
    for key,indexes in by_order.items():
        before_first, after_first = rows[indexes[0]], rows[reordered[key][0]]
        before_last, after_last = rows[indexes[-1]], rows[reordered[key][-1]]
        first_changes += before_first.orderqty != after_first.orderqty
        time_changes += before_first.time != after_first.time
        status_changes += before_last.ordstatus != after_last.ordstatus
        last_status[after_last.ordstatus] += 1
    expected = {'normal_rows': len(rows), 'orders': len(by_order),
        'mismatches_matching_timestamp_end': sum(rows[i].cumqty == group_end[i] for i in source_bad),
        'all_rows_matching_timestamp_end': sum(row.cumqty == group_end[i] for i,row in enumerate(rows)),
        'last_order_max_cumqty_mismatches': sum(max(rows[i].cumqty for i in ix) != sum(rows[i].lastqty for i in ix) for ix in by_order.values()),
        'first_observed_time_changes': time_changes, 'first_observed_quantity_changes': first_changes,
        'last_observed_status_changes': status_changes, 'cumulative_order_last_status': dict(last_status),
        'status_counts': dict(Counter(row.ordstatus for row in rows)),
        'canonical_ledger_changed': False, 'missing_executions_proven': False}
    for name,value in expected.items():
        assert summary[name] == value, name
    return {'raw_rows_verified': len(rows), 'timestamp_groups_verified': len(groups), 'prefixes_exact': True}
