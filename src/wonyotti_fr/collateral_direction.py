from __future__ import annotations

import numpy as np
import pandas as pd

SATOSHI = 100_000_000
LIMIT = np.iinfo(np.int64).max
LABELS = ('flat', 'short', 'long', 'ambiguous', 'unsupported_other_contract')


def integers(values, label):
    a = np.asarray(values)
    if a.dtype.kind not in 'iu' or any(abs(int(v)) > LIMIT for v in a):
        raise ValueError(f'{label}: 사토시·수량 정수 범위 오류')
    if sum(abs(int(v)) for v in a) > LIMIT:
        raise ValueError(f'{label}: 누적 정수 범위 초과')
    return a.astype(np.int64)


def utc_times(values, label):
    times = pd.DatetimeIndex(values)
    if times.tz is None or str(times.tz) != 'UTC' or times.hasnans:
        raise ValueError(f'{label}: UTC 시각 오류')
    return times.as_unit('ns')


def validate_inputs(events, wallet, points, opening):
    if not isinstance(opening, int) or isinstance(opening, bool) or abs(opening) > LIMIT:
        raise ValueError('초기 현금 정수 오류')
    times, queries = utc_times(events.time, '체결'), utc_times(points, '조회')
    if not len(events) or not times.is_monotonic_increasing or not queries.is_monotonic_increasing:
        raise ValueError('체결·조회 시간 순서 오류')
    if not len(queries) or queries.min() < times.min() or queries.max() > times.max()+pd.Timedelta(1, 'ns'):
        raise ValueError('원본 시각 지원 범위 오류')
    if (not events.settlcurrency.eq('XBt').all() or events.symbol.isna().any()
        or events.symbol.eq('').any() or not events.exectype.isin(['Trade', 'Settlement', 'Funding']).all()):
        raise ValueError('계약·이벤트·정산 통화 오류')
    trade = events.exectype.ne('Funding').to_numpy()
    qty = integers(events.lastqty, '수량')
    cost, fee = integers(events.execcost, '비용'), integers(events.execcomm, '수수료')
    if (qty < 0).any() or (qty[trade] <= 0).any() or not events.loc[trade, 'side'].isin(['Buy', 'Sell']).all():
        raise ValueError('체결 수량·방향 오류')
    for symbol, part in events.loc[events.exectype.eq('Trade') & events.execcost.ne(0)].groupby('symbol'):
        inverse = part.execcost.lt(0) == part.side.eq('Buy')
        if inverse.nunique() != 1 or (symbol == 'XBTUSD' and not inverse.all()):
            raise ValueError('원본 비용 방향 오류')
    if (not len(wallet) or not wallet.currency.eq('XBt').all()
        or not wallet.transactstatus.isin(['Completed', 'Canceled']).all()
        or not wallet.transacttype.isin(['Deposit', 'Withdrawal', 'RealisedPNL']).all()):
        raise ValueError('지갑 통화·유형 오류')
    amount = integers(wallet.amount, '지갑 금액')
    posted = integers(wallet.posted_amount, '지갑 기장')
    if not np.array_equal(posted, np.where(wallet.transactstatus.eq('Completed'), amount, 0)):
        raise ValueError('지갑 완료·취소 기장 오류')
    external = wallet.transacttype.ne('RealisedPNL').to_numpy()
    if ((wallet.transacttype.eq('Deposit') & (amount < 0)).any()
        or (wallet.transacttype.eq('Withdrawal') & (amount > 0)).any()):
        raise ValueError('외부 입출금 부호 오류')
    # 각 원장의 합뿐 아니라 결합된 누적 계산도 정수 범위 안에 있어야 한다.
    if abs(opening)+sum(abs(int(v)) for a in [cost, fee, posted[external]] for v in a) > LIMIT:
        raise ValueError('결합 원장 누적 범위 초과')
    dates = pd.to_datetime(wallet.date, format='%Y-%m-%d', utc=True, errors='raise')
    if dates.isna().any():
        raise ValueError('지갑 날짜 누락')
    return times, queries, trade, qty, cost, fee, posted, dates


def external_bounds(wallet, dates, posted, queries, extra_days):
    if extra_days not in (0, 1):
        raise ValueError('입출금 날짜 창 오류')
    low, high = np.zeros(len(queries), dtype=np.int64), np.zeros(len(queries), dtype=np.int64)
    count = np.zeros(len(queries), dtype=np.int64)
    for i in np.flatnonzero(wallet.transacttype.ne('RealisedPNL').to_numpy() & (posted != 0)):
        earliest = dates.iloc[i]-pd.Timedelta(days=extra_days)
        latest = dates.iloc[i]+pd.Timedelta(days=1+extra_days)
        done = queries >= latest
        # 조회 경계와 같은 시각의 이벤트는 아직 반영하지 않는다.
        uncertain = (queries > earliest) & (queries < latest)
        value = int(posted[i])
        low += np.where(done, value, np.where(uncertain, min(value, 0), 0))
        high += np.where(done, value, np.where(uncertain, max(value, 0), 0))
        count += uncertain
    return low, high, count


def direction_ledger(events, wallet, points, opening=0, extra_days=0):
    events, wallet = events.reset_index(drop=True), wallet.reset_index(drop=True)
    times, queries, trade, qty, cost, fee, posted, dates = validate_inputs(events, wallet, points, opening)
    side = np.where(events.side.eq('Buy'), 1, -1)
    changes = np.where(trade, side*qty, 0)
    balances = (-np.where(trade, cost, 0)-fee).cumsum()
    other_changes = np.zeros(len(events), dtype=np.int64)
    xbt_changes = np.where(events.symbol.eq('XBTUSD'), changes, 0)
    for symbol, indexes in events.groupby('symbol', sort=False).indices.items():
        if symbol == 'XBTUSD':
            continue
        after = changes[indexes].cumsum()
        other_changes[indexes] = (after != 0).astype(np.int64)-np.r_[0, (after[:-1] != 0).astype(np.int64)]
    indexes = np.searchsorted(times.asi8, queries.asi8, side='left')
    cost_balance = np.r_[0, balances][indexes] + opening
    xbt = np.r_[0, xbt_changes.cumsum()][indexes]
    others = np.r_[0, other_changes.cumsum()][indexes]
    if (others < 0).any():
        raise ValueError('다른 계약의 열린 수량 집계 오류')
    low, high, uncertainty = external_bounds(wallet, dates, posted, queries, extra_days)
    low, high = low+cost_balance, high+cost_balance
    labels = np.full(len(queries), 'ambiguous', dtype=object)
    labels[high < 0] = 'short'
    labels[low > 0] = 'long'
    labels[(low == 0) & (high == 0)] = 'flat'
    labels[others != 0] = 'unsupported_other_contract'
    # 다른 계약의 미실현 손익을 누락한 값을 순노출로 내보내지 않는다.
    lows, highs = pd.array(low, dtype='Int64'), pd.array(high, dtype='Int64')
    lows[others != 0], highs[others != 0] = pd.NA, pd.NA
    return pd.DataFrame({'time': queries, 'xbt_quantity': xbt, 'other_open_contracts': others,
        'delta_min_satoshi': lows, 'delta_max_satoshi': highs,
        'external_uncertain_count': uncertainty, 'account_direction': labels,
        'futures_direction': np.where(xbt == 0, 'flat', np.where(xbt < 0, 'short', 'long'))})


def summarize_rows(frame):
    supported = frame.account_direction.isin(LABELS[:3])
    comparable = frame.loc[supported]
    counts = frame.account_direction.value_counts().reindex(LABELS, fill_value=0).to_dict()
    cross = pd.crosstab(comparable.futures_direction, comparable.account_direction).reindex(
        index=LABELS[:3], columns=LABELS[:3], fill_value=0)
    different = int(comparable.account_direction.ne(comparable.futures_direction).sum())
    return {'rows': len(frame), 'direction_counts': {k: int(v) for k, v in counts.items()},
        'comparable_rows': len(comparable), 'different_rows': different,
        'different_fraction_of_comparable': different/len(comparable) if len(comparable) else None,
        'date_uncertain_rows': int(frame.external_uncertain_count.gt(0).sum()),
        'cross_table_futures_rows_account_columns': cross.to_dict(orient='index'),
        'independent_trade_count': None, 'conditional_on_wallet_date_window': True}
