from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from wonyotti_fr.collateral_direction import external_bounds, validate_inputs

STAGES = ['initial', 'marks', 'external', 'funding', 'fees', 'executions']
COMPONENTS = ['mark_change', 'external_change', 'funding_change', 'fee_change', 'execution_change']
COUNTS = ['trade_count', 'settlement_count', 'funding_count', 'external_count']


def account_snapshots(events, wallet, points):
    events, wallet = events.reset_index(drop=True), wallet.reset_index(drop=True)
    _, queries, _, _, _, _, posted, dates = validate_inputs(events, wallet, points, 0)
    if queries.has_duplicates or (queries.asi8 % (300*10**9)).any():
        raise ValueError('노출 분해 조회 경계·중복 오류')
    low, high, uncertain = external_bounds(wallet, dates, posted, queries, 1)
    completed = np.zeros(len(queries), dtype=np.int64)
    for index in np.flatnonzero(wallet.transacttype.ne('RealisedPNL').to_numpy() & (posted != 0)):
        completed += queries >= dates.iloc[index]+pd.Timedelta(days=2)
    cumulative = dict.fromkeys(['cost', 'trade_fee', 'funding_fee', 'trade_count', 'settlement_count', 'funding_count'], 0)
    quantity, result, balance = {}, {}, 0
    iterator = iter(events.itertuples(index=False))
    event = next(iterator, None)
    for index, point in enumerate(queries):
        while event is not None and event.time < point:
            fee = int(event.execcomm)
            balance -= fee
            if event.exectype == 'Funding':
                cumulative['funding_fee'] += fee
                cumulative['funding_count'] += 1
            else:
                cost = int(event.execcost)
                balance -= cost
                cumulative['cost'] += cost
                cumulative['trade_fee'] += fee
                cumulative['trade_count' if event.exectype == 'Trade' else 'settlement_count'] += 1
                amount = quantity.get(event.symbol, 0)+int(event.lastqty)*(1 if event.side == 'Buy' else -1)
                if amount:
                    quantity[event.symbol] = amount
                else:
                    quantity.pop(event.symbol, None)
            event = next(iterator, None)
        result[point] = {'balance': balance, 'quantity': quantity.copy(), **cumulative,
            'external_min': int(low[index]), 'external_max': int(high[index]),
            'external_uncertain': int(uncertain[index]), 'external_count': int(completed[index])}
    return result


def price_lookup(prices, activity):
    if prices[['symbol', 'end']].duplicated().any() or activity[['symbol', 'end']].duplicated().any():
        raise ValueError('노출 분해 가격·활동 중복')
    if activity.observed_trade.isna().any() or not all(type(v) in (bool, np.bool_) for v in activity.observed_trade):
        raise ValueError('노출 분해 거래 활동 표시 오류')
    flags = {(row.symbol, row.end): bool(row.observed_trade) for row in activity.itertuples(index=False)}
    result = {}
    for row in prices.itertuples(index=False):
        stamp = pd.Timestamp(row.end)
        if stamp.tzinfo is None or str(stamp.tz) != 'UTC' or stamp.value % (300*10**9):
            raise ValueError('노출 분해 가격 시각 오류')
        price = Decimal(str(row.close))
        key = row.symbol, stamp
        supported = row.support == 'available' and price.is_finite() and price > 0
        result[key] = (price if supported else None, flags.get(key, False))
    return result


def value(balance, quantity, point, book, specs):
    required = {'XBTUSD', *quantity}
    problems = []
    for symbol in sorted(required):
        price, observed = book.get((symbol, point), (None, False))
        if price is None:
            problems.append(symbol+':missing_price')
        elif not observed:
            problems.append(symbol+':unobserved_trade')
        if symbol in quantity:
            spec = specs.get(symbol, {})
            if not spec.get('supported') or pd.Timestamp(spec['first_time']) >= point:
                problems.append(symbol+':unsupported_contract')
    if problems:
        return None, problems
    equity = delta = Decimal(balance)
    for symbol, amount in quantity.items():
        spec = specs[symbol]
        price = book[symbol, point][0]
        mark = Decimal(amount)*Decimal(spec['multiplier_satoshi'])
        mark = -mark/price if spec['inverse'] else mark*price
        equity += mark
        if not spec['inverse'] and spec['quote'] in ('USD', 'USDT'):
            delta += mark
    if equity <= 0:
        return None, ['nonpositive_equity']
    return (equity, delta, delta/equity), []


def decompose(source, snapshots, book, specs):
    if (source.end.duplicated().any() or not source.end.is_monotonic_increasing
        or not source.label_end.sub(source.end).eq(pd.Timedelta(minutes=5)).all()):
        raise ValueError('노출 분해 원본 행 시간 경계 오류')
    results = []
    with localcontext() as ctx:
        ctx.prec = 60
        for row in source.itertuples(index=False):
            before, after = snapshots[row.end], snapshots[row.label_end]
            changes = {key: after[key]-before[key] for key in ['cost', 'trade_fee', 'funding_fee', *COUNTS]}
            if after['balance'] != before['balance']-changes['cost']-changes['trade_fee']-changes['funding_fee']:
                raise ValueError('노출 분해 전체 원가·비용 연결 오류')
            record = {key: changes[key] for key in COUNTS}
            record['external_amount_satoshi'] = after['external_min']-before['external_min']
            record.update(dict.fromkeys(COMPONENTS, np.nan))
            for stage in STAGES:
                record.update({stage+'_equity_satoshi': None, stage+'_delta_satoshi': None, stage+'_beta': np.nan})
            record.update(decomposition_supported=False, decomposition_reason='', telescoping_residual=np.nan)
            reasons = []
            if not row.change_supported:
                reasons.append('source:'+row.change_reason)
            if before['external_uncertain'] or after['external_uncertain']:
                reasons.append('uncertain_external_transfer')
            if reasons:
                record['decomposition_reason'] = '|'.join(reasons)
                results.append(record)
                continue
            k0, q0, q1 = before['balance'], before['quantity'], after['quantity']
            ext0, ext1 = before['external_min'], after['external_min']
            arguments = [(k0+ext0, q0, row.end), (k0+ext0, q0, row.label_end),
                (k0+ext1, q0, row.label_end), (k0+ext1-changes['funding_fee'], q0, row.label_end),
                (k0+ext1-changes['funding_fee']-changes['trade_fee'], q0, row.label_end),
                (after['balance']+ext1, q1, row.label_end)]
            calculated = []
            for stage, arguments_for_stage in zip(STAGES, arguments, strict=True):
                evaluated, problems = value(*arguments_for_stage, book, specs)
                if problems:
                    reasons.extend(stage+':'+reason for reason in problems)
                calculated.append(evaluated)
            if reasons:
                record['decomposition_reason'] = '|'.join(reasons)
                results.append(record)
                continue
            for stage, (equity, delta, beta) in zip(STAGES, calculated, strict=True):
                record.update({stage+'_equity_satoshi': str(equity), stage+'_delta_satoshi': str(delta), stage+'_beta': float(beta)})
            if abs(float(calculated[0][2])-row.current_net_exposure) > 1e-10 or abs(float(calculated[-1][2])-row.net_exposure) > 1e-10:
                raise ValueError('노출 분해 기존 현재·다음 정답 불일치')
            pieces = [right[2]-left[2] for left, right in zip(calculated[:-1], calculated[1:], strict=True)]
            if not changes['trade_count']+changes['settlement_count'] and abs(pieces[-1]) > Decimal('1e-45'):
                raise ValueError('체결·정산 없는 창의 기여분 오류')
            residual = sum(pieces)-(calculated[-1][2]-calculated[0][2])
            if abs(residual) > Decimal('1e-45'):
                raise ValueError('노출 분해 기여분 합계 오류')
            record.update(zip(COMPONENTS, map(float, pieces), strict=True))
            record.update(decomposition_supported=True, decomposition_reason='included', telescoping_residual=float(residual))
            results.append(record)
    out = source.reset_index(drop=True).copy()
    for key, values in pd.DataFrame(results).items():
        if key in out:
            raise ValueError('노출 분해 기존 열 덮어쓰기 거부')
        out[key] = values.to_numpy()
    return out


def summarize_components(frame):
    supported = frame.loc[frame.decomposition_supported]
    keys = ['exposure_change', *COMPONENTS]
    stats = {}
    for key in keys:
        a = supported[key].to_numpy(dtype=float)
        stats[key] = None if not len(a) else {'mean': float(a.mean()), 'mae': float(np.abs(a).mean()),
            'rms': float(np.sqrt(np.square(a).mean())), 'quantiles': dict(zip(['0', '0.01', '0.25', '0.5', '0.75', '0.99', '1'],
                map(float, np.quantile(a, [0, .01, .25, .5, .75, .99, 1])), strict=True))}
    no_execution = supported.trade_count.add(supported.settlement_count).eq(0)
    no_events = supported[COUNTS].eq(0).all(axis=1)
    return {'all_rows': len(frame), 'source_supported_rows': int(frame.change_supported.sum()),
        'decomposition_supported_rows': len(supported), 'reasons': frame.decomposition_reason.value_counts().to_dict(),
        'all_source_event_counts': {key: int(frame[key].sum()) for key in COUNTS},
        'supported_event_counts': {key: int(supported[key].sum()) for key in COUNTS},
        'no_execution_rows': int(no_execution.sum()), 'no_execution_fraction': float(no_execution.mean()) if len(supported) else None,
        'no_event_rows': int(no_events.sum()), 'no_event_fraction': float(no_events.mean()) if len(supported) else None,
        'no_execution_nonzero_total_change_rows': int((no_execution & supported.exposure_change.ne(0)).sum()),
        'statistics': stats, 'covariance_columns': keys,
        'population_covariance': np.cov(supported[keys].to_numpy().T, ddof=0).tolist() if len(supported) else None}
