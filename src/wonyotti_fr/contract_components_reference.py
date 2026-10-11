from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction

import pandas as pd


def close_fraction(actual, expected, terms=()):
    actual = Fraction(str(actual))
    scale = max(Fraction(1), abs(actual), abs(expected), sum(map(abs, terms), Fraction(0)))
    error = abs(actual-expected)
    if error > scale/Fraction(10**40):
        raise ValueError('계약 독립 유리수 검산 불일치')
    return error


def scope(symbol):
    if symbol == 'XBTUSD':
        return 'xbtusd'
    return ('dated_btc' if len(symbol) == 6 and symbol[:3] == 'XBT'
        and symbol[3] in 'HMUZ' and symbol[4:].isdigit() else 'other_contract')


def value(balance, quantity, point, prices, observed, specs):
    for symbol in {'XBTUSD', *quantity}:
        if (symbol, point) not in prices or (symbol, point) not in observed:
            raise ValueError('계약 독립 평가 가격·활동 누락')
        if symbol in quantity and (not specs[symbol]['supported'] or pd.Timestamp(specs[symbol]['first_time']) >= point):
            raise ValueError('계약 독립 명세 지원 불일치')

    def dollar_value(ratio):
        result = Fraction(balance)*ratio
        for symbol, amount in quantity.items():
            contract, price = specs[symbol], prices[symbol, point]
            multiplier = Fraction(contract['multiplier_satoshi'])
            if contract['inverse']:
                result -= amount*multiplier/price
            elif contract['quote'] == 'XBT':
                result += amount*multiplier*price
            else:
                result += amount*multiplier*price*ratio
        return result

    equity = dollar_value(Fraction(1))
    delta = (dollar_value(Fraction(1001, 1000))-dollar_value(Fraction(999, 1000)))/Fraction(2, 1000)
    return equity, delta


def verify_contracts(source, saved, details, events, wallet, price_frame, activity, specs):
    pd.testing.assert_frame_equal(source.reset_index(drop=True), saved[source.columns], check_exact=True)
    if details[['end', 'symbol']].duplicated().any():
        raise ValueError('계약 독립 세부 행 중복')
    records = {(row['end'], row['symbol']): row for row in details.to_dict('records')}
    prices = {(row.symbol, row.end): Fraction(str(row.close)) for row in price_frame.itertuples(index=False)
        if row.support == 'available' and pd.notna(row.close) and row.close > 0}
    observed = {(row.symbol, row.end) for row in activity.itertuples(index=False) if row.observed_trade}
    transfers = [(pd.Timestamp(row.date, tz='UTC')-pd.Timedelta(days=1),
        pd.Timestamp(row.date, tz='UTC')+pd.Timedelta(days=2), int(row.amount))
        for row in wallet.itertuples(index=False) if row.transactstatus == 'Completed'
        and row.transacttype in ('Deposit', 'Withdrawal') and row.amount]
    iterator = iter(events.itertuples(index=False))
    event, balance, fees, quantity = next(iterator, None), 0, 0, {}

    def advance(point, interval=None):
        nonlocal event, balance, fees
        while event is not None and event.time < point:
            commission = int(event.execcomm)
            balance -= commission
            fees += commission
            if event.exectype in ('Trade', 'Settlement'):
                change = int(event.lastqty)*(1 if event.side == 'Buy' else -1)
                balance -= int(event.execcost)
                quantity[event.symbol] = quantity.get(event.symbol, 0)+change
                if not quantity[event.symbol]:
                    del quantity[event.symbol]
                if interval is not None:
                    item = interval.setdefault(event.symbol, {'quantity': 0, 'cost': 0, 'Trade': 0, 'Settlement': 0})
                    item['quantity'] += change
                    item['cost'] += int(event.execcost)
                    item[event.exectype] += 1
            event = next(iterator, None)

    groups = ('xbtusd', 'dated_btc', 'other_contract')
    maximum, supported, compared = Fraction(0), 0, 0
    for row in saved.itertuples(index=False):
        advance(row.end)
        before, before_fees, q0 = balance, fees, quantity.copy()
        interval = {}
        advance(row.label_end, interval)
        counts = {g: 0 for g in groups}
        for symbol, item in interval.items():
            counts[scope(symbol)] += item['Trade']+item['Settlement']
        if (sum(x['Trade'] for x in interval.values()) != row.trade_count
            or sum(x['Settlement'] for x in interval.values()) != row.settlement_count
            or row.contract_count != len(interval) or row.contract_supported != row.decomposition_supported
            or row.contract_reason != row.decomposition_reason
            or any(getattr(row, g+'_event_count') != counts[g] for g in groups)):
            raise ValueError('계약 독립 사건 수·지원 불일치')
        expected_details = []
        for symbol, item in interval.items():
            record = records.pop((row.end, symbol), None)
            if record is None:
                raise ValueError('계약 독립 세부 행 누락')
            wanted = {'end': row.end, 'label_end': row.label_end, 'symbol': symbol, 'group': scope(symbol),
                'net_quantity': item['quantity'], 'cost_satoshi': item['cost'],
                'trade_count': item['Trade'], 'settlement_count': item['Settlement'],
                'supported': row.decomposition_supported, 'reason': row.decomposition_reason}
            if any(record.get(key) != target for key, target in wanted.items()):
                raise ValueError('계약 독립 세부 원가·수량·경계 불일치')
            expected_details.append((symbol, item, record))
            compared += 1
        if not row.contract_supported:
            fields = ['contract_float_error', 'contract_equity_residual_satoshi',
                'contract_delta_residual_satoshi', 'contract_change_residual', *[g+'_execution_change' for g in groups]]
            if any(not pd.isna(getattr(row, key)) for key in fields):
                raise ValueError('계약 독립 미지원 행의 숫자 출력 오류')
            for _, _, record in expected_details:
                if any(not pd.isna(record[key]) for key in ['equity_change_satoshi', 'delta_change_satoshi', 'contribution', 'contribution_float']):
                    raise ValueError('계약 독립 미지원 세부 값 오류')
            continue
        supported += 1
        if any(start < point < end for start, end, _ in transfers for point in (row.end, row.label_end)):
            raise ValueError('계약 독립 지원 행의 외부 자금 불확실성')
        external = sum(amount for _, latest, amount in transfers if row.label_end >= latest)
        before_execution = before-(fees-before_fees)+external
        e0, d0 = value(before_execution, q0, row.label_end, prices, observed, specs)
        e1, d1 = value(balance+external, quantity, row.label_end, prices, observed, specs)
        if min(e0, e1) <= 0:
            raise ValueError('계약 독립 비양수 평가액')
        for name, wanted in [('fees_equity_satoshi', e0), ('fees_delta_satoshi', d0),
            ('executions_equity_satoshi', e1), ('executions_delta_satoshi', d1)]:
            maximum = max(maximum, close_fraction(getattr(row, name), wanted))
        sums = {g: Fraction(0) for g in groups}
        changes = []
        for symbol, item, record in expected_details:
            altered = q0.copy()
            altered[symbol] = altered.get(symbol, 0)+item['quantity']
            if not altered[symbol]:
                del altered[symbol]
            hypothetical_e, hypothetical_d = value(before_execution-item['cost'], altered,
                row.label_end, prices, observed, specs)
            de, dd = hypothetical_e-e0, hypothetical_d-d0
            contribution = ((hypothetical_d-d0)-d0/e0*(hypothetical_e-e0))/e1
            for name, wanted in [('equity_change_satoshi', de), ('delta_change_satoshi', dd), ('contribution', contribution)]:
                maximum = max(maximum, close_fraction(record[name], wanted))
            if not math.isfinite(record['contribution_float']) or abs(record['contribution_float']-float(contribution)) > 1e-12:
                raise ValueError('계약 독립 세부 부동소수 값 오류')
            sums[scope(symbol)] += contribution
            changes.append((de, dd, contribution))
        totals = [sum((item[i] for item in changes), Fraction(0)) for i in range(3)]
        if totals != [e1-e0, d1-d0, d1/e1-d0/e0]:
            raise ValueError('계약 독립 유리수 회계 항등식 오류')
        for i, name in enumerate(['contract_equity_residual_satoshi', 'contract_delta_residual_satoshi', 'contract_change_residual']):
            close_fraction(getattr(row, name), Fraction(0), [item[i] for item in changes])
        for group in groups:
            actual = getattr(row, group+'_execution_change')
            if not math.isfinite(actual) or abs(actual-float(sums[group])) > 1e-12:
                raise ValueError('계약 독립 계약군 기여분 불일치')
        error = abs(float(totals[2])-row.execution_change)
        if error > 1e-10 or not math.isfinite(row.contract_float_error) or abs(row.contract_float_error-error) > 1e-12:
            raise ValueError('계약 독립 기존 정답·오차 불일치')
    if records:
        raise ValueError('계약 독립 예상 밖 세부 행')
    with localcontext() as context:
        context.prec = 70
        error = str(Decimal(maximum.numerator)/Decimal(maximum.denominator))
    return {'all_rows_compared': len(saved), 'supported_rows_compared': supported,
        'contract_rows_compared': compared, 'maximum_fraction_difference': error,
        'all_integer_events_and_rational_accounts_replayed': True, 'exact_rational_additivity': True}


def reference_summary(frame):
    groups = ('xbtusd', 'dated_btc', 'other_contract')
    all_rows = list(frame.itertuples(index=False))
    rows = [r for r in all_rows if r.contract_supported]

    def direction(value):
        return -1 if value < -1e-12 else 1 if value > 1e-12 else 0

    statistics = {}
    for key in ['execution_change', *[g+'_execution_change' for g in groups]]:
        values = [getattr(row, key) for row in rows]
        signs, ordered = Counter(map(direction, values)), sorted(map(abs, values))
        quantiles = {}
        for label, fraction in [('0', 0), ('.25', .25), ('.5', .5), ('.75', .75), ('.9', .9), ('.99', .99), ('1', 1)]:
            if ordered:
                at = (len(ordered)-1)*fraction
                lower, upper = math.floor(at), math.ceil(at)
                quantiles[label] = ordered[lower]+(ordered[upper]-ordered[lower])*(at-lower)
        statistics[key] = {'negative': signs[-1], 'zero': signs[0], 'positive': signs[1],
            'abs_mean': math.fsum(ordered)/len(ordered) if ordered else None,
            'abs_quantiles': quantiles if ordered else None}
    cross = [[0]*3 for _ in range(3)]
    for row in rows:
        cross[direction(row.xbtusd_execution_change)+1][direction(row.execution_change)+1] += 1
    cohorts = {}
    for name, chosen in [('without_xbtusd_events', [r for r in rows if not r.xbtusd_event_count]),
        ('mixed_contract_groups', [r for r in rows if sum(getattr(r, g+'_event_count') > 0 for g in groups) > 1])]:
        cohorts[name] = {'rows': len(chosen), 'nonzero_total': sum(direction(r.execution_change) != 0 for r in chosen),
            'absolute_contribution_sum': {g: math.fsum(abs(getattr(r, g+'_execution_change')) for r in chosen) for g in groups}}
    return {'all_rows': len(all_rows), 'supported_rows': len(rows), 'unsupported_rows': len(all_rows)-len(rows),
        'reasons': dict(Counter(r.contract_reason for r in all_rows)), 'statistics': statistics,
        'xbtusd_rows_total_columns_sign_cross': cross, 'cohorts': cohorts,
        'all_event_counts': {g: sum(getattr(r, g+'_event_count') for r in all_rows) for g in groups},
        'supported_event_counts': {g: sum(getattr(r, g+'_event_count') for r in rows) for g in groups},
        'maximum_float_error': max((r.contract_float_error for r in rows), default=None)}
