from __future__ import annotations

from decimal import Decimal, localcontext

import pandas as pd


def independent_contract_check(events, specs):
    observed = {}
    with localcontext() as ctx:
        ctx.prec = 70
        for row in events.itertuples(index=False):
            if row.exectype == 'Funding' or row.symbol not in specs:
                continue
            spec = specs[row.symbol]
            if not spec.get('supported'):
                continue
            quantity, price, cost = Decimal(int(row.lastqty)), Decimal(str(row.lastpx)), Decimal(int(row.execcost))
            inverse = (cost < 0) == (row.side == 'Buy')
            multiplier = Decimal(spec['multiplier_satoshi'])
            if (spec['inverse'] != inverse or spec['quote'] != row.currency or row.settlcurrency != 'XBt'
                or not multiplier.is_finite() or multiplier <= 0):
                raise ValueError('독립 계약 형태·통화·승수 불일치')
            if row.symbol not in observed:
                expected = Decimal(100000000) if inverse else abs(cost)/quantity/price
                if multiplier != expected or pd.Timestamp(spec['first_time']) != row.time:
                    raise ValueError('독립 최초 관찰 계약 단위 불일치')
                observed[row.symbol] = [0, Decimal(0)]
            residual = abs(abs(cost)/quantity-(multiplier/price if inverse else multiplier*price))
            if residual > (Decimal('0.500001') if inverse else Decimal('0.000001')):
                raise ValueError('독립 전체 계약 원가 오차 초과')
            observed[row.symbol][0] += 1
            observed[row.symbol][1] = max(observed[row.symbol][1], residual)
        for symbol, spec in specs.items():
            if not spec.get('supported'):
                continue
            if symbol not in observed or observed[symbol][0] != spec['fills_checked']:
                raise ValueError('독립 계약 전체 체결 수 불일치')
            if abs(observed[symbol][1]-Decimal(spec['maximum_unit_residual_satoshi'])) > Decimal('1e-35'):
                raise ValueError('독립 계약 최대 원가 잔차 불일치')
    return len(observed)


def independent_values(events, wallet, points, prices, specs, opening=0, extra_days=0):
    transfers = []
    for row in wallet.itertuples(index=False):
        if row.transactstatus == 'Completed' and row.transacttype in ('Deposit', 'Withdrawal'):
            day = pd.Timestamp(row.date, tz='UTC')
            transfers.append((day-pd.Timedelta(days=extra_days), day+pd.Timedelta(days=1+extra_days), int(row.amount)))
    prices_by_symbol = {}
    for symbol, part in prices.groupby('symbol'):
        prices_by_symbol[symbol] = {row.end: Decimal(str(row.close))
            for row in part.itertuples(index=False) if row.support == 'available'}
    state, convention, records = {}, {}, []
    iterator = iter(events.itertuples(index=False))
    event = next(iterator, None)
    with localcontext() as ctx:
        ctx.prec = 70
        cash = Decimal(opening)
        for point in pd.DatetimeIndex(points):
            while event is not None and event.time < point:
                cash -= Decimal(int(event.execcomm))
                if event.exectype != 'Funding':
                    old, basis = state.get(event.symbol, (0, Decimal(0)))
                    change = int(event.lastqty)*(1 if event.side == 'Buy' else -1)
                    if event.execcost:
                        inv = (event.execcost < 0) == (event.side == 'Buy')
                        if event.symbol in convention and convention[event.symbol] != inv:
                            raise ValueError('독립 포트폴리오 원가 부호 오류')
                        convention[event.symbol] = inv
                    inv = convention[event.symbol]
                    unit = abs(Decimal(int(event.execcost))/abs(change))
                    if not old or (old > 0) == (change > 0):
                        new_basis = (basis*abs(old)+unit*abs(change))/abs(old+change)
                    else:
                        closing = min(abs(old), abs(change))
                        cash += Decimal((1 if old > 0 else -1)*closing*(1 if inv else -1))*(basis-unit)
                        new_basis = unit if abs(change) > abs(old) else basis
                    state[event.symbol] = (old+change, new_basis if old+change else Decimal(0))
                event = next(iterator, None)
            a, b, count = Decimal(0), Decimal(0), 0
            for start, stop, amount in transfers:
                if point >= stop:
                    a += amount
                    b += amount
                elif point > start and amount:
                    a += min(0, amount)
                    b += max(0, amount)
                    count += 1
            stamp = point-pd.Timedelta(minutes=point.minute % 5, seconds=point.second,
                microseconds=point.microsecond, nanoseconds=point.nanosecond)
            btc = prices_by_symbol.get('XBTUSD', {}).get(stamp)
            selected, unsupported = {}, [] if btc is not None else ['XBTUSD:price']
            coefficient = cash
            for symbol, (quantity, basis) in state.items():
                if not quantity:
                    continue
                inv = convention[symbol]
                coefficient += Decimal(quantity)*basis*(1 if inv else -1)
                spec = specs.get(symbol, {})
                if not spec.get('supported') or point <= pd.Timestamp(spec['first_time']):
                    unsupported.append(symbol+':contract')
                    continue
                px = prices_by_symbol.get(symbol, {}).get(stamp)
                if px is None:
                    unsupported.append(symbol+':price')
                    continue
                selected[symbol] = (quantity, basis, inv, px, spec['quote'], Decimal(spec['multiplier_satoshi']))
            record = {'time': point, 'unsupported_reasons': '|'.join(sorted(set(unsupported))),
                'cost_balance_satoshi': int(coefficient.to_integral_value()),
                'external_min_satoshi': int(a), 'external_max_satoshi': int(b),
                'external_uncertain_count': count}
            if abs(coefficient-coefficient.to_integral_value()) > Decimal('1e-35'):
                raise ValueError('독립 현금·잔존 원가의 정수 보존 오류')
            if unsupported:
                records.append(record)
                continue

            def value(external, ratio, overrides=None, cash=cash, selected=selected, btc=btc):
                equity = cash+external
                for symbol, (q, basis, inv, px, quote, multiplier) in selected.items():
                    if overrides and symbol in overrides:
                        px *= overrides[symbol]
                    if inv:
                        mark = multiplier/(px*ratio)
                    else:
                        mark = multiplier*px/(ratio if quote == 'XBT' else Decimal(1))
                    equity += Decimal(q)*(basis-mark if inv else mark-basis)
                return equity*btc*ratio/Decimal(100_000_000)

            def sensitivity(external, overrides=None, value=value, btc=btc):
                return ((value(external, Decimal('1.001'), overrides)
                    -value(external, Decimal('0.999'), overrides))
                    / (btc*Decimal('0.002'))*Decimal(100_000_000))

            values = [value(a, Decimal(1))*Decimal(100_000_000)/btc,
                value(b, Decimal(1))*Decimal(100_000_000)/btc, sensitivity(a), sensitivity(b)]
            low_marks, high_marks = {}, {}
            for symbol, (q, _, _, _, _, _) in selected.items():
                # 양수 포지션은 두 계약 형태 모두 가격 상승에 평가액이 증가한다.
                low_marks[symbol] = Decimal('0.99') if q > 0 else Decimal('1.01')
                high_marks[symbol] = Decimal('1.01') if q > 0 else Decimal('0.99')
            values.extend([value(a, Decimal(1), low_marks)*Decimal(100_000_000)/btc,
                value(b, Decimal(1), high_marks)*Decimal(100_000_000)/btc,
                sensitivity(a, low_marks), sensitivity(b, high_marks)])
            keys = ['equity_min_satoshi', 'equity_max_satoshi', 'delta_min_satoshi', 'delta_max_satoshi',
                'stress_equity_min_satoshi', 'stress_equity_max_satoshi',
                'stress_delta_min_satoshi', 'stress_delta_max_satoshi']
            record.update(dict(zip(keys, values, strict=True)))
            record['equity_min_usd'], record['equity_max_usd'] = value(a, Decimal(1)), value(b, Decimal(1))
            records.append(record)
    return records
