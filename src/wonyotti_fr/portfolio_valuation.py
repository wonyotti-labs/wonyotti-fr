from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from wonyotti_fr.collateral_direction import external_bounds, validate_inputs

SAT = Decimal(100_000_000)
INVERSE = frozenset({'XBTUSD', 'XBTH20', 'XBTU21'})
QUANTUM = Decimal('0.000001')
VALUE_COLUMNS = ('equity_min_satoshi', 'equity_max_satoshi', 'delta_min_satoshi',
    'delta_max_satoshi', 'stress_equity_min_satoshi', 'stress_equity_max_satoshi',
    'stress_delta_min_satoshi', 'stress_delta_max_satoshi')


def contract_specs(events, symbols):
    result = {}
    with localcontext() as ctx:
        ctx.prec = 60
        for symbol in sorted(set(symbols)):
            part = events.loc[events.symbol.eq(symbol) & events.exectype.ne('Funding')]
            if part.empty:
                result[symbol] = {'supported': False, 'reason': 'no_observed_fill'}
                continue
            first = part.iloc[0]
            quote = first.currency
            inverse = symbol in INVERSE
            if not part.settlcurrency.eq('XBt').all() or not part.currency.eq(quote).all():
                raise ValueError('계약 정산·호가 통화 변경')
            if quote not in ('USD', 'USDT', 'XBT') or (inverse and quote != 'USD'):
                result[symbol] = {'supported': False, 'reason': 'unsupported_currency'}
                continue
            price = Decimal(str(first.lastpx))
            if not price.is_finite() or price <= 0 or first.lastqty <= 0 or first.execcost == 0:
                raise ValueError('계약 최초 체결 단위 오류')
            # 역선물 1달러 단위는 당시 명세를 사용하고 체결 원가는 그대로 보존한다.
            multiplier = SAT if inverse else abs(Decimal(int(first.execcost)))/Decimal(int(first.lastqty))/price
            if multiplier <= 0 or multiplier != multiplier.to_integral_value():
                result[symbol] = {'supported': False, 'reason': 'unsupported_multiplier'}
                continue
            worst, count = Decimal(0), 0
            for row in part.itertuples(index=False):
                px = Decimal(str(row.lastpx))
                if not px.is_finite() or px <= 0 or row.lastqty <= 0:
                    raise ValueError('계약 체결 가격·수량 오류')
                positive = row.side == ('Sell' if inverse else 'Buy')
                if row.side not in ('Buy', 'Sell') or row.execcost == 0 or (row.execcost > 0) != positive:
                    raise ValueError('계약 체결 비용 부호 오류')
                unit = abs(Decimal(int(row.execcost)))/Decimal(int(row.lastqty))
                theoretical = multiplier/px if inverse else multiplier*px
                residual = abs(unit-theoretical)
                # 역사적 역선물 단위 원가의 사토시 반올림만 허용한다.
                if residual > (Decimal('0.500001') if inverse else Decimal('0.000001')):
                    raise ValueError('계약 승수·체결 원가 불일치')
                worst = max(worst, residual)
                count += 1
            result[symbol] = {'supported': True, 'inverse': inverse, 'quote': quote,
                'multiplier_satoshi': str(multiplier), 'first_time': first.time.isoformat(),
                'fills_checked': count, 'maximum_unit_residual_satoshi': str(worst)}
    return result


def checked_prices(prices):
    stamps = pd.DatetimeIndex(prices.end)
    if (stamps.tz is None or str(stamps.tz) != 'UTC' or stamps.hasnans
        or prices[['symbol', 'end']].duplicated().any()
        or (stamps.as_unit('ns').asi8 % (300*10**9)).any()
        or not prices.support.isin(['available', 'invalid_close', 'missing_response']).all()):
        raise ValueError('포트폴리오 가격 경계·중복·지원 오류')
    book = {}
    for row in prices.itertuples(index=False):
        number = Decimal(str(row.close))
        if row.support == 'available' and (not number.is_finite() or number <= 0):
            raise ValueError('포트폴리오 지원 가격 오류')
        book[row.symbol, row.end.value] = number if row.support == 'available' else None
    return book


def label(low, high):
    return 'long' if low > 0 else 'short' if high < 0 else 'flat' if low == high == 0 else 'ambiguous'


def decimal_text(value):
    return str(value.quantize(QUANTUM))


def portfolio_ledger(events, wallet, points, prices, specs, opening=0, extra_days=0):
    events, wallet = events.reset_index(drop=True), wallet.reset_index(drop=True)
    _, queries, _, _, _, _, posted, dates = validate_inputs(events, wallet, points, opening)
    low, high, uncertain = external_bounds(wallet, dates, posted, queries, extra_days)
    book = checked_prices(prices)
    iterator = iter(events[['time', 'symbol', 'exectype', 'side', 'lastqty', 'execcost', 'execcomm']].itertuples(index=False))
    event = next(iterator, None)
    positions, balance, result = {}, int(opening), []
    with localcontext() as ctx:
        ctx.prec = 60
        for i, point in enumerate(queries):
            while event is not None and event.time < point:
                balance -= int(event.execcomm)
                if event.exectype != 'Funding':
                    balance -= int(event.execcost)
                    quantity = positions.get(event.symbol, 0)+int(event.lastqty)*(1 if event.side == 'Buy' else -1)
                    if quantity:
                        positions[event.symbol] = quantity
                    else:
                        positions.pop(event.symbol, None)
                event = next(iterator, None)
            first, last = Decimal(balance+int(low[i])), Decimal(balance+int(high[i]))
            equity_a, equity_b, delta_a, delta_b = first, last, first, last
            stress_a, stress_b, sd_a, sd_b = first, last, first, last
            bucket = point.floor('5min').value
            btc = book.get(('XBTUSD', bucket))
            absent = [] if btc is not None else ['XBTUSD:price']
            for symbol, quantity in positions.items():
                spec = specs.get(symbol, {})
                if not spec.get('supported', False) or pd.Timestamp(spec['first_time']) >= point:
                    absent.append(symbol+':contract')
                    continue
                price = book.get((symbol, bucket))
                if price is None:
                    absent.append(symbol+':price')
                    continue
                value = Decimal(quantity)*Decimal(spec['multiplier_satoshi'])
                if spec['inverse']:
                    value = -value/price
                    down, up = value/Decimal('0.99'), value/Decimal('1.01')
                else:
                    value *= price
                    down, up = value*Decimal('0.99'), value*Decimal('1.01')
                equity_a += value
                equity_b += value
                stress_a += min(down, up)
                stress_b += max(down, up)
                if not spec['inverse'] and spec['quote'] in ('USD', 'USDT'):
                    delta_a += value
                    delta_b += value
                    sd_a += min(down, up)
                    sd_b += max(down, up)
            xbt = positions.get('XBTUSD', 0)
            row = {'time': point, 'xbt_quantity': xbt, 'open_contracts': len(positions),
                'other_open_contracts': sum(s != 'XBTUSD' for s in positions),
                'external_uncertain_count': int(uncertain[i]), 'cost_balance_satoshi': balance,
                'external_min_satoshi': int(low[i]), 'external_max_satoshi': int(high[i]),
                'futures_direction': 'long' if xbt > 0 else 'short' if xbt < 0 else 'flat',
                'unsupported_reasons': '|'.join(sorted(set(absent)))}
            if absent:
                row.update(dict.fromkeys(VALUE_COLUMNS))
                row.update({'account_direction': 'unsupported', 'stress_direction': 'unsupported',
                    'equity_min_usd': np.nan, 'equity_max_usd': np.nan,
                    'btc_partial_leverage_min': np.nan, 'btc_partial_leverage_max': np.nan})
            else:
                row.update({key: decimal_text(value) for key, value in zip(VALUE_COLUMNS,
                    (equity_a, equity_b, delta_a, delta_b, stress_a, stress_b, sd_a, sd_b), strict=True)})
                leverage = [d/e for d in (sd_a, sd_b) for e in (stress_a, stress_b)] if stress_a > 0 else []
                row.update({'account_direction': label(delta_a, delta_b), 'stress_direction': label(sd_a, sd_b),
                    'equity_min_usd': float(btc*equity_a/SAT), 'equity_max_usd': float(btc*equity_b/SAT),
                    'btc_partial_leverage_min': float(min(leverage)) if leverage else np.nan,
                    'btc_partial_leverage_max': float(max(leverage)) if leverage else np.nan})
            result.append(row)
    frame = pd.DataFrame(result)
    frame['time'] = frame.time.astype('datetime64[ns, UTC]')
    return frame
