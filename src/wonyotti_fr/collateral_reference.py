from __future__ import annotations

from decimal import Decimal, localcontext

import pandas as pd


def independent_ledger(events, wallet, points, opening=0, extra_days=0):
    transfers = []
    for r in wallet.to_dict(orient='records'):
        if r['transacttype'] in ('Deposit', 'Withdrawal') and r['transactstatus'] == 'Completed':
            start = pd.Timestamp(r['date'], tz='UTC')-pd.Timedelta(days=extra_days)
            transfers.append((start, start+pd.Timedelta(days=1+2*extra_days), int(r['amount'])))
    convention = {}
    for r in events[['symbol', 'exectype', 'execcost', 'side']].itertuples(index=False):
        if r.exectype == 'Trade' and r.execcost:
            value = (int(r.execcost) < 0) == (r.side == 'Buy')
            if r.symbol in convention and convention[r.symbol] != value:
                raise ValueError('독립 원장 비용 방향 불일치')
            convention[r.symbol] = value
    if convention.get('XBTUSD') is not True:
        raise ValueError('XBTUSD 역선물 원장 지원 오류')
    result, state = [], {}
    iterator = iter(events.itertuples(index=False))
    row = next(iterator, None)
    with localcontext() as ctx:
        ctx.prec = 60
        cash = Decimal(opening)
        for point in pd.DatetimeIndex(points):
            while row is not None and row.time < point:
                cash -= Decimal(int(row.execcomm))
                if row.exectype == 'Funding':
                    row = next(iterator, None)
                    continue
                symbol = row.symbol
                old, basis = state.get(symbol, (0, Decimal(0)))
                change = int(row.lastqty)*(1 if row.side == 'Buy' else -1)
                unit = abs(Decimal(int(row.execcost))/Decimal(abs(change)))
                if old == 0 or (old > 0) == (change > 0):
                    new_basis = (basis*abs(old)+unit*abs(change))/abs(old+change)
                else:
                    closing = min(abs(old), abs(change))
                    multiplier = 1 if convention[symbol] else -1
                    cash += Decimal((1 if old > 0 else -1)*closing*multiplier)*(basis-unit)
                    new_basis = unit if abs(change) > abs(old) else basis
                state[symbol] = (old+change, new_basis if old+change else Decimal(0))
                row = next(iterator, None)
            quantity, xbt_basis = state.get('XBTUSD', (0, Decimal(0)))
            others = sum(q != 0 for s, (q, _) in state.items() if s != 'XBTUSD')
            a, b, uncertain = 0, 0, 0
            for start, end, value in transfers:
                if point >= end:
                    a += value
                    b += value
                elif point > start and value:
                    a += min(value, 0)
                    b += max(value, 0)
                    uncertain += 1
            coefficient = cash+Decimal(quantity)*xbt_basis
            low, high = coefficient+a, coefficient+b
            if others:
                label, low_value, high_value = 'unsupported_other_contract', None, None
            else:
                # 원가 나눗셈의 유한 정밀도 잔차만 허용하고 사토시 차이는 거부한다.
                if abs(low-low.to_integral_value()) > Decimal('1e-30') or abs(high-high.to_integral_value()) > Decimal('1e-30'):
                    raise ValueError('독립 현금·잔존 원가의 사토시 불일치')
                low_value, high_value = int(low.to_integral_value()), int(high.to_integral_value())
                label = 'long' if low_value > 0 else 'short' if high_value < 0 else 'flat' if low_value == high_value == 0 else 'ambiguous'
            result.append({'time': point, 'xbt_quantity': quantity, 'other_open_contracts': others,
                'delta_min_satoshi': low_value, 'delta_max_satoshi': high_value,
                'external_uncertain_count': uncertain, 'account_direction': label,
                'futures_direction': 'long' if quantity > 0 else 'short' if quantity < 0 else 'flat'})
    frame = pd.DataFrame(result)
    frame['time'] = frame.time.astype('datetime64[ns, UTC]')
    for col in ['delta_min_satoshi', 'delta_max_satoshi']:
        frame[col] = pd.array(frame[col], dtype='Int64')
    return frame
