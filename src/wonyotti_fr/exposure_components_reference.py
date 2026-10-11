from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd


def reference_snapshots(events, wallet, points):
    transfers = [(pd.Timestamp(row.date, tz='UTC')-pd.Timedelta(days=1),
        pd.Timestamp(row.date, tz='UTC')+pd.Timedelta(days=2), int(row.amount)) for row in wallet.itertuples(index=False)
        if row.transactstatus == 'Completed' and row.transacttype in ('Deposit', 'Withdrawal') and row.amount]
    result, positions, convention = {}, {}, {}
    counts = dict.fromkeys(['trade_count', 'settlement_count', 'funding_count'], 0)
    iterator = iter(events.itertuples(index=False))
    event = next(iterator, None)
    with localcontext() as ctx:
        ctx.prec = 70
        cash, funding, fees = Decimal(0), 0, 0
        for point in points:
            while event is not None and event.time < point:
                commission = int(event.execcomm)
                cash -= commission
                counts[{'Trade': 'trade_count', 'Settlement': 'settlement_count', 'Funding': 'funding_count'}[event.exectype]] += 1
                if event.exectype == 'Funding':
                    funding += commission
                else:
                    fees += commission
                    quantity, basis = positions.get(event.symbol, (0, Decimal(0)))
                    change = int(event.lastqty)*(1 if event.side == 'Buy' else -1)
                    if event.execcost:
                        inv = (event.execcost < 0) == (event.side == 'Buy')
                        if event.symbol in convention and convention[event.symbol] != inv:
                            raise ValueError('분해 독립 원가 부호 변경')
                        convention[event.symbol] = inv
                    inverse = convention[event.symbol]
                    unit = abs(Decimal(int(event.execcost))/abs(change))
                    if not quantity or quantity*change > 0:
                        new_basis = (abs(quantity)*basis+abs(change)*unit)/abs(quantity+change)
                    else:
                        closed = min(abs(quantity), abs(change))
                        cash += (1 if quantity > 0 else -1)*closed*(1 if inverse else -1)*(basis-unit)
                        new_basis = unit if abs(change) > abs(quantity) else basis
                    positions[event.symbol] = quantity+change, new_basis if quantity+change else Decimal(0)
                event = next(iterator, None)
            minimum, maximum, uncertainty, external_count = 0, 0, 0, 0
            for earliest, latest, amount in transfers:
                if point >= latest:
                    minimum += amount
                    maximum += amount
                    external_count += 1
                elif point > earliest:
                    minimum += min(amount, 0)
                    maximum += max(amount, 0)
                    uncertainty += 1
            result[point] = {'cash': cash, 'positions': {key: (*value, convention[key]) for key, value in positions.items() if value[0]},
                'funding': funding, 'fees': fees, **counts, 'external_min': minimum, 'external_max': maximum,
                'external_uncertain': uncertainty, 'external_count': external_count}
    return result


def reference_prices(prices, activity):
    book = {(row.symbol, row.end): Decimal(str(row.close)) for row in prices.itertuples(index=False)
        if row.support == 'available' and pd.notna(row.close) and row.close > 0}
    observed = {(row.symbol, row.end) for row in activity.itertuples(index=False) if row.observed_trade}
    return book, observed


def valuation(cash, positions, at, book, observed, specs):
    missing = []
    for symbol in sorted({'XBTUSD', *positions}):
        if (symbol, at) not in book:
            missing.append(symbol+':missing_price')
        elif (symbol, at) not in observed:
            missing.append(symbol+':unobserved_trade')
        if symbol in positions and (not specs.get(symbol, {}).get('supported') or at <= pd.Timestamp(specs[symbol]['first_time'])):
            missing.append(symbol+':unsupported_contract')
    if missing:
        return None, missing
    btc = book['XBTUSD', at]

    def usd_value(ratio):
        total = cash
        for symbol, (quantity, basis, inverse) in positions.items():
            spec, price = specs[symbol], book[symbol, at]
            multiplier = Decimal(spec['multiplier_satoshi'])
            if inverse:
                unit = multiplier/(price*ratio)
                total += quantity*(basis-unit)
            else:
                unit = multiplier*price/(ratio if spec['quote'] == 'XBT' else Decimal(1))
                total += quantity*(unit-basis)
        return total*btc*ratio/Decimal(100_000_000)

    equity = usd_value(Decimal(1))*Decimal(100_000_000)/btc
    if equity <= 0:
        return None, ['nonpositive_equity']
    # 현금·평균 원가 계좌의 달러 평가액을 가격 차분해 민감도를 검산한다.
    delta = ((usd_value(Decimal('1.001'))-usd_value(Decimal('0.999')))
        /(btc*Decimal('.002'))*Decimal(100_000_000))
    return (equity, delta, delta/equity), []


def verify_decomposition(source, saved, snapshots, prices, observed, specs):
    pd.testing.assert_frame_equal(saved[source.columns], source.reset_index(drop=True), check_exact=True)
    stages = ['initial', 'marks', 'external', 'funding', 'fees', 'executions']
    components = ['mark_change', 'external_change', 'funding_change', 'fee_change', 'execution_change']
    error, supported = Decimal(0), 0
    with localcontext() as ctx:
        ctx.prec = 70
        for original, row in zip(source.itertuples(index=False), saved.itertuples(index=False), strict=True):
            before, after = snapshots[original.end], snapshots[original.label_end]
            counts = {key: after[key]-before[key] for key in ['trade_count', 'settlement_count', 'funding_count', 'external_count']}
            for key, expected in counts.items():
                if getattr(row, key) != expected:
                    raise ValueError('분해 독립 사건 수 불일치')
            if row.external_amount_satoshi != after['external_min']-before['external_min']:
                raise ValueError('분해 독립 외부 자금 불일치')
            reasons = []
            if not original.change_supported:
                reasons.append('source:'+original.change_reason)
            if before['external_uncertain'] or after['external_uncertain']:
                reasons.append('uncertain_external_transfer')
            calculated = []
            if not reasons:
                q0, q1 = before['positions'], after['positions']
                c0, e0, e1 = before['cash'], before['external_min'], after['external_min']
                fund, fee = after['funding']-before['funding'], after['fees']-before['fees']
                states = [(c0+e0, q0, original.end), (c0+e0, q0, original.label_end),
                    (c0+e1, q0, original.label_end), (c0+e1-fund, q0, original.label_end),
                    (c0+e1-fund-fee, q0, original.label_end), (after['cash']+e1, q1, original.label_end)]
                for name, state in zip(stages, states, strict=True):
                    result, problems = valuation(*state, prices, observed, specs)
                    reasons.extend(name+':'+problem for problem in problems)
                    calculated.append(result)
            if reasons:
                if row.decomposition_supported or row.decomposition_reason != '|'.join(reasons):
                    raise ValueError('분해 독립 미지원 사유 불일치')
                if any(not pd.isna(getattr(row, key)) for key in [*components, 'telescoping_residual',
                    *[stage+'_'+suffix for stage in stages for suffix in ['equity_satoshi', 'delta_satoshi', 'beta']]]):
                    raise ValueError('분해 미지원 행의 숫자 출력 오류')
                continue
            if not row.decomposition_supported or row.decomposition_reason != 'included':
                raise ValueError('분해 독립 지원 표시 오류')
            supported += 1
            for stage, expected in zip(stages, calculated, strict=True):
                for suffix, wanted in zip(['equity_satoshi', 'delta_satoshi'], expected[:2], strict=True):
                    difference = abs(Decimal(getattr(row, stage+'_'+suffix))-wanted)
                    error = max(error, difference)
                    if difference > Decimal('1e-28'):
                        raise ValueError('분해 독립 계좌·민감도 불일치')
                if abs(getattr(row, stage+'_beta')-float(expected[2])) > 1e-12:
                    raise ValueError('분해 독립 단계 노출 불일치')
            pieces = [b[2]-a[2] for a, b in zip(calculated[:-1], calculated[1:], strict=True)]
            for name, piece in zip(components, pieces, strict=True):
                if abs(getattr(row, name)-float(piece)) > 1e-12:
                    raise ValueError('분해 독립 기여분 불일치')
            if abs(sum(getattr(row, key) for key in components)-original.exposure_change) > 1e-10:
                raise ValueError('분해 독립 전체 변경량 불일치')
            if (abs(float(calculated[0][2])-original.current_net_exposure) > 1e-10
                or abs(float(calculated[-1][2])-original.net_exposure) > 1e-10):
                raise ValueError('분해 독립 기존 정답 불일치')
            if counts['trade_count']+counts['settlement_count'] == 0 and abs(row.execution_change) > 1e-12:
                raise ValueError('분해 독립 무체결 기여분 오류')
    return {'all_rows_compared': len(saved), 'supported_rows_compared': supported,
        'maximum_satoshi_difference': str(error), 'cash_basis_and_finite_price_sensitivity_verified': True,
        'every_stage_component_reason_and_event_count_verified': True}


def independent_statistics(frame):
    selected = frame.loc[frame.decomposition_supported]
    keys = ['exposure_change', 'mark_change', 'external_change', 'funding_change', 'fee_change', 'execution_change']
    matrix = selected[keys].to_numpy()
    stats = {}
    for key in keys:
        series = selected[key]
        stats[key] = None if not len(series) else {'mean': float(series.sum()/len(series)),
            'mae': float(series.abs().sum()/len(series)), 'rms': float((series.pow(2).sum()/len(series))**.5),
            'quantiles': {str(q): float(series.quantile(q)) for q in [0, .01, .25, .5, .75, .99, 1]}}
    covariance = None
    if len(matrix):
        centered = matrix-np.mean(matrix, axis=0)
        covariance = (centered.T@centered/len(matrix)).tolist()
    return stats, covariance
