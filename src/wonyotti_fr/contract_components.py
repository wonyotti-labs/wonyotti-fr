from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext

import numpy as np
import pandas as pd

from wonyotti_fr.collateral_direction import validate_inputs

GROUPS = ('xbtusd', 'dated_btc', 'other_contract')
DETAIL_COLUMNS = ['end', 'label_end', 'symbol', 'group', 'net_quantity', 'cost_satoshi',
    'trade_count', 'settlement_count', 'supported', 'reason', 'equity_change_satoshi',
    'delta_change_satoshi', 'contribution', 'contribution_float']
RESIDUALS = ('contract_equity_residual_satoshi', 'contract_delta_residual_satoshi',
    'contract_change_residual')


def contract_group(symbol):
    if symbol == 'XBTUSD':
        return 'xbtusd'
    return 'dated_btc' if re.fullmatch(r'XBT[HMUZ]\d{2}', symbol) else 'other_contract'


def decimal_value(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, float, Decimal, np.integer, np.floating)):
        raise ValueError('계약 분해 숫자 형식 오류')
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError('계약 분해 숫자 형식 오류') from error
    if not result.is_finite():
        raise ValueError('계약 분해 비유한 숫자')
    return result


def decimal_check(actual, expected, terms=()):
    scale = max(Decimal(1), abs(actual), abs(expected), sum(map(abs, terms), Decimal(0)))
    residual = actual-expected
    if abs(residual) > Decimal('1e-40')*scale:
        raise ValueError('계약 분해 정밀 회계 불일치')
    return residual


def grouped_events(events, wallet, source):
    if (not len(source) or source.columns.duplicated().any() or source.end.duplicated().any()
        or not source.end.is_monotonic_increasing
        or not source.label_end.sub(source.end).eq(pd.Timedelta(minutes=5)).all()):
        raise ValueError('계약 분해 원본 행·시간 경계 오류')
    points = pd.DatetimeIndex(sorted(set(source.end) | set(source.label_end))).as_unit('ns')
    if (points.asi8 % (300*10**9)).any():
        raise ValueError('계약 분해 5분 경계 오류')
    times, _, trade, qty, cost, _, _, _ = validate_inputs(events, wallet, points, 0)
    if any(not isinstance(s, str) for s in events.symbol):
        raise ValueError('계약 분해 계약명 형식 오류')
    frame = pd.DataFrame({'end': times.floor('5min'), 'symbol': events.symbol.to_numpy(),
        'net_quantity': np.where(events.side.eq('Buy'), qty, -qty), 'cost_satoshi': cost,
        'trade_count': events.exectype.eq('Trade').astype(np.int64).to_numpy(),
        'settlement_count': events.exectype.eq('Settlement').astype(np.int64).to_numpy()})
    frame = frame.loc[trade & frame.end.isin(source.end)]
    grouped = frame.groupby(['end', 'symbol'], sort=True)[
        ['net_quantity', 'cost_satoshi', 'trade_count', 'settlement_count']].sum().reset_index()
    result = {}
    for row in grouped.itertuples(index=False):
        result.setdefault(row.end, []).append(row)
    return result


def marked_change(quantity, symbol, point, book, specs):
    if not quantity:
        return Decimal(0), Decimal(0)
    spec = specs.get(symbol, {})
    price, observed = book.get((symbol, point), (None, False))
    first = pd.Timestamp(spec.get('first_time'))
    if (spec.get('supported') is not True or type(spec.get('inverse')) is not bool
        or spec.get('quote') not in ('USD', 'USDT', 'XBT')
        or pd.isna(first) or first.tzinfo is None or str(first.tzinfo) != 'UTC' or first >= point):
        raise ValueError('계약 분해 승수·지원 오류')
    if price is None or observed is not True:
        raise ValueError('계약 분해 가격·거래 활동 미지원')
    price, multiplier = decimal_value(price), decimal_value(spec['multiplier_satoshi'])
    if price <= 0 or multiplier <= 0:
        raise ValueError('계약 분해 가격·승수 범위 오류')
    mark = Decimal(quantity)*multiplier
    mark = -mark/price if spec['inverse'] else mark*price
    return mark, mark if not spec['inverse'] and spec['quote'] in ('USD', 'USDT') else Decimal(0)


def decompose_contracts(source, events, wallet, book, specs):
    groups = grouped_events(events, wallet, source)
    if (source.decomposition_supported.isna().any()
        or any(type(v) not in (bool, np.bool_) for v in source.decomposition_supported)):
        raise ValueError('계약 분해 원본 지원 표시 오류')
    output_columns = ['contract_supported', 'contract_reason', 'contract_count', 'contract_float_error',
        *RESIDUALS, *[g+'_execution_change' for g in GROUPS], *[g+'_event_count' for g in GROUPS]]
    if any(name in source for name in output_columns):
        raise ValueError('계약 분해 기존 열 덮어쓰기 오류')
    rows, details = [], []
    with localcontext() as context:
        context.prec = 60
        for row in source.itertuples(index=False):
            records = groups.get(row.end, [])
            if (sum(r.trade_count for r in records) != row.trade_count
                or sum(r.settlement_count for r in records) != row.settlement_count):
                raise ValueError('계약 분해 원본 사건 수 불일치')
            item = dict.fromkeys(output_columns)
            item.update(contract_supported=bool(row.decomposition_supported),
                contract_reason=row.decomposition_reason, contract_count=len(records), contract_float_error=np.nan)
            item.update({g+'_event_count': 0 for g in GROUPS})
            item.update({g+'_execution_change': np.nan for g in GROUPS})
            values, sums = [], {g: Decimal(0) for g in GROUPS}
            if row.decomposition_supported:
                e0, d0, e1, d1 = [decimal_value(getattr(row, stage+'_'+field+'_satoshi'))
                    for stage in ['fees', 'executions'] for field in ['equity', 'delta']]
                if e0 <= 0 or e1 <= 0:
                    raise ValueError('계약 분해 비양수 평가액')
                b0 = d0/e0
            for event in records:
                group = contract_group(event.symbol)
                item[group+'_event_count'] += event.trade_count+event.settlement_count
                detail = {'end': row.end, 'label_end': row.label_end, 'symbol': event.symbol, 'group': group,
                    **{k: int(getattr(event, k)) for k in ['net_quantity', 'cost_satoshi', 'trade_count', 'settlement_count']},
                    'supported': bool(row.decomposition_supported), 'reason': row.decomposition_reason,
                    'equity_change_satoshi': None, 'delta_change_satoshi': None,
                    'contribution': None, 'contribution_float': np.nan}
                if row.decomposition_supported:
                    mark, delta_mark = marked_change(int(event.net_quantity), event.symbol, row.label_end, book, specs)
                    de, dd = mark-int(event.cost_satoshi), delta_mark-int(event.cost_satoshi)
                    contribution = (dd-b0*de)/e1
                    detail.update(equity_change_satoshi=str(de), delta_change_satoshi=str(dd),
                        contribution=str(contribution), contribution_float=float(contribution))
                    values.append((de, dd, contribution))
                    sums[group] += contribution
                details.append(detail)
            if row.decomposition_supported:
                columns = [[v[i] for v in values] for i in range(3)]
                totals = [sum(v, Decimal(0)) for v in columns]
                expected = [e1-e0, d1-d0, d1/e1-b0]
                for name, total, wanted, terms in zip(RESIDUALS, totals, expected, columns, strict=True):
                    item[name] = str(decimal_check(total, wanted, terms))
                    decimal_check(total, sum(reversed(terms), Decimal(0)), terms)
                item['contract_float_error'] = abs(float(totals[2])-float(decimal_value(row.execution_change)))
                if item['contract_float_error'] > 1e-10:
                    raise ValueError('계약 분해 기존 변경 정답 불일치')
                item.update({g+'_execution_change': float(sums[g]) for g in GROUPS})
            rows.append(item)
    out = source.reset_index(drop=True).copy()
    for name in output_columns:
        out[name] = [row[name] for row in rows]
    detail_frame = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    for name in ['end', 'label_end']:
        detail_frame[name] = pd.to_datetime(detail_frame[name], utc=True).astype('datetime64[ns, UTC]')
    return out, detail_frame


def summarize_contracts(frame):
    selected = frame.loc[frame.contract_supported]
    columns = ['execution_change', *[g+'_execution_change' for g in GROUPS]]
    statistics = {}
    for key in columns:
        values = selected[key].to_numpy(dtype=float)
        sign = np.where(values < -1e-12, -1, np.where(values > 1e-12, 1, 0))
        statistics[key] = {'negative': int((sign == -1).sum()), 'zero': int((sign == 0).sum()),
            'positive': int((sign == 1).sum()), 'abs_mean': float(np.abs(values).mean()) if len(values) else None,
            'abs_quantiles': dict(zip(['0', '.25', '.5', '.75', '.9', '.99', '1'],
                map(float, np.quantile(np.abs(values), [0, .25, .5, .75, .9, .99, 1])), strict=True)) if len(values) else None}
    cross = np.zeros((3, 3), dtype=np.int64)
    a, b = selected.xbtusd_execution_change.to_numpy(), selected.execution_change.to_numpy()
    sign_a = np.where(a < -1e-12, 0, np.where(a > 1e-12, 2, 1))
    sign_b = np.where(b < -1e-12, 0, np.where(b > 1e-12, 2, 1))
    np.add.at(cross, (sign_a, sign_b), 1)
    mixed = sum(selected[g+'_event_count'].gt(0).astype(int) for g in GROUPS).gt(1)
    without = selected.xbtusd_event_count.eq(0)
    cohorts = {}
    for name, mask in [('without_xbtusd_events', without), ('mixed_contract_groups', mixed)]:
        part = selected.loc[mask]
        cohorts[name] = {'rows': len(part), 'nonzero_total': int(part.execution_change.abs().gt(1e-12).sum()),
            'absolute_contribution_sum': {g: float(part[g+'_execution_change'].abs().sum()) for g in GROUPS}}
    return {'all_rows': len(frame), 'supported_rows': len(selected), 'unsupported_rows': len(frame)-len(selected),
        'reasons': frame.contract_reason.value_counts().to_dict(), 'statistics': statistics,
        'xbtusd_rows_total_columns_sign_cross': cross.tolist(), 'cohorts': cohorts,
        'all_event_counts': {g: int(frame[g+'_event_count'].sum()) for g in GROUPS},
        'supported_event_counts': {g: int(selected[g+'_event_count'].sum()) for g in GROUPS},
        'maximum_float_error': float(selected.contract_float_error.max()) if len(selected) else None}
