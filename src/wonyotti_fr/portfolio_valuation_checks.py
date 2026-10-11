from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

KEYS = ['equity_min_satoshi', 'equity_max_satoshi', 'delta_min_satoshi', 'delta_max_satoshi',
    'stress_equity_min_satoshi', 'stress_equity_max_satoshi', 'stress_delta_min_satoshi', 'stress_delta_max_satoshi']


def verify_values(frame, independently):
    if len(frame) != len(independently):
        raise ValueError('독립 계좌 평가 행 수 불일치')
    maximum = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 70
        for actual, ref in zip(frame.to_dict('records'), independently, strict=True):
            for key in ['time', 'unsupported_reasons', 'cost_balance_satoshi', 'external_min_satoshi',
                'external_max_satoshi', 'external_uncertain_count']:
                if actual[key] != ref[key]:
                    raise ValueError('독립 계좌 평가 경계·원가·입출금 불일치: '+key)
            if ref['unsupported_reasons']:
                if any(actual[k] is not None for k in KEYS) or actual['account_direction'] != 'unsupported':
                    raise ValueError('독립 미지원 계좌 평가 누락')
                continue
            for key in KEYS:
                difference = abs(Decimal(actual[key])-ref[key])
                maximum = max(maximum, difference)
                if difference > Decimal('0.000000500001'):
                    raise ValueError('독립 계좌 평가·가격 차분 불일치: '+key)
            for prefix, field in [('', 'account_direction'), ('stress_', 'stress_direction')]:
                a, b = ref[prefix+'delta_min_satoshi'], ref[prefix+'delta_max_satoshi']
                a = Decimal(0) if abs(a) < Decimal('1e-35') else a
                b = Decimal(0) if abs(b) < Decimal('1e-35') else b
                expected = 'long' if a > 0 else 'short' if b < 0 else 'flat' if a == b == 0 else 'ambiguous'
                if actual[field] != expected:
                    raise ValueError('독립 계좌 방향 불일치')
            for key in ['equity_min_usd', 'equity_max_usd']:
                if not np.isclose(actual[key], float(ref[key]), rtol=1e-12, atol=1e-12):
                    raise ValueError('독립 달러 평가액 불일치')
            lower, upper = ref['stress_equity_min_satoshi'], ref['stress_equity_max_satoshi']
            leverage = [d/e for d in (ref['stress_delta_min_satoshi'], ref['stress_delta_max_satoshi'])
                for e in (lower, upper)] if lower > 0 else []
            for key, expected in zip(['btc_partial_leverage_min', 'btc_partial_leverage_max'],
                [float(min(leverage)), float(max(leverage))] if leverage else [np.nan, np.nan], strict=True):
                if not np.isclose(actual[key], expected, rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise ValueError('독립 BTC 부분 레버리지 경계 불일치')
    return {'rows_verified': len(frame), 'maximum_serialization_difference_satoshi': str(maximum),
        'cash_basis_equity_and_finite_difference_all_boundaries': True}


def verify_v95_subset(frame, previous):
    old = previous.set_index('time').loc[frame.time].reset_index()
    if not np.array_equal(old.time, frame.time) or not np.array_equal(old.xbt_quantity, frame.xbt_quantity):
        raise ValueError('기존 방향 원장의 전체 경계·수량 불일치')
    matched = old.other_open_contracts.eq(0)
    for key in ['account_direction', 'futures_direction', 'external_uncertain_count']:
        if not np.array_equal(old.loc[matched, key], frame.loc[matched, key]):
            raise ValueError('XBT 단독 보유의 v95 일치 실패')
    for key in ['delta_min_satoshi', 'delta_max_satoshi']:
        if [Decimal(v) for v in frame.loc[matched, key]] != old.loc[matched, key].tolist():
            raise ValueError('XBT 단독 보유의 v95 민감도 불일치')
    return int(matched.sum())


def summarize_comparisons(frame):
    known = frame.account_direction.isin(['long', 'short', 'flat'])
    stressed = frame.stress_direction.isin(['long', 'short', 'flat'])
    before = frame.previous_account_direction.isin(['long', 'short', 'flat'])
    previous_other = frame.previous_account_direction.eq('unsupported_other_contract')
    return {'rows': len(frame), 'base_counts': frame.account_direction.value_counts().to_dict(),
        'stress_counts': frame.stress_direction.value_counts().to_dict(),
        'previous_other_contract_rows': int(previous_other.sum()),
        'added_base_known_rows': int((previous_other & known).sum()),
        'added_stress_known_rows': int((previous_other & stressed).sum()),
        'base_known_rows': int(known.sum()), 'stress_known_rows': int(stressed.sum()),
        'base_different_from_futures': int((known & frame.account_direction.ne(frame.futures_direction)).sum()),
        'stress_different_from_futures': int((stressed & frame.stress_direction.ne(frame.futures_direction)).sum()),
        'changed_previous_known_rows': int((before & frame.account_direction.ne(frame.previous_account_direction)).sum()),
        'leverage_unknown_rows': int(frame.btc_partial_leverage_min.isna().sum()),
        'base_cross_table': pd.crosstab(frame.loc[known, 'futures_direction'],
            frame.loc[known, 'account_direction']).to_dict(orient='index'),
        'independent_trade_count': None}
