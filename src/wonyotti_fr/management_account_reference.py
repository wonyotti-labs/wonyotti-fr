from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from .position_boost_verification import compare_statistics


def verify_account_state(original, ledger, activity, prices, actual):
    pd.testing.assert_frame_equal(original, actual[original.columns], check_exact=True)
    accounts = {r.time: r for r in ledger.itertuples(index=False)}
    flags = {r.end: r.all_prices_have_observed_trade for r in activity.itertuples(index=False)}
    quotes = {r.end: r for r in prices.itertuples(index=False) if r.symbol == 'XBTUSD'}
    names = ['zero', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,5]', '(5,inf)']
    with localcontext() as ctx:
        ctx.prec = 60
        for row in actual.itertuples(index=False):
            a, q = accounts.get(row.end), quotes.get(row.end)
            if a is not None:
                assert a.xbt_quantity == row.state_quantity
                assert row.account_time == row.end and row.account_quantity == a.xbt_quantity
                np.testing.assert_allclose(row.account_equity_usd, a.equity_min_usd, rtol=0, atol=0, equal_nan=True)
            else:
                assert pd.isna(row.account_time) and pd.isna(row.account_quantity) and pd.isna(row.account_equity_usd)
            np.testing.assert_allclose(row.account_price, q.close if q else np.nan, rtol=0, atol=0, equal_nan=True)
            if a is None:
                reason = 'missing_boundary'
            elif row.end not in flags:
                reason = 'missing_activity'
            elif q is None or q.support != 'available':
                reason = 'missing_price'
            elif a.unsupported_reasons:
                reason = 'unsupported_valuation'
            elif not flags[row.end]:
                reason = 'unobserved_price'
            elif a.external_uncertain_count != 0:
                reason = 'uncertain_external'
            elif a.equity_min_usd <= 0:
                reason = 'nonpositive_equity'
            else:
                reason = 'included'
            assert row.account_reason == reason
            assert row.account_supported == (reason == 'included')
            assert row.account_usable == (reason == 'included' and row.paired_usable)
            if reason != 'included':
                assert pd.isna(row.futures_leverage) and pd.isna(row.log_futures_leverage)
                assert row.account_band == 'unsupported'
                continue
            equity = Decimal(a.equity_min_satoshi)*Decimal(str(q.close))/Decimal(100_000_000)
            assert a.equity_min_satoshi == a.equity_max_satoshi and equity > 0
            np.testing.assert_allclose(float(equity), row.account_equity_usd, rtol=1e-12, atol=1e-8)
            ratio = float(Decimal(abs(int(a.xbt_quantity)))/equity)
            np.testing.assert_allclose(ratio, row.futures_leverage, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.log1p(ratio), row.log_futures_leverage, rtol=1e-12, atol=1e-12)
            # 환산 반올림과 경계 분류는 저장된 자산 기준의 정확한 비율로 별도 확인한다.
            size = abs(a.xbt_quantity)/a.equity_min_usd
            band = next((names[i] for i, limit in enumerate([0, .5, 1, 2, 5]) if size <= limit), names[-1])
            assert row.account_band == band
    return {'all_account_rows_verified': len(actual), 'original_columns_exact': True,
        'strict_same_boundary_quantity_verified': True, 'decimal_equity_conversion_verified': True}


def verify_account_summary(frame, actual):
    records = frame[['end', 'state_episode_id', 'paired_part', 'paired_usable', 'account_supported',
        'account_usable', 'account_reason', 'account_band', 'futures_leverage',
        'all_y_exit', 'all_y_reduce', 'all_y_increase']].to_dict('records')
    def count(rows):
        return {'rows': len(rows), 'episodes': len({r['state_episode_id'] for r in rows if r['state_episode_id'] is not None and r['state_episode_id'] > 0}),
            'positive': {a: sum(bool(r['all_y_'+a]) for r in rows) for a in ['exit', 'reduce', 'increase']}}
    groups = []
    for kind in ['year', 'split']:
        key = (lambda r: str(r['end'].year)) if kind == 'year' else (lambda r: r['paired_part'])
        for group in sorted({key(r) for r in records}):
            source = [r for r in records if key(r) == group]
            used = [r for r in source if r['account_usable']]
            values = sorted(r['futures_leverage'] for r in used)
            quantiles = {}
            for label, fraction in zip(['0', '.01', '.25', '.5', '.75', '.99', '1'], [0, .01, .25, .5, .75, .99, 1], strict=True):
                if values:
                    index = (len(values)-1)*fraction
                    lo, hi = int(np.floor(index)), int(np.ceil(index))
                    quantiles[label] = values[lo]+(values[hi]-values[lo])*(index-lo)
            groups.append({'kind': kind, 'group': group, 'all': count(source),
                'original_usable': count([r for r in source if r['paired_usable']]),
                'account_supported': count([r for r in source if r['account_supported']]), 'joint_usable': count(used),
                'reasons': dict(Counter(r['account_reason'] for r in source)), 'quantiles': quantiles or None,
                'bands': {band: count([r for r in used if r['account_band'] == band])
                    for band in ['zero', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,5]', '(5,inf)']}})
    compare_statistics(actual, {'all_rows': len(frame), 'groups': groups, 'model_fitted': False,
        'trading_returns_evaluated': False, 'profitability_accepted': False})
