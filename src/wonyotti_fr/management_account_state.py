from __future__ import annotations

import numpy as np
import pandas as pd

from .entry_boundary_support import utc_times
from .minute_management import ACTIONS

ACCOUNT_COLUMNS = ['account_time', 'account_quantity', 'account_equity_usd', 'account_price',
    'account_reason', 'account_supported', 'account_usable', 'futures_leverage', 'log_futures_leverage', 'account_band']
BANDS = ['zero', '(0,0.5]', '(0.5,1]', '(1,2]', '(2,5]', '(5,inf)']


def account_state_frame(original, ledger, activity, prices):
    if (original.empty or any(k in original for k in ACCOUNT_COLUMNS)
        or not {'end', 'state_quantity', 'paired_usable', 'paired_part', 'state_episode_id', *['all_y_'+a for a in ACTIONS]} <= set(original)):
        raise ValueError('계좌 관리 원본 열·빈 입력 오류')
    starts = utc_times(original.end, unique=True)
    utc_times(ledger.time, unique=True)
    utc_times(activity.end, unique=True)
    btc = prices.loc[prices.symbol.eq('XBTUSD')].copy()
    utc_times(btc.end, unique=True)
    if (starts % (300*10**9)).any():
        raise ValueError('계좌 관리 창 경계 오류')
    if (not all(type(v) in (bool, np.bool_) for v in activity.all_prices_have_observed_trade)
        or ledger.unsupported_reasons.isna().any()):
        raise ValueError('계좌 관리 지원 표시 오류')
    for key in ['xbt_quantity', 'external_uncertain_count', 'external_min_satoshi', 'external_max_satoshi']:
        v = ledger[key].to_numpy(dtype=float)
        if not np.isfinite(v).all() or (v % 1).any():
            raise ValueError('계좌 관리 정수 원장 오류')
    if (ledger.external_uncertain_count.lt(0).any() or ledger.external_min_satoshi.gt(ledger.external_max_satoshi).any()
        or (ledger.external_uncertain_count.eq(0) & ledger.external_min_satoshi.ne(ledger.external_max_satoshi)).any()):
        raise ValueError('계좌 관리 외부 현금 흐름 범위 오류')
    for key in ['equity_min_usd', 'equity_max_usd']:
        if not np.isfinite(ledger.loc[ledger.unsupported_reasons.eq(''), key].to_numpy(dtype=float)).all():
            raise ValueError('계좌 관리 평가 자산 오류')
    if (ledger.equity_min_usd.gt(ledger.equity_max_usd).any()
        or not btc.support.isin(['available', 'invalid_close', 'missing_response']).all()):
        raise ValueError('계좌 관리 평가 범위·가격 지원 오류')
    close = pd.to_numeric(btc.close, errors='raise')
    if ((~np.isfinite(close) | close.le(0)) & btc.support.eq('available')).any():
        raise ValueError('계좌 관리 지원 가격 오류')
    out = original.reset_index(drop=True).copy()
    indexed = ledger.set_index('time').reindex(out.end).reset_index()
    matched = out.end.isin(ledger.time)
    if not np.array_equal(indexed.loc[matched, 'xbt_quantity'], out.loc[matched, 'state_quantity']):
        raise ValueError('계좌 관리 동일 시각 수량 불일치')
    observed = activity.set_index('end').all_prices_have_observed_trade.reindex(out.end).reset_index(drop=True)
    quoted = btc.set_index('end').reindex(out.end).reset_index()
    same_external = indexed.external_uncertain_count.eq(0)
    if not np.array_equal(indexed.loc[matched & same_external, 'equity_min_usd'],
        indexed.loc[matched & same_external, 'equity_max_usd'], equal_nan=True):
        raise ValueError('계좌 관리 확정 현금 흐름의 자산 범위 불일치')
    reasons = np.select([~matched, observed.isna(), quoted.support.ne('available'),
        indexed.unsupported_reasons.ne(''), ~observed.eq(True), ~same_external,
        indexed.equity_min_usd.le(0)],
        ['missing_boundary', 'missing_activity', 'missing_price', 'unsupported_valuation',
            'unobserved_price', 'uncertain_external', 'nonpositive_equity'], default='included')
    out['account_time'] = out.end.where(matched)
    out['account_quantity'] = indexed.xbt_quantity
    out['account_equity_usd'] = indexed.equity_min_usd
    out['account_price'] = quoted.close
    out['account_reason'] = reasons
    out['account_supported'] = out.account_reason.eq('included')
    out['account_usable'] = out.paired_usable & out.account_supported
    out['futures_leverage'] = (out.account_quantity.abs()/out.account_equity_usd).where(out.account_supported)
    if not np.isfinite(out.loc[out.account_supported, 'futures_leverage']).all():
        raise ValueError('계좌 관리 포지션 크기 범위 오류')
    out['log_futures_leverage'] = np.log1p(out.futures_leverage)
    out['account_band'] = pd.cut(out.futures_leverage, [-np.inf, 0, .5, 1, 2, 5, np.inf], labels=BANDS).astype('object').fillna('unsupported')
    return out


def summarize_account_state(frame):
    def count(part):
        return {'rows': len(part), 'episodes': int(part.state_episode_id.loc[part.state_episode_id.gt(0)].nunique()),
            'positive': {a: int(part['all_y_'+a].sum()) for a in ACTIONS}}
    results = []
    for name, key in [('year', frame.end.dt.year.astype(str)), ('split', frame.paired_part)]:
        for group in sorted(key.unique()):
            source = frame.loc[key.eq(group)]
            usable = source.loc[source.account_usable]
            values = usable.futures_leverage.to_numpy()
            results.append({'kind': name, 'group': group, 'all': count(source),
                'original_usable': count(source.loc[source.paired_usable]),
                'account_supported': count(source.loc[source.account_supported]), 'joint_usable': count(usable),
                'reasons': source.account_reason.value_counts().sort_index().to_dict(),
                'quantiles': dict(zip(['0', '.01', '.25', '.5', '.75', '.99', '1'],
                    map(float, np.quantile(values, [0, .01, .25, .5, .75, .99, 1])), strict=True)) if len(values) else None,
                'bands': {band: count(usable.loc[usable.account_band.eq(band)]) for band in BANDS}})
    return {'all_rows': len(frame), 'groups': results, 'model_fitted': False,
        'trading_returns_evaluated': False, 'profitability_accepted': False}
