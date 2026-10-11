from __future__ import annotations

import numpy as np
import pandas as pd

from .entry_boundary_support import utc_times
from .state_action_windows import requested_orders, validate_ledger

MODELS = ('first', 'all', 'linear', 'constant')


def order_observations(raw, ledger, requested):
    required = {'lastqty', 'orderqty', 'cumqty', 'ordstatus', 'lastliquidityind'}
    if not required <= set(raw):
        raise ValueError('주문 체결 관측 열 누락')
    fresh = requested_orders(raw, ledger)
    pd.testing.assert_frame_equal(fresh, requested[fresh.columns].reset_index(drop=True), check_exact=True)
    trade = raw.loc[raw.symbol.eq('XBTUSD') & raw.exectype.eq('Trade')
        & ~raw.order_key.str.startswith('special:')].copy()
    for name in ('lastqty', 'orderqty', 'cumqty'):
        if (trade[name].dtype.kind not in 'iu' or trade[name].isna().any()
            or trade[name].lt(0 if name == 'cumqty' else 1).any()
            or sum(int(v) for v in trade[name]) > np.iinfo(np.int64).max):
            raise ValueError('주문 체결 관측 수량 오류')
    groups = trade.groupby('order_key', sort=False)
    if groups.side.nunique().ne(1).any():
        raise ValueError('같은 주문의 체결 방향 변경')
    trade['maker'] = trade.lastliquidityind.eq('AddedLiquidity')
    trade['taker'] = trade.lastliquidityind.eq('RemovedLiquidity')
    trade['cum_mismatch'] = trade.cumqty.ne(groups.lastqty.cumsum())
    observed = trade.groupby('order_key', sort=False).agg(
        first_observed_time=('time', 'first'), last_observed_time=('time', 'last'),
        observed_fills=('lastqty', 'size'), observed_quantity=('lastqty', 'sum'),
        first_order_quantity=('orderqty', 'first'), last_order_quantity=('orderqty', 'last'),
        quantity_versions=('orderqty', 'nunique'), last_cumulative_quantity=('cumqty', 'last'),
        last_observed_status=('ordstatus', lambda v: v.iloc[-1]),
        maker_fills=('maker', 'sum'), taker_fills=('taker', 'sum'),
        cumulative_mismatch_fills=('cum_mismatch', 'sum')).reset_index()
    observed['other_liquidity_fills'] = observed.observed_fills-observed.maker_fills-observed.taker_fills
    observed['fill_span_ns'] = (observed.last_observed_time-observed.first_observed_time).astype("timedelta64[ns]").array.asi8
    observed['observed_to_first_requested'] = observed.observed_quantity/observed.first_order_quantity
    actual = ledger.loc[ledger.action.ne('funding') & ~ledger.order_key.str.startswith('special:')].copy()
    actual['quantity'] = (actual.after_qty-actual.before_qty).abs()
    total = actual.groupby('order_key').quantity.sum().reindex(observed.order_key)
    if total.isna().any() or not np.array_equal(total, observed.observed_quantity):
        raise ValueError('주문별 원본 체결량과 상태 원장의 불일치')
    return requested.reset_index(drop=True).merge(observed, on='order_key', validate='one_to_one')


def link_terminations(ledger, observations):
    validate_ledger(ledger)
    if observations.order_key.duplicated().any():
        raise ValueError('종료 연결의 독립 주문 중복')
    ended = ledger.loc[ledger.action.isin(['close', 'reverse'])].copy().reset_index(drop=True)
    if ended.before_episode_id.eq(0).any() or ended.before_episode_id.duplicated().any():
        raise ValueError('에피소드 종료 사건 중복·누락')
    lookup = ended.set_index('before_episode_id')
    requests = observations.copy()
    requests['initial_episode_end_time'] = requests.before_episode_id.map(lookup.time)
    requests['initial_episode_end_order'] = requests.before_episode_id.map(lookup.order_key)
    requests['initial_episode_end_index'] = pd.array(requests.before_episode_id.map(lookup.event_index), dtype='Int64')
    first = ledger.loc[ledger.action.ne('funding')].drop_duplicates('order_key').set_index('order_key')
    requests['first_event_index'] = requests.order_key.map(first.event_index).astype('int64')
    if (requests.initial_episode_end_index.lt(requests.first_event_index).fillna(False).any()
        or requests.initial_episode_end_time.lt(requests.time).fillna(False).any()):
        raise ValueError('요청 이전의 에피소드 종료 연결')
    requests['end_owner'] = np.select([requests.before_episode_id.eq(0), requests.initial_episode_end_time.isna(),
        requests.initial_episode_end_order.eq(requests.order_key)], ['not_held', 'not_observed', 'same_order'], default='other_order')
    requests['termination_delay_ns'] = pd.array([
        None if pd.isna(end) else end.value-start.value
        for start, end in zip(requests.time, requests.initial_episode_end_time, strict=True)], dtype='Int64')
    # 오른쪽 경계의 종료는 다음 창에 속한다.
    requests['termination_in_first_window'] = requests.initial_episode_end_time.lt(requests.time.dt.floor('5min')+pd.Timedelta(minutes=5))
    known = requests.set_index('order_key')
    for target, source in [('first_request_time', 'time'), ('first_requested_action', 'requested_action'),
        ('first_request_episode', 'before_episode_id')]:
        ended[target] = ended.order_key.map(known[source])
    if ended.loc[~ended.order_key.str.startswith('special:'), 'first_request_time'].isna().any():
        raise ValueError('종료 주문의 최초 요청 누락')
    same = ended.first_request_episode.eq(ended.before_episode_id)
    ended['origin'] = np.select([ended.order_key.str.startswith('special:'), ~same,
        ended.first_requested_action.isin(['close', 'reverse'])],
        ['special_order', 'different_episode', 'exit_request_same_episode'], default='other_action_same_episode')
    ended['request_in_same_window'] = ended.first_request_time.dt.floor('5min').eq(ended.time.dt.floor('5min'))
    ended['request_to_termination_ns'] = pd.array([
        None if pd.isna(start) else end.value-start.value
        for start, end in zip(ended.first_request_time, ended.time, strict=True)], dtype='Int64')
    return requests, ended


def window_comparison(windows, requested, filled, paired, predictions):
    utc_times(windows.end, unique=True)
    utc_times(paired.end, unique=True)
    utc_times(predictions.end, unique=True)
    columns = ['end', 'state_episode_id', 'y_requested_exit', 'y_filled_exit', 'usable', 'state_quantity', 'period']
    if not windows[columns].equals(paired[columns]):
        raise ValueError('청산 비교의 원래 창·정답 변경')
    result = windows[columns].copy()
    for name, rows in [('requested', requested), ('filled', filled)]:
        positive = rows.loc[rows.exit & rows.window_index.ge(0), 'window_index']
        if not np.array_equal(result.index.isin(positive), result[f'y_{name}_exit']):
            raise ValueError('청산 사건과 창 정답의 불일치')
    result['paired_part'] = paired.paired_part
    result['paired_usable'] = paired.paired_usable
    supported = paired.loc[paired.paired_usable & paired.paired_part.isin(['calibration', 'diagnosis'])]
    if not np.array_equal(supported.end, predictions.end):
        raise ValueError('저장 예측 지원 시점 누락·추가')
    if (not np.array_equal(supported.state_episode_id, predictions.state_episode_id)
        or not np.array_equal(supported.y_requested_exit, predictions.all_y_exit)
        or not np.array_equal(supported.paired_part, predictions.paired_part)):
        raise ValueError('저장 예측의 에피소드·정답·기간 불일치')
    for name in MODELS:
        if predictions[f'{name}_exit_request'].dtype.kind != 'b' or predictions[f'{name}_exit_request'].isna().any():
            raise ValueError('저장 청산 요청의 불리언 형식 오류')
    return result.merge(predictions[['end', *[f'{name}_exit_request' for name in MODELS]]],
        on='end', how='left', validate='one_to_one')


def distribution(values):
    values = values.dropna().to_numpy(dtype=float)
    return {'count': len(values), **{name: (float(np.quantile(values, q)) if len(values) else None)
        for name, q in [('min', 0.), ('p50', .5), ('p90', .9), ('p99', .99), ('max', 1.)]}}


def binary_metrics(truth, predicted):
    y, p = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return {'positive': int(y.sum()), 'requests': int(p.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
        'f2': 5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else None}


def summarize_execution(requests, ended, windows):
    def counts(values):
        return {str(k): int(v) for k, v in values.value_counts(dropna=False).sort_index().items()}

    def orders(frame):
        exits = frame.loc[frame.requested_action.isin(['close', 'reverse'])]
        return {'orders': len(frame), 'exit_requests': len(exits), 'exit_first_filled_actions': counts(exits.first_filled_action),
            'exit_end_owner': counts(exits.end_owner), 'exit_same_window_end': int(exits.termination_in_first_window.sum()),
            'exit_same_order_same_window': int((exits.termination_in_first_window & exits.end_owner.eq('same_order')).sum()),
            'exit_delay_seconds': distribution(exits.termination_delay_ns/1e9),
            'all_fill_span_seconds': distribution(frame.fill_span_ns/1e9),
            'exit_fill_span_seconds': distribution(exits.fill_span_ns/1e9),
            'quantity_changed_orders': int(frame.quantity_versions.gt(1).sum()),
            'exit_quantity_changed_orders': int(exits.quantity_versions.gt(1).sum()),
            'exit_observed_less_than_first_requested': int(exits.observed_quantity.lt(exits.first_order_quantity).sum()),
            'exit_observed_to_first_requested': distribution(exits.observed_to_first_requested),
            'exit_last_observed_status': counts(exits.last_observed_status.fillna('unknown')),
            'observed_fills': int(frame.observed_fills.sum()), 'maker_fills': int(frame.maker_fills.sum()),
            'taker_fills': int(frame.taker_fills.sum()), 'other_liquidity_fills': int(frame.other_liquidity_fills.sum()),
            'cumulative_mismatch_fills': int(frame.cumulative_mismatch_fills.sum())}

    def terminations(frame):
        return {'events': len(frame), 'origin': counts(frame.origin), 'request_in_same_window': int(frame.request_in_same_window.sum()),
            'request_to_termination_seconds': distribution(frame.request_to_termination_ns/1e9)}

    def table(frame):
        y, z = frame.y_requested_exit, frame.y_filled_exit
        return {'windows': len(frame), 'both': int((y & z).sum()), 'requested_only': int((y & ~z).sum()),
            'filled_only': int((~y & z).sum()), 'neither': int((~y & ~z).sum())}

    def predicted(frame):
        return {name: {target: binary_metrics(frame[column], frame[f'{name}_exit_request'])
            for target, column in [('requested', 'y_requested_exit'), ('filled', 'y_filled_exit')]} for name in MODELS}

    result = {'orders': orders(requests), 'terminations': terminations(ended),
        'v107_supported_held': table(windows.loc[windows.usable & windows.state_quantity.ne(0)]),
        'paired_periods': {}, 'predictions': {}, 'yearly': [], 'monthly': [],
        'model_fitted': False, 'profitability_accepted': False}
    for part in ('training', 'calibration', 'diagnosis'):
        selected = windows.loc[windows.paired_usable & windows.paired_part.eq(part)]
        result['paired_periods'][part] = table(selected)
        if part != 'training':
            if selected[[f'{m}_exit_request' for m in MODELS]].isna().any().any():
                raise ValueError('지원 창의 저장 청산 예측 누락')
            result['predictions'][part] = predicted(selected)
    for name, format_ in [('yearly', '%Y'), ('monthly', '%Y-%m')]:
        request_keys = requests.time.dt.strftime(format_)
        end_keys = ended.time.dt.strftime(format_)
        window_keys = windows.end.dt.strftime(format_)
        keys = sorted(set(request_keys) | set(end_keys) | set(window_keys))
        for key in keys:
            selected = windows.loc[window_keys.eq(key)]
            scored = selected.loc[selected.paired_usable & selected.paired_part.isin(['calibration', 'diagnosis'])]
            result[name].append({'period': key,
                'orders': orders(requests.loc[request_keys.eq(key)]),
                'terminations': terminations(ended.loc[end_keys.eq(key)]),
                'v107_supported_held': table(selected.loc[selected.usable & selected.state_quantity.ne(0)]),
                'predictions': predicted(scored) if len(scored) else None})
    return result
