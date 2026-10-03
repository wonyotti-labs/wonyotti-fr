from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .minute_management import ACTIONS, management_orders
from .reports import table

BANDS = [0., .001, .01, .05, .25, .5, .75, 1.0000000001]


def inventory_states(actions: pd.DataFrame) -> pd.DataFrame:
    frame = actions.reset_index(drop=True).copy()
    values = frame[['before_qty', 'after_qty', 'episode_id']].to_numpy(dtype=float)
    if (frame.empty or str(frame.time.dtype) not in {'datetime64[ns, UTC]', 'datetime64[us, UTC]'}
        or not frame.time.is_monotonic_increasing or frame.time.isna().any()
        or not np.isfinite(values).all() or (values % 1 != 0).any()
        or not frame.before_qty.iloc[1:].reset_index(drop=True).equals(frame.after_qty.iloc[:-1].reset_index(drop=True))
        or frame.before_qty.iloc[0] != 0 or (frame.after_qty.ne(0) & frame.episode_id.le(0)).any()):
        raise ValueError('잔여 수량 원장의 순서·보유 연결 오류')
    frame['max_quantity_so_far'] = frame.after_qty.abs().groupby(frame.episode_id).cummax()
    frame['remaining_fraction'] = (frame.after_qty.abs() / frame.max_quantity_so_far.replace(0, np.nan)).fillna(0.)
    # 반전 행의 새 에피소드 최대값으로 직전 포지션의 크기를 나누지 않는다.
    previous = frame.max_quantity_so_far.shift(fill_value=0)
    frame['before_fraction'] = (frame.before_qty.abs() / previous.replace(0, np.nan)).fillna(0.)
    if not frame.remaining_fraction.between(0, 1).all() or not frame.before_fraction.between(0, 1).all():
        raise ValueError('과거 최대 보유 대비 잔량 비율 오류')
    return frame


def attach_inventory(minutes: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    right = states.drop_duplicates('time', keep='last')[['time', 'episode_id', 'remaining_fraction']].rename(
        columns={'time': 'inventory_time', 'episode_id': 'inventory_episode_id'})
    right['inventory_time'] = right.inventory_time.astype('datetime64[ns, UTC]')
    left = minutes.copy().astype({'end': 'datetime64[ns, UTC]'})
    joined = pd.merge_asof(left, right, left_on='end', right_on='inventory_time', direction='backward', allow_exact_matches=False)
    held = joined[joined.usable]
    if (not held.episode_id.eq(held.inventory_episode_id).all()
        or not held.remaining_fraction.between(0, 1, inclusive='right').all()):
        raise ValueError('분 경계의 보유 수량·에피소드 연결 오류')
    return joined


def order_sizes(executions: pd.DataFrame, actions: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    orders = management_orders(executions, actions)
    trade = states[states.action.ne('funding')].copy()
    trade['event_index'] = np.arange(len(trade))
    trade['executed_change'] = (trade.after_qty - trade.before_qty).abs()
    first = trade.drop_duplicates('order_key', keep='first')[['order_key', 'before_fraction']]
    grouped = trade.groupby('order_key').agg(first_time=('time', 'first'), last_time=('time', 'last'),
        action_rows=('time', 'size'), first_index=('event_index', 'first'), last_index=('event_index', 'last'),
        executed_quantity=('executed_change', 'sum'), realized_btc=('realized_btc', 'sum'), trade_fees_btc=('fee_btc', 'sum'))
    grouped['interleaved'] = grouped.last_index - grouped.first_index + 1 > grouped.action_rows
    grouped['fill_span_seconds'] = (grouped.last_time - grouped.first_time).dt.total_seconds()
    ledger = orders.merge(first, on='order_key', validate='one_to_one').merge(grouped, on='order_key', validate='one_to_one')
    fills = executions[executions.symbol.eq('XBTUSD') & executions.exectype.eq('Trade')]
    actual = pd.to_numeric(fills.lastqty, errors='raise').groupby(fills.order_key).sum().reindex(ledger.order_key).to_numpy()
    if len(ledger) != len(orders) or not np.array_equal(actual, ledger.executed_quantity):
        raise ValueError('독립 주문의 실제 체결량·원장 연결 오류')
    ledger['requested_fraction'] = ledger.orderqty / ledger.before_qty.abs().replace(0, np.nan)
    ledger['filled_fraction'] = ledger.executed_quantity / ledger.orderqty
    ledger['requested_residual_quantity'] = ledger.before_qty.abs() - ledger.orderqty
    return ledger


def inventory_groups(frame: pd.DataFrame) -> list[dict]:
    source = frame[frame.usable].copy()
    source['year'] = source.end.dt.year
    source['band'] = pd.cut(source.remaining_fraction, BANDS, include_lowest=True).astype(str)
    rows = []
    for (year, band), part in source.groupby(['year', 'band']):
        rows.append({'year': int(year), 'band': band, 'minutes': len(part),
            'mean_fraction': float(part.remaining_fraction.mean()), 'adds_at_cap_share': float(part.adds_capped.eq(5).mean()),
            **{action: {'positive_minutes': int(part[f'y_{action}'].sum()), 'rate': float(part[f'y_{action}'].mean())} for action in ACTIONS}})
    return rows


def bot_inventory(path: Path) -> dict:
    frame = pd.read_parquet(path / 'equity.parquet', columns=['time', 'quantity', 'policy_state', 'policy_event'])
    held = frame[frame.quantity.ne(0)].copy()
    held['entry'] = held.policy_state.map(lambda value: json.loads(value).get('path_entry_time'))
    missing = int(held.entry.isna().sum())
    known = held[held.entry.notna()].copy()
    known['remaining_fraction'] = known.quantity.abs() / known.quantity.abs().groupby(known.entry).cummax()
    known['band'] = pd.cut(known.remaining_fraction, BANDS, include_lowest=True).astype(str)
    return {'path': str(path), 'equity_sha256': sha256(path / 'equity.parquet'), 'held_minutes': len(held),
        'unknown_position_minutes': missing, 'groups': [{'band': band, 'minutes': len(part),
            'mean_fraction': float(part.remaining_fraction.mean()), 'events': part.policy_event.value_counts().to_dict()}
            for band, part in known.groupby('band')]}


def run_inventory_study(audit: Path, labels: Path, output: Path, bot_runs: list[Path]) -> Path:
    metadata = json.loads((labels / 'manifest.json').read_text())['settings']
    expected = json.loads((labels / 'files.json').read_text())
    inputs = {name: sha256(audit / name) for name in ['actions.parquet', 'executions.parquet', 'episodes.parquet']}
    if (inputs != metadata['audit_sha256'] or sha256(labels / 'events.parquet') != expected['events.parquet']
        or json.loads((labels / 'summary.json').read_text())['complete'] is not True):
        raise ValueError('잔여 수량 진단의 원본·분별 정답 지문 오류')
    out = new_run(output, 'inventory-study', {'audit': str(audit), 'labels': str(labels),
        'audit_sha256': inputs, 'labels_sha256': expected['events.parquet'], 'bot_runs': list(map(str, bot_runs)),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V19.md')), 'new_model_fit': False, 'profitability_test': False})
    print(f'잔여 수량·관리 의미 진단: {out}', flush=True)
    try:
        actions, executions = (pd.read_parquet(audit / f'{name}.parquet') for name in ['actions', 'executions'])
        states = inventory_states(actions)
        frame = pd.read_parquet(labels / 'events.parquet', columns=['end', 'episode_id', 'usable', 'adds_capped', *[f'y_{a}' for a in ACTIONS]])
        joined = attach_inventory(frame, states)
        sizes = order_sizes(executions, actions, states)
        states.to_parquet(out / 'source_states.parquet', index=False)
        sizes.to_parquet(out / 'order_sizes.parquet', index=False)
        save_json(out / 'source_groups.json', inventory_groups(joined))
        annual = []
        for year, part in joined[joined.usable].groupby(joined.end.dt.year):
            orders = sizes[sizes.target_time.dt.year.eq(year)]
            reduce = orders[orders.target.eq('reduce')]
            exits = orders[orders.before_qty.ne(0) & orders.target.isin(['exit', 'enter_long', 'enter_short'])]
            annual.append({'year': int(year), 'held_minutes': len(part),
                **{f'remaining_le_{threshold}': int(part.remaining_fraction.le(threshold).sum()) for threshold in [.001, .01, .05]},
                'reduce_orders': len(reduce), 'reduce_requests_ge_99pct': int(reduce.requested_fraction.ge(.99).sum()),
                'reduce_requested_median': float(reduce.requested_fraction.median()) if len(reduce) else None,
                'all_requested_filled_exact': int(orders.filled_fraction.eq(1).sum()), 'orders': len(orders),
                'interleaved_orders': int(orders.interleaved.sum()),
                'exits_from_le_1pct': int(exits.before_fraction.le(.01).sum()), 'exit_or_reversal_orders': len(exits)})
        save_json(out / 'annual.json', annual)
        money = states.groupby([states.time.dt.year.rename('year'), 'action']).agg(
            rows=('time', 'size'), realized_price_pnl_btc=('realized_btc', 'sum'), recorded_cost_btc=('fee_btc', 'sum')).reset_index()
        save_json(out / 'accounting_decomposition.json', money.to_dict('records'))
        reconstruction = json.loads((audit / 'reconstruction.json').read_text())
        trade_fees = float(states.loc[states.action.ne('funding'), 'fee_btc'].sum())
        funding = float(states.loc[states.action.eq('funding'), 'fee_btc'].sum())
        gross = float(states.realized_btc.sum())
        for key, actual in [('realized_gross_btc', gross), ('trade_fees_btc', trade_fees), ('funding_cost_btc', funding)]:
            if not np.isclose(actual, reconstruction[key], rtol=0, atol=1e-7):
                raise ValueError('관리 행동별 원본 정산 통화 회계 불일치')
        save_json(out / 'bot_groups.json', [bot_inventory(path) for path in bot_runs])
        summary = {'complete': True, 'source_rows_preserved': len(states) == len(actions), 'linked_orders': len(sizes),
            'minute_states': int(joined.usable.sum()), 'raw_accounting_exact': True, 'realized_gross_btc': gross,
            'trade_fees_btc': trade_fees, 'funding_cost_btc': funding, 'remaining_actual_contracts': int(states.after_qty.iloc[-1]),
            'threshold_selected_by_profit': False, 'new_model_fit': False, 'profitability_accepted': False}
        save_json(out / 'summary.json', summary)
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out / 'REPORT.md').write_text('# 잔여 수량과 관리 행동의 의미\n\n' + table(pd.DataFrame(annual))
            + '\n\n최대 수량은 해당 시각까지의 값이며 과거 상태에 미래 최대값을 넣지 않았다. 작은 잔량도 원본 회계·손실·펀딩에 그대로 남는다. '
            '99%는 요청량의 기술 통계이며 실제 체결률과 동시 주문을 함께 기록했다. 원본 의도·실제 평탄 상태 또는 최적 청산 기준이라는 뜻은 아니다.\n\n'
            + table(money) + '\n\n실현 가격 손익은 원본 BTC 정산 금액이며 계좌 수익률이 아니다. 마지막 열린 포지션의 평가 손익은 더하지 않았다. '
            'funding 행의 recorded_cost_btc는 펀딩이며 나머지 행은 거래 비용이다. 새 성과 모델이나 이익 문턱을 선택하지 않았다.\n')
        print(f'원본 {len(actions)}행·독립 주문 {len(sizes)}개·회계 대조 완료', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
