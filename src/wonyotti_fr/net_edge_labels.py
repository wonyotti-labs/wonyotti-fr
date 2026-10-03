from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import EngineConfig, TradingEngine
from .event_backtest import iter_events
from .event_features import MARKET_FEATURES
from .minute_data import validate_minutes
from .pullback_diagnostics import verify_fill_activity
from .pullback_policy import PullbackPolicy


def potential_entries(bars: pd.DataFrame, policy: PullbackPolicy) -> tuple[pd.DataFrame, dict]:
    if policy.offset_bps != 16 or policy.ttl_minutes != 5 or policy.baseline is not None:
        raise ValueError('v8 기회는 고정 v7 대기 기준이 필요합니다.')
    validate_minutes(bars, 1)
    if len(bars) < 2 or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError('진입 기회 자료의 연속 분봉 부족')
    frame = bars.reset_index(drop=True)
    boundary = np.flatnonzero(frame.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=5).value == 0)
    policy.prepare(frame)
    market = frame[MARKET_FEATURES].to_numpy(dtype=float)
    close = frame.close.to_numpy(dtype=float)
    state = {'direction': 0, 'halted': False, 'hold_bars': 0, 'pending': 'hold'}
    rows, counts = [], {'boundaries': len(boundary), 'base_signals': 0, 'expired': 0, 'incomplete_wait': 0,
                        'missing_signal_features': 0, 'missing_trigger_features': 0}
    for index in boundary:
        if not np.isfinite(market[index]).all():
            counts['missing_signal_features'] += 1
            continue
        intent = policy.base({'end': frame.end.iloc[index].isoformat(), 'features': market[index]}, state)
        if intent not in ('enter_long', 'enter_short'):
            continue
        counts['base_signals'] += 1
        direction = 1 if intent == 'enter_long' else -1
        last = min(index+5, len(frame)-1)
        favorable = direction * np.log(close[index] / close[index+1:last+1]) * 10000
        found = np.flatnonzero(favorable >= 16)
        if not len(found):
            counts['expired' if last == index+5 else 'incomplete_wait'] += 1
            continue
        trigger = index+1+int(found[0])
        if not np.isfinite(market[trigger]).all():
            counts['missing_trigger_features'] += 1
            continue
        rows.append({'signal_time': frame.end.iloc[index], 'decision_time': frame.end.iloc[trigger],
                     'entry_index': trigger+1, 'order_direction': direction, 'reference_price': close[index],
                     'decision_close': close[trigger], 'favorable_bps': float(favorable[found[0]]),
                     'wait_minutes': trigger-index, **dict(zip(MARKET_FEATURES, market[trigger], strict=True))})
    result = pd.DataFrame(rows, columns=['signal_time', 'decision_time', 'entry_index', 'order_direction',
                          'reference_price', 'decision_close', 'favorable_bps', 'wait_minutes', *MARKET_FEATURES])
    counts['triggered'] = len(result)
    if len(result) and result.decision_time.duplicated().any():
        raise ValueError('같은 시각에 겹친 독립 진입 기회')
    return result, counts


def isolated_outcome(window: pd.DataFrame, direction: int, config: EngineConfig) -> dict:
    if (direction not in (-1, 1) or config.bar_seconds != 60 or config.max_hold_bars != 30
        or config.signal_delay_bars != 0 or config.max_adds != 0 or len(window) != 31):
        raise ValueError('독립 순손익 정답의 방향·실행 설정·기간 오류')
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long' if direction == 1 else 'enter_short'
    trades, fills = [], []
    for index, bar in enumerate(iter_events(window)):
        result = engine.step(bar, lambda _bar, _state: 'hold', final=index == 30)
        trades.extend(result['closed_trades'])
        fills.extend(result['fills'])
    if len(trades) != 1 or len(fills) != 2 or fills[0]['reason'] != 'entry':
        raise ValueError('독립 정답의 진입·청산 횟수 오류')
    verify_fill_activity(pd.DataFrame(fills), window)
    trade = trades[0]
    notional = abs(fills[0]['delta_quantity']) * fills[0]['price']
    net = trade['net_pnl']
    if abs(engine.state['cash'] - config.initial_equity - net) > 1e-7:
        raise ValueError('독립 정답의 회계 불일치')
    return {'net_bps': net/notional*10000, 'net_pnl': net, 'entry_notional': notional,
            'entry_time': pd.Timestamp(trade['entry_time']), 'entry_price': trade['entry_price'],
            'exit_time': pd.Timestamp(trade['exit_time']), 'exit_price': trade['exit_price'],
            'exit_reason': trade['exit_reason'], 'fees': trade['fees'], 'funding_cost': trade['funding_cost'],
            'label_end': window.end.iloc[-1], 'hold_minutes': trade['hold_bars']}


def label_opportunities(bars: pd.DataFrame, opportunities: pd.DataFrame, config: EngineConfig,
                        training_end: str) -> tuple[pd.DataFrame, dict]:
    cutoff = pd.Timestamp(training_end, tz='UTC') - pd.Timedelta(days=1)
    rows, excluded = [], 0
    for opportunity in opportunities.to_dict('records'):
        index = opportunity['entry_index']
        window = bars.iloc[index:index+31]
        if len(window) != 31 or window.end.iloc[-1] >= cutoff:
            excluded += 1
            continue
        if window.time.iloc[0] != opportunity['decision_time']:
            raise ValueError('조건 판단 뒤 다음 분봉의 시작 시각 불일치')
        outcome = isolated_outcome(window, opportunity['order_direction'], config)
        close_return = opportunity['order_direction'] * np.log(window.close.iloc[29] / opportunity['decision_close']) * 10000
        rows.append({**opportunity, **outcome, 'close_markout_bps': float(close_return)})
    labeled = pd.DataFrame(rows)
    return labeled, {'opportunities': len(opportunities), 'labeled': len(labeled), 'boundary_excluded': excluded,
                     'cutoff_exclusive': cutoff, 'losing_labels_removed': False,
                     'limit': '독립 초기 계좌의 정답이며 연속 계좌 상태·겹친 기회의 독립성을 보장하지 않음'}
