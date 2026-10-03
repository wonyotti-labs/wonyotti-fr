from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import EngineConfig, PolicyDecision, TradingEngine
from .event_backtest import iter_events
from .minute_data import validate_minutes
from .minute_inventory import MinuteInventoryModels
from .streaming_backtest import ParquetRows


def paired_addition_outcome(events, start, state, config: EngineConfig, policy, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if (config.bar_seconds != 60 or config.signal_delay_bars != 0
        or type(start) is not int or not 0 <= start < len(events)
        or state['pending'] != 'increase' or not state['quantity'] or state['completed']
        or state.get('deferred_intents') or state.get('liquidity_exit_reason')
        or cutoff.tzinfo is None or cutoff.utcoffset().total_seconds()
        or cutoff != cutoff.floor('min') or pd.Timestamp(state['last_end']) >= cutoff
        or events[start]['time'] != state['last_end']):
        raise ValueError('추가 순효과 정답의 시작 상태·시간·지연 오류')
    equity = state['cash'] + state['quantity'] * state['last_close']
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError('추가 순효과 정답의 시작 순자산 오류')
    engines = {name: TradingEngine(config, state) for name in ['allow', 'skip']}
    engines['skip'].state['pending'] = 'hold'
    traces = {name: {'fills': [], 'closed_trades': [], 'max_accounting_residual': 0., 'status': 'right_censored'}
              for name in engines}

    def current_position(bar, view):
        return policy(bar, view) if view['direction'] else PolicyDecision('hold', {}, 'label_flat')

    for index in range(start, len(events)):
        event = events[index]
        if pd.Timestamp(event['end']) >= cutoff:
            break
        for name, engine in engines.items():
            trace = traces[name]
            if trace['status'] == 'closed':
                continue
            result = engine.step(event, current_position)
            if result['fills'] and (event.get('count', 1) <= 0 or event.get('volume', 1) <= 0):
                raise ValueError('추가 순효과 정답의 무거래 봉 체결 오류')
            trace['fills'].extend(result['fills'])
            trace['closed_trades'].extend(result['closed_trades'])
            trace['max_accounting_residual'] = max(trace['max_accounting_residual'], abs(result['accounting_residual']))
            if result['closed_trades']:
                if (len(trace['closed_trades']) != 1 or engine.state['quantity']
                    or trace['closed_trades'][0]['entry_time'] != state['active_trade']['entry_time']):
                    raise ValueError('추가 순효과 정답의 포지션 종료 연결 오류')
                trace.update(status='closed', label_end=pd.Timestamp(event['end']))
        if all(t['status'] == 'closed' for t in traces.values()):
            break
    for name, engine in engines.items():
        traces[name]['final_state'] = engine.snapshot()
    closed = all(t['status'] == 'closed' for t in traces.values())
    difference = engines['allow'].state['cash'] - engines['skip'].state['cash'] if closed else None
    return {'label_status': 'closed' if closed else 'right_censored',
            'label_end': max(t['label_end'] for t in traces.values()) if closed else cutoff,
            'decision_equity': float(equity), 'incremental_pnl': difference,
            'incremental_bps': difference / equity * 10000 if closed else None}, traces


def collect_addition_states(bars, policy, config, output):
    validate_minutes(bars, 1)
    if (config.bar_seconds != 60 or config.signal_delay_bars != 0
        or not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()):
        raise ValueError('추가 기회 수집의 분봉·지연 오류')
    output.mkdir(parents=True, exist_ok=False)
    writers = {name: ParquetRows(output / f'{name}.parquet', 8192) for name in ['equity', 'trades', 'fills']}
    engine, opportunities = TradingEngine(config), []
    policy.prepare(bars)
    try:
        for index, event in enumerate(iter_events(bars)):
            result = engine.step(event, policy, final=index == len(bars) - 1)
            if result['next_intent'] == 'increase' and result.get('policy_event') == 'action_increase':
                state = engine.snapshot()
                view = engine.view(event['close'])
                stored = view['policy_state']
                view['_path_bounds'] = stored['path_low'], stored['path_high']
                values = policy.feature_values(event, view)
                if values.shape != (len(MinuteInventoryModels.features),) or not np.isfinite(values).all():
                    raise ValueError('추가 기회 판단 입력의 차원·유한성 오류')
                opportunities.append({'start': index + 1, 'decision_time': event['end'],
                    'position_entry_time': state['active_trade']['entry_time'],
                    'features': values.tolist(), 'state': state,
                    'future_fill_start': writers['fills'].rows + len(result['fills']),
                    'trade_index': engine.state['closed_trades']})
            for name, key in [('trades', 'closed_trades'), ('fills', 'fills')]:
                for row in result.pop(key):
                    writers[name].append(row)
            result.pop('rejected')
            writers['equity'].append(result)
    finally:
        for writer in writers.values():
            writer.close()
    return opportunities, engine.snapshot()


def position_weights(frame):
    if frame.empty or frame.position_entry_time.isna().any():
        raise ValueError('추가 정답의 포지션 가중치 입력 오류')
    counts = frame.groupby('position_entry_time').position_entry_time.transform('size').to_numpy(dtype=float)
    weight = 1 / counts
    return weight / weight.mean()
