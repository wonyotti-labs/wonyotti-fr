from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import EngineConfig, PolicyDecision, TradingEngine
from .event_backtest import iter_events
from .minute_data import validate_minutes
from .net_edge_model import NetEdgeModel, net_values
from .pullback_diagnostics import verify_fill_activity
from .rate_policy import RateActionPolicy


class LifecycleNetPolicy:
    def __init__(self, manager: RateActionPolicy, model: NetEdgeModel, enabled: bool = True):
        if not isinstance(manager, RateActionPolicy) or model.data['alpha'] != 100 or type(enabled) is not bool:
            raise ValueError('전체 거래 순손익 정책의 고정 기반 오류')
        self.manager, self.model, self.enabled = manager, model, enabled
        self.base, self.offset_bps, self.ttl_minutes, self.baseline = manager.base, manager.offset_bps, manager.ttl_minutes, manager.baseline

    def prepare(self, frame):
        self.manager.prepare(frame)

    def __call__(self, bar, state):
        decision = self.manager(bar, state)
        if decision.event != 'triggered' or not self.enabled:
            return decision
        waiting = state['policy_state']
        favorable = waiting['direction'] * np.log(waiting['reference_price'] / bar['close']) * 10000
        minutes = (pd.Timestamp(bar['end']) - pd.Timestamp(waiting['signal_time'])).total_seconds() / 60
        features = net_values(np.asarray(bar['features'])[None, :], [waiting['direction']], [favorable], [minutes])
        score = self.model.predict(features)[0]
        return decision if np.isfinite(score) and score >= 8 else PolicyDecision('hold', {}, 'filtered')


def lifecycle_outcome(events: list[dict], start: int, direction: int, config: EngineConfig,
                      policy: RateActionPolicy, cutoff: pd.Timestamp) -> tuple[dict, list[dict]]:
    if (direction not in (-1, 1) or not isinstance(policy, RateActionPolicy) or config.bar_seconds != 60
        or config.signal_delay_bars != 0 or config.max_hold_bars != 0 or not 0 <= start < len(events)
        or cutoff.tzinfo is None or cutoff.utcoffset().total_seconds()):
        raise ValueError('전체 거래 정답의 기반·방향·시간 설정 오류')
    engine = TradingEngine(config)
    engine.state['pending'] = 'enter_long' if direction > 0 else 'enter_short'
    fills, maximum_residual = [], 0.

    def one_position(bar, state):
        return policy(bar, state) if state['direction'] else PolicyDecision('hold', {}, 'label_flat')

    for index in range(start, len(events)):
        bar = events[index]
        if pd.Timestamp(bar['end']) >= cutoff:
            break
        result = engine.step(bar, one_position)
        fills.extend(result['fills'])
        maximum_residual = max(maximum_residual, abs(result['accounting_residual']))
        if result['closed_trades']:
            if len(result['closed_trades']) != 1 or len(fills) < 2 or fills[0]['reason'] != 'entry':
                raise ValueError('전체 거래 정답의 포지션·체결 수 오류')
            trade = result['closed_trades'][0]
            notional = abs(fills[0]['delta_quantity']) * fills[0]['price']
            if engine.state['quantity'] or abs(engine.state['cash'] - config.initial_equity - trade['net_pnl']) > 1e-7:
                raise ValueError('전체 거래 정답의 회계 불일치')
            return {'label_status': 'closed', 'net_bps': trade['net_pnl'] / notional * 10000,
                    'net_pnl': trade['net_pnl'], 'entry_notional': notional,
                    'entry_time': pd.Timestamp(trade['entry_time']), 'exit_time': pd.Timestamp(trade['exit_time']),
                    'label_end': pd.Timestamp(result['time']), 'exit_reason': trade['exit_reason'],
                    'hold_minutes': trade['hold_bars'], 'adds': trade['adds'],
                    'fees': trade['fees'], 'funding_cost': trade['funding_cost'], 'fills': len(fills),
                    'max_accounting_residual': maximum_residual}, fills
    return {'label_status': 'right_censored', 'net_bps': None, 'label_end': cutoff,
            'observed_minutes': engine.state['index'], 'remaining_quantity': engine.state['quantity'],
            'max_accounting_residual': maximum_residual}, fills


def lifecycle_labels(bars: pd.DataFrame, opportunities: pd.DataFrame, config: EngineConfig,
                     policy: RateActionPolicy, training_end: str, checkpoint=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validate_minutes(bars, 1)
    if not bars.time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError('전체 거래 정답 시세의 연속성 오류')
    cutoff = pd.Timestamp(training_end, tz='UTC') - pd.Timedelta(days=1)
    if opportunities.decision_time.duplicated().any() or not opportunities.decision_time.is_monotonic_increasing:
        raise ValueError('전체 거래 정답 기회의 중복·순서 오류')
    events = list(iter_events(bars))
    rows, ledger = [], []
    for number, opportunity in enumerate(opportunities.to_dict('records')):
        start = opportunity['entry_index']
        if (type(start) is not int or start < 0 or start >= len(events)
            or bars.end.iloc[start] >= cutoff):
            ledger.append({**opportunity, 'label_status': 'outside_training_boundary'})
            continue
        if pd.Timestamp(events[start]['time']) != opportunity['decision_time']:
            raise ValueError('전체 거래 정답의 판단·진입 시점 불일치')
        outcome, fills = lifecycle_outcome(events, start, opportunity['order_direction'], config, policy, cutoff)
        stop = bars.end.searchsorted(outcome['label_end'], side='right')
        verify_fill_activity(pd.DataFrame(fills), bars.iloc[start:stop])
        row = {**opportunity, **outcome}
        ledger.append(row)
        if outcome['label_status'] == 'closed':
            rows.append(row)
        if checkpoint is not None and (number + 1) % 25 == 0:
            checkpoint(pd.DataFrame(ledger))
    frame, all_rows = pd.DataFrame(rows), pd.DataFrame(ledger)
    if checkpoint is not None:
        checkpoint(all_rows)
    if len(frame) and (frame.label_end.ge(cutoff).any() or frame.decision_time.ge(frame.label_end).any()
                       or not np.isfinite(frame.net_bps).all()):
        raise ValueError('전체 거래 정답의 시간 경계·순손익 오류')
    return frame, all_rows, {'opportunities': len(opportunities), 'closed': len(frame),
        'status_counts': all_rows.label_status.value_counts().to_dict() if len(all_rows) else {},
        'cutoff_exclusive': cutoff, 'losing_labels_removed': False, 'forced_boundary_closes': 0,
        'limit': '독립 초기 계좌의 겹친 기회이며 연속 계좌의 위험 상태와 다름'}
