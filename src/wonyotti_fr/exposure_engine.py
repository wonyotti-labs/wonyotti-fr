from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from wonyotti_fr.engine import validate_bar
from wonyotti_fr.simulator import RiskConfig


@dataclass(frozen=True)
class ExposureConfig(RiskConfig):
    bar_seconds: int = 60
    signal_delay_bars: int = 0

    def validate(self):
        super().validate()
        if self.bar_seconds != 60 or type(self.signal_delay_bars) is not int or self.signal_delay_bars not in (0, 1):
            raise ValueError('노출 실행 주기·지연 오류')


def target_quantity(cash, quantity, price, target, fee_bps, slippage_bps):
    equity = cash+quantity*price
    if (not np.isfinite([cash, quantity, price, target, fee_bps, slippage_bps]).all()
        or not np.isfinite(equity) or equity <= 0 or price <= 0 or abs(target) > 1 or min(fee_bps, slippage_bps) < 0
        or max(fee_bps, slippage_bps) > 100):
        raise ValueError('목표 노출 수량 계산 입력 오류')
    if abs(quantity*price/equity-target) <= 1e-12:
        return quantity
    direction = 1 if target*equity/price > quantity else -1
    slip, fee = slippage_bps/10000, fee_bps/10000
    cost = price*(slip+fee*(1+direction*slip))
    coefficient = target/price
    # 비용 차감 후 계좌를 분모로 쓰는 목표 수량의 선형 해다.
    result = (coefficient*equity+coefficient*cost*direction*quantity)/(1+coefficient*cost*direction)
    if not np.isfinite(result):
        raise ValueError('목표 노출 수량의 유한성 오류')
    return result


class ExposureEngine:
    def __init__(self, config, snapshot=None):
        config.validate()
        self.config = config
        if snapshot is not None:
            if (not isinstance(snapshot, dict) or set(snapshot) != {'format', 'config', 'state'} or snapshot['format'] != 'exposure_engine_state_v1'
                or snapshot['config'] != asdict(config)):
                raise ValueError('노출 실행 복원 설정 불일치')
            self.state = copy.deepcopy(snapshot['state'])
        else:
            self.state = {'cash': config.initial_equity, 'quantity': 0., 'entry_price': 0.,
                'index': 0, 'entry_index': -1, 'last_end': None, 'last_close': 0.,
                'peak': config.initial_equity, 'day': None, 'day_equity': config.initial_equity,
                'daily_halted': False, 'permanent_halted': False, 'manual_halt': False,
                'cooldown_until': -1, 'pending': None, 'deferred': [], 'liquidity_exit_reason': None,
                'active_trade': None, 'total_fees': 0., 'total_funding': 0., 'closed_net': 0.,
                'closed_trades': 0, 'wins': 0, 'sum_gains': 0., 'sum_losses': 0.,
                'max_drawdown': 0., 'completed': False}
        self._validate()

    def _validate(self):
        state = self.state
        fields = {'cash', 'quantity', 'entry_price', 'index', 'entry_index', 'last_end', 'last_close',
            'peak', 'day', 'day_equity', 'daily_halted', 'permanent_halted', 'manual_halt',
            'cooldown_until', 'pending', 'deferred', 'liquidity_exit_reason', 'active_trade',
            'total_fees', 'total_funding', 'closed_net', 'closed_trades', 'wins', 'sum_gains',
            'sum_losses', 'max_drawdown', 'completed'}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError('노출 실행 상태 필드 오류')
        for key in ['cash', 'quantity', 'entry_price', 'last_close', 'peak', 'day_equity', 'total_fees',
            'total_funding', 'closed_net', 'sum_gains', 'sum_losses', 'max_drawdown']:
            if type(state[key]) not in (float, int) or not np.isfinite(state[key]):
                raise ValueError('노출 실행 상태의 유한성 오류')
        for key in ['index', 'entry_index', 'cooldown_until', 'closed_trades', 'wins']:
            if type(state[key]) is not int or state[key] < -1:
                raise ValueError('노출 실행 상태의 횟수 오류')
        for key in ['daily_halted', 'permanent_halted', 'manual_halt', 'completed']:
            if type(state[key]) is not bool:
                raise ValueError('노출 실행 상태의 중지 표시 오류')
        if (min(state['index'], state['closed_trades'], state['wins'], state['total_fees'],
            state['sum_gains'], state['sum_losses']) < 0 or state['wins'] > state['closed_trades']
            or min(state['peak'], state['day_equity']) <= 0 or not -1 <= state['max_drawdown'] <= 0
            or state['liquidity_exit_reason'] not in (None, 'gap_stop', 'intrabar_stop', 'time_limit', 'risk_halt')):
            raise ValueError('노출 실행 상태 범위 오류')
        if state['index'] == 0:
            if state['last_end'] is not None or state['day'] is not None:
                raise ValueError('노출 실행 초기 시각 오류')
        else:
            if type(state['last_end']) is not str or type(state['day']) is not str:
                raise ValueError('노출 실행 복원 시각 형식 오류')
            stamp = pd.Timestamp(state['last_end'])
            if (stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0 or stamp.value % (60*10**9)
                or state['day'] != (stamp-pd.Timedelta(minutes=1)).date().isoformat()):
                raise ValueError('노출 실행 복원 시각 경계 오류')
        if (not isinstance(state['deferred'], list) or len(state['deferred']) > self.config.signal_delay_bars
            or (state['quantity'] == 0) != (state['active_trade'] is None)):
            raise ValueError('노출 실행 상태의 대기·포지션 오류')
        for target in [state['pending'], *state['deferred']]:
            self._target(target)
        if state['quantity'] and state['entry_price'] <= 0:
            raise ValueError('노출 실행 평균 체결가 오류')
        if state['active_trade'] is not None:
            active = state['active_trade']
            if set(active) != {'entry_time', 'entry_price', 'direction', 'gross_realized', 'fees', 'funding_cost', 'max_quantity'}:
                raise ValueError('노출 실행 열린 거래 필드 오류')
            for key in ['entry_price', 'gross_realized', 'fees', 'funding_cost', 'max_quantity']:
                if type(active[key]) not in (float, int) or not np.isfinite(active[key]):
                    raise ValueError('노출 실행 열린 거래 숫자 오류')
            if (type(active['direction']) is not int or active['direction'] not in (-1, 1)
                or active['direction']*state['quantity'] <= 0 or active['entry_price'] <= 0
                or active['fees'] < 0 or active['max_quantity'] < abs(state['quantity'])-1e-12):
                raise ValueError('노출 실행 열린 거래 수량·방향 오류')
        if state['completed'] and (state['quantity'] or state['pending'] is not None or state['deferred']):
            raise ValueError('노출 실행 완료 상태 오류')

    def _target(self, target):
        if target is not None and (type(target) not in (int, float) or not np.isfinite(target)
            or abs(target) > self.config.allocation):
            raise ValueError('노출 실행 목표 한도·형식 오류')

    def snapshot(self):
        return {'format': 'exposure_engine_state_v1', 'config': asdict(self.config), 'state': copy.deepcopy(self.state)}

    def view(self, price):
        equity = self.state['cash']+self.state['quantity']*price
        return {'equity': equity, 'quantity': self.state['quantity'],
            'net_exposure': self.state['quantity']*price/equity,
            'halted': self.state['permanent_halted'] or self.state['manual_halt']}

    def step(self, bar, decide, final=False):
        if type(final) is not bool:
            raise ValueError('노출 실행 완료 입력 형식 오류')
        event = validate_bar(bar, 60)
        trial = copy.copy(self)
        trial.state = copy.deepcopy(self.state)
        result = trial._advance(event, decide, final)
        trial._validate()
        self.state = trial.state
        return result

    def _advance(self, bar, decide, final):
        s, c = self.state, self.config
        if s['completed'] or (s['last_end'] is not None and s['last_end'] != bar['time']):
            raise ValueError('노출 실행 완료·시각 연속성 오류')
        i, opening, closing = s['index'], bar['open'], bar['close']
        tradable = bar.get('count', 1) > 0
        fills, closed, rejected = [], [], []

        def trade(delta, reference, timestamp, reason):
            if delta == 0:
                return
            before = s['quantity']
            sign = 1 if delta > 0 else -1
            price = reference*(1+sign*c.slippage_bps/10000)
            fee = abs(delta)*price*c.fee_bps/10000
            s['cash'] -= delta*price+fee
            s['total_fees'] += fee
            remaining, closing_fee = abs(delta), 0.
            if before and before*delta < 0:
                amount = min(abs(before), abs(delta))
                closing_fee = fee*amount/abs(delta)
                active = s['active_trade']
                active['gross_realized'] += (1 if before > 0 else -1)*amount*(price-s['entry_price'])
                active['fees'] += closing_fee
                remaining -= amount
                s['quantity'] = before+sign*amount
                if amount == abs(before):
                    s['quantity'] = 0.
                    net = active['gross_realized']-active['fees']-active['funding_cost']
                    active.update(exit_time=timestamp, exit_price=price, exit_reason=reason,
                        net_pnl=net, hold_bars=i-s['entry_index']+int(timestamp == bar['end']))
                    closed.append(copy.deepcopy(active))
                    s['closed_net'] += net
                    s['closed_trades'] += 1
                    s['wins'] += int(net > 0)
                    s['sum_gains'] += max(net, 0)
                    s['sum_losses'] += max(-net, 0)
                    s['active_trade'], s['entry_price'] = None, 0.
            if remaining:
                prior = abs(s['quantity'])
                s['entry_price'] = (prior*s['entry_price']+remaining*price)/(prior+remaining)
                s['quantity'] += sign*remaining
                if s['active_trade'] is None:
                    s['entry_index'] = i
                    s['active_trade'] = {'entry_time': timestamp, 'entry_price': price, 'direction': sign,
                        'gross_realized': 0., 'fees': 0., 'funding_cost': 0., 'max_quantity': remaining}
                s['active_trade']['fees'] += fee-closing_fee
                s['active_trade']['max_quantity'] = max(s['active_trade']['max_quantity'], abs(s['quantity']))
            fills.append({'time': timestamp, 'delta_quantity': delta, 'price': price, 'fee': fee, 'reason': reason})

        def force_exit(reference, timestamp, reason):
            if not s['quantity']:
                return
            if tradable:
                trade(-s['quantity'], reference, timestamp, reason)
            elif final:
                raise ValueError('노출 실행 종료 봉의 거래 부재')
            elif s['liquidity_exit_reason'] is None:
                s['liquidity_exit_reason'] = reason

        day = bar['time'][:10]
        equity = s['cash']+s['quantity']*opening
        if day != s['day']:
            s['day'], s['day_equity'], s['daily_halted'] = day, equity, False
        funding = s['quantity']*opening*bar['funding_rate']
        if s['quantity']:
            s['cash'] -= funding
            s['active_trade']['funding_cost'] += funding
            s['total_funding'] += funding
        equity = s['cash']+s['quantity']*opening
        s['peak'] = max(s['peak'], equity)
        s['daily_halted'] |= equity <= s['day_equity']*(1-c.daily_loss_limit)
        s['permanent_halted'] |= equity <= s['peak']*(1-c.max_drawdown)
        if tradable and s['liquidity_exit_reason']:
            reason = s['liquidity_exit_reason']
            force_exit(opening, bar['time'], 'liquidity_'+reason)
            s['liquidity_exit_reason'], s['pending'], s['deferred'] = None, None, []
            if reason in ('gap_stop', 'intrabar_stop', 'time_limit'):
                s['cooldown_until'] = i+c.cooldown_bars+1
        direction = 1 if s['quantity'] > 0 else -1 if s['quantity'] < 0 else 0
        stop = s['entry_price']*(1-direction*c.stop_fraction)
        if direction and c.stop_fraction and (opening <= stop if direction > 0 else opening >= stop):
            force_exit(opening, bar['time'], 'gap_stop')
            s['cooldown_until'] = i+c.cooldown_bars+1
        if s['quantity'] and c.max_hold_bars and i-s['entry_index'] >= c.max_hold_bars:
            force_exit(opening, bar['time'], 'time_limit')
            s['cooldown_until'] = i+c.cooldown_bars+1
        blocked = s['daily_halted'] or s['permanent_halted'] or s['manual_halt']
        if blocked:
            force_exit(opening, bar['time'], 'risk_halt')
        pending = s['pending']
        executed = None
        waiting = pending is not None and not tradable and not blocked
        if waiting:
            rejected.append('market_no_trades')
        elif pending is not None and not blocked and i >= s['cooldown_until']:
            desired = target_quantity(s['cash'], s['quantity'], opening, pending, c.fee_bps, c.slippage_bps)
            trade(desired-s['quantity'], opening, bar['time'], 'target_exposure')
            executed = pending
        elif pending is not None:
            rejected.append('risk_halt_or_cooldown')
        direction = 1 if s['quantity'] > 0 else -1 if s['quantity'] < 0 else 0
        stop = s['entry_price']*(1-direction*c.stop_fraction)
        if direction and c.stop_fraction and (bar['low'] <= stop if direction > 0 else bar['high'] >= stop):
            force_exit(stop, bar['end'], 'intrabar_stop')
            s['cooldown_until'] = i+c.cooldown_bars+1
        if final:
            force_exit(closing, bar['end'], 'end_of_test')
            s['completed'] = True
        equity = s['cash']+s['quantity']*closing
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError('노출 실행 계좌 파산·유한성 오류')
        s['peak'] = max(s['peak'], equity)
        drawdown = equity/s['peak']-1
        s['max_drawdown'] = min(s['max_drawdown'], drawdown)
        s['last_end'], s['last_close'], s['index'] = bar['end'], closing, i+1
        decision = None if final or blocked or s['liquidity_exit_reason'] else decide(bar, self.view(closing))
        self._target(decision)
        if final or blocked or s['liquidity_exit_reason']:
            s['pending'], s['deferred'] = None, []
        elif waiting:
            s['pending'] = pending
        elif c.signal_delay_bars:
            s['deferred'].append(decision)
            s['pending'] = s['deferred'].pop(0) if len(s['deferred']) > c.signal_delay_bars else None
        else:
            s['pending'] = decision
        active = s['active_trade']
        active_net = active['gross_realized']-active['fees']-active['funding_cost'] if active else 0.
        residual = equity-c.initial_equity-s['closed_net']-active_net-s['quantity']*(closing-s['entry_price'])
        if abs(residual) > max(1e-7, equity*1e-10):
            raise ValueError('노출 실행 현금·거래별 손익 회계 불일치')
        return {'time': bar['end'], 'equity': equity, 'quantity': s['quantity'],
            'exposure': abs(s['quantity']*closing/equity), 'net_exposure': s['quantity']*closing/equity,
            'drawdown': drawdown, 'halted': s['permanent_halted'] or s['manual_halt'],
            'daily_halted': s['daily_halted'], 'requested_target': pending, 'executed_target': executed,
            'decision': decision, 'next_target': s['pending'],
            'fills': fills, 'closed_trades': closed, 'rejected': rejected,
            'accounting_residual': residual, 'completed': final}
