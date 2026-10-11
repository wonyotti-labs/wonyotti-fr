from __future__ import annotations

import math
from datetime import datetime


def reference_target(cash, quantity, price, target, fee, slip):
    capital = cash+quantity*price
    if abs(quantity*price/capital-target) <= 1e-12:
        return quantity
    if target == 0:
        return 0.
    # 실행기의 닫힌 해와 별개로 비용 차감 후 노출의 근을 이분 탐색한다.
    left, right = -capital/price, capital/price
    for _ in range(64):
        middle = (left+right)/2
        order = middle-quantity
        fill = price*(1+math.copysign(slip/10000, order))
        after = cash-order*fill-abs(order)*fill*fee/10000+middle*price
        if middle*price/after > target:
            right = middle
        else:
            left = middle
    return (left+right)/2


def reference_policy(kind, model, bar, cash, quantity, *, constant_exposure=None):
    end = datetime.fromisoformat(bar['end'])
    if end.minute % 5 or end.second or end.microsecond:
        return None
    if kind == 'constant':
        if (type(constant_exposure) not in (int, float) or not math.isfinite(constant_exposure)
            or not -.5 <= constant_exposure <= .5):
            raise ValueError('독립 검산의 학습 대조 상수 오류')
        return float(constant_exposure)
    if bar.get('feature_end') != bar['end']:
        raise ValueError('검산 특징 시각 불일치')
    if any(value is None or not math.isfinite(value) for value in bar['features']):
        return None
    current = quantity*bar['close']/(cash+quantity*bar['close'])
    values = list(bar['features'])+([current] if kind == 'change' else [])
    predicted = model['baseline']
    for tree in model['trees']:
        at = 0
        for _ in range(3):
            if tree['left'][at] == -1:
                predicted += tree['value'][at]
                break
            branch = 'left' if values[tree['feature'][at]] <= tree['threshold'][at] else 'right'
            at = tree[branch][at]
        else:
            raise ValueError('검산 트리 깊이 오류')
    return max(-.5, min(.5, predicted+(current if kind == 'change' else 0)))


class ReferenceAccount:
    def __init__(self, config):
        self.c = config
        self.cash = config.initial_equity
        self.q = self.basis = 0.
        self.i = 0
        self.entry_i = -1
        self.active = None
        self.day = None
        self.day_start = self.peak = config.initial_equity
        self.daily = self.halted = False
        self.max_dd = self.fees = self.funding = 0.
        self.pending = None
        self.delay = []
        self.liquidate = None
        self.resume_at = -1
        self.trades = self.wins = 0
        self.gains = self.losses = self.closed_net = 0.

    def fill(self, delta, price, time, reason, end):
        if delta == 0:
            return
        direction = 1 if delta > 0 else -1
        execution = price*(1+direction*self.c.slippage_bps/10000)
        commission = abs(delta)*execution*self.c.fee_bps/10000
        self.cash -= delta*execution+commission
        self.fees += commission
        self.fills.append({'time': time, 'delta_quantity': delta, 'price': execution, 'fee': commission, 'reason': reason})
        reduce = min(abs(self.q), abs(delta)) if self.q*delta < 0 else 0.
        if reduce:
            closing_order = direction*reduce
            self.active['cash_flow'] -= closing_order*execution
            self.active['fees'] += reduce*execution*self.c.fee_bps/10000
            if reduce == abs(self.q):
                active = self.active
                pnl = active['cash_flow']-active['fees']-active['funding_cost']
                self.closed.append({'entry_time': active['entry_time'], 'entry_price': active['entry_price'],
                    'direction': active['direction'], 'gross_realized': active['cash_flow'], 'fees': active['fees'],
                    'funding_cost': active['funding_cost'], 'max_quantity': active['max_quantity'],
                    'exit_time': time, 'exit_price': execution, 'exit_reason': reason, 'net_pnl': pnl,
                    'hold_bars': self.i-self.entry_i+int(time == end)})
                self.trades += 1
                self.wins += int(pnl > 0)
                self.gains += max(pnl, 0)
                self.losses += max(-pnl, 0)
                self.closed_net += pnl
                self.q, self.basis, self.active = 0., 0., None
            else:
                self.q += closing_order
        increase = abs(delta)-reduce
        if increase:
            if self.active is None:
                self.active = {'entry_time': time, 'entry_price': execution, 'direction': direction,
                    'cash_flow': 0., 'fees': 0., 'funding_cost': 0., 'max_quantity': 0.}
                self.entry_i = self.i
            self.basis = (abs(self.q)*self.basis+increase*execution)/(abs(self.q)+increase)
            self.q += direction*increase
            self.active['cash_flow'] -= direction*increase*execution
            self.active['fees'] += increase*execution*self.c.fee_bps/10000
            self.active['max_quantity'] = max(self.active['max_quantity'], abs(self.q))

    def exit(self, price, timestamp, reason, bar, final=False):
        if not self.q:
            return
        if bar['count']:
            self.fill(-self.q, price, timestamp, reason, bar['end'])
        elif final:
            raise ValueError('검산 종료 봉 무거래')
        elif self.liquidate is None:
            self.liquidate = reason

    def stop_price(self):
        return self.basis*(1-math.copysign(self.c.stop_fraction, self.q))

    def step(self, bar, decide, final=False):
        c = self.c
        self.fills, self.closed = [], []
        rejected = []
        opening, closing = bar['open'], bar['close']
        today = bar['time'][:10]
        if self.day != today:
            self.day, self.day_start, self.daily = today, self.cash+self.q*opening, False
        payment = self.q*opening*bar['funding_rate']
        self.cash -= payment
        self.funding += payment
        if self.active is not None:
            self.active['funding_cost'] += payment
        opening_equity = self.cash+self.q*opening
        self.peak = max(self.peak, opening_equity)
        if opening_equity <= self.day_start*(1-c.daily_loss_limit):
            self.daily = True
        if opening_equity <= self.peak*(1-c.max_drawdown):
            self.halted = True
        if bar['count'] and self.liquidate:
            reason = self.liquidate
            self.exit(opening, bar['time'], 'liquidity_'+reason, bar)
            if reason in ('gap_stop', 'intrabar_stop', 'time_limit'):
                self.resume_at = self.i+c.cooldown_bars+1
            self.liquidate, self.pending, self.delay = None, None, []
        if self.q and c.stop_fraction and (opening <= self.stop_price() if self.q > 0 else opening >= self.stop_price()):
            self.exit(opening, bar['time'], 'gap_stop', bar)
            self.resume_at = self.i+c.cooldown_bars+1
        if self.q and c.max_hold_bars and self.i-self.entry_i >= c.max_hold_bars:
            self.exit(opening, bar['time'], 'time_limit', bar)
            self.resume_at = self.i+c.cooldown_bars+1
        blocked = self.daily or self.halted
        if blocked:
            self.exit(opening, bar['time'], 'risk_halt', bar)
        request = self.pending
        waiting = request is not None and not bar['count'] and not blocked
        executed = None
        if waiting:
            rejected.append('market_no_trades')
        elif request is not None and not blocked and self.i >= self.resume_at:
            after = reference_target(self.cash, self.q, opening, request, c.fee_bps, c.slippage_bps)
            self.fill(after-self.q, opening, bar['time'], 'target_exposure', bar['end'])
            executed = request
        elif request is not None:
            rejected.append('risk_halt_or_cooldown')
        if self.q and c.stop_fraction and (bar['low'] <= self.stop_price() if self.q > 0 else bar['high'] >= self.stop_price()):
            self.exit(self.stop_price(), bar['end'], 'intrabar_stop', bar)
            self.resume_at = self.i+c.cooldown_bars+1
        if final:
            self.exit(closing, bar['end'], 'end_of_test', bar, final=True)
        equity = self.cash+self.q*closing
        self.peak = max(self.peak, equity)
        drawdown = equity/self.peak-1
        self.max_dd = min(self.max_dd, drawdown)
        decision = None if final or blocked or self.liquidate else decide(bar, self.cash, self.q)
        if final or blocked or self.liquidate:
            self.pending, self.delay = None, []
        elif waiting:
            self.pending = request
        elif c.signal_delay_bars:
            self.delay.append(decision)
            self.pending = self.delay.pop(0) if len(self.delay) > c.signal_delay_bars else None
        else:
            self.pending = decision
        self.i += 1
        return {'time': bar['end'], 'equity': equity, 'quantity': self.q, 'exposure': abs(self.q*closing/equity),
            'net_exposure': self.q*closing/equity, 'drawdown': drawdown, 'halted': self.halted,
            'daily_halted': self.daily, 'requested_target': request, 'executed_target': executed,
            'decision': decision, 'next_target': self.pending, 'fills': self.fills, 'closed_trades': self.closed,
            'rejected': rejected, 'completed': final}


def require_near(actual, expected, context='root'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'검산 필드 불일치: {context}')
        for key in expected:
            require_near(actual[key], expected[key], context+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'검산 행 수 불일치: {context}')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            require_near(a, b, context+f'[{i}]')
    elif type(expected) in (int, float):
        if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f'검산 숫자 불일치: {context}: {actual} != {expected}')
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f'검산 값 불일치: {context}: {actual} != {expected}')
