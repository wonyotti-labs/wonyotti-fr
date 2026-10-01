from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .simulator import RiskConfig

INTENTS = {"hold", "enter_long", "enter_short", "increase", "reduce", "exit"}


@dataclass(frozen=True)
class EngineConfig(RiskConfig):
    bar_seconds: int = 300
    entry_fraction: float = 0.5
    addition_fraction: float = 0.5
    reduction_fraction: float = 0.5
    max_adds: int = 1
    allow_adverse_add: bool = False

    def validate(self):
        super().validate()
        if self.bar_seconds not in (60, 300, 900, 3600):
            raise ValueError("지원하지 않는 사건 간격")
        if any(not np.isfinite(v) or not 0 < v <= 1 for v in
               [self.entry_fraction, self.addition_fraction, self.reduction_fraction]):
            raise ValueError("진입·추가·축소 비율 오류")
        if type(self.max_adds) is not int or not 0 <= self.max_adds <= 10:
            raise ValueError("추가 진입 횟수 오류")
        if type(self.allow_adverse_add) is not bool:
            raise ValueError("불리한 추가 진입 설정 오류")


def validate_bar(bar: dict, seconds: int) -> dict:
    result = dict(bar)
    stamp = pd.Timestamp(bar["time"])
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("시세 시각은 명시적인 UTC여야 합니다.")
    if stamp.value % (seconds * 10**9):
        raise ValueError("시세 시작이 봉 경계와 다릅니다.")
    if "end" in bar and pd.Timestamp(bar["end"]) != stamp + pd.Timedelta(seconds=seconds):
        raise ValueError("시세 종료가 설정한 봉 간격과 다릅니다.")
    prices = [float(bar[k]) for k in ["open", "high", "low", "close"]]
    opening, high, low, closing = prices
    if not np.isfinite(prices).all() or min(prices) <= 0 or high < max(prices) or low > min(prices):
        raise ValueError("유효하지 않은 OHLC 시세")
    rate = float(bar.get("funding_rate", 0))
    if not np.isfinite(rate) or abs(rate) > 1:
        raise ValueError("유효하지 않은 펀딩률")
    result.update(dict(zip(["open", "high", "low", "close"], prices, strict=True)))
    result.update(time=stamp.isoformat(), end=(stamp + pd.Timedelta(seconds=seconds)).isoformat(), funding_rate=rate)
    return result


class TradingEngine:
    def __init__(self, config: EngineConfig, state: dict | None = None):
        config.validate()
        self.config = config
        self.state = copy.deepcopy(state) if state is not None else {
            "version": 1, "cash": config.initial_equity, "quantity": 0.0, "entry_price": 0.0,
            "peak": config.initial_equity, "day_equity": config.initial_equity,
            "day": None, "daily_halted": False, "permanent_halted": False, "manual_halt": False,
            "cooldown_until": -1, "entry_index": -1, "index": 0, "last_end": None,
            "last_close": None, "pending": "hold", "active_trade": None, "completed": False,
            "total_fees": 0.0, "total_funding": 0.0, "closed_net": 0.0, "closed_trades": 0,
            "max_drawdown": 0.0, "wins": 0, "sum_gains": 0.0, "sum_losses": 0.0,
        }
        self._validate_state()

    def _validate_state(self):
        s = self.state
        if s.get("version") != 1 or s.get("pending") not in INTENTS:
            raise ValueError("상태 형식 또는 대기 주문 오류")
        for key in ["cash", "quantity", "entry_price", "peak", "day_equity", "total_fees", "total_funding", "closed_net"]:
            if not np.isfinite(s[key]):
                raise ValueError("상태에 유한하지 않은 숫자가 있습니다.")
        if (s["quantity"] == 0) != (s["active_trade"] is None):
            raise ValueError("보유 수량과 거래 상태가 다릅니다.")
        if s["completed"] and s["quantity"] != 0:
            raise ValueError("완료 상태에 미청산 수량이 있습니다.")

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def view(self, price: float) -> dict:
        s = self.state
        direction = int(np.sign(s["quantity"]))
        return {"direction": direction, "equity": s["cash"] + s["quantity"] * price,
                "favorable_move": direction * (price / s["entry_price"] - 1) if direction else 0.0,
                "hold_bars": s["index"] - s["entry_index"] if direction else 0,
                "adds": s["active_trade"]["adds"] if direction else 0,
                "pending": s["pending"], "halted": s["permanent_halted"] or s["manual_halt"]}

    def halt(self):
        self.state["manual_halt"] = True
        self.state["pending"] = "hold"

    def step(self, bar: dict, decide: Callable[[dict, dict], str], final: bool = False) -> dict:
        event = validate_bar(bar, self.config.bar_seconds)
        # 실패한 입력이나 정책 실행은 직전 정상 상태를 바꾸지 않는다.
        trial = TradingEngine(self.config, self.state)
        result = trial._advance(event, decide, final)
        trial._validate_state()
        self.state = trial.state
        return result

    def _advance(self, bar: dict, decide: Callable[[dict, dict], str], final: bool) -> dict:
        s, c = self.state, self.config
        if s["completed"]:
            raise ValueError("완료된 실행에 새 시세를 추가할 수 없습니다.")
        if s["last_end"] is not None and pd.Timestamp(bar["time"]) != pd.Timestamp(s["last_end"]):
            raise ValueError("시세가 중복·역순이거나 누락됐습니다.")
        i, opening, high, low, closing = s["index"], bar["open"], bar["high"], bar["low"], bar["close"]
        fills, closed, rejected = [], [], []

        def reduce_position(amount: float, price: float, reason: str, at_end: bool = False):
            direction = int(np.sign(s["quantity"]))
            amount = min(amount, abs(s["quantity"]))
            if amount <= 0:
                return
            execution = price * (1 - direction * c.slippage_bps / 10000)
            fee = amount * execution * c.fee_bps / 10000
            s["cash"] += direction * amount * execution - fee
            trade = s["active_trade"]
            trade["gross_realized"] += direction * amount * (execution - s["entry_price"])
            trade["fees"] += fee
            s["total_fees"] += fee
            s["quantity"] -= direction * amount
            timestamp = bar["end"] if at_end else bar["time"]
            fills.append({"time": timestamp, "delta_quantity": -direction * amount,
                          "price": execution, "fee": fee, "reason": reason})
            if abs(s["quantity"]) < 1e-12:
                s["quantity"] = 0.0
                net = trade["gross_realized"] - trade["fees"] - trade["funding_cost"]
                trade.update(exit_time=timestamp, exit_price=execution, exit_reason=reason,
                             net_pnl=net, hold_bars=i - s["entry_index"] + int(at_end))
                s["closed_net"] += net
                s["closed_trades"] += 1
                s["wins"] += int(net > 0)
                s["sum_gains"] += max(net, 0)
                s["sum_losses"] += max(-net, 0)
                closed.append(copy.deepcopy(trade))
                s["active_trade"], s["entry_price"] = None, 0.0

        def increase_position(direction: int, fraction: float, adding: bool = False):
            equity = s["cash"] + s["quantity"] * opening
            capacity = max(0.0, equity * c.allocation - abs(s["quantity"] * opening))
            notional = min(equity * c.allocation * fraction, capacity)
            if notional <= max(1e-8, equity * 1e-10):
                rejected.append("allocation_cap")
                return
            execution = opening * (1 + direction * c.slippage_bps / 10000)
            amount = notional / (execution * (1 + c.fee_bps / 10000))
            fee = amount * execution * c.fee_bps / 10000
            before = abs(s["quantity"])
            s["entry_price"] = (before * s["entry_price"] + amount * execution) / (before + amount)
            s["quantity"] += direction * amount
            s["cash"] -= direction * amount * execution + fee
            s["total_fees"] += fee
            if s["active_trade"] is None:
                s["entry_index"] = i
                s["active_trade"] = {"entry_time": bar["time"], "entry_price": execution,
                                     "direction": direction, "initial_quantity": amount, "max_quantity": amount,
                                     "fees": 0.0, "funding_cost": 0.0, "gross_realized": 0.0, "adds": 0}
            trade = s["active_trade"]
            trade["fees"] += fee
            trade["adds"] += int(adding)
            trade["max_quantity"] = max(trade["max_quantity"], abs(s["quantity"]))
            fills.append({"time": bar["time"], "delta_quantity": direction * amount,
                          "price": execution, "fee": fee, "reason": "increase" if adding else "entry"})

        date = pd.Timestamp(bar["time"]).date().isoformat()
        equity = s["cash"] + s["quantity"] * opening
        if date != s["day"]:
            s["day"], s["day_equity"], s["daily_halted"] = date, equity, False
        if s["quantity"] and bar["funding_rate"]:
            payment = s["quantity"] * opening * bar["funding_rate"]
            s["cash"] -= payment
            s["active_trade"]["funding_cost"] += payment
            s["total_funding"] += payment
        equity = s["cash"] + s["quantity"] * opening
        s["peak"] = max(s["peak"], equity)
        s["daily_halted"] |= equity <= s["day_equity"] * (1 - c.daily_loss_limit)
        s["permanent_halted"] |= equity <= s["peak"] * (1 - c.max_drawdown)
        direction = int(np.sign(s["quantity"]))
        stop = s["entry_price"] * (1 - direction * c.stop_fraction)
        if direction and c.stop_fraction and ((direction > 0 and opening <= stop) or (direction < 0 and opening >= stop)):
            reduce_position(abs(s["quantity"]), opening, "gap_stop")
            s["cooldown_until"] = i + c.cooldown_bars + 1
        if s["quantity"] and c.max_hold_bars and i - s["entry_index"] >= c.max_hold_bars:
            reduce_position(abs(s["quantity"]), opening, "time_limit")
            s["cooldown_until"] = i + c.cooldown_bars + 1
        blocked = s["daily_halted"] or s["permanent_halted"] or s["manual_halt"]
        if blocked:
            reduce_position(abs(s["quantity"]), opening, "manual_halt" if s["manual_halt"] else "risk_halt")
        intent = s["pending"]
        if not blocked and i >= s["cooldown_until"]:
            if intent == "exit":
                reduce_position(abs(s["quantity"]), opening, "signal_exit")
            elif intent == "reduce":
                reduce_position(abs(s["quantity"]) * c.reduction_fraction, opening, "signal_reduce")
            elif intent in {"enter_long", "enter_short"}:
                requested = 1 if intent == "enter_long" else -1
                if s["quantity"] and int(np.sign(s["quantity"])) != requested:
                    reduce_position(abs(s["quantity"]), opening, "signal_reverse")
                if not s["quantity"]:
                    increase_position(requested, c.entry_fraction)
            elif intent == "increase" and s["quantity"]:
                favorable = np.sign(s["quantity"]) * (opening / s["entry_price"] - 1)
                if s["active_trade"]["adds"] >= c.max_adds:
                    rejected.append("max_adds")
                elif favorable <= 0 and not c.allow_adverse_add:
                    rejected.append("adverse_add_blocked")
                else:
                    increase_position(int(np.sign(s["quantity"])), c.addition_fraction, adding=True)
        elif intent != "hold":
            rejected.append("risk_halt_or_cooldown")
        direction = int(np.sign(s["quantity"]))
        if direction and c.stop_fraction:
            stop = s["entry_price"] * (1 - direction * c.stop_fraction)
            if (direction > 0 and low <= stop) or (direction < 0 and high >= stop):
                reduce_position(abs(s["quantity"]), stop, "intrabar_stop", at_end=True)
                s["cooldown_until"] = i + c.cooldown_bars + 1
        if final:
            reduce_position(abs(s["quantity"]), closing, "end_of_test", at_end=True)
            s["completed"] = True
        equity = s["cash"] + s["quantity"] * closing
        if equity <= 0 or not np.isfinite(equity):
            raise ValueError("파산 또는 유효하지 않은 잔고입니다. 해당 실험을 실패로 기록하세요.")
        s["peak"] = max(s["peak"], equity)
        drawdown = equity / s["peak"] - 1
        s["max_drawdown"] = min(s["max_drawdown"], drawdown)
        s["last_end"], s["last_close"] = bar["end"], closing
        s["index"] += 1
        next_intent = "hold" if final or blocked else decide(bar, self.view(closing))
        if next_intent not in INTENTS:
            raise ValueError("정책이 지원하지 않는 주문 의도를 반환했습니다.")
        s["pending"] = next_intent
        unrealized = s["quantity"] * (closing - s["entry_price"])
        active_net = (s["active_trade"]["gross_realized"] - s["active_trade"]["fees"]
                      - s["active_trade"]["funding_cost"]) if s["active_trade"] else 0.0
        residual = equity - c.initial_equity - s["closed_net"] - active_net - unrealized
        if abs(residual) > max(1e-7, equity * 1e-10):
            raise ValueError("거래별 손익과 순자산 회계가 일치하지 않습니다.")
        return {"time": bar["end"], "equity": equity, "quantity": s["quantity"],
                "exposure": abs(s["quantity"] * closing) / equity, "drawdown": drawdown,
                "halted": s["permanent_halted"] or s["manual_halt"], "daily_halted": s["daily_halted"],
                "executed_intent": intent, "next_intent": next_intent, "fills": fills, "closed_trades": closed,
                "rejected": rejected, "accounting_residual": residual, "completed": final}

    def settings(self) -> dict:
        return asdict(self.config)
