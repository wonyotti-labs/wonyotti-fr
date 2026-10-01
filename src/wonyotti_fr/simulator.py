from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 10000.0
    allocation: float = 0.5
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    stop_fraction: float = 0.03
    max_hold_bars: int = 192
    cooldown_bars: int = 4
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.25

    def validate(self):
        values = list(asdict(self).values())
        if not all(np.isfinite(v) for v in values):
            raise ValueError("설정에 유한하지 않은 수치가 있습니다.")
        if self.initial_equity <= 0 or not 0 < self.allocation <= 1:
            raise ValueError("초기 자본은 양수, 노출 비율은 0~1이어야 합니다.")
        if min(self.fee_bps, self.slippage_bps) < 0 or max(self.fee_bps, self.slippage_bps) > 100:
            raise ValueError("비용 가정이 허용 범위 밖입니다.")
        if not 0 <= self.stop_fraction < 1 or not 0 < self.daily_loss_limit <= 1 or not 0 < self.max_drawdown <= 1:
            raise ValueError("잘못된 위험 한도")
        if any(type(v) is not int or v < 0 for v in [self.max_hold_bars, self.cooldown_bars]):
            raise ValueError("보유 시간과 대기 시간은 음수가 아닌 정수여야 합니다.")


def simulate(bars: pd.DataFrame, signals: np.ndarray, funding: pd.DataFrame,
             config: RiskConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    config.validate()
    if len(bars) != len(signals) or len(bars) < 2:
        raise ValueError("시세와 신호 길이가 맞지 않습니다.")
    if not np.isin(signals, [-1, 0, 1]).all():
        raise ValueError("신호는 -1, 0, 1이어야 합니다.")
    if bars.time.duplicated().any() or not bars.time.is_monotonic_increasing:
        raise ValueError("시세는 중복 없이 시간순이어야 합니다.")
    if not ((bars.end - bars.time) == pd.Timedelta(minutes=15)).all():
        raise ValueError("시뮬레이터는 15분봉을 사용합니다.")
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("잘못된 시세")
    if ((prices[:, 1] < prices[:, [0, 2, 3]].max(axis=1))
        | (prices[:, 2] > prices[:, [0, 1, 3]].min(axis=1))).any():
        raise ValueError("OHLC 범위 오류")
    times = bars.time.to_list()
    ends = bars.end.to_list()
    normalized_funding = funding.copy()
    aligned = normalized_funding.time.dt.floor("15min")
    timestamp_offsets = (normalized_funding.time - aligned).dt.total_seconds()
    if (timestamp_offsets > 1).any():
        raise ValueError("펀딩 시각이 봉 경계에서 1초 이상 벗어납니다.")
    # 공시 기록의 1초 미만 처리 지연은 예정 정산 시각으로 정규화한다.
    normalized_funding["time"] = aligned
    relevant = normalized_funding[(normalized_funding.time >= times[0]) & (normalized_funding.time < ends[-1])]
    if relevant.time.duplicated().any() or not np.isfinite(relevant.rate).all():
        raise ValueError("잘못된 펀딩 데이터")
    # 봉 내부 펀딩은 체결 선후를 알 수 없어 현재 모델에서 허용하지 않는다.
    if not relevant.time.isin(bars.time).all():
        raise ValueError("펀딩 시각이 캔들 시작과 일치하지 않습니다. 더 세밀한 데이터가 필요합니다.")
    funding_rates = dict(zip(relevant.time, relevant.rate, strict=True))
    cash = float(config.initial_equity)
    quantity = 0.0
    entry_price = 0.0
    peak = cash
    day_equity = cash
    current_day = None
    daily_halted = False
    permanent_halted = False
    cooldown_until = -1
    entry_index = -1
    trade = None
    curve, trades, fills = [], [], []
    gap_count = 0

    def close_position(index: int, reference: float, reason: str, at_end: bool = False):
        nonlocal cash, quantity, entry_price, trade
        if quantity == 0:
            return
        executed = reference * (1 - np.sign(quantity) * config.slippage_bps / 10000)
        fee = abs(quantity) * executed * config.fee_bps / 10000
        cash += quantity * executed - fee
        assert trade is not None
        trade["exit_time"] = ends[index] if at_end else times[index]
        trade["exit_price"] = executed
        trade["fees"] += fee
        trade["net_pnl"] = quantity * (executed - entry_price) - trade["fees"] - trade["funding_cost"]
        trade["exit_reason"] = reason
        trade["hold_bars"] = index - entry_index + int(at_end)
        trades.append(trade)
        fills.append({"time": trade["exit_time"], "delta_quantity": -quantity,
                      "price": executed, "fee": fee, "reason": reason})
        quantity, entry_price, trade = 0.0, 0.0, None

    for i, (opening, high, low, closing) in enumerate(prices):
        if i and times[i] != ends[i - 1]:
            gap_count += 1
            # 알 수 없는 구간에서는 주문을 만들지 않고 다음 관측 가격으로 포지션을 정리한다.
            close_position(i, opening, "data_gap")
            cooldown_until = i + 256
        equity = cash + quantity * opening
        date = times[i].date()
        if date != current_day:
            current_day, day_equity, daily_halted = date, equity, False
        if times[i] in funding_rates and quantity:
            cost = quantity * opening * float(funding_rates[times[i]])
            cash -= cost
            assert trade is not None
            trade["funding_cost"] += cost
        equity = cash + quantity * opening
        peak = max(peak, equity)
        if equity <= day_equity * (1 - config.daily_loss_limit):
            daily_halted = True
        if equity <= peak * (1 - config.max_drawdown):
            permanent_halted = True
        stop = entry_price * (1 - np.sign(quantity) * config.stop_fraction)
        gap_stop = quantity and config.stop_fraction and ((quantity > 0 and opening <= stop) or (quantity < 0 and opening >= stop))
        if gap_stop:
            close_position(i, opening, "gap_stop")
            cooldown_until = i + config.cooldown_bars + 1
        requested = int(signals[i - 1]) if i else 0
        timed_out = quantity and config.max_hold_bars and i - entry_index >= config.max_hold_bars
        if timed_out:
            close_position(i, opening, "time_limit")
            cooldown_until = i + config.cooldown_bars + 1
        if daily_halted or permanent_halted or i < cooldown_until:
            requested = 0
        if quantity and np.sign(quantity) != requested:
            close_position(i, opening, "risk_halt" if daily_halted or permanent_halted else "signal")
        if not quantity and requested:
            equity = cash
            # 비용을 포함한 진입 금액이 설정한 자본 비율을 넘지 않도록 수량을 정한다.
            executed = opening * (1 + requested * config.slippage_bps / 10000)
            quantity = requested * equity * config.allocation / (executed * (1 + config.fee_bps / 10000))
            fee = abs(quantity) * executed * config.fee_bps / 10000
            cash -= quantity * executed + fee
            entry_price, entry_index = executed, i
            trade = {"entry_time": times[i], "entry_price": executed, "direction": requested,
                     "quantity": abs(quantity), "fees": fee, "funding_cost": 0.0}
            fills.append({"time": times[i], "delta_quantity": quantity, "price": executed, "fee": fee, "reason": "entry"})
        if quantity and config.stop_fraction:
            stop = entry_price * (1 - np.sign(quantity) * config.stop_fraction)
            touched = (quantity > 0 and low <= stop) or (quantity < 0 and high >= stop)
            if touched:
                # 봉 내부 손절 시각은 알 수 없어 봉 종료 시각으로 기록한다.
                close_position(i, stop, "intrabar_stop", at_end=True)
                cooldown_until = i + config.cooldown_bars + 1
        if i == len(prices) - 1:
            close_position(i, closing, "end_of_test", at_end=True)
        equity = cash + quantity * closing
        peak = max(peak, equity)
        curve.append({"time": ends[i], "equity": equity, "quantity": quantity,
                      "exposure": abs(quantity * closing) / equity if equity > 0 else 0.0,
                      "drawdown": equity / peak - 1, "halted": permanent_halted,
                      "close": closing, "signal": int(signals[i])})
        if equity <= 0 or not np.isfinite(equity):
            raise ValueError("순자산이 유효하지 않습니다. 파산/계산 오류를 확인하세요.")
    curve_frame = pd.DataFrame(curve)
    trade_frame = pd.DataFrame(trades)
    metrics = summarize(curve_frame, trade_frame, config.initial_equity)
    metrics["data_gaps"] = gap_count
    metrics["funding_events_in_period"] = len(relevant)
    metrics["max_funding_timestamp_normalization_seconds"] = float(timestamp_offsets.max()) if len(timestamp_offsets) else 0.0
    metrics["accounting_residual"] = float(curve_frame.iloc[-1].equity - config.initial_equity - sum(t["net_pnl"] for t in trades))
    metrics["config"] = asdict(config)
    metrics["limitations"] = ["USDT 선형 계약, 무차입 최대 1배 목표 노출", "시장가 체결 가정, 지정가 대기열 미지원",
                              "펀딩 비용의 기준 가격은 해당 봉 시가이며 실제 마크 가격과 다를 수 있음",
                              "15분봉 내부 가격 순서와 실제 체결 유동성 미관측", "위험 한도는 갭 발생 시 초과될 수 있음"]
    return curve_frame, trade_frame, pd.DataFrame(fills), metrics


def summarize(curve: pd.DataFrame, trades: pd.DataFrame, initial: float) -> dict:
    equity = curve.set_index("time").equity
    days = max((equity.index[-1] - equity.index[0] + pd.Timedelta(minutes=15)).total_seconds() / 86400, 1)
    # 자정 종료 봉은 직전 날짜의 마지막 봉에 속한다.
    daily = equity.groupby((equity.index - pd.Timedelta(nanoseconds=1)).date).last()
    returns = daily / daily.shift(1, fill_value=initial) - 1
    total_return = equity.iloc[-1] / initial - 1
    annual = (1 + total_return) ** (365.25 / days) - 1
    net = trades.net_pnl if len(trades) else pd.Series(dtype=float)
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"start": equity.index[0], "end": equity.index[-1], "bars": len(curve),
            "total_return": float(total_return), "annualized_return": float(annual),
            "max_drawdown": float(curve.drawdown.min()),
            "daily_sharpe": float(returns.mean() / returns.std() * np.sqrt(365.25)) if len(returns) > 1 and returns.std() else None,
            "closed_trades": len(trades), "win_rate": float((net > 0).mean()) if len(net) else None,
            "profit_factor": float(gains / losses) if losses else None,
            "fees": float(trades.fees.sum()) if len(trades) else 0.0,
            "funding_cost": float(trades.funding_cost.sum()) if len(trades) else 0.0,
            "average_exposure": float(curve.exposure.mean()),
            "permanent_halt": bool(curve.halted.any())}
