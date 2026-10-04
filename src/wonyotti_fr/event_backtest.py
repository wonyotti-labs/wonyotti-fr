from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256
from .engine import EngineConfig, TradingEngine
from .event_features import MARKET_FEATURES, STATE_FEATURES, event_features
from .event_model import EventModel
from .research import check_funding_coverage, load_market


def read_models(study: Path) -> tuple[EventModel, EventModel]:
    hashes = json.loads((study / 'files.json').read_text())
    models = []
    for name, features in [('entry_model.json', MARKET_FEATURES), ('management_model.json', MARKET_FEATURES + STATE_FEATURES)]:
        path = study / name
        if path.stat().st_size > 1024 * 1024 or sha256(path) != hashes[name]:
            raise ValueError('모델 크기 또는 입력 해시 오류')
        model = EventModel.from_dict(json.loads(path.read_text()))
        if model.features != features:
            raise ValueError('진입·관리 모델의 특징 구성이 바뀌었습니다.')
        models.append(model)
    return tuple(models)


def candidate_plan() -> list[dict]:
    result = []
    for entry in [0.5, 0.65]:
        for management in [0.35, 0.5]:
            for stop, hold in [(0.02, 24), (0.04, 72), (0.04, 288)]:
                result.append({'entry_threshold': entry, 'management_threshold': management,
                               'risk': asdict(EngineConfig(stop_fraction=stop, max_hold_bars=hold, cooldown_bars=3))})
    return result


class EventPolicy:
    def __init__(self, entry: EventModel, management: EventModel, entry_threshold: float,
                 management_threshold: float, baseline: str | None = None):
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in [entry_threshold, management_threshold]):
            raise ValueError('정책 신뢰도는 0~1이어야 합니다.')
        if baseline not in {None, 'cash', 'trend', 'mean_reversion'}:
            raise ValueError('알 수 없는 기준 전략입니다.')
        self.entry, self.management = entry, management
        self.entry_threshold, self.management_threshold = entry_threshold, management_threshold
        self.baseline = baseline

    def __call__(self, bar: dict, state: dict) -> str:
        if self.baseline == 'cash' or state['halted']:
            return 'hold'
        market = np.asarray(bar['features'], dtype=float)
        if market.shape != (len(MARKET_FEATURES),) or not np.isfinite(market).all():
            return 'hold'
        direction = state['direction']
        if self.baseline is not None:
            values = dict(zip(MARKET_FEATURES, market, strict=True))
            if self.baseline == 'trend':
                signs = [np.sign(values[name]) for name in ['trend_1h_4h', 'trend_4h_16h']]
                desired = int(signs[0]) if signs[0] == signs[1] else 0
                return ('enter_long' if desired > 0 else 'enter_short') if desired and desired != direction else ('exit' if not desired else 'hold')
            if not direction:
                move = values['ret_15m']
                return 'enter_long' if move < -0.005 else ('enter_short' if move > 0.005 else 'hold')
            return 'exit' if direction * values['ret_15m'] > 0 else 'hold'
        if direction:
            state_values = [direction, state['favorable_move'], np.log1p(state['hold_bars'] * 5), min(state['adds'], 5)]
            model, threshold = self.management, self.management_threshold
            features = np.concatenate([market, state_values])
        else:
            model, threshold, features = self.entry, self.entry_threshold, market
        probs = model.probabilities(features.reshape(1, -1))[0]
        allowed = {'hold', 'increase', 'reduce', 'exit', 'enter_short' if direction > 0 else 'enter_long'} if direction else {'hold', 'enter_long', 'enter_short'}
        # 부적합한 행동을 제외하되 확률을 다시 키우지 않는다.
        eligible = np.array([p if label in allowed else 0.0 for label, p in zip(model.classes, probs, strict=True)])
        index = int(eligible.argmax())
        return model.classes[index] if eligible[index] >= threshold else 'hold'


def prepare_period(market: Path, symbol: str, start: str, end: str) -> pd.DataFrame:
    bars, funding = load_market(market, symbol, '5m')
    check_funding_coverage(market, symbol, start, end, '5m')
    features = event_features(bars)
    first, last = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    if first >= last or first.value % (300 * 10**9) or last.value % (300 * 10**9):
        raise ValueError('평가 기간은 순서가 맞는 5분봉 경계여야 합니다.')
    mask = (bars.time >= first) & (bars.time < last)
    selected = bars[mask].reset_index(drop=True).copy()
    expected = pd.date_range(first, last, freq='5min', inclusive='left')
    if len(selected) != len(expected) or not np.array_equal(selected.time.astype('datetime64[ns, UTC]').array.asi8, expected.as_unit('ns').asi8):
        raise ValueError('평가 기간의 시세가 빠졌거나 중복됐습니다.')
    aligned = funding.time.dt.floor('5min')
    offsets = (funding.time - aligned).dt.total_seconds()
    if ((offsets < 0) | (offsets >= 1)).any() or not np.isfinite(funding.rate).all():
        raise ValueError('펀딩 시각 또는 값이 유효하지 않습니다.')
    normalized = funding.assign(time=aligned)
    relevant = normalized[(normalized.time >= first) & (normalized.time < last)]
    if relevant.time.duplicated().any() or not relevant.time.isin(selected.time).all():
        raise ValueError('펀딩이 중복되거나 시세 경계와 다릅니다.')
    selected['funding_rate'] = selected.time.map(relevant.set_index('time').rate).fillna(0)
    selected[MARKET_FEATURES] = features.loc[mask, MARKET_FEATURES].to_numpy()
    return selected


def iter_events(data: pd.DataFrame):
    from .minute_inputs import MINUTE_FEATURES
    names = ['time', 'end', 'open', 'high', 'low', 'close', 'funding_rate']
    if {'count', 'volume'} <= set(data.columns):
        names += ['count', 'volume']
    extra = MINUTE_FEATURES if set(MINUTE_FEATURES) <= set(data.columns) else []
    if set(MINUTE_FEATURES) & set(data.columns) and not extra:
        raise ValueError('관리용 확정 분봉 특징의 일부 누락')
    for row in data[names + MARKET_FEATURES + extra].itertuples(index=False, name=None):
        event = dict(zip(names, row[:len(names)], strict=True))
        event['time'], event['end'] = event['time'].isoformat(), event['end'].isoformat()
        split = len(names) + len(MARKET_FEATURES)
        event['features'] = [float(x) if np.isfinite(x) else None for x in row[len(names):split]]
        if extra:
            event['minute_features'] = [float(x) if np.isfinite(x) else None for x in row[split:]]
        yield event


def summarize(engine: TradingEngine, curve: pd.DataFrame, rejected: dict) -> dict:
    daily = curve.set_index(pd.to_datetime(curve.time, utc=True)).equity.resample('1D').last()
    return summarize_observations(engine, len(curve), float(curve.equity.iloc[-1]),
                                  float(curve.exposure.mean()), float(curve.accounting_residual.abs().max()), daily, rejected)


def summarize_observations(engine: TradingEngine, bars: int, final: float, average_exposure: float,
                           max_residual: float, daily: pd.Series, rejected: dict) -> dict:
    state, config = engine.state, engine.config
    elapsed_years = bars * config.bar_seconds / (365.25 * 24 * 3600)
    daily_returns = daily.pct_change().dropna()
    deviation = daily_returns.std()
    try:
        annual = (final / config.initial_equity) ** (1 / elapsed_years) - 1 if final >= 0 else None
    except OverflowError:
        annual = None
    # 짧은 급등 구간의 연율화 불능이 전체 체결·손익 저장을 중단하지 않게 한다.
    if annual is not None and not np.isfinite(annual):
        annual = None
    return {'total_return': final / config.initial_equity - 1,
            'annualized_return': annual,
            'max_drawdown': state['max_drawdown'], 'closed_trades': state['closed_trades'],
            'win_rate': state['wins'] / state['closed_trades'] if state['closed_trades'] else None,
            'profit_factor': state['sum_gains'] / state['sum_losses'] if state['sum_losses'] else None,
            'daily_sharpe': float(np.sqrt(365.25) * daily_returns.mean() / deviation) if deviation > 0 else None,
            'fees': state['total_fees'], 'funding_cost': state['total_funding'],
            'average_exposure': average_exposure, 'permanent_halt': state['permanent_halted'],
            'max_accounting_residual': max_residual,
            'rejected': rejected, 'final_equity': final, 'bars': bars}


def backtest(data: pd.DataFrame, policy: EventPolicy, config: EngineConfig, output: Path | None = None,
             *, streaming: bool = True, batch_size: int = 8192) -> dict:
    if len(data) < 2:
        raise ValueError('백테스트 시세가 부족합니다.')
    if streaming:
        from .streaming_backtest import streaming_backtest
        return streaming_backtest(data, policy, config, output, batch_size)
    if hasattr(policy, 'prepare'):
        policy.prepare(data)
    engine = TradingEngine(config)
    curve, trades, fills, rejected = [], [], [], {}
    for index, event in enumerate(iter_events(data)):
        result = engine.step(event, policy, final=index == len(data) - 1)
        trades.extend(result.pop('closed_trades'))
        fills.extend(result.pop('fills'))
        for reason in result.pop('rejected'):
            rejected[reason] = rejected.get(reason, 0) + 1
        curve.append(result)
    frame = pd.DataFrame(curve)
    metrics = summarize(engine, frame, rejected)
    if output is not None:
        from .common import save_json
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'config.json', asdict(config))
        frame.to_parquet(output / 'equity.parquet', index=False)
        pd.DataFrame(trades).to_parquet(output / 'trades.parquet', index=False)
        pd.DataFrame(fills).to_parquet(output / 'fills.parquet', index=False)
        save_json(output / 'metrics.json', metrics)
        save_json(output / 'final_state.json', engine.snapshot())
    return metrics
