from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .audit import aggregate_events
from .common import new_run, records, save_json, sha256
from .expansion_data import source_inputs
from .reports import table
from .timing_study import read_minute_history

SCENARIOS = {
    'original': {},
    'fees_5bp': {'fee_bps': 5},
    'next_minute_price': {'replace_price': True},
    'no_adds': {'no_adds': True},
    'cap_30m': {'cap_minutes': 30},
    'no_adds_cap_30m': {'no_adds': True, 'cap_minutes': 30},
    'bot_execution': {'replace_price': True, 'fee_bps': 5},
    'combined': {'no_adds': True, 'cap_minutes': 30, 'replace_price': True, 'fee_bps': 5},
}


def episode_legs(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or not events.time.is_monotonic_increasing:
        raise ValueError('원본 사건은 비어 있지 않은 시간순 자료여야 합니다.')
    rows, quantity, number = [], 0, 0
    for row in events.itertuples(index=False):
        if row.exectype == 'Funding':
            if not quantity or abs(quantity) != abs(row.quantity):
                raise ValueError('펀딩 수량과 직전 포지션 불일치')
            rows.append((number, row.time, row.order_key, 'funding', 0, abs(quantity),
                         np.sign(quantity), 0., float(row.fee_satoshi) / 1e8))
            continue
        if (row.exectype != 'Trade' or row.side not in {'Buy', 'Sell'} or row.quantity <= 0
            or not np.isfinite([row.quantity, row.cost_satoshi, row.fee_satoshi]).all()
            or row.cost_satoshi == 0):
            raise ValueError('역선물 거래 사건의 종류·수량·원가 오류')
        change = (1 if row.side == 'Buy' else -1) * int(row.quantity)
        unit = abs(float(row.cost_satoshi)) / row.quantity / 1e8
        fee_unit = float(row.fee_satoshi) / row.quantity / 1e8
        if quantity and np.sign(quantity) != np.sign(change):
            closing = min(abs(quantity), abs(change))
            rows.append((number, row.time, row.order_key, 'reduce', closing, abs(quantity),
                         np.sign(quantity), unit, fee_unit * closing))
            quantity += int(np.sign(change)) * closing
            change -= int(np.sign(change)) * closing
        if change:
            if quantity == 0:
                number += 1
            rows.append((number, row.time, row.order_key, 'increase', abs(change), abs(quantity),
                         np.sign(change), unit, fee_unit * abs(change)))
            quantity += change
    result = pd.DataFrame(rows, columns=['episode_id', 'time', 'order_key', 'kind', 'quantity',
                                        'original_before', 'direction', 'unit_btc', 'fee_btc'])
    result['time'] = result.time.astype('datetime64[ns, UTC]')
    return result


def next_prices(times: pd.Series, minute: pd.DataFrame, *, strict: bool) -> tuple[np.ndarray, np.ndarray]:
    ends = pd.to_datetime(minute.end, utc=True).astype('datetime64[ns, UTC]').array.asi8
    requested = pd.to_datetime(times, utc=True).astype('datetime64[ns, UTC]').array.asi8
    if (len(ends) < 2 or np.any(np.diff(ends) != 60 * 10**9)
        or not np.isfinite(minute.close).all() or minute.close.le(0).any()):
        raise ValueError('분봉 종가의 연속성·유한성 오류')
    offsets = np.searchsorted(ends, requested, side='right' if strict else 'left')
    if (offsets >= len(ends)).any():
        raise ValueError('설명용 체결가의 이후 분봉 부족')
    delays = (ends[offsets] - requested) / 1e9
    if (delays < 0).any() or (delays > 60).any() or (strict and (delays <= 0).any()):
        raise ValueError('설명용 체결가의 시점 경계 오류')
    return minute.close.to_numpy(dtype=float)[offsets], ends[offsets]


def replay_episode(legs: pd.DataFrame, *, no_adds: bool = False, cap_minutes: int | None = None,
                   replace_price: bool = False, fee_bps: float | None = None,
                   cap_price: float | None = None, cap_time: pd.Timestamp | None = None) -> dict:
    if (not legs.time.is_monotonic_increasing or legs.empty
        or legs.episode_id.nunique() != 1 or legs.kind.iloc[0] != 'increase'
        or fee_bps not in (None, 5) or cap_minutes not in (None, 30)):
        raise ValueError('포지션 설명 대조의 입력·설정 오류')
    if cap_minutes is not None and (cap_time is None or cap_price is None or not np.isfinite(cap_price) or cap_price <= 0):
        raise ValueError('시간 제한 청산의 분봉 근거 부족')
    first_order = legs.order_key.iloc[0]
    first_qty = float(legs.loc[legs.kind.eq('increase') & legs.order_key.eq(first_order), 'quantity'].sum())
    quantity = basis = gross = fees = funding = 0.
    direction = int(legs.direction.iloc[0])
    truncated = False

    def force_close():
        nonlocal quantity, gross, fees, truncated
        execution = cap_price * (1 - direction * 3 / 10000)
        unit = 1 / execution
        gross += direction * quantity * (basis - unit)
        fees += quantity * unit * 5 / 10000
        quantity = 0.
        truncated = True

    for row in legs.itertuples(index=False):
        # 같은 시각의 펀딩은 시간 제한 청산보다 먼저 정산한다.
        if (cap_minutes is not None and quantity and
            (row.time > cap_time or (row.time == cap_time and row.kind != 'funding'))):
            force_close()
            break
        if row.kind == 'funding':
            funding += row.fee_btc * quantity / row.original_before
            continue
        if row.kind == 'increase':
            if no_adds and row.order_key != first_order:
                continue
            amount = row.quantity
            trade_direction = direction
        else:
            amount = quantity * row.quantity / row.original_before
            trade_direction = -direction
        unit = row.unit_btc
        if replace_price:
            unit = 1 / (row.next_price * (1 + trade_direction * 3 / 10000))
        if row.kind == 'increase':
            basis = (basis * quantity + unit * amount) / (quantity + amount)
            quantity += amount
        else:
            gross += direction * amount * (basis - unit)
            quantity -= amount
            if row.quantity == row.original_before:
                quantity = 0.
        if fee_bps is None:
            # 가격만 바꾸는 대조에서는 원본 유효 수수료율을 유지한다.
            fees += row.fee_btc * amount / row.quantity * unit / row.unit_btc
        else:
            fees += amount * unit * fee_bps / 10000
    if quantity != 0:
        raise ValueError('종료 포지션의 설명 대조에 잔여 수량이 있습니다.')
    net = gross - fees - funding
    return {'gross_btc': gross, 'fees_btc': fees, 'funding_btc': funding, 'net_btc': net,
            'first_order_contracts': first_qty, 'net_btc_per_1000_first_order_contracts': net * 1000 / first_qty,
            'time_cap_applied': truncated}


def run_structure_study(audit: Path, study: Path, history: Path, minute_history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    minute, minute_hashes = read_minute_history(minute_history, history)
    destination = new_run(output, 'structure-study', {
        **hashes, **minute_hashes, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V9.md')),
        'role': 'teacher_action_counterfactual_not_executable_strategy',
        'scenarios': SCENARIOS, 'prices': 'first_complete_minute_close_after_source_fill',
        'funding': 'original_event_schedule_scaled_to_counterfactual_quantity',
    })
    print(f'원본 구조 대조: {destination}', flush=True)
    try:
        events = aggregate_events(source['executions'], 'XBTUSD')
        legs = episode_legs(events)
        trade = legs.kind.ne('funding')
        prices, stamps = next_prices(legs.loc[trade, 'time'], minute, strict=True)
        legs.loc[trade, 'next_price'] = prices
        legs.loc[trade, 'reference_delay_seconds'] = (stamps - legs.loc[trade, 'time'].array.asi8) / 1e9
        episodes = source['episodes'].set_index('episode_id')
        closed = episodes[episodes.closed]
        cap_requested = closed.entry_time + pd.Timedelta(minutes=30)
        cap_prices, cap_stamps = next_prices(cap_requested, minute, strict=False)
        cap_lookup = dict(zip(closed.index, zip(cap_prices, cap_stamps, strict=True), strict=True))
        rows = []
        max_error = 0.
        for identifier, part in legs.groupby('episode_id', sort=True):
            episode = episodes.loc[identifier]
            if not episode.closed:
                continue
            price, stamp = cap_lookup[identifier]
            for name, settings in SCENARIOS.items():
                result = replay_episode(part, **settings, cap_price=price,
                                        cap_time=pd.Timestamp(stamp, tz='UTC'))
                if name == 'original':
                    for key, original in [('net_btc', 'net_pnl_btc'), ('gross_btc', 'gross_btc'),
                                          ('fees_btc', 'fees_btc'), ('funding_btc', 'funding_btc')]:
                        max_error = max(max_error, abs(result[key] - episode[original]))
                    if max_error > 1e-7:
                        raise ValueError('원본 포지션별 회계와 설명 재생 불일치')
                rows.append({'episode_id': int(identifier), 'scenario': name, 'exit_year': episode.exit_time.year,
                             'direction': int(episode.direction), 'original_hold_minutes': episode.hold_minutes,
                             'original_profitable': bool(episode.net_pnl_btc > 0), **result})
        frame = pd.DataFrame(rows)
        if len(frame) != len(closed) * len(SCENARIOS):
            raise ValueError('설명 대조의 종료 포지션 누락')
        original = frame[frame.scenario.eq('original')].set_index('episode_id')
        frame['delta_from_original_btc'] = frame.net_btc - frame.episode_id.map(original.net_btc)
        frame['hold_band'] = pd.cut(frame.original_hold_minutes, [0, 30, 120, 1440, np.inf],
                                   labels=['0-30m', '30m-2h', '2h-1d', '1d+'], include_lowest=True).astype(str)
        frame.to_parquet(destination / 'episode_comparisons.parquet', index=False)
        legs.to_parquet(destination / 'source_legs.parquet', index=False)
        summary = frame.groupby('scenario', sort=False).agg(
            episodes=('episode_id', 'size'), net_btc=('net_btc', 'sum'),
            delta_btc=('delta_from_original_btc', 'sum'), fees_btc=('fees_btc', 'sum'),
            funding_btc=('funding_btc', 'sum'), capped_episodes=('time_cap_applied', 'sum'),
            mean_normalized_btc=('net_btc_per_1000_first_order_contracts', 'mean'),
            median_normalized_btc=('net_btc_per_1000_first_order_contracts', 'median')).reset_index()
        summary.to_csv(destination / 'summary.csv', index=False)
        breakdown = frame.groupby(['scenario', 'exit_year', 'direction', 'hold_band', 'original_profitable'],
                                  observed=True).agg(episodes=('episode_id', 'size'), net_btc=('net_btc', 'sum'),
                                                     delta_btc=('delta_from_original_btc', 'sum')).reset_index()
        breakdown.to_csv(destination / 'breakdown.csv', index=False)
        save_json(destination / 'verification.json', {
            'complete': True, 'closed_episodes': len(closed), 'open_episodes_preserved': int((~episodes.closed).sum()),
            'scenarios': len(SCENARIOS), 'max_original_accounting_error_btc': max_error,
            'source_funding_and_reversal_validated': True, 'all_losses_retained': True,
            'summary': records(summary), 'deployable_strategy': False,
            'reference_delay_seconds': {'min': float(legs.reference_delay_seconds.min()),
                                        'max': float(legs.reference_delay_seconds.max())},
        })
        (destination / 'REPORT.md').write_text(
            '# 원본 수익 구조의 설명용 대조\n\n' + table(summary) + '\n\n'
            '원본 진입·추가·축소·청산 시각과 방향을 이미 아는 설명용 재생이다. 예측 가능한 전략이나 계좌 수익률이 아니다. '
            '원본 종료 포지션의 회계를 먼저 대조하고 같은 포지션의 조건만 바꿨다. 손실도 전부 유지했다.\n\n'
            'no_adds는 첫 주문의 부분 체결을 유지하고 다른 추가 주문만 제거한다. 원본 청산 비율과 펀딩 시점에 가상 잔량을 적용한다. '
            'cap_30m은 30분 이후 첫 확정 분 종가에서 잔량을 5bp 수수료·3bp 슬리피지로 청산한다. '
            'next_minute_price는 원본 사건 시각과 펀딩 일정은 유지하고 가격만 이후 첫 분 종가·3bp로 바꾼다. '
            '이는 가격 대체 대조이며 지연 때문에 새로 생기는 펀딩 노출·호가 대기열을 재현하지 않는다. '
            'bot_execution은 가격 대체와 5bp 수수료, combined는 추가 제거·시간 제한까지 적용한다.\n\n'
            'BTC 금액에는 시기별 원래 규모가 섞인다. 정규화는 첫 진입 주문 전체 체결 1,000계약당 손익의 평균·중앙값이다. '
            '첫 주문의 최종 체결량은 사후 정규화에만 사용했으며 초기 자본·레버리지·실행 가능한 수익률로 해석하지 않는다. '
            '결합 효과를 개별 효과의 단순 합이나 현실의 인과 효과로 보지 않는다. 제한을 없애면 미래에 수익이 난다는 증거도 아니다.\n\n'
            '포지션별 결과는 episode_comparisons.parquet, 연도·방향·원래 승패·보유 구간별 결과는 breakdown.csv에 보존했다. '
            '마지막 미청산 포지션은 종료 비교에 넣지 않고 원본에 유지했다. 이 결과를 확인한 뒤 관리 정책의 후보를 별도 고정한다.\n',
            encoding='utf-8')
        save_json(destination / 'output_hashes.json', {p.name: sha256(p) for p in destination.iterdir() if p.is_file()})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return destination
