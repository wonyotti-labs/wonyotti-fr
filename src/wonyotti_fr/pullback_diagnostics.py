from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_json


def verify_fill_activity(fills: pd.DataFrame, bars: pd.DataFrame) -> dict:
    if 'count' not in bars or 'volume' not in bars:
        return {'available': False, 'checked_fills': 0}
    times = pd.to_datetime(fills.time, utc=True) if len(fills) else pd.Series([], dtype='datetime64[ns, UTC]')
    if len(fills):
        times = times - pd.to_timedelta(fills.reason.isin(['intrabar_stop', 'end_of_test']).astype(int), unit='min')
    activity = bars.set_index('time')[['count', 'volume']].reindex(times)
    if activity.isna().any().any() or activity.le(0).any().any():
        raise ValueError('거래가 없는 분봉 또는 입력 밖에서 가정한 체결이 있습니다.')
    return {'available': True, 'checked_fills': len(fills), 'zero_trade_fills': 0}


def waiting_diagnostics(directory: Path, bars: pd.DataFrame, delay: int) -> dict:
    curve = pd.read_parquet(directory / 'equity.parquet', columns=['time', 'policy_event', 'policy_state'])
    fills = pd.read_parquet(directory / 'fills.parquet')
    activity = verify_fill_activity(fills, bars)
    close = bars.set_index('end').close
    decisions, waiting = [], None
    for event in curve[curve.policy_event.isin(['armed', 'triggered', 'expired', 'cleared', 'immediate'])].itertuples(index=False):
        stamp = pd.Timestamp(event.time)
        if event.policy_event == 'armed':
            if waiting is not None:
                raise ValueError('진입 대기가 겹쳤습니다.')
            waiting = json.loads(event.policy_state)
            if pd.Timestamp(waiting['signal_time']) != stamp:
                raise ValueError('진입 대기의 시작 시각 불일치')
            continue
        if event.policy_event == 'immediate':
            decisions.append({'signal_time': stamp, 'decision_time': stamp, 'status': 'immediate',
                              'reference_price': float(close.loc[stamp]), 'decision_close': float(close.loc[stamp]),
                              'direction': None, 'wait_minutes': 0.})
            continue
        if waiting is None:
            raise ValueError('시작 없이 종료된 진입 대기')
        first = pd.Timestamp(waiting['signal_time'])
        decisions.append({'signal_time': first, 'decision_time': stamp, 'status': event.policy_event,
                          'direction': waiting['direction'], 'reference_price': waiting['reference_price'],
                          'decision_close': float(close.loc[stamp]), 'wait_minutes': (stamp - first).total_seconds() / 60})
        waiting = None
    if waiting is not None:
        raise ValueError('완료한 실행에 종료되지 않은 대기 상태가 있습니다.')
    entries = fills[fills.reason.eq('entry')].copy() if len(fills) else pd.DataFrame()
    entry_by_time = {}
    if len(entries):
        entries['stamp'] = pd.to_datetime(entries.time, utc=True)
        if entries.stamp.duplicated().any():
            raise ValueError('같은 시각에 중복 진입 체결')
        entry_by_time = {row.stamp: row for row in entries.itertuples(index=False)}
    matched = set()
    for row in decisions:
        expected = row['decision_time'] + pd.Timedelta(minutes=delay)
        entry = entry_by_time.get(expected) if row['status'] in ('triggered', 'immediate') else None
        row.update(executed=entry is not None, expected_entry_time=expected, entry_price=None,
                   favorable_at_decision_bps=None, favorable_at_fill_bps=None, fill_vs_decision_bps=None)
        if entry is not None:
            direction = int(np.sign(entry.delta_quantity))
            if row['direction'] is not None and row['direction'] != direction:
                raise ValueError('대기 방향과 실제 진입 체결 방향 불일치')
            row['direction'] = direction
            row['entry_price'] = float(entry.price)
            row['favorable_at_fill_bps'] = float(direction * np.log(row['reference_price'] / entry.price) * 10000)
            row['fill_vs_decision_bps'] = float(direction * np.log(row['decision_close'] / entry.price) * 10000)
            matched.add(expected)
        if row['direction'] is not None:
            row['favorable_at_decision_bps'] = float(row['direction'] * np.log(row['reference_price'] / row['decision_close']) * 10000)
    if len(matched) != len(entries):
        raise ValueError('대기·즉시 신호에 연결되지 않은 실제 진입 체결')
    frame = pd.DataFrame(decisions)
    frame.to_parquet(directory / 'waiting_episodes.parquet', index=False)
    counts = {str(key): int(value) for key, value in curve.policy_event.value_counts().items()}
    executed = [row for row in decisions if row['executed']]
    result = {'events': counts, 'fill_activity': activity,
              'completed_waits': sum(row['status'] != 'immediate' for row in decisions),
              'entry_fills': len(entries), 'matched_entry_fills': len(matched),
              'trigger_fraction': counts.get('triggered', 0) / counts['armed'] if counts.get('armed') else None,
              'triggered_without_fill': sum(row['status'] == 'triggered' and not row['executed'] for row in decisions),
              'mean_favorable_at_fill_bps': float(np.mean([row['favorable_at_fill_bps'] for row in executed])) if executed else None,
              'mean_fill_vs_decision_bps': float(np.mean([row['fill_vs_decision_bps'] for row in executed])) if executed else None,
              'limit': '다음 시가·슬리피지 포함 체결 가격 차이이며 수수료·청산·펀딩을 포함한 수익률이 아님'}
    save_json(directory / 'waiting_diagnostics.json', result)
    return result
