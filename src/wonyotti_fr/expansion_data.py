from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256
from .event_features import independent_orders


def source_inputs(audit: Path, study: Path, history: Path) -> tuple[dict, dict]:
    manifest = json.loads((study / 'manifest.json').read_text())['settings']
    names = ['executions', 'actions', 'episodes']
    hashes = {f'{name}.parquet': sha256(audit / f'{name}.parquet') for name in names}
    if any(hashes[key] != manifest['audit_sha256'][key] for key in hashes):
        raise ValueError('감사 입력과 사건 연구의 지문이 다릅니다.')
    files = json.loads((study / 'files.json').read_text())
    training = study / 'training_events.parquet'
    market = json.loads((history / 'manifest.json').read_text())
    market_path = (history / market['file']).resolve()
    if (not market_path.is_relative_to(history.resolve()) or sha256(market_path) != market['sha256']
        or sha256(training) != files[training.name]):
        raise ValueError('시장·학습 자료의 경로 또는 지문 오류')
    values = {name: pd.read_parquet(audit / f'{name}.parquet') for name in names}
    values.update(bars=pd.read_parquet(market_path), events=pd.read_parquet(training))
    return values, {'audit_sha256': hashes, 'history_sha256': market['sha256'],
                    'events_sha256': files[training.name]}


def expansion_targets(events: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    selected = orders[orders.target.isin(['enter_long', 'enter_short', 'increase'])].copy()
    selected['buy'] = selected.side.eq('Buy').astype(int)
    selected = selected.rename(columns={'target_episode_id': 'expansion_episode_id'})
    frame = events.drop(columns=['target_time', 'target_episode_id', 'target']).copy()
    frame = pd.merge_asof(frame, selected[['target_time', 'expansion_episode_id', 'buy']],
                          left_on='end', right_on='target_time', direction='forward',
                          tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))
    frame['active'] = frame.target_time.notna().astype(int)
    frame['target_episode_id'] = frame.expansion_episode_id.fillna(0).astype(int)
    starts, stops = frame.end.array.asi8, frame.label_end.array.asi8
    times = selected.target_time.array.asi8
    frame['expansion_count'] = np.searchsorted(times, stops) - np.searchsorted(times, starts)
    side_counts = []
    for side in ['Buy', 'Sell']:
        times = selected.loc[selected.side.eq(side), 'target_time'].array.asi8
        side_counts.append(np.searchsorted(times, stops) - np.searchsorted(times, starts))
    frame['both_directions'] = (side_counts[0] > 0) & (side_counts[1] > 0)
    return frame


def make_expansion_data(source: dict) -> pd.DataFrame:
    return expansion_targets(source['events'], independent_orders(source['executions'], source['actions']))
