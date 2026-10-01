from __future__ import annotations

from pathlib import Path

import pandas as pd


def guard_replay_period(selection: Path, frozen: dict, start: str, end: str) -> None:
    protocol = frozen.get('protocol')
    if protocol not in {'frequency_v3', 'expansion_v4', 'edge_v5'}:
        return
    first, last = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    new_start, new_end = (pd.Timestamp(value, tz='UTC') for value in frozen['new_evaluation_period'])
    if first >= last:
        raise ValueError('재생 기간의 시작·종료 순서 오류')
    if last > new_start:
        if last > new_end:
            raise ValueError('사전 계획 이후 기간의 재생은 지원하지 않습니다.')
        if protocol == 'frequency_v3':
            from .frequency_evaluation import ensure_period_allowed
            ensure_period_allowed(selection, frozen, 'new')
        else:
            from .expansion_evaluation import ensure_expansion_period
            ensure_expansion_period(selection, frozen, 'new')
