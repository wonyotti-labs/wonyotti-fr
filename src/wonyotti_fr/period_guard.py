from __future__ import annotations

from pathlib import Path

import pandas as pd


def guard_replay_period(selection: Path, frozen: dict, start: str, end: str) -> None:
    protocol = frozen.get('protocol')
    if protocol in {'pullback_v7', 'net_edge_v8', 'lifecycle_v9', 'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        first, last = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
        if (first >= last or first < pd.Timestamp('2021-01-01' if protocol in {'minute_inventory_micro_v21', 'addition_effect_v22', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'} else '2020-01-01', tz='UTC')
            or last > pd.Timestamp('2026-10-01', tz='UTC') or frozen['evaluation_end_exclusive'] != '2026-10-01'
            or frozen['unseen_evaluation_available'] is not False):
            raise ValueError('v7~v22 재생은 사전 계획에서 관찰한 기간만 지원합니다.')
        return
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
