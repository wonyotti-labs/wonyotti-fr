from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import new_run, save_json, sha256
from .edge_model import edge_values
from .event_backtest import prepare_period
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .expansion_data import source_inputs
from .period_guard import guard_replay_period
from .reports import table


def signal_support(policy, frame) -> dict:
    values = frame[MARKET_FEATURES].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    activity = policy.base.activity.probabilities(values)
    buy = policy.base.direction.probabilities(values)
    active = valid & (activity >= policy.base.activity_threshold)
    confident = (buy >= policy.base.direction_threshold) | ((1 - buy) >= policy.base.direction_threshold)
    base_pass = active & confident
    direction = np.where(buy >= .5, 1, -1)
    edge = policy.model.predict(edge_values(values, direction))
    pass_all = base_pass & np.isfinite(edge) & (edge >= 16 + policy.margin_bps)
    if not (int(pass_all.sum()) <= int(base_pass.sum()) <= int(active.sum()) <= int(valid.sum())):
        raise ValueError('신호 단계의 포함 관계 오류')
    conditional = edge[base_pass & np.isfinite(edge)]
    return {'bars': len(frame), 'valid_bars': int(valid.sum()), 'activity_pass': int(active.sum()),
            'activity_direction_pass': int(base_pass.sum()), 'all_conditions_pass': int(pass_all.sum()),
            'pass_fraction': float(pass_all.mean()),
            'base_conditional_edge_p50_bps': float(np.quantile(conditional, .5)) if len(conditional) else None,
            'base_conditional_edge_p95_bps': float(np.quantile(conditional, .95)) if len(conditional) else None}


def run_edge_diagnostics(selection: Path, audit: Path, study: Path, history: Path,
                         market: Path, recent: Path, fresh: Path, output: Path) -> Path:
    import pandas as pd

    frozen, policy = load_selection(selection)
    if frozen.get('protocol') != 'edge_v5':
        raise ValueError('v5 고정 후보가 필요합니다.')
    guard_replay_period(selection, frozen, '2026-09-01', '2026-10-01')
    source, hashes = source_inputs(audit, study, history)
    settings = json.loads((selection / 'manifest.json').read_text())['settings']
    if any(hashes[key] != settings[key] for key in hashes):
        raise ValueError('진단 원본과 v5 학습 원본이 다릅니다.')
    destination = new_run(output, 'edge-diagnostics', {**hashes, 'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'market_manifest_sha256': {str(path): sha256(path / 'manifest-5m.json') for path in [market, recent, fresh]},
        'role': 'post_evaluation_support_diagnosis_no_retuning'})
    rows = []
    try:
        for year in range(2018, 2022):
            frame = source['events']
            frame = frame[frame.usable & frame.end.dt.year.eq(year)]
            rows.append({'source': 'BitMEX', 'symbol': 'XBTUSD', 'period': str(year), **signal_support(policy, frame)})
        for start, end, path in [('2020-01-01', '2021-01-01', market), ('2021-01-01', '2022-01-01', market),
                                 ('2022-01-01', '2026-01-01', market), ('2026-01-01', '2026-09-01', recent),
                                 ('2026-09-01', '2026-10-01', fresh)]:
            for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                if start < '2022-01-01' and symbol != 'BTCUSDT':
                    continue
                frame = prepare_period(path, symbol, start, end)
                rows.append({'source': 'Binance', 'symbol': symbol, 'period': f'{start}/{end}', **signal_support(policy, frame)})
        save_json(destination / 'signal_support.json', rows)
        pd.DataFrame(rows).to_csv(destination / 'signal_support.csv', index=False)
        (destination / 'REPORT.md').write_text(
            '# 비용 조건 통과와 신호 표본의 변화\n\n' + table(pd.DataFrame(rows)) + '\n\n'
            '시장 입력만 사용해 모든 시점에서 무포지션이라고 가정한 신호 조건 통과 수다. '
            '연속 신호·보유·재진입 대기·위험 중지를 반영한 실제 거래 수가 아니다. '
            '거래가 거의 없는 후보를 안정적인 수익 전략으로 판단하지 않도록 활동·방향·예상 비용 조건을 차례로 분해했다.\n\n'
            '2018~2019년 시장과 다른 시기의 입력 분포가 다르며 예상 가격 변화 점수도 이동한다. '
            '이 표만으로 분포 변화의 원인이나 성과 개선을 증명하지 않는다. '
            '2026년 9월은 v5 선행 조건 통과 후 이미 개봉했으므로 다음 후보의 미사용 평가가 아니다. '
            '이번 진단으로 기존 후보·선택 기준·비용을 바꾸지 않았다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'신호 지원 표본 진단: {destination}', flush=True)
    return destination
