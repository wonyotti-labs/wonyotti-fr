from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import EventPolicy, backtest, candidate_plan, prepare_period, read_models
from .event_study import evaluate_imitation
from .frequency_model import FrequencyModel
from .reports import table


def run_event_selection(study: Path, market: Path, output: Path) -> Path:
    entry, management = read_models(study)
    candidates = candidate_plan()
    destination = new_run(output, 'event-selection', {
        'study_files_sha256': sha256(study / 'files.json'),
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'),
        'selection_period': ['2020-01-01', '2021-01-01'], 'symbol': 'BTCUSDT',
        'selection_rule': 'closed_trades >= 20, maximize return - 0.5 * abs(max_drawdown)',
        'candidate_count': len(candidates), 'evaluation_order': 'write plan, compare 2020, freeze, inspect 2021 imitation',
    })
    print(f'사건별 정책 선택: {destination}', flush=True)
    for name in ['entry_model.json', 'management_model.json', 'files.json']:
        (destination / name).write_bytes((study / name).read_bytes())
    save_json(destination / 'candidate_plan.json', candidates)
    try:
        bars = prepare_period(market, 'BTCUSDT', '2020-01-01', '2021-01-01')
        validation = []
        for index, candidate in enumerate(candidates):
            policy = EventPolicy(entry, management, candidate['entry_threshold'], candidate['management_threshold'])
            metrics = backtest(bars, policy, EngineConfig(**candidate['risk']), destination / f'candidate-{index:02d}')
            row = {'candidate': index, 'eligible': metrics['closed_trades'] >= 20,
                   'score': metrics['total_return'] - 0.5 * abs(metrics['max_drawdown']), **metrics}
            validation.append(row)
            save_json(destination / 'validation.json', validation)
            print(f'후보 {index + 1}/{len(candidates)}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in validation if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상인 후보 없음', 'criteria_relaxed': False})
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        frozen = {'candidate': winner['candidate'], **candidates[winner['candidate']],
                  'development_metrics': winner,
                  'model_sha256': {name: sha256(destination / name) for name in ['entry_model.json', 'management_model.json']},
                  'new_evaluation_opened': False, 'new_evaluation_period': ['2026-01-01', '2026-09-01'],
                  'observed_evaluation_period': ['2022-01-01', '2026-01-01'],
                  'selection_source': 'BTCUSDT 2020 only'}
        save_json(destination / 'frozen_selection.json', frozen)
        data_path = study / 'training_events.parquet'
        hashes = json.loads((study / 'files.json').read_text())
        if sha256(data_path) != hashes[data_path.name]:
            raise ValueError('학습 자료가 변경됐습니다.')
        data = pd.read_parquet(data_path)
        check = data[data.usable & (data.end >= '2021-01-01') & (data.label_end < '2022-01-01')]
        metrics = {name: evaluate_imitation(model, check[check.direction.eq(0) if name == 'entry' else check.direction.ne(0)])
                   for name, model in [('entry', entry), ('management', management)]}
        save_json(destination / 'imitation_2021.json', metrics)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        summary = pd.DataFrame(validation)[['candidate', 'eligible', 'score', 'total_return', 'max_drawdown', 'closed_trades']]
        (destination / 'REPORT.md').write_text(
            '# 사건별 정책의 2020년 선택\n\n' + table(summary) + '\n\n'
            f'연구 비교 대상으로 후보 {winner["candidate"]}를 고정했다. 개발 순수익은 {winner["total_return"]:.2%}다. '
            '이는 수익성 승인과 다르다. 2022~2025년은 이미 관찰한 탐색 구간이며 2026년 자료는 아직 열지 않았다.\n\n'
            '부분 체결·대기 주문·호가 대기열을 모두 모사하지 못한다. '
            '공개 원거래소 특징과 다른 거래소의 실행 시세를 사용하므로 거래소 차이도 존재한다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'고정 선택 저장: {destination}', flush=True)
    return destination


def load_selection(selection: Path) -> tuple[dict, EventPolicy]:
    frozen_path = selection / 'frozen_selection.json'
    integrity = json.loads((selection / 'frozen_integrity.json').read_text())
    if sha256(frozen_path) != integrity['frozen_selection_sha256']:
        raise ValueError('고정 선택 파일이 변경됐습니다.')
    frozen = json.loads(frozen_path.read_text())
    if frozen.get('protocol') in {'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
        from .lifecycle_edge_research import load_lifecycle_edge_selection
        return load_lifecycle_edge_selection(selection, frozen)
    if frozen.get('protocol') in {'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24'}:
        from .action_research import load_action_selection
        return load_action_selection(selection, frozen)
    if frozen.get('protocol') == 'lifecycle_v9':
        from .lifecycle_research import load_lifecycle_selection
        return load_lifecycle_selection(selection, frozen)
    if frozen.get('protocol') == 'net_edge_v8':
        from .net_edge_research import load_net_selection
        return load_net_selection(selection, frozen)
    if frozen.get('protocol') == 'pullback_v7':
        from .pullback_research import load_pullback_selection
        return load_pullback_selection(selection, frozen)
    if frozen.get('protocol') == 'edge_v5':
        from .edge_research import load_edge_selection
        return load_edge_selection(selection, frozen)
    if frozen.get('protocol') == 'expansion_v4':
        from .expansion_research import load_expansion_selection
        return load_expansion_selection(selection, frozen)
    for name, checksum in frozen['model_sha256'].items():
        if name not in {'entry_model.json', 'management_model.json'} or sha256(selection / name) != checksum:
            raise ValueError('고정 선택의 모델 지문이 다릅니다.')
    entry, management = read_models(selection)
    adjustment = frozen.get('score_adjustment')
    if adjustment is not None:
        if adjustment.get('format') != 'training_class_frequency_v1':
            raise ValueError('지원하지 않는 점수 조정 형식입니다.')
        entry = FrequencyModel.from_counts(entry, adjustment['counts']['entry'], adjustment['alpha'])
        management = FrequencyModel.from_counts(management, adjustment['counts']['management'], adjustment['alpha'])
    return frozen, EventPolicy(entry, management, frozen['entry_threshold'], frozen['management_threshold'])
