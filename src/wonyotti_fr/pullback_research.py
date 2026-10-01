from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .expansion_research import expansion_gate, load_expansion_selection
from .minute_data import prepare_minute_period
from .pullback_policy import PullbackPolicy
from .reports import table


def candidate_plan() -> list[dict]:
    return [{'offset_bps': offset, 'ttl_minutes': ttl} for offset in [8, 16, 32] for ttl in [5, 15]]


def load_pullback_selection(selection: Path, frozen: dict) -> tuple[dict, PullbackPolicy]:
    base_path = selection / 'base_selection.json'
    if (frozen.get('protocol') != 'pullback_v7' or base_path.stat().st_size > 1024 * 1024
        or sha256(base_path) != frozen['base_selection_sha256']):
        raise ValueError('진입 대기 기반 선택의 형식·크기·지문 오류')
    base = json.loads(base_path.read_text())
    if frozen['model_sha256'] != base['model_sha256']:
        raise ValueError('진입 대기와 기반 모델 지문 불일치')
    _, policy = load_expansion_selection(selection, base)
    return frozen, PullbackPolicy(policy, frozen['offset_bps'], frozen['ttl_minutes'])


def run_pullback_selection(base_selection: Path, market: Path, feature_market: Path, output: Path) -> Path:
    from .event_research import load_selection
    base_frozen, _ = load_selection(base_selection)
    if base_frozen.get('protocol') != 'expansion_v4':
        raise ValueError('진입 대기는 고정한 v4 활동·방향 모델을 사용합니다.')
    config = replace(EngineConfig(**base_frozen['risk']), bar_seconds=60, max_hold_bars=30, cooldown_bars=15, max_adds=0)
    candidates = candidate_plan()
    destination = new_run(output, 'pullback-selection', {
        'protocol': 'docs/EXPERIMENT_V7.md', 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V7.md')),
        'base_selection_sha256': sha256(base_selection / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(feature_market / 'manifest-5m.json'),
        'selection_period': ['2020-01-01', '2021-01-01'], 'candidate_count': len(candidates),
        'selection_rule': 'closed_trades>=20; max(return-0.5*abs(drawdown)); first candidate on ties',
        'all_periods_already_observed': True, 'evaluation_end_exclusive': '2026-10-01'})
    print(f'진입 대기 선택: {destination}', flush=True)
    (destination / 'base_selection.json').write_bytes((base_selection / 'frozen_selection.json').read_bytes())
    (destination / 'expansion_models.json').write_bytes((base_selection / 'expansion_models.json').read_bytes())
    save_json(destination / 'candidate_plan.json', {'candidates': candidates, 'risk': asdict(config)})
    rows = []
    try:
        data, checks = prepare_minute_period(market, feature_market, 'BTCUSDT', '2020-01-01', '2021-01-01')
        save_json(destination / 'development_data.json', checks)
        for index, candidate in enumerate(candidates):
            _, base = load_expansion_selection(destination, base_frozen)
            policy = PullbackPolicy(base, **candidate)
            metrics = backtest(data, policy, config, destination / f'candidate-{index:02d}')
            events = pd.read_parquet(destination / f'candidate-{index:02d}' / 'equity.parquet', columns=['policy_event'])
            counts = {str(key): int(value) for key, value in events.policy_event.value_counts().items()}
            save_json(destination / f'candidate-{index:02d}' / 'policy_events.json', counts)
            rows.append({'candidate': index, **candidate, 'eligible': metrics['closed_trades'] >= 20,
                         'score': metrics['total_return'] - .5 * abs(metrics['max_drawdown']), **metrics})
            save_json(destination / 'development.json', rows)
            print(f'후보 {index+1}/{len(candidates)}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상인 후보 없음', 'criteria_relaxed': False})
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        frozen = {'protocol': 'pullback_v7', 'candidate': winner['candidate'],
                  'offset_bps': winner['offset_bps'], 'ttl_minutes': winner['ttl_minutes'], 'risk': asdict(config),
                  'base_selection_sha256': sha256(destination / 'base_selection.json'),
                  'model_sha256': base_frozen['model_sha256'], 'development_metrics': winner,
                  'selection_source': 'BTCUSDT 2020, already observed',
                  'observed_evaluation_period': ['2022-01-01', '2026-01-01'],
                  'seen_2026_period': ['2026-01-01', '2026-10-01'], 'evaluation_end_exclusive': '2026-10-01',
                  'unseen_evaluation_available': False}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, policy = load_pullback_selection(destination, frozen)
        data, checks = prepare_minute_period(market, feature_market, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(destination / 'validation_data.json', checks)
        validation = backtest(data, policy, config, destination / 'validation-2021')
        gate = expansion_gate(winner, validation)
        save_json(destination / 'development_validation_checks.json', {
            'checks': gate['checks'], 'development_validation_passed': gate['may_open_new_period'],
            'unseen_period_opened': False, 'validation_metrics_sha256': sha256(destination / 'validation-2021' / 'metrics.json')})
        columns = ['candidate', 'offset_bps', 'ttl_minutes', 'total_return', 'max_drawdown', 'closed_trades', 'eligible']
        (destination / 'REPORT.md').write_text(
            '# 확정 종가에 따른 진입 대기 후보\n\n' + table(pd.DataFrame(rows)[columns]) + '\n\n'
            f'후보 {winner["candidate"]} 고정. 개발 수익 {winner["total_return"]:.2%}, '
            f'2021년 확인 수익 {validation["total_return"]:.2%}, 종료 거래 {validation["closed_trades"]}개. '
            f'개발·확인 선행 조건 통과: {gate["may_open_new_period"]}.\n\n'
            '유리한 확정 종가를 기다린 뒤 다음 1분 시가와 수수료·슬리피지로 실행했다. '
            '대기 조건 통과 가격이나 원본 체결 가격을 체결가로 주입하지 않았다. '
            '모든 기간은 이전에 관찰했으며 미사용 최종 평가나 수익성 승인이 아니다. '
            '후보·실패·대기 사건·입력 지문을 보존했다. 추가 학습과 원본 손실 표본 삭제는 없다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'진입 대기 선택 완료: {destination}', flush=True)
    return destination
