from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import EventPolicy, backtest, prepare_period, read_models
from .event_features import purged_train
from .event_research import load_selection
from .event_study import evaluate_imitation
from .frequency_model import FrequencyModel, score_quality
from .reports import table


def frequency_gate(development: dict, validation: dict) -> dict:
    checks = {'development_positive': development['total_return'] > 0,
              'validation_positive': validation['total_return'] > 0,
              'validation_enough_trades': validation['closed_trades'] >= 20,
              'validation_not_halted': not validation['permanent_halt']}
    return {'checks': checks, 'may_open_new_period': all(checks.values()), 'live_trading_approved': False}


def run_frequency_selection(study: Path, audit: Path, v2_selection: Path, market: Path, output: Path) -> Path:
    base, _ = load_selection(v2_selection)
    if base.get('protocol') is not None:
        raise ValueError('v2의 원래 고정 선택 파일이 필요합니다.')
    entry, management = read_models(study)
    for name in ['entry_model.json', 'management_model.json']:
        if sha256(study / name) != base['model_sha256'][name]:
            raise ValueError('v2에서 사용한 학습 모델과 다릅니다.')
    hashes = json.loads((study / 'files.json').read_text())
    manifest = json.loads((study / 'manifest.json').read_text())
    data_path, episodes_path = study / 'training_events.parquet', audit / 'episodes.parquet'
    if (sha256(data_path) != hashes[data_path.name]
        or sha256(episodes_path) != manifest['settings']['audit_sha256']['episodes.parquet']):
        raise ValueError('학습 자료 또는 에피소드 지문이 다릅니다.')
    data = pd.read_parquet(data_path)
    train = purged_train(data, pd.read_parquet(episodes_path), '2020-01-01')
    counts = {}
    for name in ['entry', 'management']:
        subset = train[train.direction.eq(0) if name == 'entry' else train.direction.ne(0)]
        counts[name] = {str(k): int(v) for k, v in subset.target.value_counts().items()}
    diagnostics = json.loads((study / 'diagnostics.json').read_text())
    if any(counts[name] != diagnostics[f'{name}_class_counts'] for name in counts):
        raise ValueError('모델 학습 당시와 클래스 빈도가 다릅니다.')
    candidates = [{'alpha': alpha, 'entry_threshold': entry_threshold, 'management_threshold': management_threshold}
                  for alpha in [0, 0.5, 1] for entry_threshold in [0.5, 0.65] for management_threshold in [0.35, 0.5]]
    destination = new_run(output, 'frequency-selection', {
        'protocol': 'docs/EXPERIMENT_V3.md', 'study_files_sha256': sha256(study / 'files.json'),
        'v2_selection_sha256': sha256(v2_selection / 'frozen_selection.json'),
        'episodes_sha256': sha256(episodes_path), 'market_manifest_sha256': sha256(market / 'manifest-5m.json'),
        'selection_period': ['2020-01-01', '2021-01-01'], 'candidate_count': len(candidates),
        'selection_rule': 'closed_trades >= 20; maximize return - 0.5 * abs(max_drawdown)',
        'prior_period': '2018~2019, crossing episodes and last 24h purged',
    })
    print(f'빈도 반영 후보 선택: {destination}', flush=True)
    for name in ['entry_model.json', 'management_model.json', 'files.json']:
        (destination / name).write_bytes((study / name).read_bytes())
    save_json(destination / 'candidate_plan.json', {'candidates': candidates, 'risk': base['risk']})
    save_json(destination / 'class_frequencies.json', {'counts': counts, 'training_rows': len(train),
                                                       'last_label_end': train.label_end.max()})
    validation = []
    try:
        bars = prepare_period(market, 'BTCUSDT', '2020-01-01', '2021-01-01')
        for index, candidate in enumerate(candidates):
            adjusted_entry = FrequencyModel.from_counts(entry, counts['entry'], candidate['alpha'])
            adjusted_management = FrequencyModel.from_counts(management, counts['management'], candidate['alpha'])
            policy = EventPolicy(adjusted_entry, adjusted_management, candidate['entry_threshold'], candidate['management_threshold'])
            metrics = backtest(bars, policy, EngineConfig(**base['risk']), destination / f'candidate-{index:02d}')
            row = {'candidate': index, **candidate, 'eligible': metrics['closed_trades'] >= 20,
                   'score': metrics['total_return'] - 0.5 * abs(metrics['max_drawdown']), **metrics}
            validation.append(row)
            save_json(destination / 'validation.json', validation)
            print(f'후보 {index+1}/12 alpha={candidate["alpha"]}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in validation if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상인 후보 없음', 'criteria_relaxed': False})
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        chosen = candidates[winner['candidate']]
        frozen = {'protocol': 'frequency_v3', 'candidate': winner['candidate'],
                  'entry_threshold': chosen['entry_threshold'], 'management_threshold': chosen['management_threshold'],
                  'risk': base['risk'], 'development_metrics': winner, 'model_sha256': base['model_sha256'],
                  'score_adjustment': {'format': 'training_class_frequency_v1', 'counts': counts, 'alpha': chosen['alpha']},
                  'observed_evaluation_period': ['2022-01-01', '2026-01-01'],
                  'seen_2026_period': ['2026-01-01', '2026-09-01'],
                  'new_evaluation_period': ['2026-09-01', '2026-10-01'],
                  'new_evaluation_opened': False, 'selection_source': 'BTCUSDT 2020 only'}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, fixed = load_selection(destination)
        year2021 = prepare_period(market, 'BTCUSDT', '2021-01-01', '2022-01-01')
        check_metrics = backtest(year2021, fixed, EngineConfig(**base['risk']), destination / 'validation-2021')
        check = data[data.usable & (data.end >= '2021-01-01') & (data.label_end < '2022-01-01')]
        imitation = {}
        for name, raw, adjusted in [('entry', entry, fixed.entry), ('management', management, fixed.management)]:
            subset = check[check.direction.eq(0) if name == 'entry' else check.direction.ne(0)]
            imitation[name] = {label: {'imitation': evaluate_imitation(model, subset), 'score_quality': score_quality(model, subset)}
                               for label, model in [('raw', raw), ('adjusted', adjusted)]}
        save_json(destination / 'imitation_2021.json', imitation)
        gate = frequency_gate(winner, check_metrics)
        save_json(destination / 'new_evaluation_gate.json', {**gate,
                  'selection_sha256': sha256(destination / 'frozen_selection.json'),
                  'validation_metrics_sha256': sha256(destination / 'validation-2021' / 'metrics.json')})
        summary = pd.DataFrame(validation)[['candidate', 'alpha', 'entry_threshold', 'management_threshold', 'total_return', 'max_drawdown', 'closed_trades', 'eligible']]
        (destination / 'REPORT.md').write_text(
            '# 학습 빈도 반영 후보의 선택\n\n' + table(summary) + '\n\n'
            f'후보 {winner["candidate"]}를 연구 비교 대상으로 고정했다. 개발 순수익 {winner["total_return"]:.2%}, '
            f'2021년 순수익 {check_metrics["total_return"]:.2%}, 종료 거래 {check_metrics["closed_trades"]}개다.\n\n'
            f'새 기간 개봉 조건 통과: {gate["may_open_new_period"]}. '
            '미충족이면 2026년 9월 자료를 열지 않는다. 실제 거래 승인은 하지 않는다.\n\n'
            '빈도는 2018~2019년 학습 표본에서만 계산했다. 미사용 금융 수익 확률이나 확률 보정의 보장이 아니다. '
            '원래 점수와 조정 점수의 클래스별 표본·Brier 점수·로그 손실·신뢰도 구간은 별도 JSON에 보존했다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'빈도 반영 선택 완료: {destination}', flush=True)
    return destination
