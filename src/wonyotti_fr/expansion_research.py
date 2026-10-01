from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest, prepare_period
from .event_features import MARKET_FEATURES, purged_train
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel, ExpansionPolicy
from .reports import table


def expansion_gate(development: dict, validation: dict) -> dict:
    checks = {f'{label}_{check}': bool(value) for label, metrics in [('development', development), ('validation', validation)]
              for check, value in [('positive', metrics['total_return'] > 0), ('enough_trades', metrics['closed_trades'] >= 20),
                                    ('no_halt', not metrics['permanent_halt'])]}
    return {'may_open_new_period': all(checks.values()), 'checks': checks}


def load_expansion_selection(selection: Path, frozen: dict) -> tuple[dict, ExpansionPolicy]:
    path = selection / 'expansion_models.json'
    if (frozen.get('protocol') != 'expansion_v4' or set(frozen['model_sha256']) != {path.name}
        or path.stat().st_size > 2 * 1024 * 1024 or sha256(path) != frozen['model_sha256'][path.name]):
        raise ValueError('노출 확대 모델의 형식·크기·지문 오류')
    bundle = json.loads(path.read_text())
    activity, direction = (BinaryModel.from_dict(bundle[name]) for name in ['activity', 'direction'])
    return frozen, ExpansionPolicy(activity, direction, frozen['activity_threshold'], frozen['direction_threshold'], frozen['min_hold_bars'])


def prediction_quality(models: dict, frame: pd.DataFrame) -> dict:
    active = frame.active.astype(int).to_numpy()
    activity = models['activity'].probabilities(frame[MARKET_FEATURES].to_numpy())
    subset = frame[frame.active.eq(1)]
    buys = subset.buy.astype(int).to_numpy()
    direction = models['direction'].probabilities(subset[MARKET_FEATURES].to_numpy())
    return {'bars': len(frame), 'active_windows': len(subset), 'active_rate': float(active.mean()),
            'activity_average_precision': float(average_precision_score(active, activity)) if active.sum() else None,
            'activity_brier': float(brier_score_loss(active, activity)),
            'direction_support': {'buy': int(buys.sum()), 'sell': int((buys == 0).sum())},
            'direction_balanced_accuracy': float(balanced_accuracy_score(buys, direction >= .5)) if len(np.unique(buys)) == 2 else None,
            'direction_brier': float(brier_score_loss(buys, direction)) if len(subset) else None,
            'interpretation': '체결 행동 예측이며 미래 수익 예측 성능이 아님'}


def run_expansion_selection(audit: Path, study: Path, history: Path, market: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    data = make_expansion_data(source)
    train = purged_train(data, source['episodes'], '2020-01-01')
    selected = train[train.active.eq(1)]
    candidates = [{'kind': kind, 'activity_quantile': quantile, 'direction_threshold': threshold}
                  for kind in ['logistic', 'boosted'] for quantile in [.9, .95, .975] for threshold in [.55, .65]]
    config = EngineConfig(stop_fraction=.04, max_hold_bars=72, cooldown_bars=3, max_adds=0)
    destination = new_run(output, 'expansion-selection', {**hashes,
        'protocol': 'docs/EXPERIMENT_V4.md', 'training_end_exclusive': '2020-01-01',
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'), 'candidate_count': len(candidates),
        'selection_period': ['2020-01-01', '2021-01-01'], 'selection_rule': 'trades>=20; return-0.5*abs(drawdown)',
        'retrospective_periods': ['2020', '2021', '2022~2025', '2026-01~08'],
    })
    print(f'노출 확대 선택: {destination}', flush=True)
    save_json(destination / 'candidate_plan.json', {'candidates': candidates, 'risk': asdict(config), 'min_hold_bars': 12})
    data.to_parquet(destination / 'expansion_targets.parquet', index=False)
    save_json(destination / 'training_support.json', {'train_rows': len(train), 'expansion_windows': len(selected),
        'last_label_end': train.label_end.max(), 'multiple_expansion_windows': int(train.expansion_count.gt(1).sum()),
        'both_direction_windows': int(train.both_directions.sum()), 'excluded_rows': len(data) - len(train),
        'feature_columns': MARKET_FEATURES, 'outcome_filtering': False})
    rows, diagnostics, models, thresholds = [], {}, {}, {}
    try:
        for kind in ['logistic', 'boosted']:
            models[kind], diagnostics[kind] = {}, {}
            for name, frame, label in [('activity', train, 'active'), ('direction', selected, 'buy')]:
                print(f'학습: {kind}/{name}, {len(frame):,}개', flush=True)
                models[kind][name], diagnostics[kind][name] = BinaryModel.fit(frame[MARKET_FEATURES].to_numpy(), frame[label].astype(int).to_numpy(), kind)
            scores = models[kind]['activity'].probabilities(train[MARKET_FEATURES].to_numpy())
            thresholds[kind] = {str(q): float(np.quantile(scores, q)) for q in [.9, .95, .975]}
            diagnostics[kind]['thresholds'] = thresholds[kind]
            diagnostics[kind]['training_active_fractions'] = {key: float(np.mean(scores >= value)) for key, value in thresholds[kind].items()}
            save_json(destination / f'{kind}_models.json', {name: model.to_dict() for name, model in models[kind].items()})
        save_json(destination / 'training_diagnostics.json', diagnostics)
        bars = prepare_period(market, 'BTCUSDT', '2020-01-01', '2021-01-01')
        for index, candidate in enumerate(candidates):
            kind = candidate['kind']
            threshold = thresholds[kind][str(candidate['activity_quantile'])]
            policy = ExpansionPolicy(**models[kind], activity_threshold=threshold, direction_threshold=candidate['direction_threshold'])
            metrics = backtest(bars, policy, config, destination / f'candidate-{index:02d}')
            row = {'candidate': index, **candidate, 'activity_threshold': threshold, 'eligible': metrics['closed_trades'] >= 20,
                   'score': metrics['total_return'] - .5 * abs(metrics['max_drawdown']), **metrics}
            rows.append(row)
            save_json(destination / 'validation.json', rows)
            print(f'후보 {index+1}/12 {kind}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상인 후보 없음', 'criteria_relaxed': False})
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        bundle = destination / 'expansion_models.json'
        bundle.write_bytes((destination / f'{winner["kind"]}_models.json').read_bytes())
        frozen = {'protocol': 'expansion_v4', 'candidate': winner['candidate'], 'kind': winner['kind'],
                  'activity_threshold': winner['activity_threshold'], 'direction_threshold': winner['direction_threshold'],
                  'min_hold_bars': 12, 'risk': asdict(config), 'development_metrics': winner,
                  'model_sha256': {bundle.name: sha256(bundle)}, 'selection_source': 'BTCUSDT 2020 only',
                  'observed_evaluation_period': ['2022-01-01', '2026-01-01'], 'seen_2026_period': ['2026-01-01', '2026-09-01'],
                  'new_evaluation_period': ['2026-09-01', '2026-10-01'], 'new_evaluation_opened': False}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, policy = load_expansion_selection(destination, frozen)
        check_metrics = backtest(prepare_period(market, 'BTCUSDT', '2021-01-01', '2022-01-01'), policy, config, destination / 'validation-2021')
        check = data[data.usable & (data.end >= '2021-01-01') & (data.label_end < '2022-01-01')]
        save_json(destination / 'imitation_2021.json', prediction_quality(models[winner['kind']], check))
        gate = expansion_gate(winner, check_metrics)
        save_json(destination / 'new_evaluation_gate.json', {**gate, 'selection_sha256': sha256(destination / 'frozen_selection.json'),
                  'validation_metrics_sha256': sha256(destination / 'validation-2021' / 'metrics.json')})
        summary = pd.DataFrame(rows)[['candidate', 'kind', 'activity_quantile', 'direction_threshold', 'total_return', 'max_drawdown', 'closed_trades', 'eligible']]
        (destination / 'REPORT.md').write_text(
            '# 노출 확대 시점·방향 분리 후보의 선택\n\n' + table(summary) + '\n\n'
            f'후보 {winner["candidate"]} 고정. 개발 순수익 {winner["total_return"]:.2%}, '
            f'2021년 확인 {check_metrics["total_return"]:.2%}, 종료 거래 {check_metrics["closed_trades"]}개. '
            f'새 구간 개봉 조건 통과: {gate["may_open_new_period"]}.\n\n'
            '원본 추가 진입을 방향 학습에 포함했지만 실제 봇은 한 번 진입하고 보유·반전한다. '
            '활동·방향 점수는 수익 확률이 아니다. 직전 확정 시세와 학습 구간의 활동 기준만 사용했다. '
            '원본 지정가 대기열·리베이트를 복제하지 않았으며 다음 봉 시장가 실행·비용·위험 제한을 유지했다.\n\n'
            '이미 관찰한 기간의 가설 비교다. 모든 후보·손실·입력 해시·학습 예측 일치 검사를 보존했다. '
            '기준 실패 시 다음 미사용 구간을 열지 않으며 실제 주문은 연결하지 않는다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'노출 확대 선택 완료: {destination}', flush=True)
    return destination
