from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .edge_model import EdgeModel, EdgePolicy, edge_targets
from .engine import EngineConfig
from .event_backtest import backtest, prepare_period
from .event_features import purged_train
from .event_research import load_selection
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel, ExpansionPolicy
from .expansion_research import expansion_gate
from .reports import table


def load_edge_selection(selection: Path, frozen: dict) -> tuple[dict, EdgePolicy]:
    path = selection / 'edge_models.json'
    if (frozen.get('protocol') != 'edge_v5' or set(frozen['model_sha256']) != {path.name}
        or path.stat().st_size > 2 * 1024 * 1024 or sha256(path) != frozen['model_sha256'][path.name]):
        raise ValueError('비용 예측 모델의 형식·크기·지문 오류')
    bundle = json.loads(path.read_text())
    base = ExpansionPolicy(BinaryModel.from_dict(bundle['activity']), BinaryModel.from_dict(bundle['direction']),
                           frozen['activity_threshold'], frozen['direction_threshold'])
    return frozen, EdgePolicy(base, EdgeModel.from_dict(bundle['edge']), frozen['margin_bps'])


def run_edge_selection(audit: Path, study: Path, history: Path, reference: Path, market: Path, output: Path) -> Path:
    previous, base = load_selection(reference)
    if previous.get('protocol') != 'expansion_v4':
        raise ValueError('v4의 고정 활동·방향 모델이 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    data = make_expansion_data(source)
    candidates = [{'horizon_bars': horizon, 'alpha': alpha, 'margin_bps': margin}
                  for horizon in [6, 12, 24] for alpha in [10, 100] for margin in [0, 8]]
    destination = new_run(output, 'edge-selection', {**hashes, 'protocol': 'docs/EXPERIMENT_V5.md',
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'), 'selection_period': ['2020-01-01', '2021-01-01'],
        'selection_rule': 'trades>=20; return-0.5*abs(drawdown)', 'candidate_count': len(candidates), 'outcome_filtering': False})
    print(f'보유·비용 예측 선택: {destination}', flush=True)
    save_json(destination / 'candidate_plan.json', candidates)
    rows, models, diagnostics = [], {}, {}
    try:
        for horizon in [6, 12, 24]:
            targets = edge_targets(data, source['bars'], horizon)
            train = purged_train(targets, source['episodes'], '2020-01-01')
            if train.outcome_time.max() >= pd.Timestamp('2019-12-31', tz='UTC'):
                raise ValueError('가격 정답이 학습 시간 경계를 넘었습니다.')
            train.to_parquet(destination / f'training-{horizon}.parquet', index=False)
            for alpha in [10, 100]:
                key = f'{horizon}-{alpha}'
                models[key], diagnostics[key] = EdgeModel.fit(train, alpha)
                save_json(destination / f'edge-{key}_model.json', models[key].to_dict())
        save_json(destination / 'training_diagnostics.json', diagnostics)
        bars = prepare_period(market, 'BTCUSDT', '2020-01-01', '2021-01-01')
        base_config = EngineConfig(**previous['risk'])
        for index, candidate in enumerate(candidates):
            model = models[f'{candidate["horizon_bars"]}-{candidate["alpha"]}']
            risk = replace(base_config, max_hold_bars=candidate['horizon_bars'])
            metrics = backtest(bars, EdgePolicy(base, model, candidate['margin_bps']), risk, destination / f'candidate-{index:02d}')
            rows.append({'candidate': index, **candidate, 'eligible': metrics['closed_trades'] >= 20,
                         'score': metrics['total_return'] - .5 * abs(metrics['max_drawdown']), **metrics})
            save_json(destination / 'validation.json', rows)
            print(f'보유·비용 후보 {index+1}/12: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상인 후보 없음', 'criteria_relaxed': False})
            (destination / 'REPORT.md').write_text('# 보유·비용 예측 후보의 부적격\n\n종료 거래 20개 이상인 후보가 없어 선택하지 않았다. 무거래를 수익성 확보로 해석하지 않는다.\n', encoding='utf-8')
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        bundle = {'activity': base.activity.to_dict(), 'direction': base.direction.to_dict(),
                  'edge': models[f'{winner["horizon_bars"]}-{winner["alpha"]}'].to_dict()}
        save_json(destination / 'edge_models.json', bundle)
        risk = replace(base_config, max_hold_bars=winner['horizon_bars'])
        frozen = {'protocol': 'edge_v5', 'candidate': winner['candidate'], 'risk': asdict(risk),
                  'horizon_bars': winner['horizon_bars'], 'alpha': winner['alpha'], 'margin_bps': winner['margin_bps'],
                  'activity_threshold': base.activity_threshold, 'direction_threshold': base.direction_threshold,
                  'development_metrics': winner, 'model_sha256': {'edge_models.json': sha256(destination / 'edge_models.json')},
                  'reference_sha256': sha256(reference / 'frozen_selection.json'), 'selection_source': 'BTCUSDT 2020 only',
                  'observed_evaluation_period': ['2022-01-01', '2026-01-01'], 'seen_2026_period': ['2026-01-01', '2026-09-01'],
                  'new_evaluation_period': ['2026-09-01', '2026-10-01'], 'new_evaluation_opened': False}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, policy = load_edge_selection(destination, frozen)
        check = backtest(prepare_period(market, 'BTCUSDT', '2021-01-01', '2022-01-01'), policy, risk, destination / 'validation-2021')
        gate = expansion_gate(winner, check)
        save_json(destination / 'new_evaluation_gate.json', {**gate, 'selection_sha256': sha256(destination / 'frozen_selection.json'),
                  'validation_metrics_sha256': sha256(destination / 'validation-2021' / 'metrics.json')})
        summary = pd.DataFrame(rows)[['candidate', 'horizon_bars', 'alpha', 'margin_bps', 'total_return', 'max_drawdown', 'closed_trades', 'eligible']]
        (destination / 'REPORT.md').write_text(
            '# 보유 시간·비용 예측 후보의 선택\n\n' + table(summary) + '\n\n'
            f'후보 {winner["candidate"]}를 고정했다. 개발 순수익 {winner["total_return"]:.2%}, '
            f'2021년 {check["total_return"]:.2%}, 거래 {check["closed_trades"]}개. '
            f'새 기간 개봉 조건 통과: {gate["may_open_new_period"]}.\n\n'
            '원본 노출 확대 창의 이후 가격은 학습 정답에만 사용했다. 원본 실제 체결 가격·리베이트·승패 필터는 사용하지 않았다. '
            '회귀 점수는 비용 차감 전 방향별 로그 가격 변화 추정치이며 실제 거래의 수익 확률이 아니다. '
            '학습 표본과 봇이 방문하는 시장 상태가 달라지는 한계가 있다.\n\n'
            '다음 봉 시가와 비용·펀딩·위험 제한을 적용한 실행 결과로 선택했다. '
            '2020년 이후는 이전에 관찰한 구간이며 시간순 적용만으로 미사용 성과 인증이 되지 않는다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'보유·비용 예측 선택 완료: {destination}', flush=True)
    return destination
