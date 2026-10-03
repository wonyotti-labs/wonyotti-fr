from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .event_features import MARKET_FEATURES
from .expansion_research import expansion_gate
from .minute_data import prepare_minute_period
from .net_edge_labels import label_opportunities, potential_entries
from .net_edge_model import NetEdgeModel, NetEdgePolicy, net_values
from .pullback_diagnostics import waiting_diagnostics
from .pullback_research import load_pullback_selection
from .reports import table


def candidate_plan() -> list[dict]:
    return [{'alpha': alpha, 'margin_bps': margin} for alpha in [10, 100] for margin in [0, 8]]


def load_net_selection(selection: Path, frozen: dict) -> tuple[dict, NetEdgePolicy]:
    source, model = selection / 'pullback_selection.json', selection / 'net_model.json'
    if (frozen.get('protocol') != 'net_edge_v8' or source.stat().st_size > 1024*1024
        or model.stat().st_size > 1024*1024 or sha256(source) != frozen['pullback_selection_sha256']
        or frozen['model_sha256'] != {'net_model.json': sha256(model)}):
        raise ValueError('순손익 필터의 기반·모델 크기·지문 오류')
    previous = json.loads(source.read_text())
    _, base = load_pullback_selection(selection, previous)
    if (frozen['risk'] != previous['risk'] or frozen['training_period'] != ['2020-01-01', '2021-01-01']
        or frozen['selection_period'] != ['2021-01-01', '2022-01-01']
        or frozen['confirmation_period'] != ['2022-01-01', '2023-01-01']
        or frozen['observed_evaluation_period'] != ['2023-01-01', '2026-01-01']
        or frozen['seen_2026_period'] != ['2026-01-01', '2026-10-01']
        or frozen['evaluation_end_exclusive'] != '2026-10-01' or frozen['unseen_evaluation_available'] is not False):
        raise ValueError('순손익 필터의 고정 위험·평가 범위 오류')
    fitted = NetEdgeModel.from_dict(json.loads(model.read_text()))
    if fitted.data['alpha'] != frozen['alpha']:
        raise ValueError('순손익 필터와 모델의 규제 설정 불일치')
    return frozen, NetEdgePolicy(base, fitted, frozen['margin_bps'])


def net_diagnostics(directory: Path, bars: pd.DataFrame, policy, config: EngineConfig) -> dict:
    waiting = waiting_diagnostics(directory, bars, config.signal_delay_bars)
    decomposition = decompose_run(directory, config.initial_equity)
    result = {'waiting': waiting, 'decomposition': decomposition}
    if isinstance(policy, NetEdgePolicy) and policy.enabled:
        episodes = pd.read_parquet(directory / 'waiting_episodes.parquet')
        selected = episodes[episodes.status.isin(['triggered', 'filtered'])].copy() if len(episodes) else episodes.copy()
        if len(selected):
            values = bars.set_index('end').reindex(selected.decision_time)[MARKET_FEATURES].to_numpy()
            favorable = selected.direction.to_numpy() * np.log(selected.reference_price / selected.decision_close).to_numpy()*10000
            scores = policy.model.predict(net_values(values, selected.direction, favorable, selected.wait_minutes))
            passed = np.isfinite(scores) & (scores >= policy.margin_bps)
            if not np.array_equal(passed, selected.status.eq('triggered').to_numpy()):
                raise ValueError('저장된 필터 결정과 사후 동일 입력의 예측 불일치')
            if selected.loc[~passed, 'executed'].any():
                raise ValueError('거절한 순손익 기회의 진입 체결')
            selected['predicted_net_bps'] = scores
        selected.to_parquet(directory / 'net_gate_decisions.parquet', index=False)
        result['gate'] = {'opportunities': len(selected), 'accepted': waiting['events'].get('triggered', 0),
                          'rejected': waiting['events'].get('filtered', 0), 'decisions_recomputed': True}
    save_json(directory / 'net_diagnostics.json', result)
    return result


def run_net_selection(reference: Path, market: Path, features: Path, confirmation_market: Path,
                      confirmation_features: Path, output: Path) -> Path:
    from .event_research import load_selection
    previous, base = load_selection(reference)
    if previous.get('protocol') != 'pullback_v7' or (base.offset_bps, base.ttl_minutes) != (16, 5):
        raise ValueError('v8에는 고정 v7 후보 2가 필요합니다.')
    config = EngineConfig(**previous['risk'])
    destination = new_run(output, 'net-edge-selection', {
        'protocol': 'docs/EXPERIMENT_V8.md', 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V8.md')),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features / 'manifest-5m.json'),
        'confirmation_market_sha256': sha256(confirmation_market / 'manifest-1m.json'),
        'confirmation_features_sha256': sha256(confirmation_features / 'manifest-5m.json'),
        'training': ['2020-01-01', '2021-01-01'], 'selection': ['2021-01-01', '2022-01-01'],
        'confirmation': ['2022-01-01', '2023-01-01'], 'candidate_count': 4,
        'all_periods_already_observed': True, 'outcome_filtering': False})
    print(f'순손익 필터 선택: {destination}', flush=True)
    (destination / 'pullback_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
    for name in ['base_selection.json', 'expansion_models.json']:
        (destination / name).write_bytes((reference / name).read_bytes())
    save_json(destination / 'candidate_plan.json', candidate_plan())
    try:
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2020-01-01', '2021-01-01')
        save_json(destination / 'training_input.json', checks)
        opportunities, counts = potential_entries(bars, base)
        opportunities.to_parquet(destination / 'potential_entries.parquet', index=False)
        train, labels = label_opportunities(bars, opportunities, config, '2021-01-01')
        train.to_parquet(destination / 'training_labels.parquet', index=False)
        save_json(destination / 'opportunity_summary.json', {'signals': counts, 'labels': labels})
        print(f'학습 기회 {len(opportunities)}, 경계 제거 후 {len(train)}', flush=True)
        models, diagnostics = {}, {}
        for alpha in [10, 100]:
            models[alpha], diagnostics[str(alpha)] = NetEdgeModel.fit(train, alpha)
            save_json(destination / f'net-{alpha}_model.json', models[alpha].to_dict())
        save_json(destination / 'training_diagnostics.json', diagnostics)
        save_json(destination / 'target_diagnosis.json', {
            'rows': len(train), 'mean_close_markout_bps': float(train.close_markout_bps.mean()),
            'mean_net_bps': float(train.net_bps.mean()), 'negative_net_rows': int(train.net_bps.lt(0).sum()),
            'positive_close_negative_net': int((train.close_markout_bps.gt(0) & train.net_bps.lt(0)).sum()),
            'exit_reasons': {str(k): int(v) for k, v in train.exit_reason.value_counts().items()},
            'limit': '조건부 기회·경로·비용·시장과 학습 기간 변경을 함께 포함하며 단일 원인의 효과가 아님'})
        del bars
        data, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(destination / 'development_input.json', checks)
        rows = []
        for index, candidate in enumerate(candidate_plan()):
            policy = NetEdgePolicy(base, models[candidate['alpha']], candidate['margin_bps'])
            target = destination / f'candidate-{index:02d}'
            metrics = backtest(data, policy, config, target)
            net_diagnostics(target, data, policy, config)
            rows.append({'candidate': index, **candidate, 'eligible': metrics['closed_trades'] >= 20,
                         'score': metrics['total_return'] - .5*abs(metrics['max_drawdown']), **metrics})
            save_json(destination / 'development.json', rows)
            print(f'순손익 후보 {index+1}/4: 수익 {metrics["total_return"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            save_json(destination / 'selection_failure.json', {'reason': '종료 거래 20개 이상 후보 없음', 'criteria_relaxed': False})
            (destination / 'REPORT.md').write_text('# 순손익 필터 선택 실패\n\n적격 후보가 없어 고정·후속 평가를 진행하지 않았다. 무거래를 수익성 확보로 해석하지 않는다.\n')
            return destination
        winner = max(eligible, key=lambda row: row['score'])
        save_json(destination / 'net_model.json', models[winner['alpha']].to_dict())
        frozen = {'protocol': 'net_edge_v8', 'candidate': winner['candidate'], 'alpha': winner['alpha'],
                  'margin_bps': winner['margin_bps'], 'risk': previous['risk'], 'development_metrics': winner,
                  'pullback_selection_sha256': sha256(destination / 'pullback_selection.json'),
                  'model_sha256': {'net_model.json': sha256(destination / 'net_model.json')},
                  'training_period': ['2020-01-01', '2021-01-01'], 'selection_period': ['2021-01-01', '2022-01-01'],
                  'confirmation_period': ['2022-01-01', '2023-01-01'],
                  'observed_evaluation_period': ['2023-01-01', '2026-01-01'], 'seen_2026_period': ['2026-01-01', '2026-10-01'],
                  'evaluation_end_exclusive': '2026-10-01', 'unseen_evaluation_available': False}
        save_json(destination / 'frozen_selection.json', frozen)
        save_json(destination / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(destination / 'frozen_selection.json')})
        _, policy = load_net_selection(destination, frozen)
        del data
        data, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(destination / 'confirmation_input.json', checks)
        confirmation = backtest(data, policy, config, destination / 'confirmation-2022')
        net_diagnostics(destination / 'confirmation-2022', data, policy, config)
        gate = expansion_gate(winner, confirmation)
        save_json(destination / 'development_confirmation_checks.json', {'checks': gate['checks'],
                  'passed': gate['may_open_new_period'], 'unseen_period_opened': False,
                  'confirmation_metrics_sha256': sha256(destination / 'confirmation-2022/metrics.json')})
        summary = pd.DataFrame(rows)[['candidate', 'alpha', 'margin_bps', 'total_return', 'max_drawdown', 'closed_trades', 'eligible']]
        (destination / 'REPORT.md').write_text(
            '# 실행 순손익 필터의 시간순 선택\n\n' + table(summary) + '\n\n'
            f'2020년 독립 기회 {len(train)}개로 학습하고 2021년 후보 {winner["candidate"]}를 선택했다. '
            f'개발 {winner["total_return"]:.2%}, 고정 후 2022년 {confirmation["total_return"]:.2%}·{confirmation["closed_trades"]}거래. '
            f'선행 조건 통과: {gate["may_open_new_period"]}.\n\n'
            '다음 시가·손절·시간 제한·수수료·슬리피지·펀딩을 같은 엔진으로 계산했다. '
            '각 학습 정답은 독립 초기 계좌이며 연속 계좌의 위험 상태와 다르다. '
            '겹친 기회를 독립 표본으로 보거나 이미 관찰한 기간을 미사용 평가로 부르지 않는다. '
            '손실 정답·거절 결정·선택 후보·실패와 입력 해시는 로컬에 보존한다.\n', encoding='utf-8')
        save_json(destination / 'summary.json', {'complete': True, 'selected': True, 'profitability_accepted': False})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'순손익 필터 선택 완료: {destination}', flush=True)
    return destination
