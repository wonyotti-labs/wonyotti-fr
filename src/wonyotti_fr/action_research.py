from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .action_model import ActionModels, MinuteActionPolicy
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .lifecycle_research import risk_config
from .minute_data import prepare_minute_period
from .minute_management import ACTIONS, purged_window
from .path_management import PATH_FEATURES, PathActionModels, PathActionPolicy, ReversalPathPolicy
from .pullback_diagnostics import waiting_diagnostics
from .pullback_policy import PullbackPolicy
from .pullback_research import load_pullback_selection
from .reports import table


def candidate_plan():
    return [{'kind': kind, 'multiplier': multiplier} for kind in ['logistic', 'tree'] for multiplier in [1., 1.5]]


def load_action_selection(selection: Path, frozen: dict):
    if frozen.get('protocol') == 'minute_rate_reverse_v18':
        from .rate_policy import ReversalRatePolicy
        parent_path = selection / 'rate_selection.json'
        if parent_path.stat().st_size > 1024**2 or sha256(parent_path) != frozen['rate_selection_sha256']:
            raise ValueError('누적 반전 정책의 기반 선택 지문 오류')
        parent = json.loads(parent_path.read_text())
        if (parent.get('protocol') != 'minute_rate_v14' or frozen['candidate'] != 0
            or any(frozen.get(key) != value for key, value in parent.items()
                   if key not in {'protocol', 'candidate', 'development_metrics'})):
            raise ValueError('누적 반전 정책의 고정 기반 설정 오류')
        _, policy = load_action_selection(selection, parent)
        return frozen, ReversalRatePolicy(policy, policy.manager, policy.thresholds, policy.multiplier, policy.scales)
    if frozen.get('protocol') in {'minute_reverse_v13', 'minute_rate_v14'}:
        parent_path = selection / 'path_selection.json'
        if parent_path.stat().st_size > 1024**2 or sha256(parent_path) != frozen['path_selection_sha256']:
            raise ValueError('반전 정책의 기반 선택 지문 오류')
        parent = json.loads(parent_path.read_text())
        if (parent.get('protocol') != 'minute_path_v12' or frozen['candidate'] != 0
            or any(frozen.get(key) != value for key, value in parent.items()
                   if key not in {'protocol', 'candidate', 'development_metrics'})):
            raise ValueError('반전 정책의 고정 기반 설정 오류')
        _, policy = load_action_selection(selection, parent)
        if frozen['protocol'] == 'minute_rate_v14':
            from .rate_policy import RateActionPolicy
            rate_path = selection / 'rate_calibration.json'
            if rate_path.stat().st_size > 1024**2 or sha256(rate_path) != frozen['rate_calibration_sha256']:
                raise ValueError('누적 정책의 빈도 보정 지문 오류')
            calibration = json.loads(rate_path.read_text())
            if calibration['scales'] != frozen['rate_scales']:
                raise ValueError('누적 정책의 고정 빈도 배율 불일치')
            return frozen, RateActionPolicy(policy, policy.manager, policy.thresholds, policy.multiplier, frozen['rate_scales'])
        return frozen, ReversalPathPolicy(policy, policy.manager, policy.thresholds, policy.multiplier)
    base_path, model_path = selection / 'pullback_selection.json', selection / 'action_model.json'
    path = frozen.get('protocol') == 'minute_path_v12'
    recent = frozen.get('protocol') in {'minute_action_v11', 'minute_path_v12'}
    if (frozen.get('protocol') not in {'minute_action_v10', 'minute_action_v11', 'minute_path_v12'} or base_path.stat().st_size > 1024**2
        or model_path.stat().st_size > 1024**2 or sha256(base_path) != frozen['pullback_selection_sha256']
        or frozen['model_sha256'] != {'action_model.json': sha256(model_path)}):
        raise ValueError('분별 관리 정책의 기반·모델 지문 오류')
    previous = json.loads(base_path.read_text())
    _, entry = load_pullback_selection(selection, previous)
    model = (PathActionModels if path else ActionModels).from_dict(json.loads(model_path.read_text()))
    if ((entry.offset_bps, entry.ttl_minutes) != (16, 5) or model.kind != frozen['kind']
        or frozen['training_period'] != (['2019-01-01', '2020-07-01'] if recent else ['2018-03-01', '2020-01-01'])
        or frozen['calibration_period'] != (['2020-07-01', '2021-01-01'] if recent else ['2020-01-01', '2021-01-01'])
        or frozen['selection_period'] != ['2021-01-01', '2022-01-01']
        or frozen['confirmation_period'] != ['2022-01-01', '2023-01-01']
        or frozen['observed_evaluation_period'] != ['2023-01-01', '2026-01-01']
        or frozen['seen_2026_period'] != ['2026-01-01', '2026-10-01']
        or frozen['evaluation_end_exclusive'] != '2026-10-01' or frozen['unseen_evaluation_available'] is not False
        or frozen['risk'] != asdict(risk_config(previous, frozen['sizing'], .04))):
        raise ValueError('분별 관리 정책의 고정 기간·위험 설정 오류')
    return frozen, (PathActionPolicy if path else MinuteActionPolicy)(entry, model, frozen['thresholds'], frozen['multiplier'])


def action_diagnostics(directory: Path, bars: pd.DataFrame, config: EngineConfig) -> dict:
    wait = waiting_diagnostics(directory, bars, config.signal_delay_bars, management_state=True)
    decomp = decompose_run(directory, config.initial_equity)
    trades = pd.read_parquet(directory / 'trades.parquet')
    result = {'waiting': wait, 'decomposition': decomp,
              'management_events': {k: v for k, v in wait['events'].items() if k.startswith('action_')},
              'trades_with_additions': int(trades['adds'].gt(0).sum()) if len(trades) else 0,
              'trades_over_30_minutes': int(trades.hold_bars.gt(30).sum()) if len(trades) else 0}
    save_json(directory / 'action_diagnostics.json', result)
    return result


def imitation(model, frame, thresholds, multiplier):
    scores = model.probabilities(frame[model.features].to_numpy(dtype=float))
    eligible = scores >= np.asarray([thresholds[a] * multiplier for a in ACTIONS])
    predicted = np.where(eligible.any(axis=1), np.asarray(ACTIONS)[eligible.argmax(axis=1)], 'hold')
    rows = {}
    for index, action in enumerate(ACTIONS):
        truth = frame[f'y_{action}'].to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted == action, average='binary', zero_division=0)
        rows[action] = {'support': int(truth.sum()), 'precision': precision, 'recall': recall, 'f1': f1,
                        'average_precision': average_precision_score(truth, scores[:, index]),
                        'eligible_before_priority': int(eligible[:, index].sum()),
                        'predicted_after_priority': int((predicted == action).sum())}
    return {'rows': len(frame), 'actions': rows, 'limits': '원본 상태의 행동별 모사. 복수 정답과 실행 우선순위·봇 상태는 별도'}


def run_action_selection(reference: Path, labels: Path, market: Path, features: Path,
                         confirmation_market: Path, confirmation_features: Path, output: Path,
                         regime: str = 'original') -> Path:
    from .event_research import load_selection
    previous, entry = load_selection(reference)
    if previous.get('protocol') != 'lifecycle_v9':
        raise ValueError('분별 관리 정책은 v9 고정 위험 기준을 사용합니다.')
    if regime not in ('original', 'recent', 'path'):
        raise ValueError('관리 학습 구간 설정 오류')
    path = regime == 'path'
    recent = regime in ('recent', 'path')
    model_class, policy_class = (PathActionModels, PathActionPolicy) if path else (ActionModels, MinuteActionPolicy)
    protocol = f'docs/EXPERIMENT_V{12 if path else 11 if recent else 10}.md'
    hashes = json.loads((labels / 'files.json').read_text())
    names = ['training.parquet', 'calibration.parquet', 'check_2021.parquet', 'order_ledger.parquet', 'summary.json']
    if recent:
        names.append('events.parquet')
    if any(sha256(labels / name) != hashes[name] for name in names) or not json.loads((labels / 'summary.json').read_text())['complete']:
        raise ValueError('관리 정답 자료의 무결성 오류')
    if path and json.loads((labels / 'summary.json').read_text()).get('path_features') != PATH_FEATURES:
        raise ValueError('가격 경로 특징 정답의 형식 오류')
    out = new_run(output, 'action-path-selection' if path else 'action-recent-selection' if recent else 'action-selection', {
        'labels': str(labels), 'files_sha256': sha256(labels / 'files.json'),
        'reference': str(reference), 'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path(protocol)), 'candidate_count': 4, 'regime': regime,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'분별 행동 정책 선택: {out}', flush=True)
    for name in ['pullback_selection.json', 'base_selection.json', 'expansion_models.json']:
        (out / name).write_bytes((reference / name).read_bytes())
    save_json(out / 'candidate_plan.json', candidate_plan())
    try:
        train, calibration, check = (pd.read_parquet(labels / f'{name}.parquet') for name in ['training', 'calibration', 'check_2021'])
        if recent:
            all_events = pd.read_parquet(labels / 'events.parquet')
            train = purged_window(all_events, '2019-01-01', '2020-07-01')
            calibration = purged_window(all_events, '2020-07-01', '2021-01-01')
            del all_events
        train.to_parquet(out / 'training_used.parquet', index=False)
        calibration.to_parquet(out / 'calibration_used.parquet', index=False)
        if (train.label_end.max() >= pd.Timestamp('2020-07-01' if recent else '2020-01-01', tz='UTC') - pd.Timedelta(days=1)
            or calibration.end.min() < pd.Timestamp('2020-07-02' if recent else '2020-01-02', tz='UTC')
            or calibration.label_end.max() >= pd.Timestamp('2021-01-01', tz='UTC') - pd.Timedelta(days=1)
            or check.end.min() < pd.Timestamp('2021-01-01', tz='UTC')):
            raise ValueError('분별 관리 학습·문턱·모사 기간 오류')
        fitted = {}
        for kind in ['logistic', 'tree']:
            model, thresholds, support = model_class.fit(train, calibration, kind)
            save_json(out / f'{kind}_model.json', model.to_dict())
            save_json(out / f'{kind}_thresholds.json', {'thresholds': thresholds, 'support': support})
            fitted[kind] = (model, thresholds)
            print(f'{kind} 학습·별도 문턱 조정 완료: {thresholds}', flush=True)
        del train, calibration
        base = json.loads((out / 'pullback_selection.json').read_text())
        config = risk_config(base, previous['sizing'], .04)
        bars, verified = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'development_input.json', verified)
        rows = []
        for i, candidate in enumerate(candidate_plan()):
            model, thresholds = fitted[candidate['kind']]
            policy = policy_class(PullbackPolicy(entry.base, 16, 5), model, thresholds, candidate['multiplier'])
            metrics = backtest(bars, policy, config, out / f'candidate-{i:02d}')
            action_diagnostics(out / f'candidate-{i:02d}', bars, config)
            rows.append({'candidate': i, **candidate, **metrics, 'eligible': metrics['closed_trades'] >= 20,
                         'score': metrics['total_return'] - .5 * abs(metrics['max_drawdown'])})
            save_json(out / 'development.json', rows)
            print(f'후보 {i}: {metrics["total_return"]:.2%} / 낙폭 {metrics["max_drawdown"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        eligible = [r for r in rows if r['eligible']]
        if not eligible:
            save_json(out / 'selection_failure.json', {'reason': '20거래 적격 후보 없음', 'criteria_relaxed': False})
            return out
        winner = max(eligible, key=lambda r: r['score'])
        model, thresholds = fitted[winner['kind']]
        save_json(out / 'action_model.json', model.to_dict())
        frozen = {'protocol': 'minute_path_v12' if path else 'minute_action_v11' if recent else 'minute_action_v10', 'candidate': winner['candidate'], 'kind': winner['kind'],
                  'multiplier': winner['multiplier'], 'thresholds': thresholds, 'risk': asdict(config),
                  'sizing': previous['sizing'], 'development_metrics': winner,
                  'pullback_selection_sha256': sha256(out / 'pullback_selection.json'),
                  'model_sha256': {'action_model.json': sha256(out / 'action_model.json')},
                  'training_period': ['2019-01-01', '2020-07-01'] if recent else ['2018-03-01', '2020-01-01'],
                  'calibration_period': ['2020-07-01', '2021-01-01'] if recent else ['2020-01-01', '2021-01-01'],
                  'selection_period': ['2021-01-01', '2022-01-01'], 'confirmation_period': ['2022-01-01', '2023-01-01'],
                  'observed_evaluation_period': ['2023-01-01', '2026-01-01'], 'seen_2026_period': ['2026-01-01', '2026-10-01'],
                  'evaluation_end_exclusive': '2026-10-01', 'unseen_evaluation_available': False}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_action_selection(out, frozen)
        save_json(out / 'imitation_2021.json', imitation(model, check, thresholds, winner['multiplier']))
        del bars, check
        bars, verified = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(out / 'confirmation_input.json', verified)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'selected': True, 'profitability_accepted': False})
        (out / 'REPORT.md').write_text('# 분별 관리 사건과 행동별 정책\n\n' + table(pd.DataFrame(rows)[[
            'candidate', 'kind', 'multiplier', 'total_return', 'max_drawdown', 'closed_trades']])
            + f'\n\n후보 {winner["candidate"]} 고정. 2022년 확인 {metrics["total_return"]:.2%}·{metrics["closed_trades"]}거래. '
            '수익성 채택은 후속 조건까지 별도 검증한다. 원본 상태의 행동 점수와 수익 확률을 구분한다.\n', encoding='utf-8')
        print(f'2022년 확인: {metrics["total_return"]:.2%}, {metrics["closed_trades"]}거래', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
