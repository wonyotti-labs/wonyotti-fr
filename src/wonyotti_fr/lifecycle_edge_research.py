from __future__ import annotations

from pathlib import Path

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_research import load_selection
from .lifecycle_edge import lifecycle_labels
from .minute_data import prepare_minute_period
from .net_edge_labels import potential_entries


def run_lifecycle_edge_labels(reference: Path, market: Path, features: Path, output: Path, *, direction_only: bool = False) -> Path:
    frozen, policy = load_selection(reference)
    if frozen['protocol'] != 'minute_rate_v14':
        raise ValueError('전체 거래 정답은 고정 v14 관리 정책이 필요합니다.')
    if type(direction_only) is not bool:
        raise ValueError('진입 활동 관문 선택의 형식 오류')
    if direction_only:
        from .entry_scope import direction_only_manager
        policy = direction_only_manager(policy)
    out = new_run(output, 'direction-edge-labels' if direction_only else 'lifecycle-edge-labels', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V17.md' if direction_only else 'docs/EXPERIMENT_V15.md')),
        'entry_activity_gate': not direction_only,
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features / 'manifest-5m.json'),
        'training_period': ['2021-01-01', '2022-01-01'], 'forced_boundary_closes': False})
    print(f'전체 거래 순손익 정답: {out}', flush=True)
    try:
        bars, verified = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'input_verification.json', verified)
        opportunities, counts = potential_entries(bars, policy)
        opportunities.to_parquet(out / 'potential_entries.parquet', index=False)
        print(f'독립 진입 기회 {len(opportunities)}개', flush=True)

        def checkpoint(frame):
            temporary = out / 'labels_partial.parquet.tmp'
            frame.to_parquet(temporary, index=False)
            temporary.replace(out / 'labels_partial.parquet')
            if len(frame) % 100 == 0:
                print(f'전체 거래 정답 {len(frame)}/{len(opportunities)}개 처리', flush=True)

        training, ledger, details = lifecycle_labels(bars, opportunities, EngineConfig(**frozen['risk']),
                                                     policy, '2022-01-01', checkpoint)
        training.to_parquet(out / 'training_labels.parquet', index=False)
        ledger.to_parquet(out / 'opportunity_ledger.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'signals': counts, 'labels': details,
            'losses_preserved': int(training.net_bps.lt(0).sum()) if len(training) else 0,
            'positives': int(training.net_bps.gt(0).sum()) if len(training) else 0})
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out / 'REPORT.md').write_text('# 현재 관리 정책의 전체 거래 정답\n\n'
            f'독립 기회 {len(opportunities)}개 중 자연 종료 {len(training)}개. '
            '손실·손절·위험 종료를 포함하며 학습 경계를 넘긴 포지션은 미확정 원장에 보존했다. '
            '시간 경계에서 강제 청산해 정답을 만들지 않았다. 같은 초기 계좌·고정 v14 관리의 겹친 기회이며 '
            '연속 계좌 상태나 독립 표본을 뜻하지 않는다. 2021년은 관리 정책 선택에도 사용한 전체 시스템 학습 구간이다.\n')
        print(f'정답 생성 완료: 자연 종료 {len(training)}개', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def load_lifecycle_edge_selection(selection: Path, frozen: dict):
    import json

    from .action_research import load_action_selection
    from .lifecycle_edge import LifecycleNetPolicy
    from .net_edge_model import NetEdgeModel

    parent_path, model_path = selection / 'rate_selection.json', selection / 'net_model.json'
    if (frozen.get('protocol') not in {'lifecycle_edge_v15', 'lifecycle_edge_v16'} or parent_path.stat().st_size > 1024**2
        or model_path.stat().st_size > 1024**2 or sha256(parent_path) != frozen['rate_selection_sha256']):
        raise ValueError('전체 거래 순손익 정책의 기반 지문 오류')
    parent = json.loads(parent_path.read_text())
    if (parent.get('protocol') != 'minute_rate_v14' or frozen['candidate'] != 0
        or frozen['net_training_period'] != ['2021-01-01', '2022-01-01']
        or frozen['alpha'] != 100 or frozen['margin_bps'] != 8
        or any(frozen.get(k) != v for k, v in parent.items()
               if k not in {'protocol', 'candidate', 'development_metrics', 'model_sha256'})
        or frozen['model_sha256'] != {**parent['model_sha256'], 'net_model.json': sha256(model_path)}):
        raise ValueError('전체 거래 순손익 정책의 고정 설정·모델 오류')
    if frozen['protocol'] == 'lifecycle_edge_v16':
        from .label_weighting import WEIGHTING
        if frozen.get('sample_weighting') != WEIGHTING:
            raise ValueError('중첩 가중치의 고정 방식 오류')
        for name, key in [('training_weights.parquet', 'training_weights_sha256'), ('weighting.json', 'weighting_sha256')]:
            if (selection / name).stat().st_size > 1024**2 or sha256(selection / name) != frozen[key]:
                raise ValueError('중첩 가중치의 원장·설정 지문 오류')
    _, policy = load_action_selection(selection, parent)
    model = NetEdgeModel.from_dict(json.loads(model_path.read_text()))
    return frozen, LifecycleNetPolicy(policy, model)


def lifecycle_edge_diagnostics(directory, bars, policy, config):
    import numpy as np
    import pandas as pd

    from .action_research import action_diagnostics
    from .event_features import MARKET_FEATURES
    from .lifecycle_edge import LifecycleNetPolicy
    from .net_edge_model import net_values

    details = action_diagnostics(directory, bars, config)
    if isinstance(policy, LifecycleNetPolicy) and policy.enabled:
        episodes = pd.read_parquet(directory / 'waiting_episodes.parquet')
        selected = episodes[episodes.status.isin(['triggered', 'filtered'])].copy() if len(episodes) else episodes.copy()
        if len(selected):
            market = bars.set_index('end').reindex(selected.decision_time)[MARKET_FEATURES].to_numpy()
            favorable = selected.direction.to_numpy() * np.log(selected.reference_price / selected.decision_close).to_numpy() * 10000
            scores = policy.model.predict(net_values(market, selected.direction, favorable, selected.wait_minutes))
            accepted = np.isfinite(scores) & (scores >= 8)
            if not np.array_equal(accepted, selected.status.eq('triggered')) or selected.loc[~accepted, 'executed'].any():
                raise ValueError('전체 거래 순손익 필터의 저장 결정·실행 불일치')
            selected['predicted_net_bps'] = scores
        selected.to_parquet(directory / 'net_gate_decisions.parquet', index=False)
        details['gate'] = {'opportunities': len(selected), 'accepted': details['waiting']['events'].get('triggered', 0),
                           'rejected': details['waiting']['events'].get('filtered', 0), 'decisions_recomputed': True}
    save_json(directory / 'lifecycle_edge_diagnostics.json', details)
    return details


def run_lifecycle_edge_selection(reference: Path, labels: Path, market: Path, features: Path,
                                 confirmation_market: Path, confirmation_features: Path, output: Path, *, overlap_weighted: bool = False) -> Path:
    import json

    import numpy as np
    import pandas as pd

    from .event_backtest import backtest
    from .lifecycle_edge import LifecycleNetPolicy
    from .net_edge_model import NetEdgeModel
    from .reports import table

    parent, manager = load_selection(reference)
    settings = json.loads((labels / 'manifest.json').read_text())['settings']
    hashes = json.loads((labels / 'files.json').read_text())
    if (parent['protocol'] != 'minute_rate_v14' or settings['reference_sha256'] != sha256(reference / 'frozen_selection.json')
        or settings['training_period'] != ['2021-01-01', '2022-01-01']
        or settings['market_manifest_sha256'] != sha256(market / 'manifest-1m.json')
        or settings['feature_manifest_sha256'] != sha256(features / 'manifest-5m.json')
        or any(sha256(labels / name) != hashes[name] for name in ['training_labels.parquet', 'opportunity_ledger.parquet', 'summary.json'])
        or json.loads((labels / 'summary.json').read_text())['complete'] is not True):
        raise ValueError('전체 거래 순손익 학습의 고정 기반·정답 지문 오류')
    if type(overlap_weighted) is not bool:
        raise ValueError('중첩 가중치 선택의 형식 오류')
    version = 16 if overlap_weighted else 15
    out = new_run(output, 'lifecycle-weighted-selection' if overlap_weighted else 'lifecycle-edge-selection', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path(f'docs/EXPERIMENT_V{version}.md')), 'overlap_weighted': overlap_weighted, 'candidate_count': 1, 'alpha': 100, 'margin_bps': 8,
        'in_sample_period': ['2021-01-01', '2022-01-01'], 'all_periods_already_observed': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'전체 거래 순손익 필터 학습: {out}', flush=True)
    try:
        train = pd.read_parquet(labels / 'training_labels.parquet')
        ledger = pd.read_parquet(labels / 'opportunity_ledger.parquet')
        pd.testing.assert_frame_equal(train.reset_index(drop=True),
            ledger.loc[ledger.label_status.eq('closed'), train.columns].reset_index(drop=True), check_exact=True, check_dtype=False)
        if (train.empty or train.decision_time.min() < pd.Timestamp('2021-01-01', tz='UTC')
            or train.label_end.max() >= pd.Timestamp('2021-12-31', tz='UTC')
            or train.decision_time.ge(train.label_end).any() or train.entry_notional.le(0).any()
            or not np.allclose(train.net_bps, train.net_pnl / train.entry_notional * 10000, rtol=0, atol=1e-8)):
            raise ValueError('전체 거래 순손익 정답의 시간·명목액·값 오류')
        weight_options, weight_frozen = {}, {}
        if overlap_weighted:
            from .label_weighting import WEIGHTING, lifecycle_weights
            weights, weight_ledger, weight_report = lifecycle_weights(train)
            weight_ledger.to_parquet(out / 'training_weights.parquet', index=False)
            save_json(out / 'weighting.json', weight_report)
            weight_options['sample_weight'] = weights
            weight_frozen = {'sample_weighting': WEIGHTING, 'training_weights_sha256': sha256(out / 'training_weights.parquet'),
                             'weighting_sha256': sha256(out / 'weighting.json')}
        model, support = NetEdgeModel.fit(train, 100, **weight_options)
        support.update(negative_labels=int(train.net_bps.lt(0).sum()), positive_labels=int(train.net_bps.gt(0).sum()),
                       overlapping_previous_labels=int(train.decision_time.lt(train.label_end.cummax().shift()).sum()),
                       median_hold_minutes=float(train.hold_minutes.median()),
                       monthly_counts=train.decision_time.dt.strftime('%Y-%m').value_counts().sort_index().to_dict())
        save_json(out / 'training_diagnostics.json', support)
        save_json(out / 'net_model.json', model.to_dict())
        train.to_parquet(out / 'training_used.parquet', index=False)
        for name in ['path_selection.json', 'rate_calibration.json', 'action_model.json', 'pullback_selection.json',
                     'base_selection.json', 'expansion_models.json']:
            (out / name).write_bytes((reference / name).read_bytes())
        (out / 'rate_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
        config = EngineConfig(**parent['risk'])
        policy = LifecycleNetPolicy(manager, model)
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'in_sample_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'candidate-00')
        lifecycle_edge_diagnostics(out / 'candidate-00', bars, policy, config)
        save_json(out / 'development.json', [{'candidate': 0, 'in_sample': True, 'selected_by_pnl': False, **metrics}])
        frozen = {**parent, **weight_frozen, 'protocol': f'lifecycle_edge_v{version}', 'candidate': 0, 'development_metrics': metrics,
                  'rate_selection_sha256': sha256(out / 'rate_selection.json'), 'net_training_period': ['2021-01-01', '2022-01-01'],
                  'alpha': 100, 'margin_bps': 8, 'net_training_labels_sha256': hashes['training_labels.parquet'],
                  'model_sha256': {**parent['model_sha256'], 'net_model.json': sha256(out / 'net_model.json')}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_lifecycle_edge_selection(out, frozen)
        print(f'2021년 학습 구간: {metrics["total_return"]:.2%}, {metrics["closed_trades"]}거래', flush=True)
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(out / 'confirmation_input.json', checks)
        confirmation = backtest(bars, policy, config, out / 'confirmation-2022')
        lifecycle_edge_diagnostics(out / 'confirmation-2022', bars, policy, config)
        save_json(out / 'confirmation_checks.json', {'positive': confirmation['total_return'] > 0,
            'at_least_30_trades': confirmation['closed_trades'] >= 30, 'no_halt': not confirmation['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        rows = [{'period': '2021_in_sample', 'policy': 'unfiltered_v14', **parent['development_metrics']},
                {'period': '2021_in_sample', 'policy': f'fixed_v{version}', **metrics},
                {'period': '2022_confirmation', 'policy': 'unfiltered_v14', **json.loads((reference / 'confirmation-2022/metrics.json').read_text())},
                {'period': '2022_confirmation', 'policy': f'fixed_v{version}', **confirmation}]
        save_json(out / 'comparison.json', rows)
        (out / 'REPORT.md').write_text('# 전체 거래 순손익 진입 필터\n\n' + table(pd.DataFrame(rows)[
            ['period', 'policy', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + ('\n\n동시 정답 수 역수의 구간 평균을 정규화해 표준화·Ridge에 같은 가중치를 적용했다. ' if overlap_weighted else '')
            + '\n\n2021년은 전체 시스템 학습 구간이며 같은 구간의 재생은 일반화 성과가 아니다. '
            '독립 초기 계좌의 전체 관리 결과로 alpha 100·8bp 단일 필터를 학습했다. 손실과 경계 미확정 원장을 보존했다. '
            '겹친 기회의 표본 의존성과 실제 연속 계좌의 위험 상태 차이가 남는다. 2022년 이후도 이전에 관찰한 기간이다. '
            '수익성 기준과 전체 goal 완료는 별도 검증한다.\n')
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'in_sample_reported': True})
        print(f'2022년 확인: {confirmation["total_return"]:.2%}, {confirmation["closed_trades"]}거래', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
