from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from .common import new_run, save_json, sha256
from .edge_model import EdgePolicy
from .engine import EngineConfig
from .event_backtest import backtest, prepare_period
from .event_diagnostics import decompose_run
from .event_evaluation import evaluate_gate
from .event_research import load_selection
from .expansion_model import ExpansionPolicy
from .expansion_research import expansion_gate
from .reports import table
from .robustness import block_interval

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def ensure_expansion_period(selection: Path, frozen: dict, period: str) -> tuple[str, str]:
    if frozen.get('protocol') not in {'expansion_v4', 'edge_v5'} or period not in {'observed', 'seen_2026', 'new'}:
        raise ValueError('v4·v5 선택 파일과 지원하는 평가 기간이 필요합니다.')
    if period == 'new':
        gate = json.loads((selection / 'new_evaluation_gate.json').read_text())
        metrics = selection / 'validation-2021' / 'metrics.json'
        if (sha256(selection / 'frozen_selection.json') != gate['selection_sha256']
            or sha256(metrics) != gate['validation_metrics_sha256']):
            raise ValueError('새 구간 선행 검증 지문 불일치')
        computed = expansion_gate(frozen['development_metrics'], json.loads(metrics.read_text()))
        if not computed['may_open_new_period'] or computed['checks'] != gate['checks']:
            raise ValueError('개발·2021년 조건 미충족으로 새 구간을 열지 않습니다.')
    key = {'observed': 'observed_evaluation_period', 'seen_2026': 'seen_2026_period', 'new': 'new_evaluation_period'}[period]
    return tuple(frozen[key])


def run_expansion_evaluation(selection: Path, reference: Path, market: Path, output: Path, period: str) -> Path:
    frozen, policy = load_selection(selection)
    start, end = ensure_expansion_period(selection, frozen, period)
    previous, previous_policy = load_selection(reference)
    is_edge = frozen['protocol'] == 'edge_v5'
    previous_protocol = 'expansion_v4' if is_edge else 'frequency_v3'
    previous_name = 'previous_v4' if is_edge else 'previous_v3'
    label = 'edge' if is_edge else 'expansion'
    if previous.get('protocol') != previous_protocol:
        raise ValueError('비교 대상은 직전 버전의 고정 선택 파일이어야 합니다.')
    if is_edge and sha256(reference / 'frozen_selection.json') != frozen['reference_sha256']:
        raise ValueError('v5 학습 때 사용한 v4 고정 후보와 다릅니다.')
    config = EngineConfig(**frozen['risk'])
    variants = {'fixed_policy': (policy, config), previous_name: (previous_policy, EngineConfig(**previous['risk'])),
                'cash': (lambda _bar, _state: 'hold', config),
                'cost_x2': (policy, replace(config, fee_bps=10, slippage_bps=6)),
                'cost_x3': (policy, replace(config, fee_bps=15, slippage_bps=9)),
                'extra_bar_delay': (policy, replace(config, signal_delay_bars=1))}
    if is_edge:
        variants['without_edge'] = (EdgePolicy(policy.base, policy.model, policy.margin_bps, filter_enabled=False), config)
    else:
        variants['without_activity'] = (ExpansionPolicy(policy.activity, policy.direction, 0, policy.direction_threshold), config)
    destination = new_run(output, f'{label}-evaluation-{period}', {
        'protocol': 'docs/EXPERIMENT_V5.md' if is_edge else 'docs/EXPERIMENT_V4.md', 'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'), 'period': [start, end], 'role': period,
        'retuning': False, 'variants': {name: risk.__dict__ for name, (_, risk) in variants.items()},
        'reference_difference': '직전 버전과 고정 정책 전체 비교이며 단일 원인의 인과 효과가 아님',
    })
    for name in ['frozen_selection.json', 'frozen_integrity.json', 'edge_models.json' if is_edge else 'expansion_models.json']:
        (destination / name).write_bytes((selection / name).read_bytes())
    reference_copy = destination / previous_name
    reference_copy.mkdir()
    reference_files = ['expansion_models.json'] if is_edge else ['entry_model.json', 'management_model.json', 'files.json']
    for name in ['frozen_selection.json', 'frozen_integrity.json'] + reference_files:
        (reference_copy / name).write_bytes((reference / name).read_bytes())
    save_json(destination / 'evaluation_observed.json', {'role': period, 'period': [start, end],
                                                        'status': 'started', 'do_not_reuse_as_unseen': True})
    print(f'{label} {period} 평가: {destination}', flush=True)
    rows, annual, uncertainty, decompositions = [], [], [], []
    chart, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
    try:
        for axis, symbol in zip(axes, ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], strict=True):
            bars = prepare_period(market, symbol, start, end)
            for name, (strategy, risk) in variants.items():
                target = destination / symbol / name
                metrics = backtest(bars, strategy, risk, target)
                rows.append({'symbol': symbol, 'strategy': name, **metrics})
                save_json(destination / 'results_partial.json', rows)
                print(f'{period} {symbol}/{name}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
                decompositions.append({'symbol': symbol, 'strategy': name, **decompose_run(target, risk.initial_equity)})
                if name in {'fixed_policy', previous_name, 'cash'}:
                    curve = pd.read_parquet(target / 'equity.parquet')
                    times = pd.to_datetime(curve.time, utc=True)
                    axis.plot(times, curve.equity / risk.initial_equity, label=name, lw=.8)
                    daily = curve.groupby((times - pd.Timedelta(nanoseconds=1)).dt.date).equity.last()
                    returns = (daily / daily.shift(1, fill_value=risk.initial_equity) - 1).to_numpy()
                    interval = block_interval(returns) if len(returns) >= 60 else {'status': 'insufficient_days', 'days': len(returns)}
                    uncertainty.append({'symbol': symbol, 'strategy': name, **interval})
            if period == 'observed':
                for year in range(2022, 2026):
                    part = bars[(bars.time >= f'{year}-01-01') & (bars.time < f'{year+1}-01-01')]
                    metrics = backtest(part, policy, config, destination / symbol / f'restart-{year}')
                    annual.append({'symbol': symbol, 'year': year, **metrics})
                    save_json(destination / 'annual_restart.json', annual)
            axis.set(title=f'{symbol}: {label} policy ({period})', ylabel='Equity / initial')
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        chart.savefig(destination / 'equity_comparison.png', dpi=150)
        save_json(destination / 'results.json', rows)
        pd.DataFrame(rows).drop(columns='rejected').to_csv(destination / 'results.csv', index=False)
        save_json(destination / 'bootstrap.json', uncertainty)
        save_json(destination / 'decomposition.json', decompositions)
        gate = evaluate_gate(frozen['development_metrics'], rows)
        decision = {'metric_checks': gate['checks'], 'evaluation_role': period,
                    'profitability_review_candidate': False, 'live_trading_approved': False,
                    'reason': '관찰한 구간의 탐색 결과' if period != 'new' else '한 달의 후속 점검만으로 장기 수익성을 입증하지 않음'}
        save_json(destination / 'decision.json', decision)
        save_json(destination / 'evaluation_observed.json', {'role': period, 'period': [start, end],
                                                            'status': 'completed', 'do_not_reuse_as_unseen': True})
        summary = pd.DataFrame(rows)[['symbol', 'strategy', 'total_return', 'max_drawdown', 'closed_trades', 'fees', 'permanent_halt']]
        (destination / 'REPORT.md').write_text(
            f'# {label} 후보의 {period} 평가\n\n{table(summary)}\n\n'
            '개발에서 고정한 후보를 그대로 적용했다. 직전 버전은 자신의 고정 설정으로 비교한다. '
            'without_activity는 v4의 활동 기준만, without_edge는 v5의 예상 비용 조건만 제거한다. '
            '가격 손익·비용·회전율·보유 시간은 decomposition.json에 보존했다.\n\n'
            '다음 봉 시가·수수료·슬리피지·펀딩과 위험 청산을 유지했다. '
            'maker 체결이나 리베이트를 가정하지 않았다. 원본의 추가 진입을 방향 학습에 썼지만 봇은 물타기를 하지 않는다.\n\n'
            '2022년부터 2026년 8월까지는 이미 관찰한 기간이다. 연도별 초기화는 재학습이나 연속 운용이 아니다. '
            '30일 블록 재표집은 관찰 경로에 조건부이며 모형 선택 불확실성을 포함하지 않는다. '
            '단일 시장·조건의 양수만으로 채택하지 않으며 실제 거래 승인은 하지 않는다.\n', encoding='utf-8')
        save_json(destination / 'summary.json', {'complete': True, 'decision': decision})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        plt.close(chart)
    print(f'{label} 평가 완료: {destination}', flush=True)
    return destination
