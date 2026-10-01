from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import EventPolicy, backtest, prepare_period, read_models
from .event_diagnostics import decompose_run
from .event_evaluation import evaluate_gate
from .event_research import load_selection
from .frequency_research import frequency_gate
from .reports import table
from .robustness import block_interval

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def ensure_period_allowed(selection: Path, frozen: dict, period: str) -> tuple[str, str]:
    if frozen.get('protocol') != 'frequency_v3' or period not in {'observed', 'seen_2026', 'new'}:
        raise ValueError('v3 선택 파일과 지원하는 평가 기간이 필요합니다.')
    if period == 'new':
        gate = json.loads((selection / 'new_evaluation_gate.json').read_text())
        metrics_path = selection / 'validation-2021' / 'metrics.json'
        if (sha256(selection / 'frozen_selection.json') != gate['selection_sha256']
            or sha256(metrics_path) != gate['validation_metrics_sha256']):
            raise ValueError('새 평가 구간의 선행 검증 지문이 다릅니다.')
        computed = frequency_gate(frozen['development_metrics'], json.loads(metrics_path.read_text()))
        if not computed['may_open_new_period'] or computed['checks'] != gate['checks']:
            raise ValueError('개발·2021년 조건 미충족으로 새 평가 구간을 열지 않습니다.')
    key = {'observed': 'observed_evaluation_period', 'seen_2026': 'seen_2026_period', 'new': 'new_evaluation_period'}[period]
    return tuple(frozen[key])


def run_frequency_evaluation(selection: Path, market: Path, output: Path, period: str) -> Path:
    frozen, policy = load_selection(selection)
    start, end = ensure_period_allowed(selection, frozen, period)
    raw_entry, raw_management = read_models(selection)
    raw = EventPolicy(raw_entry, raw_management, frozen['entry_threshold'], frozen['management_threshold'])
    config = EngineConfig(**frozen['risk'])
    variants = {'fixed_policy': (policy, config), 'raw_score': (raw, config),
                'cash': (EventPolicy(raw_entry, raw_management, 0, 0, 'cash'), config),
                'cost_x2': (policy, replace(config, fee_bps=10, slippage_bps=6)),
                'cost_x3': (policy, replace(config, fee_bps=15, slippage_bps=9)),
                'extra_bar_delay': (policy, replace(config, signal_delay_bars=1))}
    destination = new_run(output, f'frequency-evaluation-{period}', {
        'protocol': 'docs/EXPERIMENT_V3.md', 'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'), 'period': [start, end],
        'role': period, 'retuning': False, 'variants': {name: risk.__dict__ for name, (_, risk) in variants.items()},
        'raw_comparator': '같은 모델·신뢰도·위험 설정에서 빈도 조정만 제거',
    })
    for name in ['frozen_selection.json', 'frozen_integrity.json', 'entry_model.json', 'management_model.json', 'files.json']:
        (destination / name).write_bytes((selection / name).read_bytes())
    save_json(destination / 'evaluation_observed.json', {'role': period, 'period': [start, end],
                                                        'status': 'started', 'do_not_reuse_as_unseen': True})
    print(f'빈도 반영 {period} 평가: {destination}', flush=True)
    rows, annual, uncertainty, decomposition = [], [], [], []
    chart, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
    try:
        for axis, symbol in zip(axes, ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], strict=True):
            bars = prepare_period(market, symbol, start, end)
            for name, (strategy, risk) in variants.items():
                directory = destination / symbol / name
                metrics = backtest(bars, strategy, risk, directory)
                rows.append({'symbol': symbol, 'strategy': name, **metrics})
                save_json(destination / 'results_partial.json', rows)
                print(f'{period} {symbol}/{name}: 수익 {metrics["total_return"]:.2%}, 낙폭 {metrics["max_drawdown"]:.2%}, 거래 {metrics["closed_trades"]}', flush=True)
                if name in {'fixed_policy', 'raw_score', 'cash'}:
                    curve = pd.read_parquet(directory / 'equity.parquet')
                    times = pd.to_datetime(curve.time, utc=True)
                    axis.plot(times, curve.equity / config.initial_equity, label=name, lw=0.8)
                    daily = curve.groupby((times - pd.Timedelta(nanoseconds=1)).dt.date).equity.last()
                    returns = (daily / daily.shift(1, fill_value=config.initial_equity) - 1).to_numpy()
                    intervals = block_interval(returns) if len(returns) >= 60 else {'status': 'insufficient_days', 'days': len(returns)}
                    uncertainty.append({'symbol': symbol, 'strategy': name, **intervals})
                    decomposition.append({'symbol': symbol, 'strategy': name, **decompose_run(directory, config.initial_equity)})
            if period == 'observed':
                for year in range(2022, 2026):
                    part = bars[(bars.time >= f'{year}-01-01') & (bars.time < f'{year+1}-01-01')]
                    metrics = backtest(part, policy, config, destination / symbol / f'restart-{year}')
                    annual.append({'symbol': symbol, 'year': year, **metrics})
                    save_json(destination / 'annual_restart.json', annual)
            axis.set(title=f'{symbol}: frequency adjustment ({period})', ylabel='Equity / initial')
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        chart.savefig(destination / 'equity_comparison.png', dpi=150)
        save_json(destination / 'results.json', rows)
        pd.DataFrame(rows).drop(columns='rejected').to_csv(destination / 'results.csv', index=False)
        save_json(destination / 'bootstrap.json', uncertainty)
        save_json(destination / 'decomposition.json', decomposition)
        gate = evaluate_gate(frozen['development_metrics'], rows)
        decision = {'metric_checks': gate['checks'], 'evaluation_role': period,
                    'profitability_review_candidate': False, 'live_trading_approved': False,
                    'reason': '관찰한 구간의 탐색 결과' if period != 'new' else '한 달의 후속 점검만으로 장기 수익성을 입증하지 않음'}
        save_json(destination / 'decision.json', decision)
        save_json(destination / 'evaluation_observed.json', {'role': period, 'period': [start, end],
                                                            'status': 'completed', 'do_not_reuse_as_unseen': True})
        summary = pd.DataFrame(rows)[['symbol', 'strategy', 'total_return', 'max_drawdown', 'closed_trades', 'fees', 'permanent_halt']]
        (destination / 'REPORT.md').write_text(
            f'# 학습 행동 빈도 반영 후보의 {period} 평가\n\n{table(summary)}\n\n'
            '원래 가중 점수와 같은 신뢰도·위험 설정을 사용하여 빈도 조정의 영향을 비교했다. '
            '수수료·슬리피지·펀딩·다음 봉 실행은 유지했다. 가격 손익·회전율·보유 시간은 decomposition.json에 보존했다.\n\n'
            '비용 2~3배와 추가 한 봉 지연을 별도로 적용했다. 연도별 초기화는 연속 운용이나 재학습이 아니다. '
            '30일 블록 재표집은 관찰한 경로에 조건부이며 선택 불확실성까지 포함하지 않는다. '
            '60일 미만이면 재표집 구간을 산출하지 않는다.\n\n'
            '2022년부터 2026년 8월까지는 이미 관찰한 기간이다. '
            '빈도 조정이 확률 보정이나 금융 수익의 확률을 보장하지 않는다. '
            '무거래나 소수 거래의 손실 감소만으로 우수한 전략이라고 판단하지 않는다. 실제 거래 승인은 하지 않는다.\n', encoding='utf-8')
        save_json(destination / 'summary.json', {'complete': True, 'decision': decision})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        plt.close(chart)
    print(f'빈도 반영 평가 완료: {destination / "REPORT.md"}', flush=True)
    return destination
