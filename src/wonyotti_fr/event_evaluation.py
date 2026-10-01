from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from .common import new_run, records, save_json, sha256
from .engine import EngineConfig
from .event_backtest import EventPolicy, backtest, prepare_period
from .event_research import load_selection
from .reports import table
from .robustness import block_interval

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def evaluate_gate(development: dict, results: list[dict]) -> dict:
    fixed = [row for row in results if row['strategy'] == 'fixed_policy']
    checks = {'development_positive': development['total_return'] > 0,
              'all_markets_present': {row['symbol'] for row in fixed} == {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'},
              'enough_trades': len(fixed) == 3 and all(row['closed_trades'] >= 20 for row in fixed),
              'positive_all_markets': len(fixed) == 3 and all(row['total_return'] > 0 for row in fixed),
              'drawdown_within_limit': len(fixed) == 3 and all(row['max_drawdown'] >= -0.25 for row in fixed),
              'no_permanent_halt': len(fixed) == 3 and all(not row['permanent_halt'] for row in fixed)}
    return {'checks': checks, 'profitability_review_candidate': all(checks.values()), 'live_trading_approved': False}


def run_event_evaluation(selection: Path, market: Path, output: Path, period: str) -> Path:
    if period not in {'observed', 'new'}:
        raise ValueError('평가 구간은 observed 또는 new입니다.')
    frozen, policy = load_selection(selection)
    if frozen.get('protocol') == 'frequency_v3':
        raise ValueError('v3는 새 자료 개봉 조건을 검사하는 frequency-evaluate 명령을 사용합니다.')
    start, end = frozen['observed_evaluation_period' if period == 'observed' else 'new_evaluation_period']
    config = EngineConfig(**frozen['risk'])
    variants = {
        'fixed_policy': (policy, config),
        'trend': (EventPolicy(policy.entry, policy.management, 0, 0, 'trend'), config),
        'mean_reversion': (EventPolicy(policy.entry, policy.management, 0, 0, 'mean_reversion'), config),
        'cash': (EventPolicy(policy.entry, policy.management, 0, 0, 'cash'), config),
        'without_stop': (policy, replace(config, stop_fraction=0)),
        'without_time_limit': (policy, replace(config, max_hold_bars=0)),
        'allow_adverse_add': (policy, replace(config, allow_adverse_add=True)),
        'without_additions': (policy, replace(config, max_adds=0)),
        'cost_x2': (policy, replace(config, fee_bps=10, slippage_bps=6)),
        'cost_x3': (policy, replace(config, fee_bps=15, slippage_bps=9)),
        'extra_bar_delay': (policy, replace(config, signal_delay_bars=1)),
    }
    destination = new_run(output, f'event-evaluation-{period}', {
        'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'market_manifest_sha256': sha256(market / 'manifest-5m.json'),
        'period': [start, end], 'role': period, 'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        'variants': {name: risk.__dict__ for name, (_, risk) in variants.items()},
        'retuning': False, 'bootstrap': {'days': 30, 'samples': 1000, 'seed': 41},
        'interpretation': '고정 경로의 조건부 재표집이며 선택 불확실성을 포함하지 않음',
    })
    for name in ['frozen_selection.json', 'frozen_integrity.json', 'entry_model.json', 'management_model.json', 'files.json']:
        (destination / name).write_bytes((selection / name).read_bytes())
    save_json(destination / 'evaluation_observed.json', {'period': [start, end], 'role': period,
                                                        'status': 'started', 'do_not_reuse_as_unseen': True})
    print(f'{period} 평가 시작: {destination}', flush=True)
    rows, annual, uncertainty = [], [], []
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
                if name in {'fixed_policy', 'trend', 'mean_reversion', 'cash'}:
                    curve = pd.read_parquet(target / 'equity.parquet')
                    times = pd.to_datetime(curve.time, utc=True)
                    axis.plot(times, curve.equity / config.initial_equity, label=name, lw=0.8)
                    daily = curve.groupby((times - pd.Timedelta(nanoseconds=1)).dt.date).equity.last()
                    returns = (daily / daily.shift(1, fill_value=config.initial_equity) - 1).to_numpy()
                    uncertainty.append({'symbol': symbol, 'strategy': name, **block_interval(returns)})
            if period == 'observed':
                for year in range(2022, 2026):
                    part = bars[(bars.time >= f'{year}-01-01') & (bars.time < f'{year+1}-01-01')]
                    result = backtest(part, policy, config, destination / symbol / f'restart-{year}')
                    annual.append({'symbol': symbol, 'year': year, **result})
                    save_json(destination / 'annual_restart.json', annual)
            axis.set(title=f'{symbol}: {period} ({start} to {end})', ylabel='Equity / initial')
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        chart.savefig(destination / 'equity_comparison.png', dpi=150)
        results = pd.DataFrame(rows)
        results.drop(columns='rejected').to_csv(destination / 'results.csv', index=False)
        save_json(destination / 'results.json', rows)
        save_json(destination / 'bootstrap.json', uncertainty)
        gate = evaluate_gate(frozen['development_metrics'], rows)
        gate['evaluation_role'] = period
        if period != 'new':
            gate['profitability_review_candidate'] = False
            gate['reason'] = '이미 관찰한 탐색 구간의 진단 결과'
        save_json(destination / 'decision.json', gate)
        save_json(destination / 'evaluation_observed.json', {'period': [start, end], 'role': period,
                                                            'status': 'completed', 'do_not_reuse_as_unseen': True})
        summary = results[['symbol', 'strategy', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']]
        (destination / 'REPORT.md').write_text(
            f'# 사건별 고정 후보의 {period} 평가\n\n{table(summary)}\n\n'
            f'수익성 검토 후보 여부: {gate["profitability_review_candidate"]}. 실제 거래 승인은 하지 않는다.\n\n'
            '모델·신뢰도·위험 설정을 고정하고 수수료와 펀딩을 반영했다. '
            '손절·시간 제한·불리한 추가 진입 차단의 영향과 추가 진입 금지, 비용 2~3배, 한 봉 지연을 구분했다. '
            '추가 진입의 총 노출과 횟수 한도는 불리한 가격 허용 실험에서도 유지했다.\n\n'
            '매년 초기화한 비교는 연속 운용이나 재학습 워크포워드가 아니다. '
            '30일 블록 1,000회 재표집은 관찰한 경로에 조건부이며 미래 수익 확률·통계적 인증이 아니다. '
            '모델 선택 불확실성과 구조 변화를 모두 포함하지 못한다.\n\n'
            '기존 2022~2025년 평가를 이미 관찰했다. 새 2026년 구간도 이 실행 이후 다시 미사용 평가로 부를 수 없다. '
            '5분봉은 호가 대기열·시장 충격·봉 안의 정확한 사건 순서를 제공하지 않으며, 선형 USDT 모의 회계는 원본 역선물 계좌와 다르다.\n', encoding='utf-8')
        save_json(destination / 'summary.json', {'gate': gate, 'rows': records(summary), 'complete': True})
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        plt.close(chart)
    print(f'평가 보고서: {destination / "REPORT.md"}', flush=True)
    return destination
