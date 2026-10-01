from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json
from .context_study import execution_costs
from .event_features import independent_orders
from .expansion_data import source_inputs
from .reports import table


def markouts(orders: pd.DataFrame, fills: pd.DataFrame, bars: pd.DataFrame,
             horizons: tuple[int, ...] = (5, 15, 60, 240)) -> pd.DataFrame:
    if (bars.end.duplicated().any() or not bars.end.is_monotonic_increasing
        or not np.isfinite(bars.close).all() or (bars.close <= 0).any()):
        raise ValueError('가격 기준의 순서·유한성 오류')
    first = fills.drop_duplicates('order_key', keep='first')
    frame = orders.merge(first[['order_key', 'lastpx', 'execcomm', 'execcost', 'lastliquidityind']],
                          on='order_key', validate='one_to_one')
    if (frame.lastpx.le(0).any() or frame.execcost.eq(0).any()
        or not np.isfinite(frame[['lastpx', 'execcomm', 'execcost']]).all().all()):
        raise ValueError('최초 체결 가격·비용 오류')
    prices = bars[['end', 'close']].copy().astype({'end': 'datetime64[ns, UTC]'})
    frame = pd.merge_asof(frame.sort_values('target_time'), prices.rename(columns={'end': 'reference_time', 'close': 'reference_price'}),
                          left_on='target_time', right_on='reference_time', direction='backward',
                          allow_exact_matches=False, tolerance=pd.Timedelta(minutes=5))
    direction = np.where(frame.side.eq('Buy'), 1, -1)
    frame['reference_to_fill_bps'] = direction * np.log(frame.reference_price / frame.lastpx) * 10000
    frame['first_fill_fee_bps'] = frame.execcomm / frame.execcost.abs() * 10000
    rows = []
    for horizon in horizons:
        if type(horizon) is not int or horizon <= 0:
            raise ValueError('이후 관찰 간격은 양의 정수여야 합니다.')
        part = frame.copy()
        part['requested_end'] = part.target_time + pd.Timedelta(minutes=horizon)
        part = pd.merge_asof(part, prices.rename(columns={'end': 'outcome_time', 'close': 'outcome_price'}),
                             left_on='requested_end', right_on='outcome_time', direction='forward',
                             tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))
        part['horizon_minutes'] = horizon
        part['markout_bps'] = direction * np.log(part.outcome_price / part.lastpx) * 10000
        part['subsequent_reference_move_bps'] = direction * np.log(part.outcome_price / part.reference_price) * 10000
        part['after_first_fee_bps'] = part.markout_bps - part.first_fill_fee_bps
        part['year'] = part.target_time.dt.year
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def clustered_interval(frame: pd.DataFrame) -> dict:
    daily = frame.set_index('target_time').after_first_fee_bps.resample('1D').agg(['sum', 'count'])
    if len(daily) < 14:
        return {'ci_low_bps': None, 'ci_high_bps': None, 'calendar_days': len(daily)}
    sums, counts = daily['sum'].to_numpy(), daily['count'].to_numpy()
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(daily), size=(1000, (len(daily) + 6) // 7))
    indices = ((starts[:, :, None] + np.arange(7)) % len(daily)).reshape(1000, -1)[:, :len(daily)]
    denominators = counts[indices].sum(axis=1)
    samples = sums[indices].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {'ci_low_bps': float(lower), 'ci_high_bps': float(upper), 'calendar_days': len(daily)}


def run_execution_study(audit: Path, study: Path, history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    destination = new_run(output, 'execution-study', {**hashes, 'role': 'descriptive_not_trade_signal',
                                                    'horizons_minutes': [5, 15, 60, 240], 'bootstrap': '7-day circular blocks, 1000, seed 0'})
    try:
        fills = source['executions'].query("symbol == 'XBTUSD' and exectype == 'Trade'")
        orders = independent_orders(source['executions'], source['actions'])
        frame = markouts(orders, fills, source['bars'])
        frame.to_parquet(destination / 'order_markouts.parquet', index=False)
        valid = frame.dropna(subset=['reference_to_fill_bps', 'markout_bps'])
        rows = []
        keys = ['year', 'target', 'lastliquidityind', 'horizon_minutes']
        for group, part in valid.groupby(keys, observed=True):
            rows.append({**dict(zip(keys, group, strict=True)), 'orders': len(part),
                         **{f'mean_{name}': float(part[name].mean()) for name in
                            ['markout_bps', 'reference_to_fill_bps', 'subsequent_reference_move_bps', 'first_fill_fee_bps', 'after_first_fee_bps']},
                         **clustered_interval(part)})
        summary = pd.DataFrame(rows)
        summary.to_csv(destination / 'markout_summary.csv', index=False)
        costs = execution_costs(source['executions'])
        costs.to_csv(destination / 'all_fill_costs.csv', index=False)
        closed = source['episodes'][source['episodes'].closed]
        accounting = {name: float(closed[name].sum()) for name in ['gross_btc', 'fees_btc', 'funding_btc', 'net_pnl_btc']}
        residual = accounting['gross_btc'] - accounting['fees_btc'] - accounting['funding_btc'] - accounting['net_pnl_btc']
        if abs(residual) > 1e-6:
            raise ValueError('종료 에피소드 회계 분해 불일치')
        save_json(destination / 'summary.json', {'complete': True, 'independent_orders': len(orders),
                  'horizon_rows': len(frame), 'matched_rows': len(valid), 'closed_xbt_episodes': accounting,
                  'accounting_residual_btc': residual, 'fees_removed_counterfactual': False})
        view = summary[(summary.horizon_minutes == 60) & summary.target.isin(['enter_long', 'enter_short', 'increase'])]
        (destination / 'REPORT.md').write_text(
            '# 최초 독립 체결의 이후 가격과 비용\n\n' + table(view) + '\n\n'
            f'독립 주문 {len(orders):,}개, 네 관찰 간격 {len(frame):,}행 중 {len(valid):,}행의 가격을 연결했다. '
            '각 주문의 최초 체결을 동일 가중치로 계산했으며 계약 수량으로 가중하지 않았다.\n\n'
            'markout은 주문 방향 × log(이후 가격/최초 체결 가격) × 10,000이다. '
            '매수에 양수면 이후 상승, 매도에 양수면 이후 하락이다. 순자산 수익률·실현 손익이 아니다. '
            '최초 체결 수수료만 차감했으며 청산 비용·펀딩·다른 부분 체결은 포함하지 않는다.\n\n'
            '직전 확정 봉은 체결보다 최대 5분 앞서며 관찰 종료 가격은 요청 시점 이후 5분 미만의 첫 확정 봉이다. '
            '따라서 직전 봉과 체결의 차이는 스프레드 수익이나 체결 우월성의 직접 측정이 아니다. '
            '미체결 주문을 관측하지 못한 선택 편향이 있으며 maker 가격 경로가 taker와 다르다는 결과만으로 인과 효과를 말하지 않는다.\n\n'
            '95% 구간은 해당 연도·행동 표본의 7일 블록 재표집이며, 다중 비교·후보 선택·다른 시장으로의 이전 불확실성은 포함하지 않는다. '
            '전체 체결의 비용과 종료 XBT 에피소드 손익 분해는 별도 파일에 저장했다. '
            '이후 가격은 설명 연구에만 사용하며 v4 신호·학습 필터에는 넣지 않는다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'최초 체결 연구: {destination}', flush=True)
    return destination
