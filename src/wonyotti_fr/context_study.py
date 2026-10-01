from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, records, save_json, sha256
from .event_features import MARKET_FEATURES, event_features, independent_orders
from .reports import table
from .study import summarize_groups


def order_context(orders: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge_asof(orders.sort_values('target_time'), features.sort_values('end'),
                           left_on='target_time', right_on='end', direction='backward',
                           tolerance=pd.Timedelta(minutes=5), allow_exact_matches=False)
    result['year'] = result.target_time.dt.year
    result['order_direction'] = np.where(result.side.eq('Buy'), 1, -1)
    result['signed_prior_1h'] = result.order_direction * result.ret_1h
    result['signed_prior_4h'] = result.order_direction * result.ret_4h
    result['available'] = result[MARKET_FEATURES].notna().all(axis=1)
    return result


def execution_costs(executions: pd.DataFrame) -> pd.DataFrame:
    trades = executions[executions.exectype.eq('Trade')].copy()
    if not trades.settlcurrency.eq('XBt').all():
        raise ValueError('정산 통화가 다른 거래를 BTC 비용으로 합산하지 않습니다.')
    trades['year'] = trades.time.dt.year
    trades['notional_btc'] = trades.execcost.abs() / 1e8
    trades['fee_btc'] = trades.execcomm / 1e8
    costs = trades.groupby(['year', 'lastliquidityind', 'ordtype'], observed=True).agg(
        fills=('execid', 'size'), btc_turnover=('notional_btc', 'sum'), fees_btc=('fee_btc', 'sum')).reset_index()
    costs['notional_weighted_fee_bps'] = costs.fees_btc / costs.btc_turnover * 10000
    return costs


def run_context_study(audit: Path, history: Path, event_study: Path, output: Path) -> Path:
    manifest = json.loads((history / 'manifest.json').read_text())
    history_path = (history / manifest['file']).resolve()
    study_hashes = json.loads((event_study / 'files.json').read_text())
    training_path = event_study / 'training_events.parquet'
    if (not history_path.is_relative_to(history.resolve()) or sha256(history_path) != manifest['sha256']
        or sha256(training_path) != study_hashes[training_path.name]):
        raise ValueError('시장·사건별 입력의 지문 또는 경로 오류')
    files = ['executions.parquet', 'actions.parquet', 'episodes.parquet']
    source_hashes = {name: sha256(audit / name) for name in files}
    study_manifest = json.loads((event_study / 'manifest.json').read_text())
    if any(source_hashes[name] != study_manifest['settings']['audit_sha256'][name] for name in files):
        raise ValueError('사건별 학습과 원본 감사 입력이 다릅니다.')
    destination = new_run(output, 'full-context-study', {
        'audit_inputs': source_hashes, 'history_manifest_sha256': sha256(history / 'manifest.json'),
        'training_events_sha256': sha256(training_path),
        'role': 'descriptive_only_no_signal_relabeling', 'context_boundary': '완료 시각이 주문 체결 시각보다 엄격히 앞선 5분봉',
    })
    executions = pd.read_parquet(audit / 'executions.parquet')
    actions = pd.read_parquet(audit / 'actions.parquet')
    episodes = pd.read_parquet(audit / 'episodes.parquet')
    features = event_features(pd.read_parquet(history_path))
    context = order_context(independent_orders(executions, actions), features)
    context.to_parquet(destination / 'order_context.parquet', index=False)
    available = context[context.available]
    context_summary = available.groupby(['year', 'target'], observed=True).agg(
        orders=('order_key', 'size'), median_contracts=('orderqty', 'median'),
        p90_contracts=('orderqty', lambda s: s.quantile(0.9)),
        opposite_prior_1h_fraction=('signed_prior_1h', lambda s: (s < 0).mean()),
        opposite_prior_4h_fraction=('signed_prior_4h', lambda s: (s < 0).mean()),
        median_volume_ratio=('volume_ratio_1h', 'median'), median_volatility_1d=('vol_1d', 'median')).reset_index()
    costs = execution_costs(executions)
    closed = episodes[episodes.closed].copy()
    closed['exit_year'] = closed.exit_time.dt.year
    closed['scaled_in'] = closed.additional_entry_orders > 0
    by_addition = summarize_groups(closed, ['exit_year', 'direction', 'scaled_in'])
    examples = pd.concat([closed.nlargest(10, 'net_pnl_btc').assign(case='largest_profit'),
                          closed.nsmallest(10, 'net_pnl_btc').assign(case='largest_loss')])
    labeled = pd.read_parquet(training_path)
    comparison = labeled[labeled.usable].copy()
    comparison['year'] = comparison.end.dt.year
    controls = comparison.groupby(['year', 'direction', 'target'])[MARKET_FEATURES].median().reset_index()
    controls['bars'] = comparison.groupby(['year', 'direction', 'target']).size().to_numpy()
    for name, frame in [('order_context_summary', context_summary), ('execution_costs', costs),
                        ('scale_in_outcomes', by_addition), ('extreme_cases', examples), ('action_and_hold_context', controls)]:
        frame.to_csv(destination / f'{name}.csv', index=False)
    coverage = context.groupby('year').agg(orders=('order_key', 'size'), matched=('available', 'sum')).reset_index()
    save_json(destination / 'summary.json', {'orders': len(context), 'matched': len(available),
                                            'coverage': records(coverage), 'complete': True})
    entries = context_summary[context_summary.target.isin(['enter_long', 'enter_short'])]
    (destination / 'REPORT.md').write_text(
        '# 원거래소 전체 기간의 주문 맥락과 실행 비용\n\n'
        f'독립 주문 {len(context):,}개 중 {len(available):,}개에 직전 확정 시세를 연결했다. '
        '같은 5분 창의 첫 주문만 사용하는 모델 정답과 달리, 이 설명 자료에는 해당 창의 다른 독립 주문도 포함한다.\n\n'
        '## 연도별 연결 범위\n\n' + table(coverage) + '\n\n'
        '## 진입·반전 주문의 직전 시장\n\n' + table(entries) + '\n\n'
        '방향에 부호를 곱한 과거 수익률이 음수이면 해당 움직임의 반대 방향 주문이다. '
        '이 비율이 실제 역추세 전략이나 진입 의도의 증명은 아니다. 주문 수량은 XBTUSD 계약 수이며 자본 대비 레버리지가 아니다.\n\n'
        '## 실제 체결의 비용 구조\n\n' + table(costs) + '\n\n'
        '원본의 거래 행을 정산 통화 BTC 기준으로 합산했다. 음의 수수료는 리베이트다. '
        'Limit 주문도 유동성을 제거하며 체결될 수 있어 주문 종류와 maker를 같은 것으로 취급하지 않는다. '
        '5분 시장가 봇은 지정가 대기열·미체결·리베이트를 그대로 복제하지 못한다.\n\n'
        '## 추가 진입 여부별 결과\n\n' + table(by_addition) + '\n\n'
        '이는 사후 손익 비교다. 포지션 크기·시장·진입 조건이 다르므로 추가 진입의 인과 효과가 아니다. '
        '추가 주문이 있는 손실만 지우거나 종료 손익을 입력으로 사용하지 않는다. '
        '큰 이익·손실 각 10개 사례와 비매매 창의 시세 비교는 별도 로컬 파일에 보존했다. '
        '체결되지 않은 주문·당시 뉴스·계좌 밖 헤지·재량 판단은 관측하지 못한다.\n', encoding='utf-8')
    print(f'전체 기간 맥락 분석: {destination / "REPORT.md"}', flush=True)
    return destination
