from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .common import new_run, records, save_json, sha256
from .event_features import independent_orders
from .execution_study import markouts
from .expansion_data import source_inputs
from .reports import table

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read_minute_history(directory: Path, reference: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads((reference / 'manifest.json').read_text())
    path = (directory / manifest['file']).resolve()
    if (manifest.get('interval') != '1m' or previous.get('interval') != '5m'
        or any(manifest[key] != previous[key] for key in ['symbol', 'start', 'end_exclusive'])
        or not path.is_relative_to(directory.resolve()) or sha256(path) != manifest['sha256']):
        raise ValueError('분봉 비교 입력의 범위·간격·경로·지문 오류')
    return pd.read_parquet(path), {'minute_manifest_sha256': sha256(manifest_path),
                                  'minute_history_sha256': manifest['sha256']}


def aggregate_comparison(minute: pd.DataFrame, five: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    for bars, minutes in [(minute, 1), (five, 5)]:
        if (bars.end.duplicated().any() or not bars.end.is_monotonic_increasing
            or not (bars.end - bars.time).eq(pd.Timedelta(minutes=minutes)).all()
            or (bars.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=minutes).value).any()):
            raise ValueError('집계 자료의 시간순·봉 경계 오류')
    grouped = minute.groupby(minute.end.dt.ceil('5min')).agg(
        minute_count=('close', 'size'), close=('close', 'last'), volume=('volume', 'sum'),
        contract_volume=('contract_volume', 'sum'), trades=('trades', 'sum')).reset_index()
    result = grouped.merge(five[['end', 'close', 'volume', 'contract_volume', 'trades']],
                           on='end', how='outer', validate='one_to_one', suffixes=('_aggregated', '_official'), indicator=True)
    complete = result._merge.eq('both') & result.minute_count.eq(5)
    result['complete'] = complete
    for name, absolute in [('close', 1e-8), ('volume', 1e-7), ('contract_volume', 0), ('trades', 0)]:
        result[f'{name}_difference'] = result[f'{name}_aggregated'] - result[f'{name}_official']
        result[f'{name}_match'] = complete & np.isclose(result[f'{name}_aggregated'], result[f'{name}_official'], rtol=0, atol=absolute)
    result['all_match'] = result[[f'{name}_match' for name in ['close', 'volume', 'contract_volume', 'trades']]].all(axis=1)
    result['status'] = np.where(~complete, 'missing_or_incomplete_bucket', np.where(result.all_match, 'matched', 'value_difference'))
    summary = {'buckets': len(result), 'complete_buckets': int(complete.sum()),
               'matched_buckets': int(result.all_match.sum()), 'incomplete_buckets': int((~complete).sum()),
               'value_mismatch_buckets': int((complete & ~result.all_match).sum()),
               'absolute_tolerances': {'close': 1e-8, 'volume_btc': 1e-7, 'contracts': 0, 'trades': 0}}
    return result, summary


def compare_order_timing(orders: pd.DataFrame, fills: pd.DataFrame, minute: pd.DataFrame,
                         five: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {}
    for size, bars in [(1, minute), (5, five)]:
        frame = markouts(orders, fills, bars, interval_minutes=size)
        frame['reference_age_seconds'] = (frame.target_time - frame.reference_time).dt.total_seconds()
        frame['outcome_delay_seconds'] = (frame.outcome_time - frame.requested_end).dt.total_seconds()
        frame['available'] = frame[['reference_price', 'outcome_price']].notna().all(axis=1)
        known = frame[frame.available]
        if not ((known.reference_age_seconds > 0) & (known.reference_age_seconds <= size * 60)
                & (known.outcome_delay_seconds >= 0) & (known.outcome_delay_seconds < size * 60)).all():
            raise ValueError('연결한 시세가 정한 관찰 경계를 벗어납니다.')
        frame['resolution_minutes'] = size
        frames[size] = frame
    columns = ['order_key', 'horizon_minutes', 'available', 'reference_age_seconds', 'outcome_delay_seconds',
               'reference_price', 'reference_time', 'outcome_price', 'outcome_time', 'reference_to_fill_bps',
               'markout_bps', 'after_first_fee_bps', 'subsequent_reference_move_bps']
    paired = frames[1].merge(frames[5][columns], on=['order_key', 'horizon_minutes'], validate='one_to_one', suffixes=('_1m', '_5m'))
    paired['both_available'] = paired.available_1m & paired.available_5m
    paired['reference_difference_bps'] = paired.reference_to_fill_bps_1m - paired.reference_to_fill_bps_5m
    paired['absolute_reference_gap_reduction_bps'] = paired.reference_to_fill_bps_5m.abs() - paired.reference_to_fill_bps_1m.abs()
    coverage = pd.concat(frames.values(), ignore_index=True).groupby(
        ['resolution_minutes', 'year', 'target', 'horizon_minutes'], observed=True).agg(
            orders=('order_key', 'size'), matched=('available', 'sum')).reset_index()
    return paired, coverage


def timing_summary(paired: pd.DataFrame) -> pd.DataFrame:
    valid = paired[paired.both_available]
    rows = []
    keys = ['year', 'target', 'lastliquidityind', 'horizon_minutes']
    for group, frame in valid.groupby(keys, observed=True):
        row = {**dict(zip(keys, group, strict=True)), 'orders': len(frame)}
        for size in ['1m', '5m']:
            gap = frame[f'reference_to_fill_bps_{size}']
            row.update({f'mean_gap_{size}_bps': float(gap.mean()), f'median_gap_{size}_bps': float(gap.median()),
                        f'p10_gap_{size}_bps': float(gap.quantile(.1)), f'p90_gap_{size}_bps': float(gap.quantile(.9)),
                        f'median_absolute_gap_{size}_bps': float(gap.abs().median()),
                        f'mean_markout_{size}_bps': float(frame[f'markout_bps_{size}'].mean()),
                        f'mean_reference_age_{size}_seconds': float(frame[f'reference_age_seconds_{size}'].mean()),
                        f'mean_outcome_delay_{size}_seconds': float(frame[f'outcome_delay_seconds_{size}'].mean())})
        row['mean_absolute_gap_reduction_bps'] = float(frame.absolute_reference_gap_reduction_bps.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gap_interval(frame: pd.DataFrame) -> dict:
    names = ['reference_difference_bps', 'absolute_reference_gap_reduction_bps']
    if frame.empty or not np.isfinite(frame[names]).all().all():
        raise ValueError('짝 비교에는 유한한 관측값이 필요합니다.')
    daily = frame.set_index('target_time')[names].resample('1D').agg(['sum', 'count'])
    result = {'orders': len(frame), 'calendar_days': len(daily),
              'active_days': int(daily[(names[0], 'count')].gt(0).sum())}
    for name in names:
        result[f'mean_{name}'] = float(frame[name].mean())
        result[f'{name}_ci_low'] = None
        result[f'{name}_ci_high'] = None
    if len(daily) < 14 or result['active_days'] < 14:
        return result
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(daily), size=(1000, (len(daily) + 6) // 7))
    indices = ((starts[:, :, None] + np.arange(7)) % len(daily)).reshape(1000, -1)[:, :len(daily)]
    counts = daily[(names[0], 'count')].to_numpy()[indices].sum(axis=1)
    # 같은 날짜 블록을 두 가격 차이에 재사용해 주문 쌍과 날짜 내 의존성을 유지한다.
    for name in names:
        sums = daily[(name, 'sum')].to_numpy()[indices].sum(axis=1)
        samples = sums[counts > 0] / counts[counts > 0]
        lower, upper = np.quantile(samples, [.025, .975])
        result[f'{name}_ci_low'], result[f'{name}_ci_high'] = float(lower), float(upper)
    return result


def paired_uncertainty(paired: pd.DataFrame) -> pd.DataFrame:
    unique = paired[paired.horizon_minutes.eq(60) & paired.both_available]
    rows = []
    for scope, sample in [('all_orders', unique), ('exposure_expansion', unique[unique.target.isin(
        ['enter_long', 'enter_short', 'increase'])])]:
        for (year, liquidity), frame in sample.groupby(['year', 'lastliquidityind'], observed=True):
            rows.append({'scope': scope, 'year': year, 'lastliquidityind': liquidity,
                         **paired_gap_interval(frame),
                         **{f'mean_gap_{size}_bps': float(frame[f'reference_to_fill_bps_{size}'].mean()) for size in ['1m', '5m']}})
    return pd.DataFrame(rows)


def run_timing_study(audit: Path, study: Path, history: Path, minute_history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    minute, minute_hashes = read_minute_history(minute_history, history)
    destination = new_run(output, 'timing-study', {**hashes, **minute_hashes, 'protocol': 'docs/EXPERIMENT_V6.md',
                                                'role': 'paired_descriptive_timing_not_profitability',
                                                'paired_bootstrap': '7-day circular calendar blocks, 1000 draws, seed 0; conditional, no multiple-comparison correction'})
    print(f'동일 주문의 시간 해상도 비교: {destination}', flush=True)
    try:
        aggregation, aggregation_summary = aggregate_comparison(minute, source['bars'])
        aggregation.to_parquet(destination / 'bar_aggregation.parquet', index=False)
        aggregation[~aggregation.all_match].to_csv(destination / 'bar_mismatches.csv', index=False)
        save_json(destination / 'aggregation_summary.json', aggregation_summary)
        orders = independent_orders(source['executions'], source['actions'])
        fills = source['executions'].query("symbol == 'XBTUSD' and exectype == 'Trade'")
        paired, coverage = compare_order_timing(orders, fills, minute, source['bars'])
        paired.to_parquet(destination / 'paired_markouts.parquet', index=False)
        coverage.to_csv(destination / 'coverage.csv', index=False)
        summary = timing_summary(paired)
        summary.to_csv(destination / 'timing_summary.csv', index=False)
        uncertainty = paired_uncertainty(paired)
        uncertainty.to_csv(destination / 'paired_uncertainty.csv', index=False)
        view = summary[summary.horizon_minutes.eq(60)]
        chart, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
        try:
            unique = paired[paired.horizon_minutes.eq(60) & paired.both_available].copy()
            for axis, liquidity in zip(axes, ['AddedLiquidity', 'RemovedLiquidity'], strict=True):
                part = unique[unique.lastliquidityind.eq(liquidity)]
                if part.empty:
                    axis.set(title=liquidity)
                    axis.text(.5, .5, 'No matched orders', ha='center', va='center', transform=axis.transAxes)
                    continue
                grouped = part.groupby('year')[['reference_to_fill_bps_1m', 'reference_to_fill_bps_5m']].agg(lambda s: s.abs().median())
                grouped.rename(columns={'reference_to_fill_bps_1m': '1m reference', 'reference_to_fill_bps_5m': '5m reference'}).plot.bar(ax=axis)
                axis.set(title=liquidity, ylabel='Median absolute reference gap (bp)', xlabel='Year')
                axis.tick_params(axis='x', rotation=0)
                axis.grid(axis='y', alpha=.2)
            chart.savefig(destination / 'reference_gap_comparison.png', dpi=150)
        finally:
            plt.close(chart)
        overall = paired.groupby('horizon_minutes').agg(orders=('order_key', 'size'), both_available=('both_available', 'sum')).reset_index()
        save_json(destination / 'summary.json', {'complete': True, 'bar_aggregation': aggregation_summary, 'coverage': records(overall),
                  'strategy_selected': False, 'remaining': '시간 정밀도 차이를 원전략 재현이나 실행 가능한 이익으로 해석하지 않음'})
        columns = ['year', 'target', 'lastliquidityind', 'orders', 'mean_gap_1m_bps', 'mean_gap_5m_bps',
                   'median_absolute_gap_1m_bps', 'median_absolute_gap_5m_bps', 'mean_markout_1m_bps', 'mean_markout_5m_bps']
        (destination / 'REPORT.md').write_text(
            '# 같은 최초 체결의 1분·5분 시세 비교\n\n' + table(pd.DataFrame([aggregation_summary]).drop(columns='absolute_tolerances')) + '\n\n'
            '1분 봉을 종료 시각 기준으로 다섯 개씩 묶었다. 종가는 마지막 값, BTC·계약 거래량과 거래 수는 합계다. '
            '완전한 다섯 봉이 없는 묶음과 값 불일치는 별도로 보존한다. BTC 거래량은 합산 반올림을 위한 1e-7 BTC 허용 오차를 썼고 계약 수와 체결 수는 정확히 비교한다.\n\n'
            '## 같은 주문의 연결 범위\n\n' + table(overall) + '\n\n'
            '## 60분 관찰 기준\n\n' + table(view[columns]) + '\n\n'
            '방향별 직전 가격 차이는 방향 × log(직전 확정 종가/최초 체결 가격)이다. 양수이면 실제 체결 가격이 그 이전 종가보다 유리한 방향이다. '
            '이는 체결 전 가격 움직임·체결된 주문만 관측한 선택 효과를 포함하므로 스프레드 수익이나 maker 우월성의 측정이 아니다.\n\n'
            '1분 연결은 최대 60초 전의 엄격히 이전 확정 종가와 요청 시점 이후 60초 미만의 가격을 사용한다. '
            '5분 연결은 각각 300초와 300초 미만이다. 같은 주문·같은 관찰 간격을 연결한 쌍에서만 차이를 비교했다. '
            '연결 실패도 분모와 별도 파일에 남겼으며 특정 손익이나 유동성만 보고 제거하지 않았다.\n\n'
            '5·15·60·240분 관찰과 평균·중앙값·분위수·시각 차이는 로컬 상세 표에 보존했다. '
            'paired_uncertainty.csv는 60분 관찰이 연결된 같은 주문의 직전 가격 차이를 연도·유동성·전체 또는 노출 확대 행동으로 묶는다. '
            '방향별 차이 변화와 절대 차이 감소의 평균에 대해 7일 달력 블록을 1,000번 재표집한 조건부 95% 구간을 기록했다. '
            '같은 주문 쌍을 유지하고 거래 없는 날도 포함했다. 관측일이 14일 미만인 표본은 구간을 제시하지 않는다. '
            '이 구간은 표본 선택·다중 비교·다른 시기의 일반화 불확실성을 포함하지 않으며 수익성 검정이 아니다. '
            'markout은 실제 체결 이후 가격 변화이며 실현 손익·자본 수익률이 아니다. '
            '주문 제출·취소·대기열·다른 부분 체결을 복원하지 못한다. 어떤 이후 가격도 신호 입력에 넣지 않았다. '
            '이 분석은 다음 후보를 설계하기 위한 진단이며 수익성 검증이나 새 후보 선택이 아니다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'시점 비교 완료: {destination}', flush=True)
    return destination
