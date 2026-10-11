from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from wonyotti_fr.collateral_direction import LABELS, direction_ledger, summarize_rows
from wonyotti_fr.collateral_reference import independent_ledger
from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.position_boost_reference import checked_position_reference, position_baseline


def read(path):
    return json.loads(path.read_text())


def checked_reference(path, expected):
    if path.is_symlink() or sha256(path) != expected:
        raise ValueError('담보 방향 참조 지문 오류')
    spec = read(path)
    for name, digest in spec['sha256'].items():
        p = Path(name)
        if p.is_symlink() or sha256(p) != digest:
            raise ValueError('담보 방향 입력 변경: '+name)
    checked_position_reference(Path(spec['v44_reference']), spec['sha256'][spec['v44_reference']])
    audit, portfolio = Path(spec['audit']), Path(spec['portfolio'])
    report = read(audit/'audit.json')
    settings = read(portfolio/'manifest.json')['settings']
    if (settings['initial_position_assumption'] != 0 or report['wallet']['opening_balance_satoshi'] != 0
        or settings['executions_sha256'] != spec['sha256'][str(audit/'executions.parquet')]
        or settings['wallet_sha256'] != spec['sha256'][str(audit/'wallet.parquet')]):
        raise ValueError('담보 방향 최초 원장 연결 오류')
    support = read(Path(spec['v44'])/'training_support.json')['source_sha256']['audit_sha256']
    if support['executions.parquet'] != settings['executions_sha256']:
        raise ValueError('담보 방향과 기존 방향 정답의 원본 불일치')
    originals = {r['file']: r['sha256'] for r in report['files']}
    originals[report['wallet']['file']] = report['wallet']['sha256']
    if len(originals) != 5 or any(spec['sha256'].get(str(Path(spec['original'])/n)) != h for n, h in originals.items()):
        raise ValueError('담보 방향 원본 5개 연결 오류')
    return spec


def load_inputs(spec):
    audit, prior = Path(spec['audit']), Path(spec['v44'])
    e = pd.read_parquet(audit/'executions.parquet', columns=[
        'time', 'symbol', 'exectype', 'side', 'lastqty', 'execcost', 'execcomm', 'settlcurrency'])
    w = pd.read_parquet(audit/'wallet.parquet')
    train, val, _ = position_baseline(prior)
    parts = {'training': train, 'diagnosis': val}
    daily = pd.date_range(e.time.min().ceil('D'), e.time.max().floor('D'), freq='D', tz='UTC')
    terminal = e.time.max()+pd.Timedelta(1, 'ns')
    points = pd.DatetimeIndex(sorted(set(daily) | {terminal} | {
        v for f in parts.values() for col in ['end', 'label_end'] for v in f[col]})).as_unit('ns')
    return e, w, parts, daily.as_unit('ns'), terminal, points


def comparison_rows(parts, ledger):
    indexed = ledger.set_index('time', verify_integrity=True)
    rows = []
    for part, source in parts.items():
        for boundary, column in [('current', 'end'), ('target', 'label_end')]:
            joined = indexed.loc[source[column]].reset_index()
            if boundary == 'current':
                old = np.where(source.direction == 0, 'flat', np.where(source.direction < 0, 'short', 'long'))
            else:
                old = np.asarray(LABELS[:3], dtype=object)[source.position_target.to_numpy()]
            if not np.array_equal(old, joined.futures_direction):
                raise ValueError('기존 선물 정답과 새 원장 수량의 불일치')
            joined['part'], joined['boundary'] = part, boundary
            joined['source_end'] = source.end.to_numpy()
            joined['source_episode_id'] = source.episode_id.to_numpy()
            joined['source_target_episode_id'] = source.position_target_episode_id.to_numpy()
            joined['month'] = source.end.dt.strftime('%Y-%m').to_numpy()
            rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def grouped_summary(comparisons):
    out = []
    for columns in [('part', 'boundary'), ('part', 'boundary', 'month'), ('part', 'boundary', 'futures_direction')]:
        for keys, frame in comparisons.groupby(list(columns), sort=True):
            out.append({'group': dict(zip(columns, keys, strict=True)), **summarize_rows(frame)})
    return out


def run_collateral(reference, expected, output):
    spec = checked_reference(reference, expected)
    out = new_run(output, 'collateral-direction-diagnosis', {'reference': str(reference),
        'reference_sha256': expected, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V95.md')),
        'wallet_extra_days': [0, 1], 'new_models': 0, 'production_integrated': True,
        'trading_returns_evaluated': False, 'profitability_accepted': False})
    print(out, flush=True)
    try:
        e, w, parts, daily, terminal, points = load_inputs(spec)
        save_json(out/'reference.json', spec)
        sections = {}
        for window in [0, 1]:
            ledger = direction_ledger(e, w, points, extra_days=window)
            comparisons = comparison_rows(parts, ledger)
            ledger.to_parquet(out/f'ledger_{window}.parquet', index=False)
            comparisons.to_parquet(out/f'comparisons_{window}.parquet', index=False)
            daily_rows = ledger.set_index('time').loc[daily].reset_index()
            daily_rows.to_parquet(out/f'daily_{window}.parquet', index=False)
            rows = grouped_summary(comparisons)
            save_json(out/f'groups_{window}.json', rows)
            sections[str(window)] = {'groups': rows, 'daily': summarize_rows(daily_rows),
                'terminal': json.loads(ledger.loc[ledger.time.eq(terminal)].to_json(orient='records', date_format='iso'))[0]}
            print(f'날짜 확장 {window}일: 원본 경계 {len(ledger)}, 비교 행 {len(comparisons)}', flush=True)
        summary = {'complete': True, 'sections': sections, 'source_execution_rows': len(e),
            'previous_rows': {k: len(v) for k, v in parts.items()}, 'unique_boundaries': len(points),
            'profitability_accepted': False, 'trading_returns_evaluated': False,
            'all_periods_already_observed': True, 'wallet_date_windows_are_assumptions': True,
            'account_outside_assets_included': False, 'independent_trade_count': None}
        save_json(out/'summary.json', summary)
        checked_reference(reference, expected)
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def independent_summary_check(frame, actual):
    counts = {label: int(frame.account_direction.eq(label).sum()) for label in LABELS}
    known = frame.account_direction.isin(['flat', 'short', 'long'])
    denominator = int(known.sum())
    different = int((known & frame.futures_direction.ne(frame.account_direction)).sum())
    expected = {'rows': len(frame), 'direction_counts': counts, 'comparable_rows': denominator,
        'different_rows': different, 'different_fraction_of_comparable': different/denominator if denominator else None,
        'date_uncertain_rows': int(frame.external_uncertain_count.gt(0).sum()),
        'cross_table_futures_rows_account_columns': {f: {a: int((known & frame.futures_direction.eq(f)
            & frame.account_direction.eq(a)).sum()) for a in LABELS[:3]} for f in LABELS[:3]},
        'independent_trade_count': None, 'conditional_on_wallet_date_window': True}
    if actual != expected:
        raise ValueError('담보 방향 독립 분모·교차표 불일치')


def verify_collateral(run, expected, output):
    if sha256(run/'files.json') != expected:
        raise ValueError('담보 방향 실행 호출자 지문 오류')
    files = read(run/'files.json')
    for name, digest in files.items():
        if Path(name).name != name or (run/name).is_symlink() or sha256(run/name) != digest:
            raise ValueError('담보 방향 실행 파일 변경')
    settings = read(run/'manifest.json')['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V95.md'))
        or settings['wallet_extra_days'] != [0, 1] or settings['new_models'] != 0
        or settings['profitability_accepted'] is not False or settings['trading_returns_evaluated'] is not False):
        raise ValueError('담보 방향 계획·학습 범위 오류')
    reference = Path(settings['reference'])
    spec = checked_reference(reference, settings['reference_sha256'])
    out = new_run(output, 'collateral-direction-review', {'run': str(run), 'files_sha256': expected})
    print(out, flush=True)
    try:
        e, w, parts, daily, terminal, points = load_inputs(spec)
        summary = read(run/'summary.json')
        if (summary['previous_rows'] != {k: len(v) for k, v in parts.items()}
            or summary['source_execution_rows'] != len(e) or summary['unique_boundaries'] != len(points)
            or summary['profitability_accepted'] is not False or summary['trading_returns_evaluated'] is not False):
            raise ValueError('담보 방향 전체 지원·판정 범위 불일치')
        for window in [0, 1]:
            independently = independent_ledger(e, w, points, extra_days=window)
            saved = pd.read_parquet(run/f'ledger_{window}.parquet')
            pd.testing.assert_frame_equal(independently, saved, check_exact=True)
            joined = pd.read_parquet(run/f'comparisons_{window}.parquet')
            for part, source in parts.items():
                for boundary, column in [('current', 'end'), ('target', 'label_end')]:
                    subset = joined[joined.part.eq(part) & joined.boundary.eq(boundary)].reset_index(drop=True)
                    if len(subset) != len(source) or not np.array_equal(subset.time, source[column]):
                        raise ValueError('담보 방향 비교 행·경계 누락')
                    if not np.array_equal(subset.source_end, source.end):
                        raise ValueError('담보 방향 출처 경계 불일치')
                    if (not np.array_equal(subset.source_episode_id, source.episode_id)
                        or not np.array_equal(subset.source_target_episode_id, source.position_target_episode_id)
                        or not np.array_equal(subset.month, source.end.dt.strftime('%Y-%m'))):
                        raise ValueError('담보 방향 출처 포지션·월 불일치')
                    selected = independently.set_index('time').loc[source[column]].reset_index()
                    pd.testing.assert_frame_equal(subset[list(selected.columns)], selected, check_exact=True)
                    expected_labels = (np.where(source.direction > 0, 'long', np.where(source.direction < 0, 'short', 'flat'))
                        if boundary == 'current' else np.asarray(['flat', 'short', 'long'])[source.position_target])
                    if not np.array_equal(selected.futures_direction, expected_labels):
                        raise ValueError('담보 방향 독립 선물 정답 연결 불일치')
            if len(joined) != 2*sum(map(len, parts.values())):
                raise ValueError('담보 방향 전체 비교 행 수 오류')
            groups = read(run/f'groups_{window}.json')
            keys_seen = set()
            for group in groups:
                mask = np.ones(len(joined), dtype=bool)
                key = tuple(group['group'].items())
                if key in keys_seen:
                    raise ValueError('담보 방향 집계 중복')
                keys_seen.add(key)
                for name, value in group['group'].items():
                    mask &= joined[name].eq(value).to_numpy()
                independent_summary_check(joined.loc[mask], {k: v for k, v in group.items() if k != 'group'})
            expected_keys = {tuple(zip(columns, key, strict=True))
                for columns in [('part', 'boundary'), ('part', 'boundary', 'month'), ('part', 'boundary', 'futures_direction')]
                for key in joined.groupby(list(columns), sort=True).groups}
            if keys_seen != expected_keys or groups != summary['sections'][str(window)]['groups']:
                raise ValueError('담보 방향 집계 목록 누락')
            daily_rows = independently.set_index('time').loc[daily].reset_index()
            pd.testing.assert_frame_equal(daily_rows, pd.read_parquet(run/f'daily_{window}.parquet'))
            independent_summary_check(daily_rows, summary['sections'][str(window)]['daily'])
            ending = independently.loc[independently.time.eq(terminal)].iloc[0]
            if json.loads(independently.loc[independently.time.eq(terminal)].to_json(orient='records', date_format='iso'))[0] != summary['sections'][str(window)]['terminal']:
                raise ValueError('담보 방향 마지막 경계 요약 불일치')
            old_summaries = [read(p) for p in Path(spec['portfolio']).glob('*/summary.json')]
            external = sum(int(r.posted_amount) for r in w.itertuples() if r.transacttype != 'RealisedPNL')
            total = Decimal(external)+sum(Decimal(s['realized_net_satoshi_decimal']) for s in old_summaries)
            xbt = read(Path(spec['portfolio'])/'XBTUSD/summary.json')
            total += Decimal(xbt['final_position_contracts'])*Decimal(xbt['final_basis_satoshi_per_contract'])
            if (abs(total-Decimal(int(ending.delta_min_satoshi))) > Decimal('1e-15')
                or ending.delta_min_satoshi != ending.delta_max_satoshi or ending.other_open_contracts != 0
                or any(s['final_position_contracts'] != 0 for s in old_summaries if s != xbt)):
                raise ValueError('담보 방향 마지막 원가·현금 대조 불일치')
            print(f'날짜 확장 {window}일: 독립 Decimal 원장 {len(saved)}개 경계 일치', flush=True)
        checked_reference(reference, settings['reference_sha256'])
        save_json(out/'verification.json', {'complete': True, 'independent_all_boundary_cash_and_basis_exact': True,
            'all_original_direction_rows_and_groups_exact': True, 'date_uncertainty_and_unsupported_rows_preserved': True,
            'terminal_existing_portfolio_exact': True, 'original_files_unchanged': 5,
            'unique_boundaries': len(points), 'wallet_windows': 2,
            'profitability_accepted': False, 'trading_returns_evaluated': False})
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
