from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .event_research import load_selection
from .first_exit_reference import FIRST_EXIT_RUN_FILES
from .first_exit_research import verify_parent_replay
from .first_exit_verification import require_row, verify_execution
from .minute_data import prepare_minute_period
from .retained_weight_close_reference import checked_files
from .robustness import block_interval

STRESS_PERIODS = {'confirmation': ['2022-01-01', '2023-01-01'],
    'observed': ['2023-01-01', '2026-01-01'], 'recent': ['2026-01-01', '2026-10-01']}
STRESS_VARIANTS = ['base', 'cost_x2', 'cost_x3', 'delay_1m']


def read_json(path):
    return json.loads(path.read_text())


def seal(folder):
    save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'files.json'})


def stress_configs(risk):
    if (risk['fee_bps'], risk['slippage_bps'], risk['signal_delay_bars'], risk['bar_seconds']) != (5., 3., 0, 60):
        raise ValueError('보유 정책 비용 대조의 원래 비용·주기 오류')
    config = EngineConfig(**risk)
    return {'base': config, 'cost_x2': replace(config, fee_bps=10., slippage_bps=6.),
        'cost_x3': replace(config, fee_bps=15., slippage_bps=9.), 'delay_1m': replace(config, signal_delay_bars=1)}


def checked_stress_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 1024**2 or sha256(path) != expected:
        raise ValueError('보유 정책 비용 대조의 호출자 출처 지문 오류')
    spec = read_json(path)
    if (spec.get('profitability_accepted') is not False or spec.get('all_periods_already_observed') is not True
        or set(spec['inputs']) != set(STRESS_PERIODS) or set(spec['baselines']) != set(STRESS_PERIODS)
        or set(spec['retained_runs']) != {'2021_in_sample', '2023', '2024', '2025'}):
        raise ValueError('보유 정책 비용 대조의 고정 구간·실패 보존 오류')
    for name, digest in spec['sealed_files'].items():
        target = Path(name)
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise ValueError('보유 정책 비용 대조의 봉인 입력 변경: '+name)
    selection, previous = Path(spec['selection']), Path(spec['previous_review'])
    required = {str(selection/'frozen_selection.json'), *[str(previous/n) for n in ['manifest.json', 'verification.json', 'per_run_verification.json']]}
    frozen, _ = load_selection(selection)
    proof = read_json(previous/'verification.json')
    if (frozen['protocol'] != 'exit_state_v48' or proof.get('complete') is not True
        or proof.get('verified_runs') != 8 or proof.get('profitability_accepted') is not False
        or read_json(previous/'manifest.json')['settings']['selection'] != str(selection)):
        raise ValueError('보유 정책 비용 대조의 기존 정책·검산 연결 오류')
    stress_configs(frozen['risk'])
    records = {row['run']: row for row in read_json(previous/'per_run_verification.json')}
    if len(records) != 8:
        raise ValueError('보유 정책 비용 대조의 기존 검산 실행 수 오류')
    for folder in [*spec['baselines'].values(), *spec['retained_runs'].values()]:
        if folder not in records or records[folder]['accounting'] is not True:
            raise ValueError('보유 정책 비용 대조의 기존 계좌 근거 누락')
        for name in FIRST_EXIT_RUN_FILES:
            key = str(Path(folder)/name)
            required.add(key)
            if spec['sealed_files'].get(key) != records[folder]['hashes'][name]:
                raise ValueError('보유 정책 비용 대조의 기존 출력 검산 지문 오류')
        if read_json(Path(folder)/'config.json') != frozen['risk']:
            raise ValueError('보유 정책 비용 대조의 이전 위험 설정 변경')
    for paths in spec['inputs'].values():
        required.update(str(Path(paths[key])/name) for key, name in [('market', 'manifest-1m.json'), ('features', 'manifest-5m.json')])
    if not required <= set(spec['sealed_files']):
        raise ValueError('보유 정책 비용 대조의 필수 입력 봉인 누락')
    return spec, frozen


def stress_details(folder, initial):
    trades = pd.read_parquet(folder/'trades.parquet')
    curve = pd.read_parquet(folder/'equity.parquet', columns=['time', 'equity'])
    daily = curve.groupby((pd.to_datetime(curve.time, utc=True)-pd.Timedelta(nanoseconds=1)).dt.date).equity.last()
    returns = (daily/daily.shift(fill_value=initial)-1).to_numpy()
    uncertainty = block_interval(returns) if len(returns) >= 60 else {'unavailable_reason': 'fewer_than_60_days'}
    largest = float(trades.net_pnl.max()) if len(trades) else None
    return {'decomposition': decompose_run(folder, initial), 'largest_trade_net_pnl': largest,
        'net_without_largest_trade': float(trades.net_pnl.sum()-largest) if largest is not None else None,
        'uncertainty': uncertainty}


def stress_summary(rows):
    keys = [(row['period'], row['variant']) for row in rows]
    if len(rows) != 12 or set(keys) != {(p, v) for p in STRESS_PERIODS for v in STRESS_VARIANTS}:
        raise ValueError('보유 정책 비용 대조의 열두 실행 누락·중복')
    checks = {f"{row['period']}/{row['variant']}": {'positive': row['total_return'] > 0,
        'at_least_30_trades': row['closed_trades'] >= 30, 'no_halt': not row['permanent_halt']} for row in rows}
    return {'complete': True, 'runs': 12, 'checks': checks, 'base_replays_exact': True,
        'cross_market_research_eligible': all(checks[f'{p}/{v}']['positive'] and checks[f'{p}/{v}']['no_halt'] for p in STRESS_PERIODS for v in ['base', 'cost_x2']),
        'new_models_fitted': False, 'previous_failures_preserved': True, 'profitability_accepted': False}


def guard_run_inputs(out, reference, expected):
    spec, frozen = checked_stress_reference(reference, expected)
    manifest = read_json(out/'manifest.json')
    settings = manifest['settings']
    if (spec != settings['reference'] or frozen['risk'] != settings['risk']
        or sha256(Path('docs/EXPERIMENT_V92.md')) != settings['protocol_sha256']
        or any(sha256(Path(__file__).parent/name) != value for name, value in manifest['source_sha256'].items())):
        raise ValueError('보유 정책 비용 대조 중 입력·코드·계획 변경')
    return spec, frozen


def run_holding_stress(reference, expected, output):
    spec, frozen = checked_stress_reference(reference, expected)
    out = new_run(output, 'holding-stress', {'reference': spec, 'reference_path': str(reference),
        'reference_sha256': expected, 'risk': frozen['risk'], 'periods': STRESS_PERIODS, 'variants': STRESS_VARIANTS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V92.md')), 'all_periods_already_observed': True})
    print(out, flush=True)
    try:
        rows, runs = [], {}
        save_json(out/'retained_results.json', {key: read_json(Path(folder)/'metrics.json') for key, folder in spec['retained_runs'].items()})
        for period, bounds in STRESS_PERIODS.items():
            paths = spec['inputs'][period]
            bars, checks = prepare_minute_period(Path(paths['market']), Path(paths['features']), 'BTCUSDT', *bounds, minute_inputs=True)
            save_json(out/period/'input_verification.json', checks)
            for variant, risk in stress_configs(frozen['risk']).items():
                current, policy = load_selection(Path(spec['selection']))
                require_row(current, frozen, '고정 정책')
                folder = out/period/variant
                print(f'{period}/{variant}: 연속 계좌 시작', flush=True)
                metrics = backtest(bars, policy, risk, folder)
                if variant == 'base':
                    verify_parent_replay(folder, Path(spec['baselines'][period]))
                details = stress_details(folder, risk.initial_equity)
                save_json(folder/'details.json', details)
                seal(folder)
                runs[f'{period}/{variant}'] = sha256(folder/'files.json')
                rows.append({'period': period, 'variant': variant, **metrics, **details})
                save_json(out/'results.json', rows)
                save_json(out/'runs.json', runs)
                print(f'{period}/{variant}: {metrics["total_return"]:.6%}, {metrics["closed_trades"]}거래, 중지 {metrics["permanent_halt"]}', flush=True)
            seal(out/period)
            runs[period] = sha256(out/period/'files.json')
            del bars
        guard_run_inputs(out, reference, expected)
        save_json(out/'runs.json', runs)
        save_json(out/'summary.json', stress_summary(rows))
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def verify_holding_stress(run, expected, output):
    checked_files(run)
    if sha256(run/'files.json') != expected:
        raise ValueError('보유 정책 비용 검산의 호출자 실행 지문 오류')
    settings = read_json(run/'manifest.json')['settings']
    spec, frozen = guard_run_inputs(run, Path(settings['reference_path']), settings['reference_sha256'])
    if settings['periods'] != STRESS_PERIODS or settings['variants'] != STRESS_VARIANTS:
        raise ValueError('보유 정책 비용 검산의 기간·조건 변경')
    runs = read_json(run/'runs.json')
    expected_keys = {*STRESS_PERIODS, *[f'{p}/{v}' for p in STRESS_PERIODS for v in STRESS_VARIANTS]}
    if set(runs) != expected_keys:
        raise ValueError('보유 정책 비용 검산의 실행 봉인 목록 오류')
    for name, digest in runs.items():
        checked_files(run/name)
        if sha256(run/name/'files.json') != digest:
            raise ValueError('보유 정책 비용 검산의 하위 봉인 오류')
    out = new_run(output, 'holding-stress-review', {'source': str(run), 'source_files_sha256': expected})
    print(out, flush=True)
    try:
        verified, rows = {}, []
        for period, bounds in STRESS_PERIODS.items():
            paths = spec['inputs'][period]
            bars, checks = prepare_minute_period(Path(paths['market']), Path(paths['features']), 'BTCUSDT', *bounds, minute_inputs=True)
            require_row(checks, read_json(run/period/'input_verification.json'), '시세 연결')
            for variant, risk in stress_configs(frozen['risk']).items():
                folder = run/period/variant
                require_row(read_json(folder/'config.json'), asdict(risk), '비용·위험')
                _, policy = load_selection(Path(spec['selection']))
                verified[f'{period}/{variant}'] = verify_execution(folder, bars, policy)
                if variant == 'base':
                    verify_parent_replay(folder, Path(spec['baselines'][period]))
                details = stress_details(folder, risk.initial_equity)
                require_row(details, read_json(folder/'details.json'), '비용·집중도·구간')
                rows.append({'period': period, 'variant': variant, **read_json(folder/'metrics.json'), **details})
                save_json(out/'progress.json', verified)
                print(f'{period}/{variant}: 전체 계좌·회계 검산 일치', flush=True)
            del bars
        require_row(rows, read_json(run/'results.json'), '전체 결과')
        require_row(stress_summary(rows), read_json(run/'summary.json'), '완료·판정')
        require_row({key: read_json(Path(folder)/'metrics.json') for key, folder in spec['retained_runs'].items()}, read_json(run/'retained_results.json'), '기존 실패·독립 연도')
        guard_run_inputs(run, Path(settings['reference_path']), settings['reference_sha256'])
        checked_files(run)
        if sha256(run/'files.json') != expected:
            raise ValueError('보유 정책 비용 검산 중 실행 변경')
        for name, digest in runs.items():
            checked_files(run/name)
            if sha256(run/name/'files.json') != digest:
                raise ValueError('보유 정책 비용 검산 중 하위 실행 변경')
        save_json(out/'verification.json', {'complete': True, 'source_files_sha256': expected,
            'all_twelve_executions_independently_replayed': True, 'cash_fees_funding_and_net_verified': True,
            'previous_failed_sample_counts_preserved': True, 'runs': verified, 'profitability_accepted': False})
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
