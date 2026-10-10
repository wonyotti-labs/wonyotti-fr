from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from wonyotti_fr import holding_stress
from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.first_exit_verification import require_row, verify_execution
from wonyotti_fr.holding_ablation import HoldOnlyControl
from wonyotti_fr.holding_stress import (
    STRESS_PERIODS,
    STRESS_VARIANTS,
    checked_stress_reference,
    read_json,
    seal,
    stress_configs,
    stress_details,
    stress_summary,
)
from wonyotti_fr.minute_data import prepare_minute_period
from wonyotti_fr.retained_weight_close_reference import checked_files

SOURCE = Path(holding_stress.__file__).parent


def ablation_groups():
    groups = []
    for symbol in ['ETHUSDT', 'SOLUSDT']:
        groups.extend({'symbol': symbol, 'period': period, 'bounds': STRESS_PERIODS[period],
            'input_period': period, 'variants': STRESS_VARIANTS} for period in ['observed', 'recent'])
    groups.extend({'symbol': 'BTCUSDT', 'period': period, 'bounds': bounds,
        'input_period': period, 'variants': ['no_adds', 'hold_only']} for period, bounds in STRESS_PERIODS.items())
    for symbol in ['ETHUSDT', 'SOLUSDT']:
        groups.extend({'symbol': symbol, 'period': str(year), 'bounds': [f'{year}-01-01', f'{year+1}-01-01'],
            'input_period': 'observed', 'variants': ['base']} for year in [2023, 2024, 2025])
    return groups


def checked_seal(folder, expected, required):
    files = checked_files(folder)
    if sha256(folder/'files.json') != expected or not set(required) <= set(files):
        raise ValueError('보유 관리 기여 대조의 출처 봉인·파일 누락')
    return files


def checked_ablation_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 1024**2 or sha256(reference) != expected:
        raise ValueError('보유 관리 기여 대조의 호출자 출처 지문 오류')
    spec = read_json(reference)
    if (spec.get('profitability_accepted') is not False or spec.get('all_periods_already_observed') is not True
        or set(spec['inputs']) != {'ETHUSDT', 'SOLUSDT'}
        or any(set(periods) != {'observed', 'recent'} for periods in spec['inputs'].values())):
        raise ValueError('보유 관리 기여 대조의 시장·관찰 범위 오류')
    run, review = Path(spec['stress_run']), Path(spec['stress_review'])
    checked_seal(run, spec['stress_run_sha256'], ['manifest.json', 'summary.json', 'results.json', 'runs.json', 'retained_results.json'])
    checked_seal(review, spec['stress_review_sha256'], ['manifest.json', 'verification.json'])
    manifest, proof = read_json(run/'manifest.json'), read_json(review/'verification.json')
    settings = manifest['settings']
    prior, frozen = checked_stress_reference(Path(settings['reference_path']), settings['reference_sha256'])
    if (prior != settings['reference'] or frozen['risk'] != settings['risk']
        or settings['periods'] != STRESS_PERIODS or settings['variants'] != STRESS_VARIANTS
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V92.md'))):
        raise ValueError('보유 관리 기여 대조의 기존 정책·기간 연결 오류')
    expected_keys = {f'{p}/{v}' for p in STRESS_PERIODS for v in STRESS_VARIANTS}
    if (proof.get('complete') is not True or proof.get('all_twelve_executions_independently_replayed') is not True
        or proof.get('cash_fees_funding_and_net_verified') is not True
        or proof.get('previous_failed_sample_counts_preserved') is not True
        or proof.get('profitability_accepted') is not False or set(proof['runs']) != expected_keys
        or proof['source_files_sha256'] != spec['stress_run_sha256']
        or read_json(review/'manifest.json')['settings'] != {'source': str(run), 'source_files_sha256': spec['stress_run_sha256']}):
        raise ValueError('보유 관리 기여 대조의 열두 독립 검산 근거 오류')
    # 새 명령 등록 외에는 기존 정책·회계 구현을 바꾸지 않는다.
    for name, digest in manifest['source_sha256'].items():
        if (Path(name).name != name or sha256(run/'code_snapshot'/name) != digest
            or (name != 'cli.py' and sha256(SOURCE/name) != digest)):
            raise ValueError('보유 관리 기여 대조의 기존 실행 코드 변경')
    runs, rows = read_json(run/'runs.json'), read_json(run/'results.json')
    if set(runs) != expected_keys | set(STRESS_PERIODS):
        raise ValueError('보유 관리 기여 대조의 기존 실행 목록 오류')
    for key, digest in runs.items():
        checked_seal(run/key, digest, ['input_verification.json'] if key in STRESS_PERIODS else ['config.json', 'metrics.json', 'equity.parquet', 'fills.parquet', 'trades.parquet', 'final_state.json', 'details.json'])
    for row in rows:
        key = f"{row['period']}/{row['variant']}"
        metric = read_json(run/key/'metrics.json')
        require_row(read_json(run/key/'config.json'), asdict(stress_configs(frozen['risk'])[row['variant']]), '기존 위험·비용')
        require_row(row, {'period': row['period'], 'variant': row['variant'], **metric, **read_json(run/key/'details.json')}, '기존 집계')
        record = proof['runs'][key]
        if (record.get('all_policy_decisions_fills_trades_equity_and_state_exact') is not True
            or record.get('cash_fees_funding_and_net_verified') is not True
            or record['trades'] != metric['closed_trades'] or record['bars'] != metric['bars']):
            raise ValueError('보유 관리 기여 대조의 기존 개별 검산 오류')
    summary = stress_summary(rows)
    require_row(summary, read_json(run/'summary.json'), '기존 완료·분기')
    require_row(read_json(run/'retained_results.json'),
        {key: read_json(Path(folder)/'metrics.json') for key, folder in prior['retained_runs'].items()}, '기존 실패·연도 근거')
    if not summary['cross_market_research_eligible']:
        raise ValueError('보유 관리 기여 대조의 사전 비용·기간 조건 미달')
    required = {str(Path(paths[key])/name) for periods in spec['inputs'].values() for paths in periods.values()
        for key, name in [('market', 'manifest-1m.json'), ('features', 'manifest-5m.json')]}
    if not required <= set(spec['sealed_files']):
        raise ValueError('보유 관리 기여 대조의 시장 입력 봉인 누락')
    for name, digest in spec['sealed_files'].items():
        target = Path(name)
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise ValueError('보유 관리 기여 대조의 시장 입력 지문 변경')
    return spec, prior, frozen


def ablation_config(risk, variant):
    configs = stress_configs(risk)
    if variant == 'no_adds':
        return replace(configs['base'], max_adds=0)
    if variant == 'hold_only':
        return configs['base']
    if variant not in configs:
        raise ValueError('보유 관리 기여 대조의 조건 오류')
    return configs[variant]


def ablation_policy(selection, frozen, variant):
    current, parent = load_selection(selection)
    require_row(current, frozen, '고정 부모')
    return HoldOnlyControl(parent) if variant == 'hold_only' else parent


def ablation_summary(rows):
    expected = {(g['symbol'], g['period'], v) for g in ablation_groups() for v in g['variants']}
    if len(rows) != 28 or {(r['symbol'], r['period'], r['variant']) for r in rows} != expected:
        raise ValueError('보유 관리 기여 대조의 스물여덟 실행 누락·중복')
    checks = {f"{r['symbol']}/{r['period']}/{r['variant']}": {'positive': r['total_return'] > 0,
        'at_least_30_trades': r['closed_trades'] >= 30, 'no_halt': not r['permanent_halt']} for r in rows}
    return {'complete': True, 'new_runs': 28, 'retained_btc_runs': 12, 'retained_legacy_runs': 4,
        'checks': checks, 'new_models_fitted': False, 'automatic_candidate_selection': False,
        'previous_failures_preserved': True, 'profitability_accepted': False}


def ablation_retained(spec):
    run = Path(spec['stress_run'])
    return {'btc_stress': read_json(run/'results.json'), 'legacy': read_json(run/'retained_results.json')}


def guard_ablation_inputs(out, reference, expected):
    spec, prior, frozen = checked_ablation_reference(reference, expected)
    manifest = read_json(out/'manifest.json')
    settings = manifest['settings']
    if (settings['reference'] != spec or settings['risk'] != frozen['risk']
        or settings['groups'] != ablation_groups()
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V93.md'))
        or any(sha256(SOURCE/name) != value for name, value in manifest['source_sha256'].items())):
        raise ValueError('보유 관리 기여 대조 중 코드·계획·입력 변경')
    return spec, prior, frozen


def group_bars(group, spec, prior):
    paths = prior['inputs'][group['input_period']] if group['symbol'] == 'BTCUSDT' else spec['inputs'][group['symbol']][group['input_period']]
    return prepare_minute_period(Path(paths['market']), Path(paths['features']), group['symbol'], *group['bounds'], minute_inputs=True)


def run_holding_ablation(reference, expected, output):
    spec, prior, frozen = checked_ablation_reference(reference, expected)
    out = new_run(output, 'holding-ablation', {'reference': spec, 'reference_path': str(reference),
        'reference_sha256': expected, 'risk': frozen['risk'], 'groups': ablation_groups(),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V93.md')), 'all_periods_already_observed': True})
    print(out, flush=True)
    try:
        rows, runs = [], {}
        save_json(out/'retained_results.json', ablation_retained(spec))
        for group in ablation_groups():
            key = f"{group['symbol']}/{group['period']}"
            bars, checks = group_bars(group, spec, prior)
            save_json(out/key/'input_verification.json', checks)
            for variant in group['variants']:
                label = f'{key}/{variant}'
                folder = out/label
                risk = ablation_config(frozen['risk'], variant)
                policy = ablation_policy(Path(prior['selection']), frozen, variant)
                print(label+': 연속 계좌 시작', flush=True)
                metrics = backtest(bars, policy, risk, folder)
                details = stress_details(folder, risk.initial_equity)
                save_json(folder/'details.json', details)
                seal(folder)
                runs[label] = sha256(folder/'files.json')
                rows.append({'symbol': group['symbol'], 'period': group['period'], 'variant': variant, **metrics, **details})
                save_json(out/'results.json', rows)
                save_json(out/'runs.json', runs)
                print(f'{label}: {metrics["total_return"]:.6%}, {metrics["closed_trades"]}거래, 중지 {metrics["permanent_halt"]}', flush=True)
            seal(out/key)
            runs[key] = sha256(out/key/'files.json')
            del bars
        guard_ablation_inputs(out, reference, expected)
        save_json(out/'runs.json', runs)
        save_json(out/'summary.json', ablation_summary(rows))
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def verify_holding_ablation(run, expected, output):
    checked_seal(run, expected, ['manifest.json', 'summary.json', 'runs.json', 'results.json', 'retained_results.json'])
    settings = read_json(run/'manifest.json')['settings']
    spec, prior, frozen = guard_ablation_inputs(run, Path(settings['reference_path']), settings['reference_sha256'])
    runs = read_json(run/'runs.json')
    expected_keys = {f"{g['symbol']}/{g['period']}" for g in ablation_groups()} | {f"{g['symbol']}/{g['period']}/{v}" for g in ablation_groups() for v in g['variants']}
    if set(runs) != expected_keys:
        raise ValueError('보유 관리 기여 검산의 실행 목록 오류')
    for name, digest in runs.items():
        checked_seal(run/name, digest, [])
    out = new_run(output, 'holding-ablation-review', {'source': str(run), 'source_files_sha256': expected})
    print(out, flush=True)
    try:
        verified, rows = {}, []
        for group in ablation_groups():
            key = f"{group['symbol']}/{group['period']}"
            bars, checks = group_bars(group, spec, prior)
            require_row(checks, read_json(run/key/'input_verification.json'), '다른 시장 시세')
            for variant in group['variants']:
                label = f'{key}/{variant}'
                folder = run/label
                risk = ablation_config(frozen['risk'], variant)
                require_row(read_json(folder/'config.json'), asdict(risk), '대조 위험·비용')
                policy = ablation_policy(Path(prior['selection']), frozen, variant)
                verified[label] = verify_execution(folder, bars, policy)
                details = stress_details(folder, risk.initial_equity)
                require_row(details, read_json(folder/'details.json'), '대조 비용·집중도·구간')
                reasons = details['decomposition']['fills_by_reason']
                if ((variant == 'no_adds' and 'increase' in reasons)
                    or (variant == 'hold_only' and {'increase', 'signal_reduce', 'signal_exit', 'signal_reverse'} & set(reasons))):
                    raise ValueError('보유 관리 기여 검산의 금지된 체결 발견')
                rows.append({'symbol': group['symbol'], 'period': group['period'], 'variant': variant, **read_json(folder/'metrics.json'), **details})
                save_json(out/'progress.json', verified)
                print(label+': 전체 계좌·회계 검산 일치', flush=True)
            del bars
        require_row(rows, read_json(run/'results.json'), '스물여덟 결과')
        require_row(ablation_summary(rows), read_json(run/'summary.json'), '기여 대조 판정')
        require_row(ablation_retained(spec), read_json(run/'retained_results.json'), '기존 실패·비용·연도')
        guard_ablation_inputs(run, Path(settings['reference_path']), settings['reference_sha256'])
        checked_seal(run, expected, [])
        for name, digest in runs.items():
            checked_seal(run/name, digest, [])
        save_json(out/'verification.json', {'complete': True, 'source_files_sha256': expected,
            'all_twenty_eight_executions_independently_replayed': True,
            'cash_fees_funding_and_net_verified': True, 'previous_failures_preserved': True,
            'runs': verified, 'profitability_accepted': False})
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
