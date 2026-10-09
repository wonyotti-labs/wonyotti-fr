from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_diagnostics import decompose_run
from .event_research import load_selection
from .first_exit_control import FirstExitControl
from .first_exit_reference import (
    FIRST_EXIT_BASELINES,
    FIRST_EXIT_PERIODS,
    FIRST_EXIT_RUN_FILES,
    checked_first_exit_reference,
)
from .minute_data import prepare_minute_period
from .streaming_backtest import ParquetRows


def verify_parent_replay(folder, previous):
    for name in FIRST_EXIT_RUN_FILES:
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(folder/name), pd.read_parquet(previous/name), check_exact=True)
        elif json.loads((folder/name).read_text()) != json.loads((previous/name).read_text()):
            raise ValueError('첫 청산 계좌 대조의 원래 부모 재생 불일치: '+name)


def first_exit_breakdown(folder):
    trades = pd.read_parquet(folder/'trades.parquet')
    if trades.empty:
        return []
    result = []
    for kind, groups in [('direction', trades.groupby('direction')),
        ('entry_month', trades.groupby(pd.to_datetime(trades.entry_time, utc=True).dt.strftime('%Y-%m')))]:
        for name, part in groups:
            result.append({'kind': kind, 'group': str(name), 'closed_trades': len(part),
                'net_pnl': float(part.net_pnl.sum()), 'gross_pnl_after_slippage': float(part.gross_realized.sum()),
                'fees': float(part.fees.sum()), 'funding_cost': float(part.funding_cost.sum())})
    return result


def run_first_exit_control(reference, verification, verification_sha256, output):
    evidence = checked_first_exit_reference(reference, verification, verification_sha256)
    parent = Path(evidence['parent'])
    settings = {**evidence, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V91.md')), 'periods': FIRST_EXIT_PERIODS,
        'symbol': 'BTCUSDT', 'policies': ['parent', 'always_first_exit'], 'new_models_fitted': False,
        'failed_model_applied': False, 'all_periods_already_observed': True, 'profitability_accepted': False}
    out = new_run(output, 'first-exit-control', settings)
    print(f'첫 청산 기준의 연속 계좌 대조: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', evidence)
        results, runs = [], {}
        risk = EngineConfig(**evidence['risk'])
        for year, bounds in FIRST_EXIT_PERIODS.items():
            phase = out/year
            phase.mkdir(mode=0o700)
            inputs = evidence['inputs'][year]
            bars, checks = prepare_minute_period(Path(inputs['market']), Path(inputs['features']), 'BTCUSDT', *bounds, minute_inputs=True)
            save_json(phase/'input_verification.json', checks)
            for name in ['parent', 'always_first_exit']:
                frozen, original = load_selection(parent)
                if frozen['risk'] != evidence['risk']:
                    raise ValueError('첫 청산 계좌 대조 실행 중 위험 설정 변경')
                trace = ParquetRows(phase/'control_decisions.parquet', 8192) if name == 'always_first_exit' else None
                policy = FirstExitControl(original, enabled=trace is not None, record=trace.append if trace else None)
                folder = phase/name
                print(f'{year} {name}: 연속 계좌 재생 시작', flush=True)
                try:
                    metrics = backtest(bars, policy, risk, folder)
                finally:
                    if trace is not None:
                        trace.close()
                if name == 'parent':
                    verify_parent_replay(folder, parent/FIRST_EXIT_BASELINES[year])
                decomposition = decompose_run(folder, risk.initial_equity)
                save_json(folder/'decomposition.json', decomposition)
                save_json(folder/'breakdown.json', first_exit_breakdown(folder))
                save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
                runs[f'{year}/{name}'] = sha256(folder/'files.json')
                results.append({'year': year, 'policy': name, **metrics, 'decomposition': decomposition})
                save_json(out/'results.json', results)
                print(f'{year} {name}: 수익 {metrics["total_return"]:.6%}, {metrics["closed_trades"]}거래, 중지 {metrics["permanent_halt"]}', flush=True)
            save_json(phase/'files.json', {p.name: sha256(p) for p in phase.iterdir() if p.is_file()})
            runs[year] = sha256(phase/'files.json')
            del bars
        if checked_first_exit_reference(reference, verification, verification_sha256) != evidence:
            raise ValueError('첫 청산 계좌 대조 실행 중 입력·출처 변경')
        manifest = json.loads((out/'manifest.json').read_text())
        if (any(sha256(Path(__file__).parent/name) != value for name, value in manifest['source_sha256'].items())
            or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V91.md'))):
            raise ValueError('첫 청산 계좌 대조 실행 중 코드·계획 변경')
        control = [row for row in results if row['policy'] == 'always_first_exit']
        checks = {row['year']: {'positive': row['total_return'] > 0, 'at_least_30_trades': row['closed_trades'] >= 30,
            'no_halt': not row['permanent_halt']} for row in control}
        save_json(out/'runs.json', runs)
        save_json(out/'summary.json', {'complete': True, 'runs': len(results), 'parent_replays_exact': True,
            'checks': checks, 'further_evaluation_eligible': all(all(group.values()) for group in checks.values()),
            'new_models_fitted': False, 'failed_model_applied': False, 'profitability_accepted': False})
        (out/'REPORT.md').write_text('# 첫 청산 기준의 연속 계좌 대조\n\n'
            '같은 진입·비용·위험 한도로 조기 청산 후 재진입까지 두 기간에 재생했다. '
            '기존 부모의 분별 계좌·체결·거래·최종 상태를 대조했다. '
            '이미 본 두 기간의 설명 대조이며 실패한 청산 모델의 적용이나 수익성 채택이 아니다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
