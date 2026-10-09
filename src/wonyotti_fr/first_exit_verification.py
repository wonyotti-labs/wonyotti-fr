from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .close_effect import CLOSE_FEATURES
from .common import new_run, save_json, sha256
from .engine import EngineConfig, PolicyDecision, TradingEngine
from .event_backtest import iter_events
from .event_diagnostics import decompose_run
from .event_research import load_selection
from .first_exit_reference import (
    FIRST_EXIT_BASELINES,
    FIRST_EXIT_PERIODS,
    checked_first_exit_reference,
)
from .first_exit_research import first_exit_breakdown, verify_parent_replay
from .journal import canonical
from .minute_data import prepare_minute_period
from .retained_weight_close_reference import checked_files


def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def require_row(actual, expected, kind):
    if actual != expected:
        raise ValueError('첫 청산 독립 재생의 저장 행 불일치: '+kind)


class RecordedControlAudit:
    def __init__(self, parent, trace):
        self.parent, self.trace = parent, trace
        self.calls = 0

    def prepare(self, bars):
        self.parent.prepare(bars)

    def __call__(self, bar, state):
        if self.trace is None:
            return self.parent(bar, state)
        calls = []
        original = self.parent.feature_values

        def observed(current, context):
            values = original(current, context)
            calls.append(np.array(values, dtype=float, copy=True))
            return values

        self.parent.feature_values = observed
        try:
            original_decision = self.parent(bar, state)
        finally:
            self.parent.feature_values = original
        if len(calls) > 1 or any(row.shape != (len(CLOSE_FEATURES),) for row in calls):
            raise ValueError('첫 청산 독립 재생의 부모 특징 호출 오류')
        supported = bool(len(calls) == 1 and all(math.isfinite(float(value)) for value in calls[0]))
        if state['direction'] == 0:
            reason = 'flat'
        elif state['halted']:
            reason = 'halted'
        elif not calls:
            reason = 'no_management_call'
        elif not supported:
            reason = 'nonfinite_features'
        elif original_decision.intent == 'exit':
            reason = 'original_exit'
        else:
            reason = 'first_eligible_exit'
        changed = reason == 'first_eligible_exit'
        expected = {'decision_time': bar['end'], 'direction': state['direction'], 'feature_calls': len(calls),
            'features_supported': supported, 'original_intent': original_decision.intent,
            'final_intent': 'exit' if changed else original_decision.intent,
            'original_event': original_decision.event, 'changed': changed, 'reason': reason}
        saved = next(self.trace, None)
        require_row(saved, expected, '대조 판단')
        self.calls += 1
        if changed:
            return PolicyDecision(saved['final_intent'], original_decision.state, 'first_exit_control')
        return original_decision


def verify_execution(folder, bars, parent, trace_path=None):
    config = EngineConfig(**json.loads((folder/'config.json').read_text()))
    policy = RecordedControlAudit(parent, parquet_rows(trace_path) if trace_path is not None else None)
    policy.prepare(bars)
    engine = TradingEngine(config)
    curves, fills, trades = [parquet_rows(folder/(name+'.parquet')) for name in ['equity', 'fills', 'trades']]
    fill_flows, fee_values, funding_values, net_values, gross_values = [], [], [], [], []
    rejects, fill_count, trade_count = {}, 0, 0
    for index, bar in enumerate(iter_events(bars)):
        funding_values.append(engine.state['quantity']*bar['open']*bar['funding_rate'])
        result = engine.step(bar, policy, final=index == len(bars)-1)
        for item in result.pop('fills'):
            require_row(next(fills, None), item, '체결')
            if bar.get('count', 1) <= 0 or bar.get('volume', 1) <= 0:
                raise ValueError('첫 청산 독립 재생의 무거래 체결')
            np.testing.assert_allclose(item['fee'], abs(item['delta_quantity'])*item['price']*config.fee_bps/10000, rtol=0, atol=1e-9)
            fill_flows.append(item['delta_quantity']*item['price'])
            fee_values.append(item['fee'])
            fill_count += 1
        for item in result.pop('closed_trades'):
            require_row(next(trades, None), item, '종료 거래')
            np.testing.assert_allclose(item['gross_realized']-item['fees']-item['funding_cost'], item['net_pnl'], rtol=0, atol=1e-9)
            net_values.append(item['net_pnl'])
            gross_values.append(item['gross_realized'])
            trade_count += 1
        for name in result.pop('rejected'):
            rejects[name] = rejects.get(name, 0)+1
        require_row(next(curves, None), result, '분별 계좌')
    if any(next(stream, None) is not None for stream in [curves, fills, trades]):
        raise ValueError('첫 청산 독립 재생의 저장 행 초과')
    if policy.trace is not None and next(policy.trace, None) is not None:
        raise ValueError('첫 청산 독립 재생의 판단 행 초과')
    require_row(engine.snapshot(), json.loads((folder/'final_state.json').read_text()), '최종 상태')
    metrics = json.loads((folder/'metrics.json').read_text())
    cash = config.initial_equity-math.fsum(fill_flows)-math.fsum(fee_values)-math.fsum(funding_values)
    if engine.state['quantity'] != 0 or not engine.state['completed']:
        raise ValueError('첫 청산 독립 재생의 미종료 계좌')
    np.testing.assert_allclose([cash, math.fsum(net_values), -math.fsum(fill_flows), math.fsum(fee_values), math.fsum(funding_values)],
        [engine.state['cash'], cash-config.initial_equity, math.fsum(gross_values), metrics['fees'], metrics['funding_cost']], rtol=0, atol=1e-7)
    np.testing.assert_allclose([metrics['total_return'], metrics['final_equity'], metrics['max_drawdown']],
        [cash/config.initial_equity-1, cash, engine.state['max_drawdown']], rtol=0, atol=1e-9)
    if (metrics['closed_trades'] != trade_count or metrics['bars'] != len(bars) or metrics['rejected'] != rejects
        or metrics['permanent_halt'] != engine.state['permanent_halted']):
        raise ValueError('첫 청산 독립 재생의 전체 지표 오류')
    return {'all_policy_decisions_fills_trades_equity_and_state_exact': True, 'cash_fees_funding_and_net_verified': True,
        'bars': len(bars), 'fills': fill_count, 'trades': trade_count, 'control_decisions': policy.calls}


def verify_first_exit_run(run, expected_sha256, output):
    if sha256(run/'files.json') != expected_sha256:
        raise ValueError('첫 청산 독립 검산의 호출자 지정 지문 불일치')
    files = checked_files(run)
    manifest = json.loads((run/'manifest.json').read_text())
    settings = manifest['settings']
    evidence = checked_first_exit_reference(Path(settings['reference']), Path(settings['verification']), settings['verification_sha256'])
    if (any(settings.get(key) != value for key, value in evidence.items())
        or settings['periods'] != FIRST_EXIT_PERIODS or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V91.md'))
        or settings['policies'] != ['parent', 'always_first_exit'] or settings['failed_model_applied'] is not False
        or settings['new_models_fitted'] is not False or settings['profitability_accepted'] is not False):
        raise ValueError('첫 청산 독립 검산의 계획·부모·설정 오류')
    runs = json.loads((run/'runs.json').read_text())
    expected_runs = {year for year in FIRST_EXIT_PERIODS} | {f'{year}/{name}' for year in FIRST_EXIT_PERIODS for name in settings['policies']}
    if set(runs) != expected_runs:
        raise ValueError('첫 청산 독립 검산의 실행 목록 오류')
    for name, value in runs.items():
        checked_files(run/name)
        if sha256(run/name/'files.json') != value:
            raise ValueError('첫 청산 독립 검산의 하위 출력 지문 변경')
    if any(sha256(Path(__file__).parent/name) != value for name, value in manifest['source_sha256'].items()):
        raise ValueError('첫 청산 독립 검산의 실행 코드 변경')
    out = new_run(output, 'first-exit-review', {'source': str(run), 'source_files_sha256': expected_sha256,
        'protocol_sha256': settings['protocol_sha256'], 'independent_control_replay': True})
    print(f'첫 청산 기준의 독립 순차 검산: {out}', flush=True)
    try:
        verified = {}
        for year, bounds in FIRST_EXIT_PERIODS.items():
            inputs = evidence['inputs'][year]
            bars, checks = prepare_minute_period(Path(inputs['market']), Path(inputs['features']), 'BTCUSDT', *bounds, minute_inputs=True)
            if canonical(checks) != canonical(json.loads((run/year/'input_verification.json').read_text())):
                raise ValueError('첫 청산 독립 검산의 시세 연결 변경')
            for name in settings['policies']:
                frozen, parent = load_selection(Path(evidence['parent']))
                folder = run/year/name
                if frozen['risk'] != evidence['risk'] or json.loads((folder/'config.json').read_text()) != evidence['risk']:
                    raise ValueError('첫 청산 독립 검산의 위험 설정 변경')
                trace = run/year/'control_decisions.parquet' if name == 'always_first_exit' else None
                verified[f'{year}/{name}'] = verify_execution(folder, bars, parent, trace)
                if name == 'parent':
                    verify_parent_replay(folder, Path(evidence['parent'])/FIRST_EXIT_BASELINES[year])
                require_row(decompose_run(folder, evidence['risk']['initial_equity']), json.loads((folder/'decomposition.json').read_text()), '손익 분해')
                require_row(first_exit_breakdown(folder), json.loads((folder/'breakdown.json').read_text()), '방향·월별 집계')
                save_json(out/'progress.json', verified)
                print(f'{year} {name}: 전체 판단·체결·계좌 검산 일치', flush=True)
            del bars
        summary = json.loads((run/'summary.json').read_text())
        results = json.loads((run/'results.json').read_text())
        if len(results) != 4 or {(row['year'], row['policy']) for row in results} != {(y, p) for y in FIRST_EXIT_PERIODS for p in settings['policies']}:
            raise ValueError('첫 청산 독립 검산의 집계 실행 목록 오류')
        checks = {}
        for row in results:
            metrics = json.loads((run/row['year']/row['policy']/'metrics.json').read_text())
            if any(row.get(key) != value for key, value in metrics.items()):
                raise ValueError('첫 청산 독립 검산의 실행 지표 집계 오류')
            require_row(row['decomposition'], json.loads((run/row['year']/row['policy']/'decomposition.json').read_text()), '손익 분해 집계')
            if row['policy'] == 'always_first_exit':
                checks[row['year']] = {'positive': metrics['total_return'] > 0, 'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt']}
        if summary != {'complete': True, 'runs': 4, 'parent_replays_exact': True, 'checks': checks,
            'further_evaluation_eligible': all(all(group.values()) for group in checks.values()),
            'new_models_fitted': False, 'failed_model_applied': False, 'profitability_accepted': False}:
            raise ValueError('첫 청산 독립 검산의 최종 판정 오류')
        if checked_files(run) != files or sha256(run/'files.json') != expected_sha256:
            raise ValueError('첫 청산 독립 검산 중 출력 변경')
        for name, value in runs.items():
            checked_files(run/name)
            if sha256(run/name/'files.json') != value:
                raise ValueError('첫 청산 독립 검산 중 하위 출력 변경')
        if checked_first_exit_reference(Path(settings['reference']), Path(settings['verification']), settings['verification_sha256']) != evidence:
            raise ValueError('첫 청산 독립 검산 중 출처 변경')
        save_json(out/'verification.json', {'complete': True, 'source_files_sha256': expected_sha256,
            'all_four_executions_independently_replayed': True, 'all_policy_decisions_fills_trades_equity_and_state_exact': True,
            'cash_fees_funding_and_net_verified': True, 'runs': verified, 'profitability_accepted': False})
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
