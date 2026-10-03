from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import new_run, save_json
from .engine import EngineConfig, TradingEngine
from .event_backtest import iter_events, prepare_period
from .event_replay import replay_identity
from .event_research import load_selection
from .journal import EventJournal, canonical
from .minute_data import prepare_minute_period
from .period_guard import guard_replay_period


def crash_worker(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text())
    frozen, policy = load_selection(Path(payload['selection']))
    with EventJournal(Path(payload['journal']), EngineConfig(**frozen['risk']), payload['identity']) as journal:
        journal.process(payload['event'], policy, before_commit=lambda: os._exit(73))
    raise RuntimeError('강제 종료 주입 지점에 도달하지 못했습니다.')


def verify_stress(events: list[dict], selection: Path, directory: Path, identity: dict,
                  interruption_kind: str = 'position', require_state: bool = False) -> dict:
    if len(events) < 4:
        raise ValueError('장애 검증 사건이 부족합니다.')
    if interruption_kind not in {'position', 'waiting', 'liquidity', 'addition'}:
        raise ValueError('지원하지 않는 중단 상태')
    frozen, policy = load_selection(selection)
    config = EngineConfig(**frozen['risk'])
    memory = TradingEngine(config)
    expected = [memory.step(event, policy, final=index == len(events) - 1) for index, event in enumerate(events)]
    candidates = [index + 1 for index, result in enumerate(expected[:-2])
                  if (result['quantity'] != 0 if interruption_kind == 'position' else
                      ('market_no_trades' in result['rejected'] and result['next_intent'] != 'hold')
                      if interruption_kind == 'liquidity' else
                      result.get('policy_event') in {'action_increase', 'action_add_rejected'}
                      if interruption_kind == 'addition' else bool(json.loads(result.get('policy_state', '{}'))))]
    if not candidates and (require_state or interruption_kind in {'waiting', 'liquidity', 'addition'}):
        raise ValueError(f'요청한 {interruption_kind} 상태가 없어 장애 검증을 실행할 수 없습니다.')
    cut = candidates[0] if candidates else min(24, len(events) - 2)
    journal_path = directory / 'crash.sqlite'
    checks = {}
    with EventJournal(journal_path, config, identity) as journal:
        for event in events[:cut]:
            journal.process(event, policy)
        before = journal.snapshot()
        duplicate = journal.process(events[cut - 1], policy)
        checks['duplicate_not_reexecuted'] = duplicate['duplicate'] and canonical(before) == canonical(journal.snapshot())
        altered = {**events[cut - 1], 'close': events[cut - 1]['close'] * 1.0001}
        altered['high'] = max(altered['high'], altered['close'])
        invalid = {**events[cut], 'high': 0}
        for name, event in [('changed_duplicate_rejected', altered), ('missing_bar_rejected', events[cut + 1]), ('invalid_ohlc_rejected', invalid)]:
            try:
                journal.process(event, policy)
            except ValueError:
                checks[name] = canonical(before) == canonical(journal.snapshot())
            else:
                checks[name] = False
    payload = directory / 'crash_input.json'
    save_json(payload, {'selection': str(selection.resolve()), 'journal': str(journal_path.resolve()),
                        'identity': identity, 'event': events[cut]})
    process = subprocess.run([sys.executable, '-m', 'wonyotti_fr.engine_stress', '--worker', str(payload.resolve())],
                             capture_output=True, text=True, timeout=30, check=False)
    save_json(directory / 'crash_process.json', {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
    if process.returncode != 73:
        raise ValueError('실제 프로세스 종료 검증에 실패했습니다.')
    with EventJournal(journal_path, config, identity) as journal:
        checks['process_exit_before_commit_rolled_back'] = canonical(before) == canonical(journal.snapshot())
        for index in range(cut, len(events)):
            journal.process(events[index], policy, final=index == len(events) - 1)
        journal.verify()
        checks['crash_restart_full_results_equal'] = canonical(journal.results()) == canonical(expected)
        checks['crash_restart_final_state_equal'] = canonical(journal.snapshot()) == canonical(memory.snapshot())
    tradable_cut = next((i for i in range(cut, len(events)) if events[i].get('count', 1) > 0), cut)
    manual_memory = TradingEngine(config)
    manual_expected = []
    with EventJournal(directory / 'manual.sqlite', config, {**identity, 'manual_control': True}) as journal:
        for index, event in enumerate(events):
            if index == cut:
                journal.halt()
                manual_memory.halt()
            expected_row = manual_memory.step(event, policy, final=index == len(events) - 1)
            manual_expected.append(expected_row)
            actual = journal.process(event, policy, final=index == len(events) - 1)
            if index == tradable_cut:
                checks['manual_halt_liquidates_when_executable'] = actual['quantity'] == 0 and actual['next_intent'] == 'hold'
        checks['manual_halt_full_results_equal'] = canonical(journal.results()) == canonical(manual_expected)
        checks['manual_halt_no_reentry'] = all(row['quantity'] == 0 for row in manual_expected[tradable_cut:])
    if not all(checks.values()):
        raise ValueError(f'장애 검증 미통과: {[name for name, passed in checks.items() if not passed]}')
    return {'checks': checks, 'bars': len(events), 'interruption_after_bars': cut,
            'position_open_at_interruption': before['quantity'] != 0,
            'waiting_at_interruption': bool(before.get('policy_state')), 'interruption_kind': interruption_kind,
            'child_process_exit_code': process.returncode, 'all_passed': True}


def run_engine_stress(selection: Path, market: Path, output: Path, start: str = '2020-03-10', end: str = '2020-03-14',
                       feature_market: Path | None = None) -> Path:
    frozen, _ = load_selection(selection)
    guard_replay_period(selection, frozen, start, end)
    identity = replay_identity(selection, market, 'BTCUSDT', start, end, feature_market)
    destination = new_run(output, 'engine-stress', {**identity,
                                                   'checks': '가격 급변 시세, 실제 프로세스 종료, 중복·누락·잘못된 입력, 수동 중지'})
    try:
        if frozen.get('protocol') in {'pullback_v7', 'net_edge_v8', 'lifecycle_v9', 'minute_action_v10', 'minute_action_v11', 'minute_path_v12', 'minute_reverse_v13', 'minute_rate_v14', 'minute_rate_reverse_v18', 'minute_inventory_v19', 'minute_inventory_recent_v20', 'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29', 'lifecycle_edge_v15', 'lifecycle_edge_v16', 'lifecycle_edge_v17'}:
            bars, input_checks = prepare_minute_period(market, feature_market, 'BTCUSDT', start, end,
                **({'minute_inputs': True} if frozen['protocol'] in {'minute_inventory_micro_v21', 'addition_effect_v22', 'recent_entry_v23', 'realized_exit_v24', 'new_position_v25', 'position_prior_v26', 'position_direction_v27', 'boosted_direction_v28', 'probe_entry_v29'} else {}))
            save_json(destination / 'input_verification.json', input_checks)
            events = list(iter_events(bars))
            cases = {}
            for kind in ['waiting', 'position']:
                directory = destination / kind
                directory.mkdir()
                cases[kind] = verify_stress(events, selection, directory, {**identity, 'interruption_kind': kind}, kind, True)
            save_json(destination / 'verification.json', {'all_passed': all(value['all_passed'] for value in cases.values()),
                                                         'bars': len(events), 'cases': cases})
            (destination / 'REPORT.md').write_text(
                '# 진입 대기·보유 중 실제 프로세스 종료 복구\n\n'
                f'{start}부터 {end} 직전까지 {len(events):,}개 1분 시세와 확정 5분 특징을 사용했다. '
                f'대기 중 {cases["waiting"]["interruption_after_bars"]}봉, 보유 중 {cases["position"]["interruption_after_bars"]}봉 뒤 '
                '서로 다른 저널의 프로세스를 커밋 직전에 실제 종료했다.\n\n'
                '두 경우 모두 전체 결과·최종 상태가 단일 실행과 일치했다. 중복·변경된 중복·누락·잘못된 시세를 검사했고, '
                '수동 중지는 대기를 비우고 다음 거래 가능 시세에서 보유분을 청산한 뒤 재진입을 차단했다. '
                '각 상태가 실제로 발생하지 않으면 이 검증은 통과하지 않는다. '
                '거래소 주문·인증·출금이나 실제 체결·호가·시장 충격은 검증 범위가 아니다.\n', encoding='utf-8')
            print(f'대기·보유 중 실제 장애 검증: {destination}', flush=True)
            return destination
        bars = prepare_period(market, 'BTCUSDT', start, end)
        result = verify_stress(list(iter_events(bars)), selection, destination, identity)
        save_json(destination / 'verification.json', result)
        (destination / 'REPORT.md').write_text(
            '# 실제 급변 구간의 오프라인 실행 장애 검증\n\n'
            f'{start}부터 {end} 직전까지 BTC 시세 {len(bars):,}봉을 사용했다. '
            f'{result["interruption_after_bars"]}봉 처리 후 중단을 주입했으며 보유 포지션 존재: {result["position_open_at_interruption"]}.\n\n'
            '독립 Python 프로세스를 트랜잭션 커밋 직전에 종료 코드 73으로 강제 종료했다. '
            '저널 재개 후 모든 체결·잔고·최종 상태가 단일 실행과 일치했다. '
            '중복·변경된 중복·누락 봉·잘못된 OHLC를 구분했고 거부된 입력이 상태를 바꾸지 않음을 확인했다. '
            '수동 중지는 다음 유효 시세에서 청산하고 이후 재진입하지 않았다.\n\n'
            '이는 해당 고정 시세와 모의 체결 가정의 기능 검증이다. 거래소 실제 체결·네트워크 장애·호가·시장 충격의 검증은 아니다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'실제 급변·장애 검증: {destination / "REPORT.md"}', flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    crash_worker(parser.parse_args().worker)
