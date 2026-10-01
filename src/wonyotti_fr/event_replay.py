from __future__ import annotations

import json
from pathlib import Path

from .common import new_run, save_json, sha256
from .engine import EngineConfig, TradingEngine
from .event_backtest import iter_events, prepare_period
from .event_research import load_selection
from .journal import EventJournal, canonical, digest


def replay_identity(selection: Path, market: Path, symbol: str, start: str, end: str) -> dict:
    source = Path(__file__).parent
    return {'selection_sha256': sha256(selection / 'frozen_selection.json'),
            'market_manifest_sha256': sha256(market / 'manifest-5m.json'),
            'symbol': symbol, 'start': start, 'end_exclusive': end,
            'source_sha256': {path.name: sha256(path) for path in sorted(source.glob('*.py'))},
            'mode': 'offline_only'}


def run_event_replay(selection: Path, market: Path, symbol: str, start: str, end: str,
                     journal_path: Path, output: Path, max_bars: int | None = None,
                     halt: bool = False, verify_memory: bool = False) -> Path:
    if max_bars is not None and (type(max_bars) is not int or max_bars < 0):
        raise ValueError('최대 처리 봉 수는 음수가 아닌 정수여야 합니다.')
    frozen, policy = load_selection(selection)
    identity = replay_identity(selection, market, symbol, start, end)
    config = EngineConfig(**frozen['risk'])
    bars = prepare_period(market, symbol, start, end)
    if hasattr(policy, 'prepare'):
        policy.prepare(bars)
    destination = new_run(output, 'event-replay', {**identity, 'journal': str(journal_path.resolve()),
                                                 'max_bars': max_bars, 'halt': halt, 'verify_memory': verify_memory})
    try:
        with EventJournal(journal_path, config, identity) as journal:
            if halt:
                journal.halt()
            previous = journal.snapshot()
            offset = previous['index']
            if offset > len(bars):
                raise ValueError('저장한 진행 위치가 자료 길이를 벗어납니다.')
            stop = len(bars) if max_bars is None else min(len(bars), offset + max_bars)
            if previous['completed'] and offset != len(bars):
                raise ValueError('저장한 완료 상태와 자료 끝이 다릅니다.')
            for index, event in enumerate(iter_events(bars.iloc[offset:stop]), start=offset):
                journal.process(event, policy, final=index == len(bars) - 1)
            journal.verify()
            state = journal.snapshot()
            parity = None
            if verify_memory:
                if state['manual_halt']:
                    raise ValueError('수동 중지 이력은 제어 사건 없는 메모리 실행과 비교할 수 없습니다.')
                memory = TradingEngine(config)
                expected = []
                for index, event in enumerate(iter_events(bars.iloc[:state['index']])):
                    expected.append(memory.step(event, policy, final=index == len(bars) - 1))
                recorded = journal.results()
                parity = (canonical(expected) == canonical(recorded) and canonical(memory.snapshot()) == canonical(state))
                if not parity:
                    raise ValueError('중단·재개 결과가 단일 메모리 실행과 다릅니다.')
            report = {'processed_this_run': stop - offset, 'processed_total': state['index'],
                      'total_bars': len(bars), 'completed': state['completed'],
                      'memory_parity': parity, 'manual_halt': state['manual_halt'],
                      'state_sha256': digest(canonical(state)), 'final_state': state,
                      'journal_chain_verified': True,
                      'hash_limit': '로컬 변경 탐지용 해시이며 서명·외부 계정 인증·완전한 삭제 탐지 수단이 아님'}
            save_json(destination / 'replay.json', report)
            (destination / 'REPORT.md').write_text(
                '# 영속 상태를 사용하는 오프라인 재생\n\n'
                f'이번 처리 {stop - offset:,}봉, 누적 {state["index"]:,}/{len(bars):,}봉. '
                f'전체 기간 완료: {state["completed"]}. 단일 메모리 실행 일치: {parity}.\n\n'
                '중간 종료에서는 강제 청산하지 않는다. 다음 실행은 같은 모델·자료·설정·소스의 저널을 이어서 처리한다. '
                '같은 사건의 재전달은 재체결하지 않으며 다른 값의 같은 사건은 거부한다. '
                '실제 주문이나 인증 연결은 없다.\n\n'
                '해시 연결은 로컬 손상 탐지용이다. 공격자가 전체 저널을 재작성하거나 끝을 삭제하는 상황을 외부 증거 없이 모두 인증하지 못한다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(json.dumps({key: report[key] for key in ['processed_this_run', 'processed_total', 'completed', 'memory_parity']}, ensure_ascii=False), flush=True)
    print(f'재생 기록: {destination}', flush=True)
    return destination
