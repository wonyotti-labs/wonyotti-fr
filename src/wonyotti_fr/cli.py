from __future__ import annotations

import argparse
from pathlib import Path

from .action_research import run_action_selection
from .audit import aggregate_events, write_audit
from .bitmex_history import fetch_bitmex_history
from .common import new_run, save_json
from .context_study import run_context_study
from .edge_diagnostics import run_edge_diagnostics
from .edge_research import run_edge_selection
from .engine_stress import run_engine_stress
from .event_diagnostics import run_event_diagnostics
from .event_evaluation import run_event_evaluation
from .event_replay import run_event_replay
from .event_research import run_event_selection
from .event_study import run_event_study
from .event_walkforward import run_event_walkforward
from .execution_study import run_execution_study
from .expansion_evaluation import run_expansion_evaluation
from .expansion_research import run_expansion_selection
from .frequency_evaluation import run_frequency_evaluation
from .frequency_research import run_frequency_selection
from .lifecycle_research import run_lifecycle_selection
from .market import fetch_market, repair_gaps
from .minute_repair import repair_minute_market
from .net_edge_research import run_net_selection
from .offline import demo, replay
from .paired_repair import load_repair_targets, repair_paired_market
from .portfolio import reconstruct_portfolio
from .pullback_evaluation import run_pullback_evaluation
from .pullback_research import run_pullback_selection
from .reconciliation import reconcile_wallet
from .reconstruct import reconstruct
from .reports import write_reconstruction_report
from .research import run_research
from .robustness import run_robustness
from .structure_study import run_structure_study
from .study import run_study
from .timing_study import run_timing_study
from .verification import verify_samples


def run_audit(args) -> None:
    source = args.source.resolve(strict=True)
    destination = new_run(args.output, "audit", {"source": str(source), "timezone": args.timezone})
    print(f"감사 시작: {destination}", flush=True)
    data, wallet, audit = write_audit(source, destination, args.timezone)
    events = aggregate_events(data, "XBTUSD")
    events.to_parquet(destination / "xbtusd_events.parquet", index=False)
    episodes, actions, summary = reconstruct(events)
    wallet_pnl = int(wallet.loc[(wallet.address == "XBTUSD") & (wallet.transacttype == "RealisedPNL"), "amount"].sum())
    summary["wallet_xbtusd_pnl_btc"] = wallet_pnl / 1e8
    summary["wallet_difference_btc"] = summary["realized_net_btc"] - wallet_pnl / 1e8
    daily, summary["wallet_reconciliation"] = reconcile_wallet(actions, wallet)
    daily.to_csv(destination / "wallet_daily_reconciliation.csv", index=False)
    save_json(destination / "reconstruction.json", summary)
    episodes.to_parquet(destination / "episodes.parquet", index=False)
    actions.to_parquet(destination / "actions.parquet", index=False)
    write_reconstruction_report(destination, audit, summary, episodes, actions)
    print(f"보고서: {destination / 'REPORT.md'}", flush=True)
    print(f"에피소드 {len(episodes):,}개 / 같은 기간 대조 잔차 {summary['wallet_reconciliation']['aligned_residual_satoshi']:.6f} 사토시", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wonyotti: offline research only")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="전체 입력 검사 및 XBTUSD 복원")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--timezone", default="UTC", help="독립 검증 전까지 명시적 가정")
    audit.add_argument("--output", type=Path, default=Path("artifacts"))
    audit.set_defaults(func=run_audit)
    portfolio = commands.add_parser("portfolio", help="원본 정산 통화와 비용으로 전체 계약 복원")
    portfolio.add_argument("--audit-run", type=Path, required=True)
    portfolio.add_argument("--output", type=Path, default=Path("artifacts"))
    portfolio.set_defaults(func=lambda a: reconstruct_portfolio(a.audit_run, a.output))
    verify = commands.add_parser("verify", help="공식 공개 체결로 원본 표본을 외부 대조")
    verify.add_argument("--audit-run", type=Path, required=True)
    verify.add_argument("--output", type=Path, default=Path("artifacts"))
    verify.add_argument("--cache", type=Path, default=Path("data/bitmex-verification"))
    verify.set_defaults(func=lambda a: verify_samples(a.audit_run, a.output, a.cache))
    history = commands.add_parser("bitmex-history", help="원거래소 공식 1분·5분봉을 특징 연구용으로 수집")
    history.add_argument("--start", default="2018-03-01")
    history.add_argument("--end", default="2022-01-01")
    history.add_argument("--output", type=Path, default=Path("data/bitmex-history"))
    history.add_argument("--interval", choices=['1m', '5m'], default='5m')
    history.set_defaults(func=lambda a: fetch_bitmex_history(a.output, a.start, a.end, a.interval))
    market = commands.add_parser("market", help="공식 시세/펀딩 자료를 체크섬 검증 후 저장")
    market.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    market.add_argument("--start", default="2019-09")
    market.add_argument("--end", default="2025-12")
    market.add_argument("--interval", default="15m", choices=["1m", "5m", "15m", "1h"])
    market.add_argument("--output", type=Path, default=Path("data/market"))
    market.set_defaults(func=lambda a: fetch_market(a.output, a.symbols, a.start, a.end, a.interval))
    repair = commands.add_parser("market-repair", help="월별 자료의 내부 결측을 공식 일별 자료로 보완")
    repair.add_argument("--market", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--interval", default="15m", choices=["1m", "5m", "15m", "1h"])
    repair.add_argument("--start", help="선택 보완 범위 UTC 시작, 기존 자료 포함")
    repair.add_argument("--end", help="선택 보완 범위 UTC 종료, 미포함")
    repair.set_defaults(func=lambda a: repair_gaps(a.market, a.output, a.interval, a.start, a.end))
    minute_repair = commands.add_parser("minute-repair", help="공식 원체결과 5분 대조로 불일치 분봉을 별도 복원")
    minute_repair.add_argument("--market", type=Path, required=True)
    minute_repair.add_argument("--feature-market", type=Path, required=True)
    minute_repair.add_argument("--output", type=Path, required=True)
    minute_repair.add_argument("--cache", type=Path, default=Path("data/minute-repair-trades"))
    minute_repair.set_defaults(func=lambda a: repair_minute_market(a.market, a.feature_market, a.output, a.cache))
    paired = commands.add_parser("paired-repair", help="두 형식의 원체결 대조 후 1분·5분 자료를 함께 복원")
    paired.add_argument("--market", type=Path, required=True)
    paired.add_argument("--feature-market", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--feature-output", type=Path, required=True)
    paired.add_argument("--cache", type=Path, default=Path("data/minute-repair-trades"))
    paired.add_argument("--start", help="새 정규화 자료의 UTC 시작 날짜, 포함")
    paired.add_argument("--end", help="새 정규화 자료의 UTC 종료 날짜, 미포함")
    paired.add_argument("--extra-targets", type=Path, help="이미 발견한 오류의 심볼별 5분 종료 시각 JSON")
    paired.add_argument("--symbols", nargs='+', help="독립적으로 검증·복원할 입력 심볼")
    paired.add_argument("--verify-unchanged-minutes", action='store_true', help="미연결 체결의 미수정 봉 값을 원체결·기존·일별·월별 경로로 정확 대조")
    paired.set_defaults(func=lambda a: repair_paired_market(a.market, a.feature_market, a.output, a.feature_output, a.cache,
                                                            a.start, a.end, load_repair_targets(a.extra_targets), a.symbols,
                                                            a.verify_unchanged_minutes))
    research = commands.add_parser("research", help="학습·검증·평가를 분리하여 모사 후보를 비교")
    research.add_argument("--audit-run", type=Path, required=True)
    research.add_argument("--market", type=Path, default=Path("data/market"))
    research.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    research.add_argument("--output", type=Path, default=Path("artifacts"))
    research.set_defaults(func=lambda a: run_research(a.audit_run, a.market, a.output, a.symbols))
    study = commands.add_parser("study", help="추가 진입·보유 시간·시장 맥락의 사후 행동 분석")
    study.add_argument("--audit-run", type=Path, required=True)
    study.add_argument("--market", type=Path, default=Path("data/market-complete"))
    study.add_argument("--output", type=Path, default=Path("artifacts"))
    study.set_defaults(func=lambda a: run_study(a.audit_run, a.market, a.output))
    events = commands.add_parser("event-study", help="독립 주문의 진입·관리 학습 자료와 모델 생성")
    events.add_argument("--audit-run", type=Path, required=True)
    events.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    events.add_argument("--output", type=Path, default=Path("artifacts"))
    events.set_defaults(func=lambda a: run_event_study(a.audit_run, a.history, a.output))
    selection = commands.add_parser("event-select", help="2020년에서 사건별 정책 선택 후 설정 고정")
    selection.add_argument("--study-run", type=Path, required=True)
    selection.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    selection.add_argument("--output", type=Path, default=Path("artifacts"))
    selection.set_defaults(func=lambda a: run_event_selection(a.study_run, a.market, a.output))
    evaluation = commands.add_parser("event-evaluate", help="고정 사건별 후보의 다년·다시장·위험 제거 평가")
    evaluation.add_argument("--selection-run", type=Path, required=True)
    evaluation.add_argument("--market", type=Path, required=True)
    evaluation.add_argument("--period", choices=["observed", "new"], required=True)
    evaluation.add_argument("--output", type=Path, default=Path("artifacts"))
    evaluation.set_defaults(func=lambda a: run_event_evaluation(a.selection_run, a.market, a.output, a.period))
    diagnostics = commands.add_parser("event-diagnose", help="완료한 사건별 평가의 매매 빈도·가격 손익·비용 분해")
    diagnostics.add_argument("--study-run", type=Path, required=True)
    diagnostics.add_argument("--evaluation-runs", type=Path, nargs="+", required=True)
    diagnostics.add_argument("--output", type=Path, default=Path("artifacts"))
    diagnostics.set_defaults(func=lambda a: run_event_diagnostics(a.study_run, a.evaluation_runs, a.output))
    walkforward = commands.add_parser("event-walkforward", help="사건별 분류기의 확장 학습 모사 검증")
    walkforward.add_argument("--study-run", type=Path, required=True)
    walkforward.add_argument("--audit-run", type=Path, required=True)
    walkforward.add_argument("--output", type=Path, default=Path("artifacts"))
    walkforward.set_defaults(func=lambda a: run_event_walkforward(a.study_run, a.audit_run, a.output))
    frequency = commands.add_parser("frequency-select", help="학습 행동 빈도를 반영한 12개 후보 선택")
    frequency.add_argument("--study-run", type=Path, required=True)
    frequency.add_argument("--audit-run", type=Path, required=True)
    frequency.add_argument("--v2-selection-run", type=Path, required=True)
    frequency.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    frequency.add_argument("--output", type=Path, default=Path("artifacts"))
    frequency.set_defaults(func=lambda a: run_frequency_selection(a.study_run, a.audit_run, a.v2_selection_run, a.market, a.output))
    frequency_evaluation = commands.add_parser("frequency-evaluate", help="빈도 반영 고정 후보의 관찰 기간·조건부 새 기간 평가")
    frequency_evaluation.add_argument("--selection-run", type=Path, required=True)
    frequency_evaluation.add_argument("--market", type=Path, required=True)
    frequency_evaluation.add_argument("--period", choices=["observed", "seen_2026", "new"], required=True)
    frequency_evaluation.add_argument("--output", type=Path, default=Path("artifacts"))
    frequency_evaluation.set_defaults(func=lambda a: run_frequency_evaluation(a.selection_run, a.market, a.output, a.period))
    stress = commands.add_parser("engine-stress", help="실제 급변 시세와 프로세스 종료·중복·누락·중지 검증")
    stress.add_argument("--selection-run", type=Path, required=True)
    stress.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    stress.add_argument("--feature-market", type=Path, help="v7 1분 실행의 별도 5분 특징 자료")
    stress.add_argument("--start", default="2020-03-10")
    stress.add_argument("--end", default="2020-03-14")
    stress.add_argument("--output", type=Path, default=Path("artifacts"))
    stress.set_defaults(func=lambda a: run_engine_stress(a.selection_run, a.market, a.output, a.start, a.end, a.feature_market))
    context = commands.add_parser("context-study", help="원거래소 전체 기간의 주문 맥락과 체결 비용 분석")
    context.add_argument("--audit-run", type=Path, required=True)
    context.add_argument("--study-run", type=Path, required=True)
    context.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    context.add_argument("--output", type=Path, default=Path("artifacts"))
    context.set_defaults(func=lambda a: run_context_study(a.audit_run, a.history, a.study_run, a.output))
    execution = commands.add_parser("execution-study", help="최초 독립 체결의 이후 가격·비용·유동성 분석")
    execution.add_argument("--audit-run", type=Path, required=True)
    execution.add_argument("--study-run", type=Path, required=True)
    execution.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    execution.add_argument("--output", type=Path, default=Path("artifacts"))
    execution.set_defaults(func=lambda a: run_execution_study(a.audit_run, a.study_run, a.history, a.output))
    expansion = commands.add_parser("expansion-select", help="노출 확대의 시점·방향 분리 후보 선택")
    expansion.add_argument("--audit-run", type=Path, required=True)
    expansion.add_argument("--study-run", type=Path, required=True)
    expansion.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    expansion.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    expansion.add_argument("--output", type=Path, default=Path("artifacts"))
    expansion.set_defaults(func=lambda a: run_expansion_selection(a.audit_run, a.study_run, a.history, a.market, a.output))
    expansion_evaluation = commands.add_parser("expansion-evaluate", help="노출 확대 후보의 다년·비용·지연 비교")
    expansion_evaluation.add_argument("--selection-run", type=Path, required=True)
    expansion_evaluation.add_argument("--v3-selection-run", type=Path, required=True)
    expansion_evaluation.add_argument("--market", type=Path, required=True)
    expansion_evaluation.add_argument("--period", choices=["observed", "seen_2026", "new"], required=True)
    expansion_evaluation.add_argument("--output", type=Path, default=Path("artifacts"))
    expansion_evaluation.set_defaults(func=lambda a: run_expansion_evaluation(a.selection_run, a.v3_selection_run, a.market, a.output, a.period))
    edge = commands.add_parser("edge-select", help="보유 시간·비용 예측 후보 선택")
    edge.add_argument("--audit-run", type=Path, required=True)
    edge.add_argument("--study-run", type=Path, required=True)
    edge.add_argument("--v4-selection-run", type=Path, required=True)
    edge.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    edge.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    edge.add_argument("--output", type=Path, default=Path("artifacts"))
    edge.set_defaults(func=lambda a: run_edge_selection(a.audit_run, a.study_run, a.history, a.v4_selection_run, a.market, a.output))
    edge_evaluation = commands.add_parser("edge-evaluate", help="보유 시간·비용 예측 후보의 고정 비교")
    edge_evaluation.add_argument("--selection-run", type=Path, required=True)
    edge_evaluation.add_argument("--v4-selection-run", type=Path, required=True)
    edge_evaluation.add_argument("--market", type=Path, required=True)
    edge_evaluation.add_argument("--period", choices=["observed", "seen_2026", "new"], required=True)
    edge_evaluation.add_argument("--output", type=Path, default=Path("artifacts"))
    edge_evaluation.set_defaults(func=lambda a: run_expansion_evaluation(a.selection_run, a.v4_selection_run, a.market, a.output, a.period))
    edge_diagnostics = commands.add_parser("edge-diagnose", help="비용 예측 신호의 단계별 표본 변화 분석")
    edge_diagnostics.add_argument("--selection-run", type=Path, required=True)
    edge_diagnostics.add_argument("--audit-run", type=Path, required=True)
    edge_diagnostics.add_argument("--study-run", type=Path, required=True)
    edge_diagnostics.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    edge_diagnostics.add_argument("--market", type=Path, default=Path("data/market-5m-complete"))
    edge_diagnostics.add_argument("--recent-market", type=Path, required=True)
    edge_diagnostics.add_argument("--new-market", type=Path, required=True)
    edge_diagnostics.add_argument("--output", type=Path, default=Path("artifacts"))
    edge_diagnostics.set_defaults(func=lambda a: run_edge_diagnostics(a.selection_run, a.audit_run, a.study_run, a.history,
                                                                     a.market, a.recent_market, a.new_market, a.output))
    structure = commands.add_parser("structure-study", help="원본 회계·보유 제한·추가 진입·체결 조건의 설명 대조")
    structure.add_argument("--audit-run", type=Path, required=True)
    structure.add_argument("--study-run", type=Path, required=True)
    structure.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    structure.add_argument("--minute-history", type=Path, default=Path("data/bitmex-history-1m"))
    structure.add_argument("--output", type=Path, default=Path("artifacts"))
    structure.set_defaults(func=lambda a: run_structure_study(a.audit_run, a.study_run, a.history, a.minute_history, a.output))
    timing = commands.add_parser("timing-study", help="동일 주문의 1분·5분 시세와 체결 가격 차이 비교")
    timing.add_argument("--audit-run", type=Path, required=True)
    timing.add_argument("--study-run", type=Path, required=True)
    timing.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    timing.add_argument("--minute-history", type=Path, default=Path("data/bitmex-history-1m"))
    timing.add_argument("--output", type=Path, default=Path("artifacts"))
    timing.set_defaults(func=lambda a: run_timing_study(a.audit_run, a.study_run, a.history, a.minute_history, a.output))
    pullback = commands.add_parser("pullback-select", help="확정 1분 종가의 진입 대기 후보 여섯 개 선택")
    pullback.add_argument("--v4-selection-run", type=Path, required=True)
    pullback.add_argument("--market", type=Path, required=True)
    pullback.add_argument("--feature-market", type=Path, required=True)
    pullback.add_argument("--output", type=Path, default=Path("artifacts"))
    pullback.set_defaults(func=lambda a: run_pullback_selection(a.v4_selection_run, a.market, a.feature_market, a.output))
    net_select = commands.add_parser("net-edge-select", help="실행 순손익 필터의 학습·시간순 후보 선택")
    net_select.add_argument("--v7-selection-run", type=Path, required=True)
    net_select.add_argument("--market", type=Path, required=True)
    net_select.add_argument("--feature-market", type=Path, required=True)
    net_select.add_argument("--confirmation-market", type=Path, required=True)
    net_select.add_argument("--confirmation-features", type=Path, required=True)
    net_select.add_argument("--output", type=Path, default=Path("artifacts"))
    net_select.set_defaults(func=lambda a: run_net_selection(a.v7_selection_run, a.market, a.feature_market,
                                                           a.confirmation_market, a.confirmation_features, a.output))
    lifecycle = commands.add_parser("lifecycle-select", help="원본 관리 행동의 시간순 학습·후보 선택")
    lifecycle.add_argument("--v7-selection-run", type=Path, required=True)
    lifecycle.add_argument("--audit-run", type=Path, required=True)
    lifecycle.add_argument("--study-run", type=Path, required=True)
    lifecycle.add_argument("--history", type=Path, default=Path("data/bitmex-history"))
    lifecycle.add_argument("--market", type=Path, required=True)
    lifecycle.add_argument("--feature-market", type=Path, required=True)
    lifecycle.add_argument("--confirmation-market", type=Path, required=True)
    lifecycle.add_argument("--confirmation-features", type=Path, required=True)
    lifecycle.add_argument("--output", type=Path, default=Path("artifacts"))
    lifecycle.set_defaults(func=lambda a: run_lifecycle_selection(a.v7_selection_run, a.audit_run, a.study_run,
                           a.history, a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output))
    from .management_labels import run_management_labels, run_path_labels
    labels = commands.add_parser("action-labels", help="원본 분별 관리 정답과 연결 원장 생성")
    for name in ['audit-run', 'study-run', 'history', 'minute-history']:
        labels.add_argument(f'--{name}', type=Path, required=True)
    labels.add_argument('--output', type=Path, default=Path('artifacts'))
    labels.set_defaults(func=lambda a: print(run_management_labels(a.audit_run, a.study_run, a.history, a.minute_history, a.output)))
    path_labels = commands.add_parser('path-labels', help='관리 정답에 과거 보유 가격 경로 추가')
    path_labels.add_argument('--labels-run', type=Path, required=True)
    path_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    path_labels.set_defaults(func=lambda a: print(run_path_labels(a.labels_run, a.output)))
    from .inventory_study import run_inventory_study
    inventory = commands.add_parser('inventory-study', help='원본 잔여 수량·축소 크기·실현 손익의 연결 진단')
    inventory.add_argument('--audit-run', type=Path, required=True)
    inventory.add_argument('--labels-run', type=Path, required=True)
    inventory.add_argument('--bot-runs', nargs='*', type=Path, default=[])
    inventory.add_argument('--output', type=Path, default=Path('artifacts'))
    inventory.set_defaults(func=lambda a: print(run_inventory_study(a.audit_run, a.labels_run, a.output, a.bot_runs)))
    from .inventory_labels import run_inventory_labels
    inventory_labels = commands.add_parser('inventory-labels', help='잔여 수량 특징과 실제 체결 축소 크기 정답 생성')
    inventory_labels.add_argument('--path-labels-run', type=Path, required=True)
    inventory_labels.add_argument('--inventory-study-run', type=Path, required=True)
    inventory_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    inventory_labels.set_defaults(func=lambda a: print(run_inventory_labels(a.path_labels_run, a.inventory_study_run, a.output)))
    from .minute_inventory import run_minute_inventory_labels
    minute_labels = commands.add_parser('minute-inventory-labels', help='수량 관리 정답에 현재 확정 분봉 입력 추가')
    for name in ['labels-run', 'minute-history', 'history']:
        minute_labels.add_argument(f'--{name}', type=Path, required=True)
    minute_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    minute_labels.set_defaults(func=lambda a: print(run_minute_inventory_labels(a.labels_run, a.minute_history, a.history, a.output)))
    from .addition_research import run_addition_labels
    addition_labels = commands.add_parser('addition-effect-labels', help='같은 보유 상태에서 추가 유지·취소의 순효과 정답 생성')
    for name in ['selection-run', 'market', 'feature-market']:
        addition_labels.add_argument(f'--{name}', type=Path, required=True)
    addition_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    addition_labels.set_defaults(func=lambda a: print(run_addition_labels(a.selection_run, a.market, a.feature_market, a.output)))
    from .addition_research import run_addition_selection
    addition_select = commands.add_parser('addition-effect-select', help='추가 순효과 모델의 고정 학습과 이후 확인')
    for name in ['selection-run', 'labels-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        addition_select.add_argument(f'--{name}', type=Path, required=True)
    addition_select.add_argument('--output', type=Path, default=Path('artifacts'))
    addition_select.set_defaults(func=lambda a: print(run_addition_selection(a.selection_run, a.labels_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .realized_exit import run_realized_exit_labels
    exit_labels = commands.add_parser('realized-exit-labels', help='실제 보유 종료로 청산 정답만 수정')
    for name in ['labels-run', 'audit-run']:
        exit_labels.add_argument(f'--{name}', type=Path, required=True)
    exit_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    exit_labels.set_defaults(func=lambda a: print(run_realized_exit_labels(a.labels_run, a.audit_run, a.output)))
    from .realized_exit_research import run_realized_exit_selection
    exit_select = commands.add_parser('realized-exit-select', help='실제 종료 청산 모델의 고정 학습과 확인')
    for name in ['selection-run', 'labels-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        exit_select.add_argument(f'--{name}', type=Path, required=True)
    exit_select.add_argument('--output', type=Path, default=Path('artifacts'))
    exit_select.set_defaults(func=lambda a: print(run_realized_exit_selection(a.selection_run, a.labels_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .position_prior_research import run_position_prior_selection
    prior_select = commands.add_parser('position-prior-select', help='신규 포지션 방향의 원본 빈도 차이만 보정해 고정 비교')
    for name in ['selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        prior_select.add_argument(f'--{name}', type=Path, required=True)
    prior_select.add_argument('--output', type=Path, default=Path('artifacts'))
    prior_select.set_defaults(func=lambda a: print(run_position_prior_selection(a.selection_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .history_calibration_diagnostics import run_history_calibration_diagnostics
    history_calibration = commands.add_parser('history-calibration-diagnose', help='과거 주문 입력과 세 구간 빈도 보정의 결합 진단')
    history_calibration.add_argument('--history-run', type=Path, required=True)
    history_calibration.add_argument('--calibration-run', type=Path, required=True)
    history_calibration.add_argument('--output', type=Path, default=Path('artifacts'))
    history_calibration.set_defaults(func=lambda a: print(run_history_calibration_diagnostics(a.history_run, a.calibration_run, a.output)))
    from .order_history_boost import run_order_history_boost_diagnostics
    history_boost = commands.add_parser('order-history-boost-diagnose', help='고정 과거 주문 입력의 선형·비선형 시간순 대조')
    history_boost.add_argument('--history-run', type=Path, required=True)
    history_boost.add_argument('--output', type=Path, default=Path('artifacts'))
    history_boost.set_defaults(func=lambda a: print(run_order_history_boost_diagnostics(a.history_run, a.output)))
    from .order_history_diagnostics import run_order_history_diagnostics
    history_diagnosis = commands.add_parser('order-history-diagnose', help='과거 독립 증가·축소 주문 맥락의 예측 진단')
    for name in ['selection-run', 'labels-run', 'audit-run']:
        history_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    history_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    history_diagnosis.set_defaults(func=lambda a: print(run_order_history_diagnostics(a.selection_run, a.labels_run, a.audit_run, a.output)))
    from .calibration_diagnostics import run_calibration_diagnostics
    calibration_diagnosis = commands.add_parser('management-calibration-diagnose', help='학습·빈도 보정·진단을 세 구간으로 분리해 관리 점수 대조')
    for name in ['selection-run', 'labels-run', 'diagnosis-run']:
        calibration_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    calibration_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    calibration_diagnosis.set_defaults(func=lambda a: print(run_calibration_diagnostics(a.selection_run, a.labels_run, a.diagnosis_run, a.output)))
    from .management_diagnostics import run_management_diagnostics
    management_diagnosis = commands.add_parser('management-model-diagnose', help='원본 관리 행동의 시간순 선형·부스팅 예측 대조')
    management_diagnosis.add_argument('--selection-run', type=Path, required=True)
    management_diagnosis.add_argument('--labels-run', type=Path, required=True)
    management_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    management_diagnosis.set_defaults(func=lambda a: print(run_management_diagnostics(a.selection_run, a.labels_run, a.output)))
    from .direction_diagnostics import run_direction_diagnostics
    direction_diagnosis = commands.add_parser('direction-model-diagnose', help='원본 신규 방향의 시간순 선형·부스팅 예측 대조')
    for name in ['audit-run', 'study-run', 'history']:
        direction_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    direction_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    direction_diagnosis.set_defaults(func=lambda a: print(run_direction_diagnostics(a.audit_run, a.study_run, a.history, a.output)))
    from .history_window import run_history_window_diagnosis
    history_window = commands.add_parser('history-window-diagnose', help='과거 두 연도 추가의 관리 예측 진단')
    history_window.add_argument('--diagnosis-run', type=Path, required=True)
    history_window.add_argument('--output', type=Path, default=Path('artifacts'))
    history_window.set_defaults(func=lambda a: print(run_history_window_diagnosis(a.diagnosis_run, a.output)))
    from .exit_direct import run_exit_direct_diagnosis
    exit_direct = commands.add_parser('exit-direct-diagnose', help='청산 문턱의 가산 제거 진단')
    for name in ['selection-run', 'diagnosis-run']:
        exit_direct.add_argument(f'--{name}', type=Path, required=True)
    exit_direct.add_argument('--output', type=Path, default=Path('artifacts'))
    exit_direct.set_defaults(func=lambda a: print(run_exit_direct_diagnosis(a.selection_run, a.diagnosis_run, a.output)))
    from .first_management_direct import run_first_management_direct_diagnosis
    first_direct = commands.add_parser('first-management-direct-diagnose', help='첫 관리 문턱의 가산 제거 진단')
    first_direct.add_argument('--diagnosis-run', type=Path, required=True)
    first_direct.add_argument('--output', type=Path, default=Path('artifacts'))
    first_direct.set_defaults(func=lambda a: print(run_first_management_direct_diagnosis(a.diagnosis_run, a.output)))
    from .first_management import run_first_management_diagnosis
    first_diagnosis = commands.add_parser('first-management-diagnose', help='첫 관리 주문의 문턱 분리 진단')
    for name in ['selection-run', 'diagnosis-run']:
        first_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    first_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    first_diagnosis.set_defaults(func=lambda a: print(run_first_management_diagnosis(a.selection_run, a.diagnosis_run, a.output)))
    from .exit_state_research import run_exit_state_selection
    exit_state = commands.add_parser('exit-state-select', help='청산 문턱 직접 적용의 고정 매매 대조')
    for name in ['selection-run', 'diagnosis-run', 'market', 'features', 'confirmation-market', 'confirmation-features']:
        exit_state.add_argument(f'--{name}', type=Path, required=True)
    exit_state.add_argument('--output', type=Path, default=Path('artifacts'))
    exit_state.set_defaults(func=lambda a: print(run_exit_state_selection(a.selection_run, a.diagnosis_run,
        a.market, a.features, a.confirmation_market, a.confirmation_features, a.output)))
    from .activity_ablation_research import run_activity_ablation_selection
    activity_ablation = commands.add_parser('activity-ablation-select', help='신규 활동 관문 제거의 고정 비교')
    for name in ['selection-run', 'market', 'features', 'confirmation-market', 'confirmation-features']:
        activity_ablation.add_argument(f'--{name}', type=Path, required=True)
    activity_ablation.add_argument('--output', type=Path, default=Path('artifacts'))
    activity_ablation.set_defaults(func=lambda a: print(run_activity_ablation_selection(a.selection_run,
        a.market, a.features, a.confirmation_market, a.confirmation_features, a.output)))
    from .context_position import run_context_diagnostics
    context_diagnosis = commands.add_parser('context-position-diagnose', help='보유 방향의 다일 시장 맥락 진단')
    context_diagnosis.add_argument('--diagnosis-run', type=Path, required=True)
    context_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    context_diagnosis.set_defaults(func=lambda a: print(run_context_diagnostics(a.diagnosis_run, a.output)))
    from .position_target import run_position_diagnostics
    position_diagnosis = commands.add_parser('position-target-diagnose', help='다음 경계의 보유 방향 재구성 진단')
    position_diagnosis.add_argument('--activity-run', type=Path, required=True)
    position_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    position_diagnosis.set_defaults(func=lambda a: print(run_position_diagnostics(a.activity_run, a.output)))
    from .joint_research import run_joint_diagnostics
    joint_diagnosis = commands.add_parser('joint-entry-diagnose', help='신규 진입 활동·방향의 결합 예측 진단')
    for name in ['activity-run', 'direction-run']:
        joint_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    joint_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    joint_diagnosis.set_defaults(func=lambda a: print(run_joint_diagnostics(a.activity_run, a.direction_run, a.output)))
    from .activity_diagnostics import run_activity_diagnostics
    activity_diagnosis = commands.add_parser('activity-model-diagnose', help='신규 진입 활동의 시간순 모형 비교')
    for name in ['audit-run', 'study-run', 'history']:
        activity_diagnosis.add_argument(f'--{name}', type=Path, required=True)
    activity_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    activity_diagnosis.set_defaults(func=lambda a: print(run_activity_diagnostics(a.audit_run, a.study_run, a.history, a.output)))
    from .episode_balance import run_episode_balance_diagnosis
    episode_diagnosis = commands.add_parser('episode-balance-diagnose', help='포지션별 관리 학습 기여 균등화 진단')
    episode_diagnosis.add_argument('--diagnosis-run', type=Path, required=True)
    episode_diagnosis.add_argument('--output', type=Path, default=Path('artifacts'))
    episode_diagnosis.set_defaults(func=lambda a: print(run_episode_balance_diagnosis(a.diagnosis_run, a.output)))
    from .holding_support_research import run_holding_support_selection
    holding_select = commands.add_parser('holding-support-select', help='관찰한 보유 시간 한도의 고정 비교')
    for name in ['selection-run', 'diagnosis-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        holding_select.add_argument(f'--{name}', type=Path, required=True)
    holding_select.add_argument('--output', type=Path, default=Path('artifacts'))
    holding_select.set_defaults(func=lambda a: print(run_holding_support_selection(a.selection_run, a.diagnosis_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .first_state_research import run_first_state_selection
    first_select = commands.add_parser('first-state-select', help='첫 관리 문턱의 자체 체결 적용 비교')
    for name in ['selection-run', 'diagnosis-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        first_select.add_argument(f'--{name}', type=Path, required=True)
    first_select.add_argument('--output', type=Path, default=Path('artifacts'))
    first_select.set_defaults(func=lambda a: print(run_first_state_selection(a.selection_run, a.diagnosis_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .history_state_research import run_history_state_selection
    history_select = commands.add_parser('history-state-select', help='보정된 주문 이력 모델의 자체 체결 관리 비교')
    for name in ['selection-run', 'diagnosis-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        history_select.add_argument(f'--{name}', type=Path, required=True)
    history_select.add_argument('--output', type=Path, default=Path('artifacts'))
    history_select.set_defaults(func=lambda a: print(run_history_state_selection(a.selection_run, a.diagnosis_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .current_state_research import run_current_state_selection
    current_select = commands.add_parser('current-state-select', help='누적량 대신 현재 점수와 기존 문턱으로 관리 판단 비교')
    for name in ['selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        current_select.add_argument(f'--{name}', type=Path, required=True)
    current_select.add_argument('--output', type=Path, default=Path('artifacts'))
    current_select.set_defaults(func=lambda a: print(run_current_state_selection(a.selection_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .probe_entry_research import run_probe_entry_selection
    probe_select = commands.add_parser('probe-entry-select', help='최초 노출만 줄인 단일 후보와 비례 위험 축소 대조')
    for name in ['selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        probe_select.add_argument(f'--{name}', type=Path, required=True)
    probe_select.add_argument('--output', type=Path, default=Path('artifacts'))
    probe_select.set_defaults(func=lambda a: print(run_probe_entry_selection(a.selection_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .boosted_direction_research import run_boosted_direction_selection
    boosted_select = commands.add_parser('boosted-direction-select', help='시간순 진단을 통과한 고정 방향 부스팅 학습')
    for name in ['selection-run', 'diagnosis-run', 'audit-run', 'study-run', 'history', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        boosted_select.add_argument(f'--{name}', type=Path, required=True)
    boosted_select.add_argument('--output', type=Path, default=Path('artifacts'))
    boosted_select.set_defaults(func=lambda a: print(run_boosted_direction_selection(a.selection_run, a.diagnosis_run, a.audit_run,
        a.study_run, a.history, a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .position_direction_research import run_position_direction_selection
    direction_select = commands.add_parser('position-direction-select', help='전체 원본의 실제 신규 포지션 방향만 직접 학습')
    for name in ['selection-run', 'audit-run', 'study-run', 'history', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        direction_select.add_argument(f'--{name}', type=Path, required=True)
    direction_select.add_argument('--output', type=Path, default=Path('artifacts'))
    direction_select.set_defaults(func=lambda a: print(run_position_direction_selection(a.selection_run, a.audit_run,
        a.study_run, a.history, a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .new_position_research import run_new_position_selection
    new_position = commands.add_parser('new-position-select', help='새 포지션 시작으로 진입 활동만 다시 학습해 고정 비교')
    for name in ['selection-run', 'audit-run', 'study-run', 'history', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        new_position.add_argument(f'--{name}', type=Path, required=True)
    new_position.add_argument('--output', type=Path, default=Path('artifacts'))
    new_position.set_defaults(func=lambda a: print(run_new_position_selection(a.selection_run, a.audit_run,
        a.study_run, a.history, a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .recent_entry import run_recent_entry_selection
    recent_entry = commands.add_parser('recent-entry-select', help='최근 두 해 원본으로 신규 진입만 다시 학습해 고정 비교')
    for name in ['selection-run', 'audit-run', 'study-run', 'history', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        recent_entry.add_argument(f'--{name}', type=Path, required=True)
    recent_entry.add_argument('--output', type=Path, default=Path('artifacts'))
    recent_entry.set_defaults(func=lambda a: print(run_recent_entry_selection(a.selection_run, a.audit_run,
        a.study_run, a.history, a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .minute_inventory_research import run_minute_inventory_selection
    minute_select = commands.add_parser('minute-inventory-select', help='확정 분봉 입력을 추가한 관리 모델의 학습과 확인')
    for name in ['inventory-selection-run', 'labels-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        minute_select.add_argument(f'--{name}', type=Path, required=True)
    minute_select.add_argument('--output', type=Path, default=Path('artifacts'))
    minute_select.set_defaults(func=lambda a: print(run_minute_inventory_selection(a.inventory_selection_run, a.labels_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output)))
    from .inventory_research import run_inventory_selection
    inventory_select = commands.add_parser('inventory-select', help='잔여 수량·실제 축소 크기의 고정 학습과 확인')
    for name in ['rate-selection-run', 'labels-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        inventory_select.add_argument(f'--{name}', type=Path, required=True)
    inventory_select.add_argument('--latest-source', action='store_true', help='v20의 2020~2021년 상반기 학습·하반기 보정')
    inventory_select.add_argument('--output', type=Path, default=Path('artifacts'))
    inventory_select.set_defaults(func=lambda a: print(run_inventory_selection(a.rate_selection_run, a.labels_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output, latest_source=a.latest_source)))
    action = commands.add_parser("action-select", help="분별 관리 사건과 행동별 정책 선택")
    action.add_argument("--v9-selection-run", type=Path, required=True)
    action.add_argument("--labels-run", type=Path, required=True)
    action.add_argument("--regime", choices=["original", "recent", "path"], default="original")
    action.add_argument("--market", type=Path, required=True)
    action.add_argument("--feature-market", type=Path, required=True)
    action.add_argument("--confirmation-market", type=Path, required=True)
    action.add_argument("--confirmation-features", type=Path, required=True)
    action.add_argument("--output", type=Path, default=Path("artifacts"))
    action.set_defaults(func=lambda a: run_action_selection(a.v9_selection_run, a.labels_run, a.market,
        a.feature_market, a.confirmation_market, a.confirmation_features, a.output, a.regime))
    from .reversal_research import run_reversal_selection
    reverse = commands.add_parser('reversal-select', help='고정 v12 모델 청산의 반전 실행 대조')
    for name in ['path-selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        reverse.add_argument(f'--{name}', type=Path, required=True)
    reverse.add_argument('--output', type=Path, default=Path('artifacts'))
    reverse.set_defaults(func=lambda a: run_reversal_selection(a.path_selection_run, a.market, a.feature_market,
        a.confirmation_market, a.confirmation_features, a.output))
    rate_reverse = commands.add_parser('rate-reversal-select', help='고정 v14 누적 관리 청산의 반전 실행 대조')
    for name in ['rate-selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        rate_reverse.add_argument(f'--{name}', type=Path, required=True)
    rate_reverse.add_argument('--output', type=Path, default=Path('artifacts'))
    rate_reverse.set_defaults(func=lambda a: run_reversal_selection(a.rate_selection_run, a.market, a.feature_market,
        a.confirmation_market, a.confirmation_features, a.output, rate_based=True))
    from .rate_research import run_rate_selection
    rate = commands.add_parser('rate-select', help='고정 v12 모델의 사건 빈도 누적 대조')
    for name in ['path-selection-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        rate.add_argument(f'--{name}', type=Path, required=True)
    rate.add_argument('--output', type=Path, default=Path('artifacts'))
    rate.set_defaults(func=lambda a: run_rate_selection(a.path_selection_run, a.market, a.feature_market,
        a.confirmation_market, a.confirmation_features, a.output))
    from .lifecycle_edge_research import run_lifecycle_edge_labels, run_lifecycle_edge_selection
    lifecycle_labels = commands.add_parser('lifecycle-edge-labels', help='고정 v14 관리의 전체 거래 순손익 정답 생성')
    for name in ['rate-selection-run', 'market', 'feature-market']:
        lifecycle_labels.add_argument(f'--{name}', type=Path, required=True)
    lifecycle_labels.add_argument('--direction-only', action='store_true', help='v17의 활동 관문을 제거한 진입 기회')
    lifecycle_labels.add_argument('--output', type=Path, default=Path('artifacts'))
    lifecycle_labels.set_defaults(func=lambda a: print(run_lifecycle_edge_labels(a.rate_selection_run,
        a.market, a.feature_market, a.output, direction_only=a.direction_only)))
    lifecycle_select = commands.add_parser('lifecycle-edge-select', help='전체 거래 정답의 고정 진입 필터 학습·확인')
    for name in ['rate-selection-run', 'labels-run', 'market', 'feature-market', 'confirmation-market', 'confirmation-features']:
        lifecycle_select.add_argument(f'--{name}', type=Path, required=True)
    lifecycle_select.add_argument('--direction-only', action='store_true', help='v17의 방향 기반 기회와 중첩 가중치 학습')
    lifecycle_select.add_argument('--overlap-weighted', action='store_true', help='v16의 고정 중첩 가중치 적용')
    lifecycle_select.add_argument('--output', type=Path, default=Path('artifacts'))
    lifecycle_select.set_defaults(func=lambda a: print(run_lifecycle_edge_selection(a.rate_selection_run, a.labels_run,
        a.market, a.feature_market, a.confirmation_market, a.confirmation_features, a.output, overlap_weighted=a.overlap_weighted, direction_only=a.direction_only)))
    pullback_eval = commands.add_parser("pullback-evaluate", aliases=["net-edge-evaluate", "lifecycle-evaluate", "action-evaluate"], help="고정 v7~v21 후보의 다시장·비용·지연 비교")
    pullback_eval.add_argument("--selection-run", type=Path, required=True)
    pullback_eval.add_argument("--market", type=Path, required=True)
    pullback_eval.add_argument("--feature-market", type=Path, required=True)
    pullback_eval.add_argument("--period", choices=['observed', 'seen_2026', 'verified_2022_2024'], required=True)
    pullback_eval.add_argument("--symbols", nargs='+', choices=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    pullback_eval.add_argument("--output", type=Path, default=Path("artifacts"))
    pullback_eval.add_argument("--diagnostic-only", action="store_true", help="v12~v21 확인 실패 후 사전 계획한 BTC 고정·연간 비교")
    pullback_eval.set_defaults(func=lambda a: run_pullback_evaluation(a.selection_run, a.market, a.feature_market, a.output, a.period, a.symbols, a.diagnostic_only))
    journal = commands.add_parser("event-replay", help="영속 저널을 이용한 사건별 봇 재생과 중단 복원")
    journal.add_argument("--selection-run", type=Path, required=True)
    journal.add_argument("--market", type=Path, required=True)
    journal.add_argument("--feature-market", type=Path, help="v7 1분 실행의 별도 5분 특징 자료")
    journal.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT", "SOLUSDT"], default="BTCUSDT")
    journal.add_argument("--start", required=True)
    journal.add_argument("--end", required=True)
    journal.add_argument("--journal", type=Path, required=True)
    journal.add_argument("--max-bars", type=int)
    journal.add_argument("--halt", action="store_true")
    journal.add_argument("--verify-memory", action="store_true")
    journal.add_argument("--output", type=Path, default=Path("artifacts"))
    journal.set_defaults(func=lambda a: run_event_replay(a.selection_run, a.market, a.symbol, a.start, a.end,
                                                       a.journal, a.output, a.max_bars, a.halt, a.verify_memory, a.feature_market))
    robustness = commands.add_parser("robustness", help="고정 후보의 연도별 재시작과 블록 재표집 진단")
    robustness.add_argument("--research-run", type=Path, required=True)
    robustness.add_argument("--market", type=Path, default=Path("data/market-complete"))
    robustness.add_argument("--output", type=Path, default=Path("artifacts"))
    robustness.set_defaults(func=lambda a: run_robustness(a.research_run, a.market, a.output))
    playback = commands.add_parser("replay", help="고정한 모델로 로컬 시세를 오프라인 재생")
    playback.add_argument("--research-run", type=Path, required=True)
    playback.add_argument("--market", type=Path, default=Path("data/market-complete"))
    playback.add_argument("--symbol", default="BTCUSDT")
    playback.add_argument("--start", required=True, help="UTC 시작, 포함")
    playback.add_argument("--end", required=True, help="UTC 종료, 미포함")
    playback.add_argument("--output", type=Path, default=Path("artifacts"))
    playback.set_defaults(func=lambda a: replay(a.research_run, a.market, a.symbol, a.start, a.end, a.output))
    sample = commands.add_parser("demo", help="원본과 네트워크 없이 합성 시세로 동작 확인")
    sample.add_argument("--output", type=Path, default=Path("artifacts"))
    sample.set_defaults(func=lambda a: demo(a.output))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
