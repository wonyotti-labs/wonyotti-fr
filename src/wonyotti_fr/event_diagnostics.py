from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, records, save_json, sha256
from .reports import table


def decompose_run(directory: Path, initial_equity: float) -> dict:
    curve = pd.read_parquet(directory / 'equity.parquet')
    trades = pd.read_parquet(directory / 'trades.parquet')
    fills = pd.read_parquet(directory / 'fills.parquet')
    metrics = json.loads((directory / 'metrics.json').read_text())
    final = json.loads((directory / 'final_state.json').read_text())
    config_path = directory / 'config.json'
    settings = json.loads(config_path.read_text()) if config_path.exists() else {'bar_seconds': 300, 'initial_equity': initial_equity}
    if settings['initial_equity'] != initial_equity or settings['bar_seconds'] not in (60, 300, 900, 3600):
        raise ValueError('손익 분석의 초기 자본·시세 간격이 실행과 다릅니다.')
    if not final['completed'] or final['quantity'] != 0:
        raise ValueError('완료하고 청산한 실행만 손익 분해에 사용합니다.')
    net = float(trades.net_pnl.sum()) if len(trades) else 0.0
    gross = float(trades.gross_realized.sum()) if len(trades) else 0.0
    fees = float(trades.fees.sum()) if len(trades) else 0.0
    funding = float(trades.funding_cost.sum()) if len(trades) else 0.0
    delta = float(curve.equity.iloc[-1] - initial_equity)
    if max(abs(net - delta), abs(gross - fees - funding - net), abs(fees - metrics['fees'])) > max(1e-7, initial_equity * 1e-10):
        raise ValueError('저장된 체결 손익·비용과 최종 잔고가 일치하지 않습니다.')
    halted = curve.halted.astype(bool)
    first_halt = int(np.flatnonzero(halted)[0]) if halted.any() else len(curve) - 1
    active = curve.iloc[:first_halt + 1]
    turnover = float((fills.delta_quantity.abs() * fills.price).sum()) if len(fills) else 0.0
    reasons = {str(k): int(v) for k, v in fills.reason.value_counts().items()} if len(fills) else {}
    return {'net_pnl': net, 'price_pnl_after_slippage_before_fees': gross,
            'fees': fees, 'funding_cost': funding, 'turnover_over_initial_equity': turnover / initial_equity,
            'average_trade_pnl': float(trades.net_pnl.mean()) if len(trades) else None,
            'median_hold_minutes': float(trades.hold_bars.median() * settings['bar_seconds'] / 60) if len(trades) else None,
            'first_halt_time': curve.time.iloc[first_halt] if halted.any() else None,
            'flat_fraction_before_halt': float(active.quantity.eq(0).mean()),
            'trade_additions': int(trades['adds'].sum()) if len(trades) else 0,
            'fills_by_reason': reasons, 'closed_trades': len(trades),
            'fees_over_abs_net_loss': fees / abs(net) if net < 0 else None,
            'top10_loss_fraction': (float(trades.loc[trades.net_pnl < 0, 'net_pnl'].nsmallest(10).sum()
                                           / trades.loc[trades.net_pnl < 0, 'net_pnl'].sum())
                                    if len(trades) and (trades.net_pnl < 0).any() else None)}


def run_event_diagnostics(study: Path, evaluations: list[Path], output: Path) -> Path:
    hashes = json.loads((study / 'files.json').read_text())
    data_path = study / 'training_events.parquet'
    if sha256(data_path) != hashes[data_path.name]:
        raise ValueError('사건별 학습 자료 무결성 오류')
    input_hashes = {'study': sha256(study / 'files.json')}
    for evaluation in evaluations:
        summary = json.loads((evaluation / 'summary.json').read_text())
        if not summary['complete']:
            raise ValueError('완료한 평가만 분석합니다.')
        input_hashes[str(evaluation)] = sha256(evaluation / 'results.json')
    destination = new_run(output, 'event-diagnostics', {'input_sha256': input_hashes,
                                                       'role': 'post_evaluation_diagnosis_without_retuning'})
    data = pd.read_parquet(data_path)
    valid = data[data.usable].copy()
    valid['year'] = valid.end.dt.year
    source_rows = []
    for year, frame in valid.groupby('year'):
        flat = frame[frame.direction.eq(0)]
        source_rows.append({'year': int(year), 'bars': len(frame), 'flat_bars': len(flat),
                            'flat_fraction': len(flat) / len(frame),
                            'flat_entry_targets': int(flat.target.ne('hold').sum()),
                            'action_fraction': float(frame.target.ne('hold').mean()),
                            'multiple_order_windows': int(frame.orders_in_window.gt(1).sum())})
    rows, file_hashes = [], {}
    for evaluation in evaluations:
        frozen = json.loads((evaluation / 'frozen_selection.json').read_text())
        role = json.loads((evaluation / 'evaluation_observed.json').read_text())['role']
        for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            directory = evaluation / symbol / 'fixed_policy'
            values = decompose_run(directory, frozen['risk']['initial_equity'])
            rows.append({'period': role, 'symbol': symbol, **values})
            file_hashes[str(directory)] = {path.name: sha256(path) for path in directory.iterdir() if path.is_file()}
    save_json(destination / 'input_result_hashes.json', file_hashes)
    save_json(destination / 'diagnostics.json', {'teacher_distribution': source_rows, 'pnl_decomposition': rows})
    source = pd.DataFrame(source_rows)
    results = pd.DataFrame(rows)
    source.to_csv(destination / 'teacher_distribution.csv', index=False)
    results.drop(columns='fills_by_reason').to_csv(destination / 'pnl_decomposition.csv', index=False)
    brief = results[['period', 'symbol', 'net_pnl', 'price_pnl_after_slippage_before_fees', 'fees',
                     'closed_trades', 'median_hold_minutes', 'turnover_over_initial_equity']]
    (destination / 'REPORT.md').write_text(
        '# 사건별 후보의 실패 원인 분해\n\n'
        '고정 평가를 끝낸 뒤 수행한 사후 분석이다. 결과에 맞춰 기존 후보나 평가를 수정하지 않았다.\n\n'
        '## 학습 자료의 행동 분포\n\n' + table(source) + '\n\n'
        '5분봉 경계의 무포지션 비율과 진입 정답 수가 연도에 따라 크게 다르다. '
        '원본은 지속 보유·반전이 많고 봇은 손절·축소·비용으로 다른 상태를 방문한다. '
        '이 표는 관측한 분포 차이다. 분포 차이가 손실의 유일한 원인이라고 증명하지 않는다.\n\n'
        '## 가격 손익과 비용\n\n' + table(brief) + '\n\n'
        '가격 손익은 슬리피지가 반영된 체결 가격의 차이며 수수료와 펀딩 차감 전이다. '
        '따라서 이를 무비용 백테스트나 원래 계좌의 수익으로 해석하지 않는다. '
        '매매 횟수·보유 시간·회전율과 청산 사유는 상세 JSON에 남겼다.\n\n'
        '## 확인한 한계와 다음 가설\n\n'
        '- 클래스 균형 가중치로 학습한 점수를 실제 행동 빈도나 수익 확률로 취급할 수 없다. '
        '빈도 보정 또는 별도 거래 여부 판단은 후속 개발 가설이며 아직 개선이 입증되지 않았다.\n'
        '- 새 진입 표본이 적은 연도에는 클래스별 성능 검증이 부족하다. 높은 일부 재현율로 전체 전략 복원을 주장할 수 없다.\n'
        '- 최초 체결과 주문 제출 시각은 다르다. 체결되지 않은 주문·호가·뉴스·재량 판단은 입력에 없다.\n'
        '- 실패를 피하기 위해 과거 손실 거래를 삭제하거나 손실 후에만 알 수 있는 값으로 필터링하지 않는다. '
        '다음 버전은 이미 관찰한 기간을 공개하고 새 계획부터 작성한다.\n', encoding='utf-8')
    save_json(destination / 'summary.json', {'completed': True, 'teacher': records(source), 'decomposition': records(brief)})
    print(f'실패 원인 분석: {destination / "REPORT.md"}', flush=True)
    return destination
