from __future__ import annotations

from pathlib import Path

from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_research import load_selection
from .lifecycle_edge import lifecycle_labels
from .minute_data import prepare_minute_period
from .net_edge_labels import potential_entries


def run_lifecycle_edge_labels(reference: Path, market: Path, features: Path, output: Path) -> Path:
    frozen, policy = load_selection(reference)
    if frozen['protocol'] != 'minute_rate_v14':
        raise ValueError('전체 거래 정답은 고정 v14 관리 정책이 필요합니다.')
    out = new_run(output, 'lifecycle-edge-labels', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V15.md')),
        'market_manifest_sha256': sha256(market / 'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features / 'manifest-5m.json'),
        'training_period': ['2021-01-01', '2022-01-01'], 'forced_boundary_closes': False})
    print(f'전체 거래 순손익 정답: {out}', flush=True)
    try:
        bars, verified = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'input_verification.json', verified)
        opportunities, counts = potential_entries(bars, policy)
        opportunities.to_parquet(out / 'potential_entries.parquet', index=False)
        print(f'독립 진입 기회 {len(opportunities)}개', flush=True)

        def checkpoint(frame):
            temporary = out / 'labels_partial.parquet.tmp'
            frame.to_parquet(temporary, index=False)
            temporary.replace(out / 'labels_partial.parquet')
            if len(frame) % 100 == 0:
                print(f'전체 거래 정답 {len(frame)}/{len(opportunities)}개 처리', flush=True)

        training, ledger, details = lifecycle_labels(bars, opportunities, EngineConfig(**frozen['risk']),
                                                     policy, '2022-01-01', checkpoint)
        training.to_parquet(out / 'training_labels.parquet', index=False)
        ledger.to_parquet(out / 'opportunity_ledger.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'signals': counts, 'labels': details,
            'losses_preserved': int(training.net_bps.lt(0).sum()) if len(training) else 0,
            'positives': int(training.net_bps.gt(0).sum()) if len(training) else 0})
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out / 'REPORT.md').write_text('# 현재 관리 정책의 전체 거래 정답\n\n'
            f'독립 기회 {len(opportunities)}개 중 자연 종료 {len(training)}개. '
            '손실·손절·위험 종료를 포함하며 학습 경계를 넘긴 포지션은 미확정 원장에 보존했다. '
            '시간 경계에서 강제 청산해 정답을 만들지 않았다. 같은 초기 계좌·고정 v14 관리의 겹친 기회이며 '
            '연속 계좌 상태나 독립 표본을 뜻하지 않는다. 2021년은 관리 정책 선택에도 사용한 전체 시스템 학습 구간이다.\n')
        print(f'정답 생성 완료: 자연 종료 {len(training)}개', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
