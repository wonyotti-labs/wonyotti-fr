from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .inventory_labels import verify_files
from .minute_inventory import load_minute_inventory_labels

EXIT_LABEL_FILES = ['exit_targets.parquet', 'ending_ledger.parquet', 'summary.json']


def realized_exit_targets(frame, actions):
    if (frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or not frame.label_end.sub(frame.end).eq(pd.Timedelta(minutes=1)).all()
        or (frame.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=1).value).any()
        or not actions.time.is_monotonic_increasing
        or not np.isfinite(actions[['before_qty', 'after_qty']]).all().all()
        or not actions.before_qty.iloc[1:].reset_index(drop=True).equals(actions.after_qty.iloc[:-1].reset_index(drop=True))):
        raise ValueError('실제 보유 종료의 시각·수량 연속성 오류')
    ending = actions.action.isin(['close', 'reverse'])
    actual = actions.before_qty.ne(0) & (np.sign(actions.before_qty) * np.sign(actions.after_qty)).le(0)
    if not ending.equals(actual):
        raise ValueError('실제 보유 종료의 행동·수량 불일치')
    ledger = actions.assign(before_episode_id=np.where(actions.before_qty.ne(0), actions.episode_id.shift(fill_value=0), 0))[ending].copy()
    if ledger.before_episode_id.le(0).any():
        raise ValueError('실제 보유 종료의 이전 에피소드 누락')
    ledger['window_end'] = ledger.time.dt.floor('min')
    ledger = ledger.merge(frame[['end', 'label_end', 'episode_id', 'usable']], left_on='window_end', right_on='end',
                           how='left', validate='many_to_one', suffixes=('_after', '_boundary'))
    ledger['reason'] = np.select([
        ledger.end.isna(), ledger.before_episode_id.ne(ledger.episode_id_boundary), ledger.usable.ne(True),
    ], ['outside_minutes', 'different_episode', 'unusable_features_or_range'], default='linked')
    linked = ledger[ledger.reason.eq('linked')]
    if linked.time.lt(linked.end).any() or linked.time.ge(linked.label_end).any():
        raise ValueError('실제 보유 종료의 다음 분 정답 범위 오류')
    counts = linked.groupby('window_end').size()
    targets = frame[['end', 'y_exit', 'exit_count']].rename(columns={'y_exit': 'requested_y_exit', 'exit_count': 'requested_exit_count'}).copy()
    targets['exit_count'] = targets.end.map(counts).fillna(0).astype(int)
    targets['y_exit'] = targets.exit_count.gt(0).astype(int)
    if targets.exit_count.sum() != len(linked):
        raise ValueError('실제 보유 종료의 원장·정답 합계 불일치')
    yearly = []
    for year, rows in targets[frame.usable.to_numpy()].groupby(targets.end.dt.year):
        old, new = rows.requested_y_exit.eq(1), rows.y_exit.eq(1)
        yearly.append({'year': int(year), 'minutes': len(rows), 'requested_exit_minutes': int(old.sum()),
            'actual_end_minutes': int(new.sum()), 'same_minute': int((old & new).sum()),
            'requested_without_actual_end': int((old & ~new).sum()), 'actual_end_without_requested': int((~old & new).sum())})
    return targets, ledger, {'complete': True, 'minute_rows': len(frame), 'actual_events': len(ledger),
        'linked_actual_events': len(linked), 'reasons': ledger.reason.value_counts().to_dict(), 'yearly': yearly,
        'only_exit_targets_changed': True, 'observed_execution_outcomes_not_decision_times': True}


def load_realized_exit_labels(root):
    meta = json.loads((root / 'manifest.json').read_text())['settings']
    source = Path(meta['minute_labels'])
    if sha256(source / 'files.json') != meta['minute_files_sha256']:
        raise ValueError('실제 청산 정답의 원래 분별 입력 지문 오류')
    verify_files(root, EXIT_LABEL_FILES)
    frame, sizes = load_minute_inventory_labels(source)
    targets = pd.read_parquet(root / 'exit_targets.parquet')
    if (not targets.end.equals(frame.end) or not targets.requested_y_exit.equals(frame.y_exit)
        or not targets.requested_exit_count.equals(frame.exit_count)):
        raise ValueError('실제 청산 정답의 원래 행·정답 연결 오류')
    result = frame.copy()
    result[['y_exit', 'exit_count']] = targets[['y_exit', 'exit_count']]
    return result, sizes


def run_realized_exit_labels(labels: Path, audit: Path, output: Path) -> Path:
    meta = json.loads((labels / 'manifest.json').read_text())['settings']
    source = Path(meta['inventory_labels'])
    audit_digest = json.loads((source / 'manifest.json').read_text())['settings']['audit_sha256']
    if {n: sha256(audit / n) for n in ['actions.parquet', 'executions.parquet', 'episodes.parquet']} != audit_digest:
        raise ValueError('실제 청산 정답과 분별 입력의 감사 원본 불일치')
    frame, _ = load_minute_inventory_labels(labels)
    out = new_run(output, 'realized-exit-labels', {'minute_labels': str(labels), 'audit': str(audit),
        'minute_files_sha256': sha256(labels / 'files.json'), 'audit_sha256': audit_digest,
        'actions_sha256': sha256(audit / 'actions.parquet'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V24.md'))})
    print(f'실제 보유 종료 정답: {out}', flush=True)
    try:
        targets, ledger, summary = realized_exit_targets(frame, pd.read_parquet(audit / 'actions.parquet'))
        targets.to_parquet(out / 'exit_targets.parquet', index=False)
        ledger.to_parquet(out / 'ending_ledger.parquet', index=False)
        save_json(out / 'summary.json', summary)
        save_json(out / 'files.json', {n: sha256(out / n) for n in EXIT_LABEL_FILES})
        (out / 'REPORT.md').write_text('# 실제 보유 종료 정답\n\n청산 요청의 첫 체결과 실제 포지션 종료를 구분했다. '
            '이후 종료 정보는 정답에만 쓰고 당시 입력·상태·지원 행은 유지했다. '
            '같은 분 안의 다른 포지션·범위 밖 종료도 원장에 보존했다. 마지막 부분 체결은 관측된 실행 결과이며 판단 시각의 복원이 아니다.\n')
        print(f'실제 종료 {len(ledger)}개, 연결 {summary["linked_actual_events"]}개', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
