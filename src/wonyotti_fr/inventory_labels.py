from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .inventory_management import InventoryActionModels
from .inventory_study import attach_inventory
from .minute_management import purged_window
from .path_management import PATH_FEATURES


def sizing_labels(frame, orders, sizes):
    if orders.order_key.duplicated().any() or sizes.order_key.duplicated().any():
        raise ValueError('축소 크기 원장의 독립 주문 중복')
    extra = sizes[['order_key', 'last_time', 'executed_quantity', 'interleaved']].copy()
    ledger = orders.merge(extra, on='order_key', validate='one_to_one', how='left')
    if len(ledger) != len(sizes) or ledger.last_time.isna().any():
        raise ValueError('축소 크기 원장의 주문 연결 누락')
    ledger = ledger.merge(frame[['end', 'inventory_quantity']], left_on='window_end', right_on='end',
                          how='left', validate='many_to_one', suffixes=('', '_inventory'))
    counts = orders[orders.reason.eq('linked')].groupby('window_end').size()
    ledger['linked_count'] = ledger.window_end.map(counts).fillna(0).astype(int)
    ledger['reduction_target'] = ledger.executed_quantity / ledger.before_qty.abs().replace(0, np.nan)
    ledger['size_label_end'] = pd.concat([ledger.window_end + pd.Timedelta(minutes=1),
                                         ledger.last_time.dt.ceil('min')], axis=1).max(axis=1)
    ledger['size_reason'] = np.select([
        ledger.reason.ne('linked'), ledger.action.ne('reduce'), ledger.linked_count.ne(1),
        ledger.before_qty.ne(ledger.inventory_quantity), ledger.interleaved,
        ~ledger.reduction_target.between(0, 1, inclusive='right'),
    ], ['unlinked_action', 'not_reduce', 'multiple_management_orders', 'boundary_quantity_changed',
        'interleaved_fills', 'invalid_filled_fraction'], default='supported')
    linked = ledger[ledger.size_reason.eq('supported')]
    if linked.window_end.duplicated().any():
        raise ValueError('축소 크기 정답의 분 경계 중복')
    return ledger


def sizing_training(frame, ledger):
    supported = ledger[ledger.size_reason.eq('supported')].set_index('window_end')
    full = frame[['end', 'entry_time', 'episode_id', 'label_end', 'usable']].copy()
    full['reduction_target'] = full.end.map(supported.reduction_target)
    full['size_label_end'] = full.end.map(supported.size_label_end)
    full['usable'] &= full.reduction_target.notna()
    full['label_end'] = full.size_label_end.fillna(full.label_end)
    # 포지션 경계 제거는 축소 표본만이 아니라 전체 시계열에서 판정한다.
    selected = purged_window(full, '2019-01-01', '2020-07-01')
    selected = selected.merge(frame[['end', *InventoryActionModels.features]], on='end', validate='one_to_one')
    selected['order_key'] = selected.end.map(supported.order_key)
    return selected.drop(columns='size_label_end')


def verify_files(root, names):
    hashes = json.loads((root / 'files.json').read_text())
    if any(sha256(root / name) != hashes[name] for name in names):
        raise ValueError('수량 정답 입력의 지문 오류')
    if json.loads((root / 'summary.json').read_text()).get('complete') is not True:
        raise ValueError('수량 정답 입력의 미완료 상태')


def load_inventory_labels(root):
    metadata = json.loads((root / 'manifest.json').read_text())['settings']
    source = Path(metadata['path_labels'])
    if sha256(source / 'files.json') != metadata['path_files_sha256']:
        raise ValueError('기반 가격 경로 정답의 지문 오류')
    verify_files(source, ['events.parquet', 'order_ledger.parquet', 'summary.json'])
    verify_files(root, ['inventory_features.parquet', 'size_ledger.parquet', 'summary.json'])
    original = pd.read_parquet(source / 'events.parquet')
    features = pd.read_parquet(root / 'inventory_features.parquet')
    if not original.end.equals(features.end):
        raise ValueError('수량 특징과 원본 분 경계 불일치')
    return pd.concat([original, features.drop(columns='end')], axis=1), pd.read_parquet(root / 'size_ledger.parquet')


def run_inventory_labels(path_labels: Path, study: Path, output: Path) -> Path:
    verify_files(path_labels, ['events.parquet', 'order_ledger.parquet', 'summary.json'])
    verify_files(study, ['source_states.parquet', 'order_sizes.parquet', 'summary.json'])
    path_meta = json.loads((path_labels / 'manifest.json').read_text())['settings']
    plain = Path(path_meta['labels'])
    if sha256(plain / 'files.json') != path_meta['files_sha256']:
        raise ValueError('가격 경로 입력 계층의 지문 오류')
    source = json.loads((plain / 'manifest.json').read_text())['settings']['audit_sha256']
    study_source = json.loads((study / 'manifest.json').read_text())['settings']['audit_sha256']
    if source != study_source or json.loads((path_labels / 'summary.json').read_text()).get('path_features') != PATH_FEATURES:
        raise ValueError('수량·가격 경로 입력의 원본 계층 불일치')
    out = new_run(output, 'inventory-management-labels', {'path_labels': str(path_labels), 'study': str(study),
        'path_files_sha256': sha256(path_labels / 'files.json'), 'study_files_sha256': sha256(study / 'files.json'),
        'audit_sha256': source, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V19.md'))})
    print(f'잔여 수량과 실제 축소 크기 정답: {out}', flush=True)
    try:
        original = pd.read_parquet(path_labels / 'events.parquet')
        frame = attach_inventory(original, pd.read_parquet(study / 'source_states.parquet'))
        pd.testing.assert_frame_equal(frame[original.columns], original, check_exact=True)
        features = frame[['end', 'remaining_fraction', 'inventory_quantity', 'inventory_time', 'inventory_episode_id']]
        features.to_parquet(out / 'inventory_features.parquet', index=False)
        ledger = sizing_labels(frame, pd.read_parquet(path_labels / 'order_ledger.parquet'), pd.read_parquet(study / 'order_sizes.parquet'))
        ledger.to_parquet(out / 'size_ledger.parquet', index=False)
        training = sizing_training(frame, ledger)
        training.to_parquet(out / 'size_training.parquet', index=False)
        summary = {'complete': True, 'original_minute_rows': len(original), 'original_actions_unchanged': True,
            'independent_orders_preserved': len(ledger), 'size_reasons': ledger.size_reason.value_counts().to_dict(),
            'size_training_rows': len(training), 'size_training_days': int((training.end.max()-training.end.min()).days),
            'size_training_first': training.end.min(), 'size_training_last_label_end': training.label_end.max(),
            'required_support': bool(len(training) >= 100 and training.end.max()-training.end.min() >= pd.Timedelta(days=90)),
            'features': InventoryActionModels.features, 'profitability_accepted': False}
        save_json(out / 'summary.json', summary)
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out / 'REPORT.md').write_text('# 실제 체결 축소 크기의 대응 정답\n\n'
            f'원본 {len(original):,}분·{len(ledger):,}독립 주문 보존. 크기 학습 {len(training):,}개.\n\n'
            '분 경계의 보유량과 최초 체결 직전 보유량이 같고, 관리 주문 하나이며 다른 주문이 끼지 않은 실제 체결만 대응했다. '
            '대응 불가 사유·원본 행동·손익은 그대로 보존했다. 크기 정답 종료는 마지막 체결까지 연장하고 포지션·24시간 경계를 제거했다. '
            '여러 부분 체결의 총량을 다음 거래 가능 시가에서 한 번에 실행하는 근사이며 원래 체결 대기열의 복제가 아니다.\n')
        print(f'원본 행동 불변, 크기 학습 {len(training)}개 / {summary["size_training_days"]}일', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
