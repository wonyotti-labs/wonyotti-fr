from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .inventory_labels import load_inventory_labels, verify_files
from .inventory_management import InventoryActionModels, InventoryRatePolicy, ReductionModel
from .minute_inputs import MINUTE_FEATURES, minute_features
from .timing_study import read_minute_history

CONTEXT_FEATURES = MINUTE_FEATURES + [f'directional_{name}' for name in MINUTE_FEATURES]


class MinuteInventoryModels(InventoryActionModels):
    features = InventoryActionModels.features + CONTEXT_FEATURES
    format = 'minute_inventory_context_action_v1'


class MinuteReductionModel(ReductionModel):
    features = MinuteInventoryModels.features
    format = 'minute_inventory_context_reduction_v1'


class MinuteInventoryPolicy(InventoryRatePolicy):
    def feature_values(self, bar, state):
        minute = np.asarray(bar['minute_features'], dtype=float)
        if minute.shape != (len(MINUTE_FEATURES),):
            raise ValueError('관리용 확정 분봉 입력의 차원 오류')
        return np.r_[super().feature_values(bar, state), minute, minute * state['direction']]


def attach_minute_inputs(frame, features):
    if not frame.end.astype('datetime64[ns, UTC]').equals(features.end.astype('datetime64[ns, UTC]')):
        raise ValueError('원본 관리 정답과 확정 분봉 입력 시각 불일치')
    result = pd.concat([frame, features.drop(columns='end')], axis=1)
    for name in MINUTE_FEATURES:
        result[f'directional_{name}'] = result[name] * result.direction
    if not np.isfinite(result.loc[result.usable, CONTEXT_FEATURES]).all().all():
        raise ValueError('기존 관리 지원 행의 분봉 입력 누락')
    return result


def load_minute_inventory_labels(root):
    metadata = json.loads((root / 'manifest.json').read_text())['settings']
    previous = Path(metadata['inventory_labels'])
    if sha256(previous / 'files.json') != metadata['inventory_files_sha256']:
        raise ValueError('분봉 입력의 기반 수량 정답 지문 오류')
    verify_files(root, ['minute_features.parquet', 'summary.json'])
    frame, ledger = load_inventory_labels(previous)
    return attach_minute_inputs(frame, pd.read_parquet(root / 'minute_features.parquet')), ledger


def run_minute_inventory_labels(labels: Path, minute_history: Path, history: Path, output: Path) -> Path:
    frame, ledger = load_inventory_labels(labels)
    minute, hashes = read_minute_history(minute_history, history)
    parent = Path(json.loads((labels / 'manifest.json').read_text())['settings']['path_labels'])
    plain = Path(json.loads((parent / 'manifest.json').read_text())['settings']['labels'])
    original = json.loads((plain / 'manifest.json').read_text())['settings']
    if any(original[key] != digest for key, digest in hashes.items()):
        raise ValueError('분별 관리 정답의 원본 1분 시세 지문 불일치')
    out = new_run(output, 'minute-inventory-labels', {'inventory_labels': str(labels),
        'inventory_files_sha256': sha256(labels / 'files.json'), **hashes,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V21.md'))})
    print(f'관리용 확정 분봉 입력: {out}', flush=True)
    try:
        if not np.array_equal(frame.close.to_numpy(), minute.close.to_numpy()):
            raise ValueError('분별 관리 정답과 원본 종가 불일치')
        features = minute_features(minute)
        attached = attach_minute_inputs(frame, features)
        pd.testing.assert_frame_equal(attached[frame.columns], frame, check_exact=True)
        features.to_parquet(out / 'minute_features.parquet', index=False)
        summary = {'complete': True, 'rows': len(frame), 'usable_rows': int(frame.usable.sum()),
            'independent_orders': len(ledger), 'original_labels_unchanged': True,
            'features': MinuteInventoryModels.features, 'new_profitability_test': False}
        save_json(out / 'summary.json', summary)
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out / 'REPORT.md').write_text('# 확정 1분 시장 입력과 원본 정답 연결\n\n'
            f'{len(frame):,}분과 {len(ledger):,}독립 주문 보존. 기존 지원 행·행동 정답·축소 크기는 그대로 유지했다. '
            '현재 확정 종가·고가·저가·거래량과 과거 확정 시세만 사용했다. 거래량 기준은 현재 분을 제외한 이전 60분이다. '
            '이 결과는 입력의 연결 검사이며 수익성 검증이 아니다.\n')
        print(f'{len(frame)}분 원본 정답·지원 행 불변', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
