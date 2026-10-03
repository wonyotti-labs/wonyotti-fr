from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .event_research import load_selection
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .management_diagnostics import management_admission, management_metrics, management_split
from .minute_inventory import MinuteInventoryModels, load_minute_inventory_labels
from .minute_management import ACTIONS, purged_window
from .order_history import HISTORY_FEATURES, OrderHistoryModels, attach_order_history
from .reports import table


def history_admission(metrics):
    mapped = {a: {'logistic': m['original'], 'histogram': m['history'], 'constant': m['constant']}
              for a, m in metrics.items()}
    decision = management_admission(mapped)
    decision['history_admitted'] = decision.pop('histogram_admitted')
    return decision


def run_order_history_diagnostics(selection: Path, labels: Path, audit: Path, output: Path) -> Path:
    frozen, original = load_selection(selection)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('과거 주문 맥락 진단에는 고정 v21 모델이 필요합니다.')
    if json.loads((selection / 'manifest.json').read_text())['settings']['files_sha256'] != sha256(labels / 'files.json'):
        raise ValueError('과거 주문 맥락과 기존 학습의 정답 지문 오류')
    parent = Path(json.loads((labels / 'manifest.json').read_text())['settings']['inventory_labels'])
    expected = json.loads((parent / 'manifest.json').read_text())['settings']['audit_sha256']
    if any(sha256(audit / name) != digest for name, digest in expected.items()):
        raise ValueError('과거 주문 맥락과 정답 원본의 감사 지문 오류')
    out = new_run(output, 'order-history-diagnosis', {'selection': str(selection), 'labels': str(labels), 'audit': str(audit),
        'selection_sha256': sha256(selection / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json'),
        'audit_sha256': expected, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V33.md')),
        'training_period': TRAINING_PERIODS[1], 'diagnosis_period': CALIBRATION_PERIODS[1],
        'new_features': HISTORY_FEATURES, 'trading_returns_evaluated': False})
    print(f'과거 독립 관리 주문의 입력 진단: {out}', flush=True)
    try:
        frame, _ = load_minute_inventory_labels(labels)
        old_train, old_validation = management_split(frame, selection)
        attached, ledger = attach_order_history(frame, pd.read_parquet(audit / 'actions.parquet'))
        train = purged_window(attached, *TRAINING_PERIODS[1]).reset_index(drop=True)
        validation = purged_window(attached, *CALIBRATION_PERIODS[1]).reset_index(drop=True)
        pd.testing.assert_frame_equal(train[frame.columns], old_train, check_exact=True)
        pd.testing.assert_frame_equal(validation[frame.columns], old_validation, check_exact=True)
        features = attached[['end', *HISTORY_FEATURES]]
        features.to_parquet(out / 'history_features.parquet', index=False)
        ledger.to_parquet(out / 'first_execution_ledger.parquet', index=False)
        train.to_parquet(out / 'training_used.parquet', index=False)
        validation.to_parquet(out / 'diagnosis_used.parquet', index=False)
        del frame, attached, old_train, old_validation, features
        model, thresholds, support = OrderHistoryModels.fit(train, validation, 'logistic')
        scores = {'original': original.manager.probabilities(validation[MinuteInventoryModels.features].to_numpy()),
            'history': model.probabilities(validation[OrderHistoryModels.features].to_numpy()),
            'constant': np.tile(train[[f'y_{a}' for a in ACTIONS]].mean().to_numpy(), (len(validation), 1))}
        predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        metrics = {}
        for i, action in enumerate(ACTIONS):
            metrics[action] = {}
            for name, values in scores.items():
                predictions[f'{action}_{name}'] = values[:, i]
                metrics[action][name] = management_metrics(validation[f'y_{action}'], values[:, i])
            print(f'{action}: 기존 {metrics[action]["original"]["log_loss"]:.6f}, 이력 추가 {metrics[action]["history"]["log_loss"]:.6f}', flush=True)
        for name, value in [('model', model.to_dict()), ('thresholds_unused', thresholds), ('training_support', support), ('metrics', metrics)]:
            save_json(out / f'{name}.json', value)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        decision = history_admission(metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'rows': len(train) + len(validation),
            'training_rows': len(train), 'diagnosis_rows': len(validation), 'past_independent_orders': len(ledger),
            'original_rows_labels_features_exact': True, **decision})
        values = [{'action': a, 'model': k, **v} for a, group in metrics.items() for k, v in group.items()]
        (out / 'REPORT.md').write_text('# 과거 독립 관리 주문의 입력 진단\n\n' + table(pd.DataFrame(values))
            + '\n\n각 독립 주문의 첫 체결에서 이미 관측한 증가·축소만 같은 포지션의 엄격한 과거로 연결했다. '
            '기존 행·정답·50개 중 기존 44개 입력·학습 기간은 유지했다. 문턱은 기존 학습 함수가 산출한 미사용 참고값이며 매매 후보를 선택하지 않았다. '
            '모든 기간을 이미 관찰했으며 이 결과는 원본 상태의 예측 진단이다. 봇 자체 체결 이력의 실행·복원은 별도 검증이 필요하다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 연구 허용: {decision["history_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
