from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .event_research import load_selection
from .histogram_management import HistogramManagementModels
from .inventory_labels import verify_files
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .management_diagnostics import management_admission, management_metrics
from .minute_inventory import MinuteInventoryModels
from .minute_management import ACTIONS
from .order_history import HISTORY_FEATURES, OrderHistoryModels
from .reports import table


class OrderHistoryBoostModels(HistogramManagementModels):
    features = OrderHistoryModels.features
    format = 'order_history_histogram_v1'


def history_boost_admission(metrics):
    comparisons = {}
    for name in ['original', 'history']:
        comparisons[name] = management_admission({a: {'logistic': m[name], 'histogram': m['boosted'], 'constant': m['constant']}
                                                  for a, m in metrics.items()})
    return {'history_boost_admitted': all(v['histogram_admitted'] for v in comparisons.values()),
        'comparisons': comparisons, 'profitability_accepted': False, 'trading_returns_evaluated': False,
        'all_source_periods_already_observed': True}


def run_order_history_boost_diagnostics(history: Path, output: Path) -> Path:
    previous = json.loads((history / 'manifest.json').read_text())['settings']
    if (previous['training_period'] != list(TRAINING_PERIODS[1])
        or previous['diagnosis_period'] != list(CALIBRATION_PERIODS[1]) or previous['new_features'] != HISTORY_FEATURES):
        raise ValueError('과거 주문 부스팅의 고정 기간·입력 오류')
    verify_files(history, list(json.loads((history / 'files.json').read_text())))
    selection, labels, audit = (Path(previous[k]) for k in ['selection', 'labels', 'audit'])
    if (sha256(selection / 'frozen_selection.json') != previous['selection_sha256']
        or sha256(labels / 'files.json') != previous['labels_files_sha256']
        or any(sha256(audit / n) != h for n, h in previous['audit_sha256'].items())):
        raise ValueError('과거 주문 부스팅의 원본·모델 지문 오류')
    frozen, original = load_selection(selection)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('과거 주문 부스팅에는 원래 v21 모델이 필요합니다.')
    out = new_run(output, 'order-history-boost-diagnosis', {'history': str(history),
        'history_files_sha256': sha256(history / 'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V34.md')),
        'training_period': TRAINING_PERIODS[1], 'diagnosis_period': CALIBRATION_PERIODS[1],
        'new_models': 1, 'trading_returns_evaluated': False})
    print(f'과거 주문 맥락의 고정 부스팅 진단: {out}', flush=True)
    try:
        train, validation = (pd.read_parquet(history / name) for name in ['training_used.parquet', 'diagnosis_used.parquet'])
        for part, name in [(train, 'training_used'), (validation, 'calibration_used')]:
            pd.testing.assert_frame_equal(part.drop(columns=HISTORY_FEATURES), pd.read_parquet(selection / f'{name}.parquet'), check_exact=True)
            y = part[[f'y_{a}' for a in ACTIONS]].to_numpy()
            if (len(part) < (1000 if name == 'training_used' else 100)
                or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any()
                or not np.isfinite(part[OrderHistoryModels.features]).all().all()):
                raise ValueError('과거 주문 부스팅의 입력·지원 부족')
        if set(train.episode_id) & set(validation.episode_id) or train.label_end.max() >= validation.end.min():
            raise ValueError('과거 주문 부스팅의 시간·포지션 분리 오류')
        reference, thresholds, support = OrderHistoryModels.fit(train, validation, 'logistic')
        if (reference.to_dict() != json.loads((history / 'model.json').read_text())
            or thresholds != json.loads((history / 'thresholds_unused.json').read_text())):
            raise ValueError('과거 주문 로지스틱의 학습 재현 불일치')
        x, vx = (part[OrderHistoryModels.features].to_numpy(dtype=float) for part in [train, validation])
        y = train[[f'y_{a}' for a in ACTIONS]].to_numpy(dtype=int)
        model, boost_support = OrderHistoryBoostModels.fit(x, y, vx)
        scores = {'original': original.manager.probabilities(validation[MinuteInventoryModels.features].to_numpy()),
            'history': reference.probabilities(vx), 'boosted': model.probabilities(vx),
            'constant': np.tile(y.mean(axis=0), (len(validation), 1))}
        old_scores = pd.read_parquet(history / 'predictions.parquet')
        columns = ['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]
        pd.testing.assert_frame_equal(old_scores[columns], validation[columns], check_exact=True)
        for name in ['original', 'history', 'constant']:
            np.testing.assert_array_equal(scores[name], old_scores[[f'{a}_{name}' for a in ACTIONS]].to_numpy())
        for name in ['training_used.parquet', 'diagnosis_used.parquet']:
            (out / name).write_bytes((history / name).read_bytes())
        predictions = validation[columns].copy()
        metrics = {}
        for i, action in enumerate(ACTIONS):
            metrics[action] = {}
            for name, values in scores.items():
                predictions[f'{action}_{name}'] = values[:, i]
                metrics[action][name] = management_metrics(validation[f'y_{action}'], values[:, i])
            print(f'{action}: 이력 선형 {metrics[action]["history"]["log_loss"]:.6f}, 부스팅 {metrics[action]["boosted"]["log_loss"]:.6f}', flush=True)
        for name, value in [('model', model.to_dict()), ('metrics', metrics),
            ('training_support', {'history': support, 'boosted': boost_support})]:
            save_json(out / f'{name}.json', value)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        decision = history_boost_admission(metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'original_rows_labels_features_exact': True,
            'original_and_history_predictions_exact': True, 'history_model_and_thresholds_refitted_exact': True,
            'training_rows': len(train), 'diagnosis_rows': len(validation), **decision})
        values = [{'action': a, 'model': k, **v} for a, group in metrics.items() for k, v in group.items()]
        (out / 'REPORT.md').write_text('# 과거 주문과 가격 상태의 고정 부스팅 진단\n\n' + table(pd.DataFrame(values))
            + '\n\nv33의 모든 행·정답·50개 입력·기간을 그대로 사용했다. 고정 부스팅 하나를 두 로지스틱과 상수에 대조했다. '
            '세 행동 모두가 두 로지스틱 각각의 사전 개선 조건을 통과해야 하며 일부 행동을 선택하지 않는다. '
            '이미 관찰한 원본 상태의 진단이며 봇 자신의 체결 이력·실제 매매·수익성 입증을 포함하지 않는다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 연구 허용: {decision["history_boost_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
