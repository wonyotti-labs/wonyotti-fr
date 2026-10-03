from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .common import new_run, save_json, sha256
from .event_research import load_selection
from .histogram_management import HistogramManagementModels
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .minute_inventory import MinuteInventoryModels, load_minute_inventory_labels
from .minute_management import ACTIONS, purged_window
from .reports import table


def management_metrics(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or y.shape != p.shape or set(np.unique(y)) != {0, 1}
        or min(y.sum(), (1 - y).sum()) < 20 or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()):
        raise ValueError('관리 진단의 정답·점수·지원 부족')
    return {'rows': len(y), 'positive': int(y.sum()), 'negative': int((1 - y).sum()),
        'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)),
        'average_precision': float(average_precision_score(y, p)), 'roc_auc': float(roc_auc_score(y, p)),
        'actual_positive_fraction': float(y.mean()), 'predicted_mean': float(p.mean())}


def management_admission(metrics):
    comparisons = {}
    for action in ACTIONS:
        old, new, baseline = (metrics[action][k] for k in ['logistic', 'histogram', 'constant'])
        values = [old['log_loss'], new['log_loss'], baseline['log_loss'], old['average_precision'], new['average_precision']]
        if not np.isfinite(values).all() or min(values[:3]) < 0 or not all(0 <= v <= 1 for v in values[3:]):
            raise ValueError('관리 진단 판단의 지표 범위 오류')
        gain = 1 - new['log_loss'] / old['log_loss'] if old['log_loss'] > 0 else 0.
        comparisons[action] = {'relative_log_loss_gain': gain,
            'gain_passed': bool(gain > .01 and not np.isclose(gain, .01, atol=1e-12, rtol=0)),
            'constant_beaten': new['log_loss'] < baseline['log_loss'],
            'average_precision_preserved': new['average_precision'] >= old['average_precision']}
    admitted = all(r['gain_passed'] and r['constant_beaten'] and r['average_precision_preserved'] for r in comparisons.values())
    return {'histogram_admitted': admitted, 'actions': comparisons, 'required_relative_gain': .01,
            'profitability_accepted': False, 'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def management_split(frame, reference):
    train = purged_window(frame, *TRAINING_PERIODS[1]).reset_index(drop=True)
    validation = purged_window(frame, *CALIBRATION_PERIODS[1]).reset_index(drop=True)
    for name, rows, minimum in [('training_used', train, 1000), ('calibration_used', validation, 100)]:
        pd.testing.assert_frame_equal(rows, pd.read_parquet(reference / f'{name}.parquet'), check_exact=True)
        y = rows[[f'y_{a}' for a in ACTIONS]].to_numpy()
        if (len(rows) < minimum or not np.isfinite(rows[MinuteInventoryModels.features]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1 - y).sum(axis=0) < 20).any()
            or rows.end.duplicated().any() or not rows.end.is_monotonic_increasing):
            raise ValueError('관리 시간순 진단의 지원·특징·순서 오류')
    if (set(train.episode_id) & set(validation.episode_id)) - {0} or train.label_end.max() >= validation.end.min():
        raise ValueError('관리 학습·진단의 포지션·시간 중첩')
    return train, validation


def run_management_diagnostics(reference: Path, labels: Path, output: Path) -> Path:
    frozen, original = load_selection(reference)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('관리 시간순 진단에는 고정 v21 기반이 필요합니다.')
    previous = json.loads((reference / 'manifest.json').read_text())['settings']
    if previous['files_sha256'] != sha256(labels / 'files.json'):
        raise ValueError('관리 진단과 기존 학습의 원본 입력 지문 불일치')
    out = new_run(output, 'management-model-diagnosis', {'reference': str(reference), 'labels': str(labels),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'labels_files_sha256': sha256(labels / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V30.md')), 'training_period': TRAINING_PERIODS[1],
        'diagnosis_period': CALIBRATION_PERIODS[1], 'all_source_periods_already_observed': True,
        'model_families': ['logistic', 'histogram'], 'trading_returns_evaluated': False})
    print(f'관리 행동의 시간순 예측 진단: {out}', flush=True)
    try:
        frame, _ = load_minute_inventory_labels(labels)
        train, validation = management_split(frame, reference)
        del frame
        train.to_parquet(out / 'training_used.parquet', index=False)
        validation.to_parquet(out / 'diagnosis_used.parquet', index=False)
        logistic, thresholds, support = MinuteInventoryModels.fit(train, validation, 'logistic')
        if logistic.to_dict() != original.manager.to_dict() or thresholds != frozen['inventory_thresholds']:
            raise ValueError('기존 관리 모형·문턱의 학습 재현 불일치')
        x = train[MinuteInventoryModels.features].to_numpy(dtype=float)
        y = train[[f'y_{a}' for a in ACTIONS]].to_numpy(dtype=int)
        vx = validation[MinuteInventoryModels.features].to_numpy(dtype=float)
        histogram, hist_support = HistogramManagementModels.fit(x, y, vx)
        del x
        save_json(out / 'models.json', {'logistic': logistic.to_dict(), 'histogram': histogram.to_dict()})
        save_json(out / 'training_support.json', {'logistic': support, 'histogram': hist_support})
        scores = {'logistic': logistic.probabilities(vx), 'histogram': histogram.probabilities(vx),
                  'constant': np.tile(y.mean(axis=0), (len(validation), 1))}
        predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        metrics = {}
        for i, action in enumerate(ACTIONS):
            metrics[action] = {}
            for kind, values in scores.items():
                predictions[f'{action}_{kind}'] = values[:, i]
                metrics[action][kind] = management_metrics(validation[f'y_{action}'], values[:, i])
            print(f'{action}: 로그 손실 기존 {metrics[action]["logistic"]["log_loss"]:.6f}, 부스팅 {metrics[action]["histogram"]["log_loss"]:.6f}', flush=True)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'metrics.json', metrics)
        decision = management_admission(metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'training_rows': len(train), 'diagnosis_rows': len(validation),
            'episode_intersection': 0, 'existing_model_and_thresholds_exact': True, **decision})
        rows = [{'action': action, 'model': name, **value} for action, comparisons in metrics.items() for name, value in comparisons.items()]
        (out / 'REPORT.md').write_text('# 관리 행동의 시간순 예측 진단\n\n' + table(pd.DataFrame(rows))
            + '\n\n이미 관찰하고 기존 문턱·빈도 보정에 쓴 원본 내부 기간이다. 행동 예측 점수이며 매매 수익성 결과가 아니다. '
            '세 행동 모두의 사전 조건을 통과해야 별도로 계획한 관리 후보를 허용한다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 관리 후보 허용: {decision["histogram_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
