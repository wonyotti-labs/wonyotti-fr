from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .event_research import load_selection
from .first_management import BETAS, FIRST_ACTIONS, decision_metrics
from .inventory_labels import verify_files
from .minute_management import ACTIONS
from .reports import table

MODEL_FILES = ['history_manager.json', 'history_offset.json', 'history_thresholds.json',
               'first_policy_thresholds.json']


def exit_direct_admission(metrics, unchanged):
    old, direct = (metrics['exit'][k] for k in ['original', 'direct'])
    values = [m[k] for m in [old, direct] for k in ['recall', 'f_beta']]
    if (not np.isfinite(values).all() or any(not 0 <= v <= 1 for v in values)
        or set(unchanged) != set(FIRST_ACTIONS) or any(type(v) is not bool for v in unchanged.values())):
        raise ValueError('청산 직접 문턱 진단의 지표·보존 조건 오류')
    checks = {'exit_recall_improved': direct['recall'] > old['recall'] + 1e-12,
              'exit_f2_improved': direct['f_beta'] > old['f_beta'] + 1e-12}
    return {'exit_direct_admitted': all(checks.values()) and all(unchanged.values()),
        'checks': checks, 'unchanged_masks': unchanged, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def exit_direct_diagnosis(calibration, validation, manager, thresholds, first_thresholds, multiplier):
    if (multiplier != 1.5 or set(thresholds) != set(ACTIONS)
        or set(first_thresholds) != set(FIRST_ACTIONS)
        or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
               for v in [*thresholds.values(), *first_thresholds.values()])):
        raise ValueError('청산 직접 문턱 진단의 기존 문턱·배율 오류')
    for part, name in [(calibration, 'calibration'), (validation, 'diagnosis')]:
        first, last = (pd.Timestamp(v, tz='UTC') for v in PERIODS[name])
        if (len(part) < 100 or part.end.min() < first + pd.Timedelta(days=1)
            or part.label_end.max() >= last - pd.Timedelta(days=1) or part.entry_time.min() < first
            or part.end.duplicated().any() or not part.end.is_monotonic_increasing
            or part[['end', 'label_end', 'entry_time', 'episode_id']].isna().any().any()
            or (part.label_end <= part.end).any() or (part.entry_time >= part.end).any()
            or not np.isfinite(part[manager.features]).all().all()):
            raise ValueError('청산 직접 문턱 진단의 시간·입력 오류')
        for action in ACTIONS:
            y = part[f'y_{action}'].to_numpy()
            if not np.isin(y, [0, 1]).all() or min(y.sum(), len(y) - y.sum()) < 20:
                raise ValueError('청산 직접 문턱 진단의 정답 지원 부족')
        if any(not part[f'past_{a}_exists'].isin([0, 1]).all() for a in FIRST_ACTIONS):
            raise ValueError('청산 직접 문턱 진단의 과거 체결 존재 여부 오류')
    if (set(calibration.episode_id) & set(validation.episode_id)
        or calibration.label_end.max() >= validation.end.min()):
        raise ValueError('청산 직접 문턱 진단의 포지션·시간 중첩')
    cp, vp = (manager.probabilities(p[manager.features].to_numpy()) for p in [calibration, validation])
    for scores, part in [(cp, calibration), (vp, validation)]:
        if (scores.shape != (len(part), len(ACTIONS)) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
            raise ValueError('청산 직접 문턱 진단의 확률 차원·범위 오류')
    threshold, support = select_threshold(calibration.y_exit, cp[:, ACTIONS.index('exit')], 2.)
    if threshold != thresholds['exit']:
        raise ValueError('청산 직접 문턱의 상반기 선택 재현 불일치')
    columns = ['end', 'episode_id', *[f'y_{a}' for a in ACTIONS],
               'past_reduce_exists', 'past_increase_exists']
    predictions, metrics, unchanged = validation[columns].copy(), {}, {}
    for i, action in enumerate(ACTIONS):
        boundary = np.full(len(validation), thresholds[action] * multiplier)
        if action in FIRST_ACTIONS:
            boundary[validation[f'past_{action}_exists'].eq(0)] = first_thresholds[action]
        original = vp[:, i] >= boundary
        direct = vp[:, i] >= threshold if action == 'exit' else original.copy()
        if action in FIRST_ACTIONS:
            unchanged[action] = bool(np.array_equal(original, direct))
        predictions[f'{action}_score'] = vp[:, i]
        predictions[f'{action}_original'] = original
        predictions[f'{action}_direct'] = direct
        metrics[action] = {kind: decision_metrics(validation[f'y_{action}'], p, BETAS[action])
                           for kind, p in [('original', original), ('direct', direct)]}
    return support, predictions, metrics, exit_direct_admission(metrics, unchanged)


def run_exit_direct_diagnosis(selection: Path, diagnosis: Path, output: Path) -> Path:
    frozen, policy = load_selection(selection)
    if frozen['protocol'] != 'activity_ablation_v46':
        raise ValueError('청산 직접 문턱 진단에는 고정 v46 후보가 필요합니다.')
    files = json.loads((diagnosis / 'files.json').read_text())
    required = {'manifest.json', 'summary.json', 'calibration_used.parquet', 'diagnosis_used.parquet'}
    if (not required <= set(files)
        or any(not (diagnosis / n).resolve().is_relative_to(diagnosis.resolve()) for n in files)):
        raise ValueError('청산 직접 문턱 진단의 파일 경로·목록 오류')
    verify_files(diagnosis, list(files))
    evidence = json.loads((selection / 'history_admission.json').read_text())
    old = json.loads((selection / 'history_thresholds.json').read_text())
    if (sha256(diagnosis / 'files.json') != evidence['files_sha256']
        or old['calibration_sha256'] != sha256(diagnosis / 'calibration_used.parquet')
        or old['calibration_period'] != list(PERIODS['calibration'])
        or old['minimum_predicted_positive'] != 20 or old['thresholds'] != policy.thresholds):
        raise ValueError('청산 직접 문턱 진단의 원본 입력·선택 지문 불일치')
    out = new_run(output, 'exit-direct-diagnosis', {'selection': str(selection), 'diagnosis': str(diagnosis),
        'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V47.md')), 'new_models_fitted': False,
        'exit_multiplier': 1., 'original_multiplier': 1.5,
        'calibration_period': PERIODS['calibration'], 'diagnosis_period': PERIODS['diagnosis']})
    print(f'청산 문턱 가산 제거 진단: {out}', flush=True)
    try:
        calibration, validation = (pd.read_parquet(diagnosis / f'{n}_used.parquet') for n in ['calibration', 'diagnosis'])
        support, predictions, metrics, decision = exit_direct_diagnosis(
            calibration, validation, policy.manager, policy.thresholds, policy.first_thresholds, policy.multiplier)
        if support != old['support']['exit'] or len(calibration) != old['rows']:
            raise ValueError('청산 직접 문턱의 상반기 선택 기록 재현 불일치')
        for name in MODEL_FILES:
            (out / name).write_bytes((selection / name).read_bytes())
        save_json(out / 'exit_threshold.json', {'threshold': policy.thresholds['exit'], 'support': support,
            'exit_multiplier': 1., 'original_multiplier': 1.5, 'minimum_predicted_positive': 20,
            'calibration_period': PERIODS['calibration'], 'beta': 2.})
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'metrics.json', metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'calibration_rows': len(calibration),
            'diagnosis_rows': len(validation), 'episode_intersection': 0, 'new_models_fitted': False, **decision})
        rows = [{'action': a, 'threshold': k, **m} for a, pairs in metrics.items() for k, m in pairs.items()]
        (out / 'REPORT.md').write_text('# 청산 문턱의 가산 제거 진단\n\n' + table(pd.DataFrame(rows))
            + '\n\n기존 모델·절편·상반기 문턱을 유지하고 청산 배율만 1.5에서 1.0으로 변경했다. '
            '첫·반복 추가와 축소 요청은 그대로다. 이미 관찰한 원본 상태의 진단이며 실제 매매 수익성의 증거가 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 청산 매매 연구 허용: {decision["exit_direct_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
