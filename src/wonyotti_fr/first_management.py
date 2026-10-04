from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .event_research import load_selection
from .inventory_labels import verify_files
from .minute_management import ACTIONS
from .reports import table

FIRST_ACTIONS = ['reduce', 'increase']
BETAS = {'exit': 2., 'reduce': 1., 'increase': .5}


def decision_metrics(labels, predicted, beta):
    y, p = np.asarray(labels), np.asarray(predicted)
    if (y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all()
        or p.dtype != np.dtype(bool) or not len(y) or beta not in (.5, 1., 2.)):
        raise ValueError('첫 관리 주문 진단의 정답·요청 오류')
    tp, fp, fn = int(((y == 1) & p).sum()), int(((y == 0) & p).sum()), int(((y == 1) & ~p).sum())
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    denominator = (1 + beta**2) * tp + beta**2 * fn + fp
    return {'rows': len(y), 'positive': int(y.sum()), 'predicted_positive': int(p.sum()),
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'precision': precision,
        'recall': recall, 'f_beta': (1 + beta**2) * tp / denominator if denominator else 0.}


def first_management_admission(metrics, unchanged):
    checks = {}
    for action in FIRST_ACTIONS:
        first, all_rows = metrics[action]['first'], metrics[action]['all']
        values = [first[k][v] for k in ['original', 'separate'] for v in ['recall', 'f_beta']]
        values += [all_rows[k]['f_beta'] for k in ['original', 'separate']]
        if not np.isfinite(values).all() or any(not 0 <= v <= 1 for v in values):
            raise ValueError('첫 관리 주문 진단의 지표 범위 오류')
        checks[action] = {
            'first_recall_improved': first['separate']['recall'] > first['original']['recall'] + 1e-12,
            'first_f_improved': first['separate']['f_beta'] > first['original']['f_beta'] + 1e-12,
            'all_f_preserved': all_rows['separate']['f_beta'] >= all_rows['original']['f_beta'] - 1e-12}
    if set(unchanged) != {'exit', 'repeat_reduce', 'repeat_increase'} or any(type(v) is not bool for v in unchanged.values()):
        raise ValueError('첫 관리 주문 진단의 보존 조건 오류')
    return {'first_thresholds_admitted': all(all(v.values()) for v in checks.values()) and all(unchanged.values()),
        'checks': checks, 'unchanged_masks': unchanged, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def first_management_diagnosis(calibration, validation, manager, thresholds, multiplier):
    if multiplier != 1.5 or set(thresholds) != set(ACTIONS):
        raise ValueError('첫 관리 주문 진단의 기존 문턱·배율 오류')
    for part, name in [(calibration, 'calibration'), (validation, 'diagnosis')]:
        first, last = (pd.Timestamp(v, tz='UTC') for v in PERIODS[name])
        if (len(part) < 100 or part.end.min() < first + pd.Timedelta(days=1)
            or part.label_end.max() >= last - pd.Timedelta(days=1) or part.entry_time.min() < first
            or part.end.duplicated().any() or not part.end.is_monotonic_increasing
            or not np.isfinite(part[manager.features]).all().all()):
            raise ValueError('첫 관리 주문 진단의 시간·입력 오류')
        for action in FIRST_ACTIONS:
            flags = part[f'past_{action}_exists']
            if not flags.isin([0, 1]).all():
                raise ValueError('첫 관리 주문의 과거 존재 여부 오류')
            for exists in [0, 1]:
                y = part.loc[flags.eq(exists), f'y_{action}'].to_numpy()
                if not np.isin(y, [0, 1]).all() or min(y.sum(), len(y)-y.sum()) < 20:
                    raise ValueError('첫 관리 주문의 단계별 지원 부족')
    if (set(calibration.episode_id) & set(validation.episode_id)
        or calibration.label_end.max() >= validation.end.min()):
        raise ValueError('첫 관리 주문 진단의 포지션·시간 중첩')
    cx, vx = (part[manager.features].to_numpy() for part in [calibration, validation])
    calibration_scores, scores = manager.probabilities(cx), manager.probabilities(vx)
    for scores_part, part in [(calibration_scores, calibration), (scores, validation)]:
        if (scores_part.shape != (len(part), 3) or not np.isfinite(scores_part).all()
            or ((scores_part < 0) | (scores_part > 1)).any()):
            raise ValueError('첫 관리 주문 진단의 확률 차원·범위 오류')
    first_thresholds, support = {}, {}
    for action in FIRST_ACTIONS:
        i = ACTIONS.index(action)
        mask = calibration[f'past_{action}_exists'].eq(0).to_numpy()
        first_thresholds[action], support[action] = select_threshold(
            calibration.loc[mask, f'y_{action}'], calibration_scores[mask, i], BETAS[action])
    columns = ['end', 'episode_id', *[f'y_{a}' for a in ACTIONS],
               'past_reduce_exists', 'past_increase_exists']
    predictions, metrics, unchanged = validation[columns].copy(), {}, {}
    for i, action in enumerate(ACTIONS):
        original = scores[:, i] >= thresholds[action] * multiplier
        separate = original.copy()
        groups = {'all': np.ones(len(validation), dtype=bool)}
        if action in FIRST_ACTIONS:
            first = validation[f'past_{action}_exists'].eq(0).to_numpy()
            separate[first] = scores[first, i] >= first_thresholds[action] * multiplier
            groups.update(first=first, repeat=~first)
            unchanged[f'repeat_{action}'] = bool(np.array_equal(separate[~first], original[~first]))
        else:
            unchanged['exit'] = bool(np.array_equal(separate, original))
        predictions[f'{action}_score'] = scores[:, i]
        predictions[f'{action}_original'] = original
        predictions[f'{action}_separate'] = separate
        metrics[action] = {name: {kind: decision_metrics(validation.loc[mask, f'y_{action}'], p[mask], BETAS[action])
            for kind, p in [('original', original), ('separate', separate)]} for name, mask in groups.items()}
    return first_thresholds, support, predictions, metrics, first_management_admission(metrics, unchanged)


def run_first_management_diagnosis(selection: Path, diagnosis: Path, output: Path) -> Path:
    frozen, policy = load_selection(selection)
    if frozen['protocol'] != 'history_state_v36':
        raise ValueError('첫 관리 주문 진단에는 고정 v36 후보가 필요합니다.')
    files = json.loads((diagnosis / 'files.json').read_text())
    if any(not (diagnosis / n).resolve().is_relative_to(diagnosis.resolve()) for n in files):
        raise ValueError('첫 관리 주문 진단의 파일 경로 오류')
    verify_files(diagnosis, list(files))
    evidence = json.loads((selection / 'history_admission.json').read_text())
    if sha256(diagnosis / 'files.json') != evidence['files_sha256']:
        raise ValueError('첫 관리 주문 진단의 원본 입력 지문 불일치')
    out = new_run(output, 'first-management-diagnosis', {'selection': str(selection), 'diagnosis': str(diagnosis),
        'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V37.md')), 'new_models_fitted': False,
        'calibration_period': PERIODS['calibration'], 'diagnosis_period': PERIODS['diagnosis']})
    print(f'첫 관리 주문과 반복 주문 문턱 진단: {out}', flush=True)
    try:
        calibration, validation = (pd.read_parquet(diagnosis / f'{n}_used.parquet') for n in ['calibration', 'diagnosis'])
        first, support, predictions, metrics, decision = first_management_diagnosis(
            calibration, validation, policy.manager, policy.thresholds, policy.multiplier)
        for name in ['history_manager.json', 'history_offset.json', 'history_thresholds.json']:
            (out / name).write_bytes((selection / name).read_bytes())
        save_json(out / 'first_thresholds.json', {'thresholds': first, 'support': support,
            'betas': {a: BETAS[a] for a in FIRST_ACTIONS}, 'multiplier': policy.multiplier,
            'calibration_period': PERIODS['calibration'], 'minimum_predicted_positive': 20})
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'metrics.json', metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'calibration_rows': len(calibration),
            'diagnosis_rows': len(validation), 'new_models_fitted': False, **decision})
        rows = [{'action': a, 'phase': phase, 'threshold': kind, **m}
                for a, phases in metrics.items() for phase, comparison in phases.items() for kind, m in comparison.items()]
        (out / 'REPORT.md').write_text('# 첫 관리 주문 문턱의 분리 진단\n\n' + table(pd.DataFrame(rows))
            + '\n\n같은 관리 모형·절편·배율을 유지하고 첫 추가·축소 문턱만 상반기 정답으로 분리했다. '
            '청산과 반복 주문 요청은 그대로다. 원본 상태 고정 진단이며 실제 매매 수익성이나 체결 재현이 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 매매 연구 허용: {decision["first_thresholds_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
