from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .episode_balance import load_episode_reference, verified_files
from .management_calibration import ManagementOffset
from .management_diagnostics import management_admission, management_metrics
from .minute_inventory import load_minute_inventory_labels
from .minute_management import ACTIONS, purged_window
from .order_history import HISTORY_FEATURES
from .order_history_boost import OrderHistoryBoostModels
from .reports import table

TRAINING_YEARS = (2018, 2019, 2020)


class HistoryWindowModels(OrderHistoryBoostModels):
    format = 'expanded_history_histogram_v1'


def expanded_splits(frame, history, reference):
    if (history.columns.tolist() != ['end', *HISTORY_FEATURES] or history.end.duplicated().any()
        or history.end.isna().any() or not history.end.is_monotonic_increasing
        or not np.isfinite(history[HISTORY_FEATURES]).all().all() or history[HISTORY_FEATURES].lt(0).any().any()
        or frame.end.isna().any() or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing):
        raise ValueError('관리 학습 이력 확장의 시각·과거 체결 입력 오류')
    indexed = history.set_index('end')
    parts, support, ledgers = {}, {}, []
    windows = {str(y): (f'{y}-01-01', f'{y+1}-01-01') for y in TRAINING_YEARS}
    windows.update({n: PERIODS[n] for n in ['calibration', 'diagnosis']})
    for name, period in windows.items():
        part = purged_window(frame, *period).reset_index(drop=True)
        if not part.end.isin(indexed.index).all():
            raise ValueError('관리 학습 이력 확장의 과거 체결 행 누락')
        part[HISTORY_FEATURES] = indexed.loc[part.end, HISTORY_FEATURES].to_numpy()
        y = part[[f'y_{a}' for a in ACTIONS]].to_numpy()
        ids = part.episode_id.to_numpy()
        if (len(part) < (1000 if name.isdigit() else 100)
            or not np.isfinite(part[HistoryWindowModels.features]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any()
            or part[['end', 'entry_time', 'label_end', 'episode_id']].isna().any().any()
            or not np.isfinite(ids).all() or ((ids <= 0) | (ids >= 2**53) | (ids != np.floor(ids))).any()
            or part.entry_time.ge(part.end).any() or part.end.gt(part.label_end).any()):
            raise ValueError('관리 학습 이력 확장의 연도별 지원·입력 오류')
        if name in ['2020', 'calibration', 'diagnosis']:
            pd.testing.assert_frame_equal(part, reference['training' if name == '2020' else name], check_exact=True)
        parts[name] = part
        support[name] = {'period': list(period), 'rows': len(part), 'episodes': int(part.episode_id.nunique()),
            'positive_minutes': {a: int(part[f'y_{a}'].sum()) for a in ACTIONS},
            'first_end': part.end.min(), 'last_label_end': part.label_end.max()}
        if name.isdigit():
            first, last = (pd.Timestamp(t, tz='UTC') for t in period)
            ledger = frame.loc[frame.end.ge(first) & frame.end.lt(last),
                ['end', 'entry_time', 'label_end', 'episode_id', 'usable']].copy()
            crossing = frame.loc[frame.entry_time.lt(last) & frame.end.ge(last), 'episode_id'].unique()
            ledger['reason'] = np.select([~ledger.usable, ledger.entry_time.isna(),
                ledger.end.lt(first+pd.Timedelta(days=1)), ledger.entry_time.lt(first),
                ledger.label_end.ge(last-pd.Timedelta(days=1)), ledger.episode_id.isin(crossing)],
                ['unusable', 'no_position', 'start_embargo', 'entry_before_period', 'end_embargo', 'crossing_episode'], default='included')
            ledger['included'] = ledger.reason.eq('included')
            if set(ledger.loc[ledger.included, 'end']) != set(part.end):
                raise ValueError('관리 학습 이력 확장의 포함·배제 원장 불일치')
            ledger['year'] = int(name)
            support[name]['exclusions'] = ledger.reason.value_counts().to_dict()
            ledgers.append(ledger)
    names = list(parts)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            a, b = parts[first], parts[second]
            if set(a.episode_id) & set(b.episode_id) or a.label_end.max() >= b.end.min():
                raise ValueError('관리 학습 이력 확장의 시간·포지션 중첩')
    rows = {'training': pd.concat([parts[str(y)] for y in TRAINING_YEARS], ignore_index=True),
            'calibration': parts['calibration'], 'diagnosis': parts['diagnosis']}
    return rows, support, pd.concat(ledgers, ignore_index=True)


def fit_history_window(rows):
    train, calibration, validation = (rows[n] for n in PERIODS)
    x, cx, vx = (v[HistoryWindowModels.features].to_numpy(dtype=float) for v in [train, calibration, validation])
    y, cy = (v[[f'y_{a}' for a in ACTIONS]].to_numpy() for v in [train, calibration])
    model, fitted = HistoryWindowModels.fit(x, y, np.vstack([cx, vx]))
    offset, calibrated = ManagementOffset.fit(model.probabilities(cx), cy)
    raw = model.probabilities(vx)
    return model, offset, {'model': fitted, 'offset': calibrated}, raw, offset.predict(raw)


def history_window_admission(metrics):
    comparison = management_admission({a: {'logistic': m['history_calibrated'],
        'histogram': m['expanded_calibrated'], 'constant': m['constant']} for a, m in metrics.items()})
    preserved = {}
    for a in ACTIONS:
        raw, calibrated = (metrics[a][k]['log_loss'] for k in ['expanded_raw', 'expanded_calibrated'])
        if not np.isfinite([raw, calibrated]).all() or min(raw, calibrated) < 0:
            raise ValueError('관리 학습 이력 확장의 보정 지표 오류')
        preserved[a] = calibrated <= raw
    return {'history_window_admitted': comparison['histogram_admitted'] and all(preserved.values()),
        'comparison': comparison, 'raw_loss_preserved': preserved, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def load_history_window_inputs(source):
    reference, scores = load_episode_reference(source)
    settings = json.loads((source / 'manifest.json').read_text())['settings']
    labels, history = Path(settings['labels']), Path(settings['history'])
    files = verified_files(history)
    if not {'manifest.json', 'summary.json', 'history_features.parquet'} <= set(files):
        raise ValueError('관리 학습 이력 확장의 과거 입력 목록 누락')
    metadata = json.loads((history / 'manifest.json').read_text())['settings']
    if (sha256(labels / 'files.json') != settings['labels_files_sha256']
        or sha256(history / 'files.json') != settings['history_files_sha256']
        or metadata['labels_files_sha256'] != settings['labels_files_sha256']
        or metadata['selection_sha256'] != settings['selection_sha256']
        or metadata['new_features'] != HISTORY_FEATURES
        or any(sha256(Path(metadata['audit']) / n) != h for n, h in metadata['audit_sha256'].items())):
        raise ValueError('관리 학습 이력 확장의 원본·과거 입력 연결 오류')
    frame, _ = load_minute_inventory_labels(labels)
    features = pd.read_parquet(history / 'history_features.parquet')
    rows, support, ledger = expanded_splits(frame, features, reference)
    return rows, scores, support, ledger


def run_history_window_diagnosis(source: Path, output: Path) -> Path:
    out = new_run(output, 'history-window-diagnosis', {'source': str(source),
        'source_files_sha256': sha256(source / 'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V49.md')),
        'training_years': list(TRAINING_YEARS), 'new_model_families': 1, 'trading_returns_evaluated': False})
    print(f'관리 학습 이력 확장 진단: {out}', flush=True)
    try:
        rows, scores, splits, ledger = load_history_window_inputs(source)
        for name, frame in rows.items():
            frame.to_parquet(out / f'{name}_used.parquet', index=False)
        ledger.to_parquet(out / 'training_inclusion.parquet', index=False)
        model, offset, support, raw, calibrated = fit_history_window(rows)
        scores.update(expanded_raw=raw, expanded_calibrated=calibrated)
        validation = rows['diagnosis']
        predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        metrics = {}
        for i, a in enumerate(ACTIONS):
            metrics[a] = {}
            for name, values in scores.items():
                predictions[a+'_'+name] = values[:, i]
                metrics[a][name] = management_metrics(validation['y_'+a], values[:, i])
        decision = history_window_admission(metrics)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        for name, value in [('model', model.to_dict()), ('offset', offset.to_dict()),
            ('training_support', {**support, 'splits': splits}), ('metrics', metrics), ('decision', decision)]:
            save_json(out / f'{name}.json', value)
        save_json(out / 'summary.json', {'complete': True, 'baseline_rows_and_scores_exact': True,
            'training_rows': len(rows['training']), 'episode_intersection': 0, **decision})
        (out / 'REPORT.md').write_text('# 관리 학습 이력의 확장 진단\n\n' + table(pd.DataFrame([
            {'action': a, 'model': n, **m} for a, group in metrics.items() for n, m in group.items()]))
            + '\n\n기존 모형 설정·입력·보정·진단을 유지하고 과거 두 연도의 지원 학습 행만 추가했다. '
            '연도별 경계·포지션 배제를 보존했다. 이미 관찰한 원본 상태의 예측 진단이며 매매 수익성의 증거가 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'확장 이력 관리의 후속 연구 허용: {decision["history_window_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
