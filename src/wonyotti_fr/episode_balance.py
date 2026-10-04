from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .event_research import load_selection
from .histogram_management import HistogramManagementModels
from .history_calibration_diagnostics import history_calibration_admission
from .inventory_labels import verify_files
from .management_calibration import ManagementOffset
from .management_diagnostics import management_admission, management_metrics
from .minute_inventory import MinuteInventoryModels
from .minute_management import ACTIONS, purged_window
from .order_history import OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels
from .reports import table


def effective_count(weights):
    values = np.asarray(weights, dtype=float)
    return float(values.sum()**2 / (values @ values)) if len(values) else 0.


def episode_weights(episode_ids):
    ids = np.asarray(episode_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
        or (ids <= 0).any()):
        raise ValueError('포지션 균등 가중치의 식별자 오류')
    episodes, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = len(ids) / (len(episodes) * counts[inverse].astype(float))
    sums = np.bincount(inverse, weights=weights)
    if (not np.isfinite(weights).all() or (weights <= 0).any()
        or not np.isclose(weights.mean(), 1., atol=1e-12, rtol=0)
        or not np.allclose(sums, len(ids)/len(episodes), atol=1e-8, rtol=1e-12)
        or effective_count(weights) < 1000):
        raise ValueError('포지션 균등 가중치의 총합·유효 표본 부족')
    return weights, {'format': 'equal_episode_weight_v1', 'rows': len(ids), 'episodes': len(episodes),
        'sum': float(weights.sum()), 'mean': float(weights.mean()), 'minimum': float(weights.min()),
        'maximum': float(weights.max()), 'effective_rows': effective_count(weights),
        'episode_sum_max_error': float(np.max(np.abs(sums-len(ids)/len(episodes))))}


class EpisodeBalanceModels(OrderHistoryBoostModels):
    format = 'episode_balance_histogram_v1'

    @classmethod
    def fit(cls, values, labels, validation_values, episode_ids):
        weights, support = episode_weights(episode_ids)
        if len(weights) != len(values):
            raise ValueError('포지션 가중치와 학습 행 수 불일치')
        model, checks = super().fit(values, labels, validation_values, sample_weight=weights)
        y = np.asarray(labels)
        support['effective_positive'] = {a: effective_count(weights[y[:, i] == 1]) for i, a in enumerate(ACTIONS)}
        support['effective_negative'] = {a: effective_count(weights[y[:, i] == 0]) for i, a in enumerate(ACTIONS)}
        return model, {**checks, 'weighting': support}, weights


def validate_episode_splits(rows):
    support = {}
    for name, period in PERIODS.items():
        frame = rows[name]
        if frame[['end', 'entry_time', 'label_end', 'episode_id']].isna().any().any():
            raise ValueError('포지션 균등 학습의 시각·식별자 누락')
        pd.testing.assert_frame_equal(frame, purged_window(frame, *period).reset_index(drop=True), check_exact=True)
        y = frame[[f'y_{a}' for a in ACTIONS]].to_numpy()
        if (len(frame) < (1000 if name == 'training' else 100)
            or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
            or not np.isfinite(frame[EpisodeBalanceModels.features]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any()
            or frame.entry_time.ge(frame.end).any() or frame.end.gt(frame.label_end).any()):
            raise ValueError('포지션 균등 학습의 기간·정답·입력 지원 부족')
        support[name] = {'rows': len(frame), 'episodes': int(frame.episode_id.nunique()), 'period': list(period)}
    for first, second in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        if (set(rows[first].episode_id) & set(rows[second].episode_id)
            or rows[first].label_end.max() >= rows[second].end.min()):
            raise ValueError('포지션 균등 학습의 시간·포지션 중첩')
    return support


def fit_episode_candidate(rows):
    support = validate_episode_splits(rows)
    train, calibration, validation = (rows[n] for n in PERIODS)
    x, cx, vx = (v[EpisodeBalanceModels.features].to_numpy(dtype=float) for v in [train, calibration, validation])
    y, cy = (v[[f'y_{a}' for a in ACTIONS]].to_numpy() for v in [train, calibration])
    model, fitted, weights = EpisodeBalanceModels.fit(x, y, np.vstack([cx, vx]), train.episode_id.to_numpy())
    offset, calibrated = ManagementOffset.fit(model.probabilities(cx), cy)
    raw = model.probabilities(vx)
    return model, offset, weights, {'splits': support, 'model': fitted, 'offset': calibrated}, raw, offset.predict(raw)


def episode_balance_admission(metrics):
    decision = history_calibration_admission(metrics)
    comparison = management_admission({a: {'logistic': m['history_calibrated'],
        'histogram': m['histogram_calibrated'], 'constant': m['constant']} for a, m in metrics.items()})
    decision['comparisons']['history_calibrated'] = comparison
    decision['episode_balanced_admitted'] = decision.pop('calibrated_histogram_admitted') and comparison['histogram_admitted']
    return decision


def episode_log_loss(frame, scores):
    p = np.clip(np.asarray(scores), np.finfo(float).eps, 1-np.finfo(float).eps)
    result = {}
    for i, a in enumerate(ACTIONS):
        y = frame['y_'+a].to_numpy()
        losses = pd.Series(-y*np.log(p[:, i])-(1-y)*np.log1p(-p[:, i]))
        result[a] = float(losses.groupby(frame.episode_id.to_numpy()).mean().mean())
    return result


def verified_files(root):
    files = json.loads((root / 'files.json').read_text())
    if any(not (root / n).resolve().is_relative_to(root.resolve()) for n in files):
        raise ValueError('포지션 균등 진단의 입력 경로 오류')
    verify_files(root, list(files))
    return files


def load_episode_reference(source):
    files = verified_files(source)
    required = {'manifest.json', 'summary.json', 'models.json', 'offsets.json', 'predictions.parquet',
        *[f'{n}_used.parquet' for n in PERIODS]}
    if not required <= set(files):
        raise ValueError('포지션 균등 진단의 기준 파일 누락')
    settings = json.loads((source / 'manifest.json').read_text())['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V35.md'))
        or settings['periods'] != {k: list(v) for k, v in PERIODS.items()}
        or settings['features'] != EpisodeBalanceModels.features
        or json.loads((source / 'summary.json').read_text()).get('complete') is not True):
        raise ValueError('포지션 균등 진단의 기존 계획·기간 오류')
    rows = {n: pd.read_parquet(source / f'{n}_used.parquet') for n in PERIODS}
    validate_episode_splits(rows)
    selection = Path(settings['selection'])
    if sha256(selection / 'frozen_selection.json') != settings['selection_sha256']:
        raise ValueError('포지션 균등 진단의 원래 관리 지문 오류')
    frozen, original = load_selection(selection)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('포지션 균등 진단의 원래 관리 모형 오류')
    previous = Path(settings['calibration_reference'])
    verified_files(previous)
    if sha256(previous / 'files.json') != settings['calibration_files_sha256']:
        raise ValueError('포지션 균등 진단의 이전 보정 연결 오류')
    models, offsets = (json.loads((source / f'{n}.json').read_text()) for n in ['models', 'offsets'])
    old_model = HistogramManagementModels.from_dict(json.loads((previous / 'models.json').read_text())['histogram'])
    old_offset = ManagementOffset.from_dict(json.loads((previous / 'offsets.json').read_text())['histogram'])
    validation = rows['diagnosis']
    vx = validation[EpisodeBalanceModels.features].to_numpy()
    scores = {'original_v21': original.manager.probabilities(validation[MinuteInventoryModels.features].to_numpy()),
        'previous_calibrated': old_offset.predict(old_model.probabilities(validation[old_model.features].to_numpy())),
        'constant': np.tile(rows['calibration'][[f'y_{a}' for a in ACTIONS]].mean().to_numpy(), (len(vx), 1))}
    for name, cls, label in [('histogram', OrderHistoryBoostModels, 'history_calibrated'), ('logistic', OrderHistoryModels, 'logistic_calibrated')]:
        scores[label] = ManagementOffset.from_dict(offsets[name]).predict(cls.from_dict(models[name]).probabilities(vx))
    predictions = pd.read_parquet(source / 'predictions.parquet')
    columns = ['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]
    pd.testing.assert_frame_equal(predictions[columns], validation[columns], check_exact=True)
    for name, values in scores.items():
        key = 'histogram_calibrated' if name == 'history_calibrated' else name
        np.testing.assert_array_equal(values, predictions[[a+'_'+key for a in ACTIONS]].to_numpy())
    return rows, scores


def run_episode_balance_diagnosis(source: Path, output: Path) -> Path:
    out = new_run(output, 'episode-balance-diagnosis', {'source': str(source),
        'source_files_sha256': sha256(source / 'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V41.md')),
        'weighting': 'N/(E*n_episode)', 'new_model_families': 1, 'trading_returns_evaluated': False})
    print(f'포지션별 학습 기여의 고정 비교: {out}', flush=True)
    try:
        rows, scores = load_episode_reference(source)
        model, offset, weights, support, raw, calibrated = fit_episode_candidate(rows)
        scores.update(histogram_raw=raw, histogram_calibrated=calibrated)
        validation = rows['diagnosis']
        predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        metrics = {}
        for i, a in enumerate(ACTIONS):
            metrics[a] = {}
            for name, values in scores.items():
                predictions[a+'_'+name] = values[:, i]
                metrics[a][name] = management_metrics(validation['y_'+a], values[:, i])
        decision = episode_balance_admission(metrics)
        weighting = rows['training'][['end', 'episode_id']].copy()
        weighting['weight'] = weights
        weighting.to_parquet(out / 'training_weights.parquet', index=False)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        for name, value in [('model', model.to_dict()), ('offset', offset.to_dict()), ('training_support', support),
            ('metrics', metrics), ('decision', decision),
            ('episode_metrics', {n: episode_log_loss(validation, v) for n, v in scores.items()})]:
            save_json(out / f'{name}.json', value)
        save_json(out / 'summary.json', {'complete': True, 'baseline_scores_exact': True, **decision})
        (out / 'REPORT.md').write_text('# 포지션별 학습 기여 균등화 진단\n\n' + table(pd.DataFrame([
            {'action': a, 'model': n, **m} for a, group in metrics.items() for n, m in group.items()]))
            + '\n\n같은 학습 포지션의 분봉 가중 합을 같게 하고 기존 설정·행·정답을 유지했다. '
            '상반기 절편 보정과 하반기 평가는 원래 분봉 기여로 유지했다. 포지션별 평균 손실은 설명 자료이며 사전 주 판정을 대체하지 않는다. '
            '이미 관찰한 원본 상태의 예측 진단이며 봇 수익성을 입증하지 않는다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 연구 허용: {decision["episode_balanced_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
