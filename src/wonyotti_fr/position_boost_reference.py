from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from wonyotti_fr import common
from wonyotti_fr.activity_diagnostics import ACTIVITY_PERIODS
from wonyotti_fr.common import sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.first_exit_verification import require_row
from wonyotti_fr.position_target import PositionTargetModel, position_admission, position_metrics
from wonyotti_fr.retained_weight_close_reference import checked_files

POSITION_FILES = {'manifest.json', 'summary.json', 'model.json', 'metrics.json', 'monthly.json',
    'predictions.parquet', 'decision.json', 'training_support.json', 'REPORT.md',
    *[f'{part}_{suffix}.parquet' for part in ['training', 'diagnosis']
      for suffix in ['used', 'ledger', 'position_support']]}
POSITION_REVIEW_CHECKS = ['complete', 'independent_strict_asof_current_and_future_states_exact',
    'period_ledgers_rows_and_position_support_exact', 'independent_sklearn_refit_and_softmax_exact',
    'manual_losses_precision_brier_confusions_months_and_decision_exact']


def read(path):
    return json.loads(path.read_text())


def checked_runtime_sources(root):
    sources = read(root/'manifest.json')['source_sha256']
    current = Path(common.__file__).parent
    if not sources or any(Path(name).name != name or not name.endswith('.py')
        or (root/'code_snapshot'/name).is_symlink() or (current/name).is_symlink()
        or sha256(root/'code_snapshot'/name) != digest or sha256(current/name) != digest
        for name, digest in sources.items()):
        raise ValueError('보유 방향 부스팅의 실행 코드 사본·현재 지문 오류')


def checked_position_reference(reference, expected):
    if reference.is_symlink() or reference.stat().st_size > 1024**2 or sha256(reference) != expected:
        raise ValueError('보유 방향 부스팅의 호출자 출처 지문 오류')
    spec = read(reference)
    if spec.get('profitability_accepted') is not False or spec.get('all_source_periods_already_observed') is not True:
        raise ValueError('보유 방향 부스팅의 출처 판단 범위 오류')
    source, review = Path(spec['source']), Path(spec['review'])
    files = checked_files(source)
    if set(files) != POSITION_FILES or sha256(source/'files.json') != spec['source_files_sha256']:
        raise ValueError('보유 방향 부스팅의 기존 파일 목록·지문 오류')
    if review.is_symlink() or set(spec['review_sha256']) != {'manifest.json', 'verification.json', 'reproduce.py'}:
        raise ValueError('보유 방향 부스팅의 기존 검산 목록 오류')
    for name, digest in spec['review_sha256'].items():
        if (review/name).is_symlink() or sha256(review/name) != digest:
            raise ValueError('보유 방향 부스팅의 기존 검산 지문 오류')
    proof = read(review/'verification.json')
    if (any(proof.get(key) is not True for key in POSITION_REVIEW_CHECKS)
        or proof.get('original_files_unchanged') != 5 or proof.get('position_target_admitted') is not False
        or proof.get('profitability_accepted') is not False
        or read(review/'manifest.json')['settings'] != {'diagnosis': str(source), 'files_sha256': spec['source_files_sha256']}):
        raise ValueError('보유 방향 부스팅의 기존 독립 검산 연결 오류')
    settings, summary = read(source/'manifest.json')['settings'], read(source/'summary.json')
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V44.md'))
        or settings['periods'] != ACTIVITY_PERIODS or settings['new_model_count'] != 1
        or settings['request_threshold'] != .65 or settings['trading_returns_evaluated'] is not False
        or summary['complete'] is not True or summary['position_target_admitted'] is not False
        or summary['profitability_accepted'] is not False):
        raise ValueError('보유 방향 부스팅의 기존 계획·기간·실패 보존 오류')
    return spec, files


def position_comparisons(frame, scores):
    metrics = {name: position_metrics(frame.position_target.to_numpy(), values) for name, values in scores.items()}
    predictions = frame[['end', 'episode_id', 'position_target_episode_id', 'position_target']].copy()
    for name, values in scores.items():
        for index, label in enumerate(PositionTargetModel.classes):
            predictions[name+'_'+label] = values[:, index]
    monthly = []
    for month, part in predictions.groupby(predictions.end.dt.strftime('%Y-%m')):
        item = {'month': month, 'rows': len(part), 'class_counts': np.bincount(part.position_target, minlength=3).tolist(),
            'current_positions': int(part.episode_id[part.episode_id.ne(0)].nunique()),
            'target_positions': int(part.position_target_episode_id[part.position_target_episode_id.ne(0)].nunique())}
        for name in scores:
            item[name+'_log_loss'] = float(log_loss(part.position_target,
                part[[name+'_'+label for label in PositionTargetModel.classes]].to_numpy(), labels=[0, 1, 2]))
        monthly.append(item)
    return predictions, metrics, monthly


def position_baseline(source):
    parts = {name: pd.read_parquet(source/f'{name}_used.parquet') for name in ACTIVITY_PERIODS}
    for name, rows in parts.items():
        first, last = (pd.Timestamp(value, tz='UTC') for value in ACTIVITY_PERIODS[name])
        y = rows.position_target.to_numpy()
        if (len(rows) < (1000 if name == 'training' else 100) or rows.end.duplicated().any()
            or not rows.end.is_monotonic_increasing or rows.end.min() < first + pd.Timedelta(days=1)
            or rows.label_end.max() >= last - pd.Timedelta(days=1)
            or not rows.label_end.sub(rows.end).eq(pd.Timedelta(minutes=5)).all()
            or not np.isfinite(rows[MARKET_FEATURES]).all().all() or y.dtype.kind not in 'iu'
            or set(np.unique(y)) != {0, 1, 2} or (np.bincount(y, minlength=3) < 20).any()):
            raise ValueError('보유 방향 부스팅의 기존 행·시간·정답 지원 오류')
    train, validation = parts['training'], parts['diagnosis']
    sets = [set(frame.episode_id) | set(frame.position_target_episode_id) for frame in parts.values()]
    if (sets[0] & sets[1]) - {0} or train.label_end.max() >= validation.end.min():
        raise ValueError('보유 방향 부스팅의 포지션·시간 중첩')
    model = PositionTargetModel.from_dict(read(source/'model.json'))
    frequencies = np.bincount(train.position_target, minlength=3) / len(train)
    scores = {'position': model.probabilities(validation[MARKET_FEATURES].to_numpy()),
        'constant': np.tile(frequencies, (len(validation), 1))}
    predictions, metrics, monthly = position_comparisons(validation, scores)
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(source/'predictions.parquet'), check_exact=True)
    for value, name in [(metrics, 'metrics'), (monthly, 'monthly'), (position_admission(metrics, monthly), 'decision')]:
        require_row(value, read(source/f'{name}.json'), '기존 보유 방향 '+name)
    np.testing.assert_array_equal(frequencies, read(source/'training_support.json')['training_frequencies'])
    if read(source/'summary.json')['rows'] != {name: len(rows) for name, rows in parts.items()}:
        raise ValueError('보유 방향 부스팅의 기존 전체 행 수 오류')
    return train, validation, scores
