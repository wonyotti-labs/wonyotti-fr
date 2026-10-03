from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_diagnostics import PERIODS, calibration_admission, run_calibration_diagnostics
from .common import sha256
from .inventory_labels import verify_files
from .inventory_management import CALIBRATION_PERIODS, TRAINING_PERIODS
from .management_diagnostics import management_admission
from .order_history import HISTORY_FEATURES, OrderHistoryModels
from .order_history_boost import OrderHistoryBoostModels


def history_calibration_admission(metrics):
    decision = calibration_admission(metrics)
    comparison = management_admission({a: {'logistic': m['previous_calibrated'],
        'histogram': m['histogram_calibrated'], 'constant': m['constant']} for a, m in metrics.items()})
    decision['comparisons']['previous_calibrated'] = comparison
    decision['calibrated_histogram_admitted'] &= comparison['histogram_admitted']
    decision['history_features_added'] = True
    return decision


def history_calibration_context(history, reference):
    for root in [history, reference]:
        verify_files(root, list(json.loads((root / 'files.json').read_text())))
    previous = json.loads((reference / 'manifest.json').read_text())['settings']
    source = json.loads((history / 'manifest.json').read_text())['settings']
    if (previous['periods'] != {k: list(v) for k, v in PERIODS.items()}
        or source['training_period'] != list(TRAINING_PERIODS[1]) or source['diagnosis_period'] != list(CALIBRATION_PERIODS[1])
        or source['new_features'] != HISTORY_FEATURES
        or source['selection_sha256'] != previous['selection_sha256']
        or source['labels_files_sha256'] != previous['labels_files_sha256']
        or any(sha256(Path(source['audit']) / n) != h for n, h in source['audit_sha256'].items())):
        raise ValueError('주문 이력 보정의 기간·입력·원본 연결 오류')
    features = pd.read_parquet(history / 'history_features.parquet')
    if (features.columns.tolist() != ['end', *HISTORY_FEATURES] or features.end.duplicated().any()
        or features.end.isna().any() or not features.end.is_monotonic_increasing
        or not np.isfinite(features[HISTORY_FEATURES]).all().all() or features[HISTORY_FEATURES].lt(0).any().any()):
        raise ValueError('주문 이력 보정의 과거 특징 시각·숫자 오류')
    return {'reference': reference, 'features': features.set_index('end'),
        'model_class': OrderHistoryModels, 'boost_class': OrderHistoryBoostModels,
        'previous_predictions': pd.read_parquet(reference / 'predictions.parquet'),
        'settings': previous, 'metadata': {'history': str(history), 'calibration_reference': str(reference),
            'history_files_sha256': sha256(history / 'files.json'), 'calibration_files_sha256': sha256(reference / 'files.json')}}


def attach_history_splits(rows, context):
    result = {}
    for name, part in rows.items():
        pd.testing.assert_frame_equal(part, pd.read_parquet(context['reference'] / f'{name}_used.parquet'), check_exact=True)
        features = context['features']
        if not part.end.isin(features.index).all():
            raise ValueError('주문 이력 보정의 기존 분할 시각 누락')
        attached = part.copy()
        attached[HISTORY_FEATURES] = features.loc[part.end, HISTORY_FEATURES].to_numpy()
        pd.testing.assert_frame_equal(attached[part.columns], part, check_exact=True)
        result[name] = attached
    return result


def run_history_calibration_diagnostics(history: Path, reference: Path, output: Path) -> Path:
    context = history_calibration_context(history, reference)
    previous = context['settings']
    return run_calibration_diagnostics(Path(previous['selection']), Path(previous['labels']),
        Path(previous['diagnosis']), output, _history_context=context)
