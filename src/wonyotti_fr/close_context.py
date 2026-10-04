from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_utility import UtilityCloseModel, utility_admission
from .common import sha256
from .context_position import CONTEXT_FEATURES, CONTEXT_WINDOW, context_features
from .event_features import MARKET_FEATURES, event_features
from .minute_data import validate_minutes
from .research import load_market


class ContextCloseModel(UtilityCloseModel):
    features = UtilityCloseModel.features+CONTEXT_FEATURES
    format = 'context_cost_weighted_close_v1'


def close_context_source(reference):
    visited, current = set(), Path(reference)
    for _ in range(16):
        identity = current.resolve()
        if identity in visited:
            raise ValueError('청산 시장 맥락의 원본 참조 순환')
        visited.add(identity)
        settings = json.loads((current/'manifest.json').read_text())['settings']
        if 'labels' in settings:
            labels = Path(settings['labels'])
            files = json.loads((labels/'files.json').read_text())
            if (sha256(labels/'files.json') != settings['labels_files_sha256']
                or (labels/'manifest.json').is_symlink() or sha256(labels/'manifest.json') != files['manifest.json']):
                raise ValueError('청산 시장 맥락의 정답 원본 지문 오류')
            label_settings = json.loads((labels/'manifest.json').read_text())['settings']
            market = Path(label_settings['features'])
            manifest_path = market/'manifest-5m.json'
            if manifest_path.is_symlink() or sha256(manifest_path) != label_settings['feature_manifest_sha256']:
                raise ValueError('청산 시장 맥락의 원래 특징 시세 지문 오류')
            bars, _ = load_market(market, 'BTCUSDT', '5m')
            return bars, {'labels': str(labels), 'labels_files_sha256': sha256(labels/'files.json'),
                'market': str(market), 'market_manifest_sha256': sha256(manifest_path),
                'new_features': CONTEXT_FEATURES, 'warmup_bars': CONTEXT_WINDOW,
                'original_market_features_reconstruction_required': True}
        parent = Path(settings['reference'])
        if settings.get('reference_files_sha256') != sha256(parent/'files.json'):
            raise ValueError('청산 시장 맥락의 이전 결과 연결 오류')
        current = parent
    raise ValueError('청산 시장 맥락의 정답 참조 깊이 초과')


def attach_close_context(rows, bars):
    if not rows or any(frame.empty for frame in rows.values()):
        raise ValueError('청산 시장 맥락의 원래 행 누락')
    last = max(frame.decision_time.max() for frame in rows.values())
    # 마지막 판단 뒤의 시세는 계산 범위에 넣지 않는다.
    history = bars[bars.end.le(last)].reset_index(drop=True)
    validate_minutes(history, 5)
    old = event_features(history).set_index('end')
    extended = context_features(history).set_index('end')
    output = {}
    for name, frame in rows.items():
        if (frame.decision_time.isna().any() or frame.decision_time.duplicated().any()
            or not frame.decision_time.is_monotonic_increasing or set(CONTEXT_FEATURES) & set(frame)
            or not frame.decision_time.isin(old.index).all()):
            raise ValueError('청산 시장 맥락의 확정 판단 시각·중복 입력 오류')
        original = old.loc[frame.decision_time, MARKET_FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, frame[MARKET_FEATURES].reset_index(drop=True), check_exact=True)
        extra = extended.loc[frame.decision_time, CONTEXT_FEATURES].reset_index(drop=True)
        if not np.isfinite(extra).all().all():
            raise ValueError('청산 시장 맥락의 연속 시세·준비 기간 지원 부족')
        output[name] = pd.concat([frame.reset_index(drop=True), extra], axis=1)
        pd.testing.assert_frame_equal(output[name].drop(columns=CONTEXT_FEATURES), frame.reset_index(drop=True), check_exact=True)
    return output


def context_close_admission(metrics, probability, first, intervals, utility_intervals):
    base = utility_admission(metrics, probability, first, intervals, candidate_name='context')
    checks = dict(base['checks'])
    candidate, previous = probability['context'], probability['utility']
    low = utility_intervals['intervals']['paired_difference']['lower']
    checks.update(cost_log_loss_vs_utility=candidate['cost_log_loss'] is not None and previous['cost_log_loss'] is not None and candidate['cost_log_loss'] < previous['cost_log_loss']*.99,
        cost_brier_vs_utility=candidate['cost_brier'] is not None and previous['cost_brier'] is not None and candidate['cost_brier'] <= previous['cost_brier']+1e-12,
        weighted_regret_vs_utility=metrics['context']['weighted_regret_bps'] < metrics['utility']['weighted_regret_bps'],
        first_mean_vs_utility=first['context']['all_position_mean_common_bps'] > first['utility']['all_position_mean_common_bps'],
        positive_utility_paired_interval_lower=bool(low is not None and np.isfinite(low) and low > 0))
    return {'checks': checks, 'context_admitted': all(checks.values()), 'trading_returns_evaluated': False}
