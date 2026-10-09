from __future__ import annotations

import numpy as np
import pandas as pd

from .close_context import (
    ContextCloseModel,
    attach_close_context,
    close_context_source,
    context_close_admission,
)
from .context_position import CONTEXT_FEATURES
from .minute_data import validate_minutes

FLOW_WINDOWS = {'taker_imbalance_5m': 1, 'taker_imbalance_15m': 3, 'taker_imbalance_1h': 12,
    'taker_imbalance_4h': 48, 'taker_imbalance_1d': 288}
FLOW_FEATURES = [*FLOW_WINDOWS, *['directional_'+name for name in FLOW_WINDOWS]]


def validate_flow_matrix(values, features):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(features):
        raise ValueError('청산 체결 방향의 입력 차원 오류')
    raw = matrix[:, [features.index(name) for name in FLOW_WINDOWS]]
    directional = matrix[:, [features.index('directional_'+name) for name in FLOW_WINDOWS]]
    direction = matrix[:, features.index('direction')]
    if (not np.isfinite(raw).all() or (np.abs(raw) > 1).any() or not np.isin(direction, [-1, 1]).all()
        or not np.allclose(directional, raw*direction[:, None], rtol=0, atol=1e-12)):
        raise ValueError('청산 체결 방향의 비율·포지션 부호 오류')


class FlowCloseModel(ContextCloseModel):
    features = ContextCloseModel.features+FLOW_FEATURES
    format = 'flow_cost_weighted_close_v1'

    @classmethod
    def fit(cls, values, target, weights, validation_values):
        for matrix in [values, validation_values]:
            validate_flow_matrix(matrix, cls.features)
        return super().fit(values, target, weights, validation_values)

    def probabilities(self, values):
        validate_flow_matrix(values, self.features)
        return super().probabilities(values)


def flow_features(bars):
    if bars.empty:
        raise ValueError('청산 체결 방향의 시세 누락')
    validate_minutes(bars, 5)
    frame = bars.reset_index(drop=True)
    volume, buy = (frame[name].astype(float) for name in ['volume', 'taker_buy_volume'])
    if not np.isfinite(buy).all() or buy.lt(0).any() or buy.gt(volume).any():
        raise ValueError('청산 체결 방향의 매수 거래량 범위 오류')
    segments = frame.time.diff().ne(pd.Timedelta(minutes=5)).cumsum()
    result = frame[['end']].copy().astype({'end': 'datetime64[ns, UTC]'})
    for name, window in FLOW_WINDOWS.items():
        total = volume.groupby(segments).transform(lambda part, size=window: part.rolling(size, min_periods=size).sum())
        bought = buy.groupby(segments).transform(lambda part, size=window: part.rolling(size, min_periods=size).sum())
        # 무거래 창은 체결 방향이 없으며 준비 부족의 결측과 구분한다.
        value = (2*bought/total.replace(0, np.nan)-1).mask(total.eq(0), 0.)
        known = total.notna()
        if not np.isfinite(value[known]).all() or value[known].abs().gt(1+1e-12).any():
            raise ValueError('청산 체결 방향의 창 합계·비율 범위 오류')
        result[name] = value.clip(-1, 1)
    return result


def close_flow_source(reference):
    bars, proof = close_context_source(reference)
    return bars, {**proof, 'flow_features': FLOW_FEATURES, 'flow_windows': FLOW_WINDOWS,
        'flow_definition': '2_times_sum_taker_buy_base_volume_over_sum_base_volume_minus_1',
        'zero_volume_window_value': 0., 'original_context_features_reconstruction_required': True}


def attach_close_flow(rows, bars):
    if not rows or any(set(FLOW_FEATURES) & set(frame) for frame in rows.values()):
        raise ValueError('청산 체결 방향의 원래 행 누락·중복 입력 오류')
    reconstructed = attach_close_context({name: frame.drop(columns=CONTEXT_FEATURES) for name, frame in rows.items()}, bars)
    last = max(frame.decision_time.max() for frame in rows.values())
    flow = flow_features(bars[bars.end.le(last)].reset_index(drop=True)).set_index('end')
    output = {}
    for name, frame in rows.items():
        pd.testing.assert_frame_equal(reconstructed[name], frame.reset_index(drop=True), check_exact=True)
        extra = flow.loc[frame.decision_time, list(FLOW_WINDOWS)].reset_index(drop=True)
        if not np.isfinite(extra).all().all() or not frame.direction.isin([-1, 1]).all():
            raise ValueError('청산 체결 방향의 연속 시세·준비·방향 지원 부족')
        for feature in FLOW_WINDOWS:
            extra['directional_'+feature] = extra[feature]*frame.direction.to_numpy()
        output[name] = pd.concat([frame.reset_index(drop=True), extra], axis=1)
        pd.testing.assert_frame_equal(output[name].drop(columns=FLOW_FEATURES), frame.reset_index(drop=True), check_exact=True)
    return output


def flow_close_admission(metrics, probability, first, intervals, utility_intervals, context_intervals):
    base = context_close_admission(metrics, probability, first, intervals, utility_intervals, candidate_name='flow')
    checks = dict(base['checks'])
    candidate, previous = probability['flow'], probability['context']
    low = context_intervals['intervals']['paired_difference']['lower']
    checks.update(cost_log_loss_vs_context=candidate['cost_log_loss'] is not None and previous['cost_log_loss'] is not None and candidate['cost_log_loss'] < previous['cost_log_loss']*.99,
        cost_brier_vs_context=candidate['cost_brier'] is not None and previous['cost_brier'] is not None and candidate['cost_brier'] <= previous['cost_brier']+1e-12,
        weighted_regret_vs_context=metrics['flow']['weighted_regret_bps'] < metrics['context']['weighted_regret_bps'],
        first_mean_vs_context=first['flow']['all_position_mean_common_bps'] > first['context']['all_position_mean_common_bps'],
        positive_context_paired_interval_lower=bool(low is not None and np.isfinite(low) and low > 0))
    return {'checks': checks, 'flow_admitted': all(checks.values()), 'trading_returns_evaluated': False}
