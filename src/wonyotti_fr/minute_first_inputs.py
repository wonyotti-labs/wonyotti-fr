from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256
from .first_linear_close import FirstLinearCloseModel
from .managed_first_model import ManagedFirstLinearModel
from .managed_first_parent import pinned_json
from .minute_data import validate_minutes
from .probability_first_model import checked_existing_manager
from .research import load_market

MINUTE_FLOW_FEATURES = ['minute_taker_imbalance_1m', 'directional_minute_taker_imbalance_1m']


class MinuteFirstLinearModel(FirstLinearCloseModel):
    features = [*ManagedFirstLinearModel.features, *MINUTE_FLOW_FEATURES]
    format = 'minute_flow_first_opportunity_cost_logistic_v1'

    @classmethod
    def matrix(cls, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(cls.features) or not np.isfinite(matrix).all():
            raise ValueError('확정 분봉 첫 모델의 입력 차원·유한성 오류')
        ManagedFirstLinearModel.matrix(matrix[:, :-2])
        raw, directional = matrix[:, -2], matrix[:, -1]
        if (np.abs(raw) > 1).any() or not np.allclose(directional, raw*matrix[:, cls.features.index('direction')], rtol=0, atol=1e-12):
            raise ValueError('확정 분봉 첫 모델의 체결 불균형·방향 오류')
        return matrix


def checked_minute_first_parent(reference):
    proof = json.loads((reference/'reference_evidence.json').read_text())
    previous = Path(proof['reference'])
    for name in ['reference_evidence.json', 'manager_evidence.json']:
        pinned_json(previous/name, proof['files'][name])
    manager, parent = checked_existing_manager(previous)
    if parent != json.loads((reference/'manager_evidence.json').read_text()):
        raise ValueError('확정 분봉 첫 모델의 원래 부모 불일치')
    return manager, parent


def load_first_minute_source(reference):
    generation = json.loads((reference/'generation_manifest.json').read_text())['settings']
    market = Path(generation['market'])
    manifest = pinned_json(market/'manifest-1m.json', generation['market_manifest_sha256'])
    if manifest['interval'] != '1m':
        raise ValueError('확정 분봉 첫 입력의 시세 간격 오류')
    bars, _ = load_market(market, 'BTCUSDT', '1m')
    return bars, {'market': str(market), 'symbol': 'BTCUSDT', 'interval': '1m',
        'market_manifest_sha256': generation['market_manifest_sha256'],
        'generation_manifest_sha256': sha256(reference/'generation_manifest.json'),
        'normalized_files': manifest['summary']['BTCUSDT'], 'future_candles_used': False}


def attach_first_minute_flow(first, bars):
    if first.empty or any(name in first for name in MINUTE_FLOW_FEATURES):
        raise ValueError('확정 분봉 첫 입력의 원래 행·중복 특징 오류')
    ManagedFirstLinearModel.matrix(first[ManagedFirstLinearModel.features].to_numpy())
    times = first.decision_time
    if (not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != 'UTC' or times.isna().any()
        or times.duplicated().any() or not times.is_monotonic_increasing
        or (times.astype('datetime64[ns, UTC]').array.asi8 % (60*10**9)).any()):
        raise ValueError('확정 분봉 첫 입력의 UTC·시각 순서·분 경계 오류')
    # 판단 이후의 봉은 입력 검증과 계산에서 함께 제외한다.
    history = bars[bars.end.le(times.max())].reset_index(drop=True)
    for name in ['time', 'end']:
        if not isinstance(history[name].dtype, pd.DatetimeTZDtype) or str(history[name].dt.tz) != 'UTC' or history[name].isna().any():
            raise ValueError('확정 분봉 시세의 UTC 시각 오류')
    validate_minutes(history, 1)
    buy, total = history.taker_buy_volume, history.volume
    if not np.isfinite(buy).all() or buy.lt(0).any() or buy.gt(total).any():
        raise ValueError('확정 분봉 매수 체결량의 유한성·범위 오류')
    indices = pd.Index(history.end).get_indexer(times)
    if (indices < 0).any():
        raise ValueError('첫 판단 시각에 끝난 분봉 누락')
    selected = history.iloc[indices].reset_index(drop=True).copy()
    volume = selected.volume.to_numpy(dtype=float)
    bought = selected.taker_buy_volume.to_numpy(dtype=float)
    imbalance = np.divide(2*bought, volume, out=np.ones(len(selected)), where=volume != 0)-1
    result = first.reset_index(drop=True).copy()
    result[MINUTE_FLOW_FEATURES[0]] = imbalance
    result[MINUTE_FLOW_FEATURES[1]] = imbalance*result.direction.to_numpy()
    MinuteFirstLinearModel.matrix(result[MinuteFirstLinearModel.features].to_numpy())
    selected['input_row'] = np.arange(len(first))
    selected['decision_time'] = result.decision_time
    selected['position_entry_time'] = result.position_entry_time
    return result, selected
