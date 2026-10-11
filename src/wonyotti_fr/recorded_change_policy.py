from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ['end', 'label_end', 'contract_supported', 'contract_reason',
    'execution_change', 'xbtusd_execution_change']
TARGETS = ('execution_change', 'xbtusd_execution_change')
TIMINGS = ('window_start', 'window_end')


def metadata(target, timing, historical_lookahead=False):
    if target not in TARGETS or timing not in TIMINGS or type(historical_lookahead) is not bool:
        raise ValueError('기록 변경량 진단 설정 오류')
    if historical_lookahead != (timing == 'window_start'):
        raise ValueError('창 시작 진단의 미래 정보 공급 표시 필요')
    return {'kind': 'recorded_change_diagnostic', 'target': target, 'timing': timing,
        'historical_lookahead': historical_lookahead, 'diagnostic_only': True,
        'model_fitted': False, 'source_known_replay': True, 'allocation': .5,
        'zero_threshold': 1e-12, 'initial_original_state_injected': False}


def validated_source(source):
    if (not isinstance(source, pd.DataFrame) or source.columns.duplicated().any()
        or not set(SOURCE_COLUMNS).issubset(source.columns)):
        raise ValueError('기록 변경량 원본 열 오류')
    for key in ['end', 'label_end']:
        series = source[key]
        if (not isinstance(series.dtype, pd.DatetimeTZDtype) or str(series.dt.tz) != 'UTC'
            or series.isna().any() or not series.is_monotonic_increasing or series.duplicated().any()
            or not series.eq(series.dt.floor('5min')).all()):
            raise ValueError('기록 변경량 원본 시각·정렬·경계 오류')
    if not source.label_end.sub(source.end).eq(pd.Timedelta(minutes=5)).all():
        raise ValueError('기록 변경량 원본 창 길이 오류')
    if (source.contract_supported.dtype != bool or source.contract_supported.isna().any()
        or not source.contract_reason.map(lambda x: isinstance(x, str) and bool(x)).all()):
        raise ValueError('기록 변경량 지원·사유 오류')
    for key in TARGETS:
        if (not pd.api.types.is_numeric_dtype(source[key]) or pd.api.types.is_bool_dtype(source[key])
            or pd.api.types.is_complex_dtype(source[key])
            or not np.isfinite(source.loc[source.contract_supported, key].to_numpy(dtype=float)).all()):
            raise ValueError('지원된 기록 변경량의 유한성 오류')
    return source.copy(deep=True).reset_index(drop=True)


class RecordedChangePolicy:
    def __init__(self, source, *, target, timing, historical_lookahead=False):
        self._metadata = metadata(target, timing, historical_lookahead)
        self.source = validated_source(source)
        self.audit_sink = None
        key = 'end' if timing == 'window_start' else 'label_end'
        self._rows = {row[key].isoformat(): row for row in
            self.source[SOURCE_COLUMNS].to_dict('records')}

    def to_dict(self):
        return dict(self._metadata)

    def __call__(self, bar, own):
        end = datetime.fromisoformat(bar['end'])
        if end.tzinfo is None or end.utcoffset().total_seconds() != 0:
            raise ValueError('기록 변경량 판단의 UTC 시각 필요')
        if end.second or end.microsecond or end.minute % 5:
            return None
        equity, quantity, price = own['equity'], own['quantity'], bar['close']
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in [equity, quantity, price])
            or equity <= 0 or price <= 0):
            raise ValueError('기록 변경량 자체 계좌 값 오류')
        current = quantity*price/equity
        if not math.isfinite(current):
            raise ValueError('기록 변경량 자체 노출의 유한성 오류')
        source = self._rows.get(end.isoformat())
        row = {'time': bar['end'], 'source_end': None, 'source_label_end': None, 'available_at': None,
            'lookahead_seconds': 0, 'historical_lookahead': self._metadata['historical_lookahead'],
            'source_supported': None, 'source_reason': None, 'supplied_change': None,
            'equity': equity, 'quantity': quantity, 'current_exposure': current,
            'unclipped_target': None, 'target': None, 'clipped': False, 'reason': 'missing_source'}
        if source is not None:
            row.update(source_end=source['end'].isoformat(), source_label_end=source['label_end'].isoformat(),
                available_at=source['label_end'].isoformat(),
                lookahead_seconds=max(0, int((source['label_end'].to_pydatetime()-end).total_seconds())),
                source_supported=source['contract_supported'], source_reason=source['contract_reason'],
                reason='unsupported_source')
            if source['contract_supported']:
                change = float(source[self._metadata['target']])
                row.update(supplied_change=change, reason='zero_change')
                if abs(change) > 1e-12:
                    desired = current+change
                    if not math.isfinite(desired):
                        raise ValueError('기록 변경량 목표의 유한성 오류')
                    target = max(-.5, min(.5, desired))
                    row.update(unclipped_target=desired, target=target,
                        clipped=target != desired, reason='target_change')
        if self.audit_sink is not None:
            self.audit_sink(row)
        return row['target']
