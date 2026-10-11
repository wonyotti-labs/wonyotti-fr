from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.recorded_change_policy import validated_source


def period_bounds(first, last):
    bounds = [pd.Timestamp(v) for v in [first, last]]
    if any(v.tzinfo is None or v.utcoffset().total_seconds() != 0 or v != v.floor('5min') for v in bounds) or bounds[0] >= bounds[1]:
        raise ValueError('행동 정렬 기간의 UTC·경계·순서 오류')
    return bounds


def checked_market(features):
    if (not isinstance(features, pd.DataFrame) or features.columns.duplicated().any()
        or not {'end', *MARKET_FEATURES}.issubset(features.columns)):
        raise ValueError('행동 정렬 시장 열 오류')
    stamps = features.end
    if (not isinstance(stamps.dtype, pd.DatetimeTZDtype) or str(stamps.dt.tz) != 'UTC'
        or stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing
        or not stamps.eq(stamps.dt.floor('5min')).all()):
        raise ValueError('행동 정렬 시장 시각 오류')
    for key in MARKET_FEATURES:
        if (not pd.api.types.is_numeric_dtype(features[key]) or pd.api.types.is_bool_dtype(features[key])
            or pd.api.types.is_complex_dtype(features[key])):
            raise ValueError('행동 정렬 시장 특징 형식 오류')
    return features[['end', *MARKET_FEATURES]].copy()


def alignment_columns():
    return [*[f'{side}_market_{name}' for side in ['start', 'end'] for name in MARKET_FEATURES],
        *[f'{side}_{field}' for side in ['start', 'end'] for field in ['market_present', 'market_finite', 'in_period']],
        'alignment_supported', 'alignment_reason']


def align_market(source, features, first, last):
    first, last = period_bounds(first, last)
    out, market = validated_source(source), checked_market(features)
    if set(alignment_columns()).intersection(out.columns):
        raise ValueError('행동 정렬 기존 열 덮어쓰기 오류')
    book = market.set_index('end')[MARKET_FEATURES]
    supported = out.contract_supported.to_numpy().copy()
    reasons = [[] if ok else ['source:'+reason] for ok, reason in zip(out.contract_supported, out.contract_reason, strict=True)]
    for side, key in [('start', 'end'), ('end', 'label_end')]:
        stamps = out[key]
        values = book.reindex(pd.DatetimeIndex(stamps)).to_numpy(dtype=float)
        present = stamps.isin(market.end).to_numpy()
        finite = np.isfinite(values).all(axis=1)
        inside = ((stamps >= first) & (stamps < last)).to_numpy()
        out[[f'{side}_market_{name}' for name in MARKET_FEATURES]] = values
        out[f'{side}_market_present'] = present
        out[f'{side}_market_finite'] = finite
        out[f'{side}_in_period'] = inside
        supported &= present & finite & inside
        for i, (p, f, valid) in enumerate(zip(present, finite, inside, strict=True)):
            if not p:
                reasons[i].append(side+':missing_market')
            elif not f:
                reasons[i].append(side+':nonfinite_market')
            if not valid:
                reasons[i].append(side+':outside_period')
    out['alignment_supported'] = supported
    out['alignment_reason'] = ['|'.join(parts) or 'included' for parts in reasons]
    return out


def aligned_matrix(frame, timing):
    if timing not in ('window_start', 'window_end'):
        raise ValueError('행동 정렬 학습 시각 오류')
    side = 'start' if timing == 'window_start' else 'end'
    return frame.loc[frame.alignment_supported, [f'{side}_market_{name}' for name in MARKET_FEATURES]].to_numpy(dtype=float)


def verify_alignment(source, features, saved, first, last):
    first, last = period_bounds(first, last)
    original, market = validated_source(source), checked_market(features)
    if set(saved.columns) != set(original.columns) | set(alignment_columns()):
        raise ValueError('행동 정렬 검산 열 누락·추가 오류')
    try:
        pd.testing.assert_frame_equal(saved[original.columns], original, check_exact=True)
    except AssertionError as error:
        raise ValueError('행동 정렬 원본 행 변경') from error
    times = list(market.end)
    values = market[MARKET_FEATURES].to_numpy(dtype=float)
    expected = {name: [] for name in alignment_columns()}
    for source_row in original.itertuples(index=False):
        reasons = [] if source_row.contract_supported else ['source:'+source_row.contract_reason]
        available = bool(source_row.contract_supported)
        for side, stamp in [('start', source_row.end), ('end', source_row.label_end)]:
            at = bisect_left(times, stamp)
            present = at < len(times) and times[at] == stamp
            row = values[at] if present else np.full(len(MARKET_FEATURES), np.nan)
            finite = all(np.isfinite(v) for v in row)
            inside = first <= stamp < last
            for name, value in zip(MARKET_FEATURES, row, strict=True):
                expected[f'{side}_market_{name}'].append(value)
            for name, value in [('market_present', present), ('market_finite', finite), ('in_period', inside)]:
                expected[f'{side}_{name}'].append(value)
            if not present:
                reasons.append(side+':missing_market')
            elif not finite:
                reasons.append(side+':nonfinite_market')
            if not inside:
                reasons.append(side+':outside_period')
            available &= present and finite and inside
        expected['alignment_supported'].append(available)
        expected['alignment_reason'].append('|'.join(reasons) or 'included')
    for name, values in expected.items():
        try:
            np.testing.assert_array_equal(saved[name].to_numpy(), np.asarray(values))
        except AssertionError as error:
            raise ValueError('행동 정렬 검산 불일치: '+name) from error
    return {'complete': True, 'all_rows': len(saved), 'jointly_supported': int(saved.alignment_supported.sum()),
        'all_original_columns_preserved': True, 'independent_exact_time_lookup': True}
