from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from wonyotti_fr.action_model import ActionModels, select_threshold
from wonyotti_fr.entry_boundary_support import utc_times
from wonyotti_fr.event_features import MARKET_FEATURES, teacher_states
from wonyotti_fr.histogram_management import HistogramManagementModels
from wonyotti_fr.minute_management import ACTIONS, FEATURES

PERIODS = {'training': ('2018-01-01', '2021-01-01'),
    'calibration': ('2021-01-01', '2021-07-01'), 'diagnosis': ('2021-07-01', '2022-01-01')}


class PairedManagementModel(HistogramManagementModels):
    features = FEATURES
    format = 'paired_five_minute_management_v1'


class PairedManagementLinear(ActionModels):
    features = FEATURES
    format = 'paired_five_minute_management_linear_v1'

    @classmethod
    def fit(cls, train, calibration, kind='logistic'):
        # 배열 저장 순서에 따른 합산 오차도 재현 가능하도록 고정한다.
        x = np.ascontiguousarray(train[cls.features].to_numpy(dtype=float))
        y = train[[f'y_{a}' for a in ACTIONS]].to_numpy(dtype=int)
        vx = np.ascontiguousarray(calibration[cls.features].to_numpy(dtype=float))
        if (kind != 'logistic' or len(x) < 1000 or not np.isfinite(x).all() or not len(vx)
            or not np.isfinite(vx).all() or not np.isin(y, [0, 1]).all()
            or (y.sum(axis=0) < 20).any() or ((1-y).sum(axis=0) < 20).any()
            or train.label_end.max() >= calibration.end.min()):
            raise ValueError('동일 관리 선형 모형의 지원·입력·기간 오류')
        with threadpool_limits(limits=1):
            scaler = StandardScaler().fit(x)
            learners = [LogisticRegression(C=.1, max_iter=2000, random_state=0).fit(scaler.transform(x), y[:, i]) for i in range(3)]
            if any(m.n_iter_.max() >= 2000 for m in learners):
                raise ValueError('동일 관리 선형 모형 수렴 실패')
            expected = np.column_stack([m.predict_proba(scaler.transform(vx))[:, 1] for m in learners])
        model = cls.from_dict({'format': cls.format, 'kind': 'logistic', 'features': cls.features, 'actions': ACTIONS,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': [m.coef_[0].tolist() for m in learners], 'intercept': [float(m.intercept_[0]) for m in learners]})
        scores = model.probabilities(vx)
        error = float(np.max(np.abs(scores-expected)))
        if error > 1e-12:
            raise ValueError('동일 관리 선형 숫자 내보내기 오류')
        thresholds, details = {}, {}
        for i, action in enumerate(ACTIONS):
            thresholds[action], details[action] = select_threshold(calibration['y_'+action], scores[:, i], (2., 1., .5)[i])
        return model, thresholds, {'rows': len(x), 'validation_export_max_error': error,
            'threshold_selection': details, 'matrix_order': 'C', 'iterations': [int(m.n_iter_[0]) for m in learners]}


def management_frame(windows, filled, requested, events, actions, episodes, bars):
    if not {'end', 'label_end', 'usable', 'state_direction', 'state_episode_id', 'state_time'} <= set(windows):
        raise ValueError('동일 관리 비교의 원본 창 열 누락')
    starts = utc_times(windows.end, unique=True)
    utc_times(requested.time)
    utc_times(bars.end, unique=True)
    if (not windows.label_end.sub(windows.end).eq(pd.Timedelta(minutes=5)).all()
        or requested.order_key.duplicated().any() or not set(MARKET_FEATURES) <= {k.removeprefix('start_market_') for k in windows}
        or episodes.episode_id.duplicated().any()):
        raise ValueError('동일 관리 비교의 창·주문·특징 오류')
    states = teacher_states(actions, events)
    positions = np.searchsorted(utc_times(states.state_time, unique=True), starts, side='left')-1
    source = states.iloc[np.maximum(positions, 0)].reset_index(drop=True)
    frame = windows.reset_index(drop=True).copy()
    known = positions >= 0
    if (not np.array_equal(source.direction.where(known, 0), frame.state_direction.fillna(0))
        or not np.array_equal(source.episode_id.where(known, 0), frame.state_episode_id.fillna(0))):
        raise ValueError('동일 관리 비교의 원본 상태 연결 불일치')
    frame['basis_state_time'] = source.state_time.where(known)
    if not frame.basis_state_time.astype('datetime64[ns, UTC]').equals(frame.state_time.astype('datetime64[ns, UTC]')):
        raise ValueError('동일 관리 비교의 엄격한 이전 상태 시각 불일치')
    for key in ('average_entry', 'entry_time', 'adds'):
        frame['basis_'+key] = source[key].where(known)
    frame['observed_close'] = bars.set_index('end').close.reindex(frame.end).to_numpy()
    frame['direction'] = frame.state_direction.fillna(0).astype('int64')
    frame['favorable_move'] = (frame.observed_close/frame.basis_average_entry.replace(0, np.nan)-1)*frame.direction
    hold = (frame.end-frame.basis_entry_time).dt.total_seconds()/60
    frame['log_hold_minutes'] = np.log1p(hold.clip(lower=0))
    frame['adds_capped'] = frame.basis_adds.clip(upper=5)
    for name in MARKET_FEATURES:
        frame[name] = frame['start_market_'+name]
        frame['directional_'+name] = frame[name]*frame.direction
    first = requested.loc[requested.window_index.ge(0)].drop_duplicates('window_index', keep='first').set_index('window_index')
    for action in ACTIONS:
        frame['first_y_'+action] = first[action].reindex(frame.index, fill_value=False).astype(bool)
        frame['all_y_'+action] = frame['y_requested_'+action].astype(bool)
        if (frame['first_y_'+action] & ~frame['all_y_'+action]).any():
            raise ValueError('첫 관리 정답이 전체 정답에 포함되지 않음')
    frame['paired_part'], frame['paired_reason'] = 'outside_periods', 'outside_periods'
    for name, (begin, finish) in PERIODS.items():
        left, right = pd.Timestamp(begin, tz='UTC'), pd.Timestamp(finish, tz='UTC')
        period = frame.end.ge(left) & frame.end.lt(right)
        crossing = []
        for boundary in (left, right):
            ids = episodes.loc[episodes.entry_time.lt(boundary)
                & (~episodes.closed | episodes.exit_time.isna() | episodes.exit_time.ge(boundary)), 'episode_id']
            linked = filled.loc[(filled.before_episode_id.isin(ids) | filled.after_episode_id.isin(ids)) & filled.window_index.ge(0), 'window_index']
            crossing.append(frame.state_episode_id.isin(ids) | frame.index.isin(linked))
        reasons = np.select([~frame.usable, frame.direction.eq(0), ~np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(axis=1),
            frame.end.lt(left+pd.Timedelta(days=1)), frame.label_end.ge(right-pd.Timedelta(days=1)), crossing[0], crossing[1]],
            ['original_unsupported', 'flat_state', 'nonfinite_state_features', 'left_embargo', 'right_embargo',
                'left_episode_boundary', 'right_episode_boundary'], default='included')
        frame.loc[period, 'paired_part'] = name
        frame.loc[period, 'paired_reason'] = reasons[period]
    frame['paired_usable'] = frame.paired_reason.eq('included')
    return frame


def split_support(frame):
    rows, support = {}, {}
    for part in PERIODS:
        chosen = frame.loc[frame.paired_part.eq(part) & frame.paired_usable].copy().reset_index(drop=True)
        rows[part] = chosen
        counts = {mode: {action: int(chosen[f'{mode}_y_{action}'].sum()) for action in ACTIONS} for mode in ('first', 'all')}
        scopes = ('first', 'all') if part == 'training' else ('all',)
        support[part] = {'rows': len(chosen), 'episodes': int(chosen.state_episode_id.nunique()), 'positive': counts,
            'row_minimum_passed': len(chosen) >= (1000 if part == 'training' else 100),
            'classes_supported': all(20 <= counts[mode][action] <= len(chosen)-20 for mode in scopes for action in ACTIONS)}
    for left, right in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        if (set(rows[left].state_episode_id) & set(rows[right].state_episode_id)
            or len(rows[left]) and len(rows[right]) and rows[left].label_end.max() >= rows[right].end.min()):
            raise ValueError('동일 관리 비교의 기간·에피소드 중첩')
    return rows, {'periods': support, 'supported': all(x['row_minimum_passed'] and x['classes_supported'] for x in support.values()),
        'reasons': {part: {str(k): int(v) for k, v in frame.loc[frame.paired_part.eq(part), 'paired_reason'].value_counts().items()} for part in PERIODS}}


def direct_requests(scores, thresholds):
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any() or set(thresholds) != set(ACTIONS)
        or not np.isfinite(list(thresholds.values())).all() or any(not 0 <= v <= 1 for v in thresholds.values())):
        raise ValueError('직접 관리 요청의 점수·문턱 오류')
    masks = values >= np.array([thresholds[a] for a in ACTIONS])
    chosen = np.select([masks[:, i] for i in range(3)], ACTIONS, default='hold')
    return masks, chosen
