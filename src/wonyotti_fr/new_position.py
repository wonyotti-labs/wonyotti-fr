from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .addition_research import copy_minute_parent
from .common import sha256
from .event_features import MARKET_FEATURES
from .expansion_model import BinaryModel
from .recent_entry import ENTRY_FILES, ENTRY_PERIOD, RecentEntryPolicy, load_recent_entry_selection

POSITION_FILES = ['new_position_activity.json', 'new_position_training.json']


def new_position_targets(data, actions):
    data = data.astype({'end': 'datetime64[ns, UTC]', 'label_end': 'datetime64[ns, UTC]'})
    if (data.end.duplicated().any() or not data.end.is_monotonic_increasing
        or not data.label_end.sub(data.end).eq(pd.Timedelta(minutes=5)).all()
        or (data.end.astype('datetime64[ns, UTC]').array.asi8 % pd.Timedelta(minutes=5).value).any()
        or not actions.time.is_monotonic_increasing
        or not np.isfinite(actions[['before_qty', 'after_qty']]).all().all()
        or not actions.before_qty.iloc[1:].reset_index(drop=True).equals(actions.after_qty.iloc[:-1].reset_index(drop=True))):
        raise ValueError('신규 포지션 목표의 시간·수량 연속성 오류')
    transition = actions.after_qty.ne(0) & (actions.before_qty.eq(0)
        | np.sign(actions.before_qty).ne(np.sign(actions.after_qty)))
    if not transition.equals(actions.action.isin(['open', 'reverse'])):
        raise ValueError('신규 포지션 목표의 행동·수량 전이 불일치')
    starts = actions.loc[transition, ['time', 'episode_id', 'before_qty', 'after_qty', 'action']].copy()
    if starts.episode_id.duplicated().any() or starts.episode_id.le(0).any():
        raise ValueError('신규 포지션 목표의 에피소드 중복·누락')
    starts = starts.rename(columns={'time': 'target_time', 'episode_id': 'target_episode_id'}).reset_index(drop=True)
    starts['target_time'] = starts.target_time.astype('datetime64[ns, UTC]')
    starts['target_event_index'] = np.arange(len(starts))
    starts['buy'] = starts.after_qty.gt(0).astype(int)
    frame = data.drop(columns=['target_time', 'target_episode_id', 'active', 'buy',
                              'expansion_episode_id', 'expansion_count', 'both_directions']).copy()
    frame = pd.merge_asof(frame, starts[['target_time', 'target_episode_id', 'target_event_index', 'buy']],
        left_on='end', right_on='target_time', direction='forward',
        tolerance=pd.Timedelta(minutes=5) - pd.Timedelta(nanoseconds=1))
    frame['active'] = frame.target_time.notna().astype(int)
    frame['target_episode_id'] = frame.target_episode_id.fillna(0).astype(int)
    frame['target_event_index'] = frame.target_event_index.fillna(-1).astype(int)
    first, last = (frame[n].astype('datetime64[ns, UTC]').array.asi8 for n in ['end', 'label_end'])
    counts = []
    for side in [0, 1]:
        times = starts.loc[starts.buy.eq(side), 'target_time'].array.asi8
        counts.append(np.searchsorted(times, last, side='left') - np.searchsorted(times, first, side='left'))
    frame['new_position_count'] = counts[0] + counts[1]
    frame['new_position_both_directions'] = (counts[0] > 0) & (counts[1] > 0)
    starts['window_end'] = starts.target_time.dt.floor('5min')
    ledger = starts.merge(frame[['end', 'usable', 'target_event_index']], left_on='window_end', right_on='end',
                           how='left', validate='many_to_one', suffixes=('', '_selected'))
    ledger['reason'] = np.select([ledger.end.isna(), ledger.usable.ne(True),
        ledger.target_event_index.ne(ledger.target_event_index_selected)],
        ['outside_windows', 'unusable_original_window', 'later_event_in_window'], default='first_supported')
    pd.testing.assert_frame_equal(frame[MARKET_FEATURES + ['end', 'usable', 'episode_id']],
                                  data[MARKET_FEATURES + ['end', 'usable', 'episode_id']].reset_index(drop=True), check_exact=True)
    return frame, ledger


def copy_recent_parent(reference, out):
    copy_minute_parent(reference, out)
    for name in ['minute_selection.json', *ENTRY_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'recent_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_new_position_selection(selection, frozen):
    path = selection / 'recent_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['recent_selection_sha256']:
        raise ValueError('신규 포지션 활동의 기존 진입 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'new_position_v25' or parent.get('protocol') != 'recent_entry_v23'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['position_files_sha256']) != set(POSITION_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['position_files_sha256'][n]
               for n in POSITION_FILES)):
        raise ValueError('신규 포지션 활동의 고정 설정·모델 지문 오류')
    _, original = load_recent_entry_selection(selection, parent)
    activity = BinaryModel.from_dict(json.loads((selection / 'new_position_activity.json').read_text()))
    metadata = json.loads((selection / 'new_position_training.json').read_text())
    if (activity.data['kind'] != 'logistic' or metadata['training_period'] != ENTRY_PERIOD
        or metadata['activity_quantile'] != .975 or metadata['activity_threshold'] != frozen['position_activity_threshold']
        or metadata['direction_unchanged'] is not True):
        raise ValueError('신규 포지션 활동의 학습·방향 보존 설정 오류')
    return frozen, RecentEntryPolicy(original, activity, original.base.direction, frozen['position_activity_threshold'])
